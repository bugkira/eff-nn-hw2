"""Fused ParaTitans Newton: cell + diagonal J + scan (Alg. 1).

``W_x(x)`` is ``(B, T, 5 d_h)`` = ``(α_pre, θ_pre, k, v, n_pre)``. Diagonal
gains are ``u`` (tanh residual) and ``r`` (recurrent mix into tanh).
Gates ``α=σ(α_pre)`` and ``θ=softplus(θ_pre)`` are independent of ``h``:
written once in the App. A init kernel; Newton iters reload them and
still evaluate grad/tanh from ``h``. Full-T ``j_loc``/``r_loc`` stay off
HBM: chunk aggregates only, apply recomputes the local scan (or a single
launch fuses cell+scan+ω when ``T≤BLOCK_T``). CUDA fp16/fp32 (bf16 on
CC ≥ 8.0); cell+scan algebra in fp32.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels._fused_common import (
    _bt_row,
    _softplus,
    _tanh,
    fused_early_exit_hit,
    gather_h0_heads,
    is_seg_head,
    log_fused_done,
    log_fused_iter,
    mark_fused_iters_done,
    prepare_h0_block_table,
    time_tiles,
)
from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors
from pararnn.kernels.scan_diag import scan_chunk_aggregates

log = logging.getLogger(__name__)

_BLOCK_T = 128
_BLOCK_D = 32
_CHUNK_PAD = 64


@triton.jit
def _compose_diag(j_left, r_left, j_right, r_right):
    return j_right * j_left, j_right * r_left + r_right


@triton.jit
def _titans_pred(h_prev, alpha, theta, k, v, n_pre, u, r):
    """App. A / step only — no Jacobian (init path)."""
    grad = (h_prev * k - v) * k
    h_lin = (1.0 - alpha) * h_prev - theta * grad
    n = _tanh(n_pre + r * h_prev)
    return h_lin + u * n


@triton.jit
def _titans_pred_j(h_prev, alpha, theta, k, v, n_pre, u, r):
    """Titans L=1 memory step + diag J from precomputed ``α``, ``θ``."""
    grad = (h_prev * k - v) * k
    h_lin = (1.0 - alpha) * h_prev - theta * grad
    n = _tanh(n_pre + r * h_prev)
    h_new = h_lin + u * n
    n_p = 1.0 - n * n
    j = (1.0 - alpha) - theta * k * k + u * n_p * r
    return h_new, j


@triton.jit
def _load_wx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, sb, st, sd):
    base = wx_ptr + pid_b * sb + offs_t[:, None] * st
    alpha_pre = load_acc(base + offs_d[None, :] * sd, mask, 0.0)
    theta_pre = load_acc(base + (offs_d[None, :] + d_h) * sd, mask, 0.0)
    k = load_acc(base + (offs_d[None, :] + 2 * d_h) * sd, mask, 0.0)
    v = load_acc(base + (offs_d[None, :] + 3 * d_h) * sd, mask, 0.0)
    n_pre = load_acc(base + (offs_d[None, :] + 4 * d_h) * sd, mask, 0.0)
    return alpha_pre, theta_pre, k, v, n_pre


@triton.jit
def _load_kvn(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, sb, st, sd):
    """Load ``(k, v, n_pre)`` only (Newton iters reuse ``α``, ``θ``)."""
    base = wx_ptr + pid_b * sb + offs_t[:, None] * st
    k = load_acc(base + (offs_d[None, :] + 2 * d_h) * sd, mask, 0.0)
    v = load_acc(base + (offs_d[None, :] + 3 * d_h) * sd, mask, 0.0)
    n_pre = load_acc(base + (offs_d[None, :] + 4 * d_h) * sd, mask, 0.0)
    return k, v, n_pre


@triton.jit
def _titans_init_kernel(
    wx_ptr,
    h_ptr,
    alpha_ptr,
    theta_ptr,
    u_ptr,
    r_ptr,
    h0_ptr,
    d_h,
    time,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_ab,
    stride_at,
    stride_ad,
    stride_thb,
    stride_tht,
    stride_thd,
    stride_h0b,
    stride_h0d,
    cs_ptr,
    n_seq,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """App. A guess + write ``α``, ``θ`` once for the Newton K-loop."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    alpha_pre, theta_pre, k, v, n_pre = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    u = load_acc(u_ptr + offs_d, dmask, 0.0)
    r = load_acc(r_ptr + offs_d, dmask, 0.0)
    alpha = tl.sigmoid(alpha_pre)
    theta = _softplus(theta_pre)
    store_acc(
        alpha_ptr + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_d[None, :] * stride_ad,
        alpha,
        mask,
    )
    store_acc(
        theta_ptr
        + pid_b * stride_thb
        + offs_t[:, None] * stride_tht
        + offs_d[None, :] * stride_thd,
        theta,
        mask,
    )
    if HAS_CU:
        h_prev = gather_h0_heads(
            h0_ptr,
            offs_t,
            offs_d,
            dmask,
            cs_ptr,
            n_seq,
            stride_h0b,
            stride_h0d,
            bt_ptr,
            BLOCK_T,
            BLOCK_D,
            HAS_BT,
        )
    else:
        row = _bt_row(pid_b, bt_ptr, HAS_BT)
        h0 = load_acc(h0_ptr + row * stride_h0b + offs_d * stride_h0d, dmask, 0.0)
        h_prev = tl.where((offs_t == 0)[:, None], h0[None, :], 0.0)
    h_new = _titans_pred(h_prev, alpha, theta, k, v, n_pre, u, r)
    store_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        h_new,
        mask,
    )


@triton.jit
def _titans_chunk_jr(
    h_ptr,
    wx_ptr,
    alpha_ptr,
    theta_ptr,
    u_ptr,
    r_ptr,
    h0_ptr,
    pid_b,
    offs_t,
    offs_d,
    time,
    d_h,
    mask,
    dmask,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_h0b,
    stride_h0d,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_ab,
    stride_at,
    stride_ad,
    stride_thb,
    stride_tht,
    stride_thd,
    cs_ptr,
    n_seq,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """Cell + diag J + local inclusive scan for one ``(BLOCK_T, BLOCK_D)`` tile.

    Returns scanned ``(j_s, r_s)`` and the raw Newton residual ``F`` (pre-scan).
    Stays in registers — callers write only aggregates or apply ``ω`` in-place.
    """
    h = load_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        mask,
        0.0,
    )
    offs_tm1 = offs_t - 1
    mask_prev = (offs_tm1[:, None] >= 0) & (offs_tm1[:, None] < time) & (offs_d[None, :] < d_h)
    h_prev = load_acc(
        h_ptr + pid_b * stride_hb + offs_tm1[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        mask_prev,
        0.0,
    )
    row = _bt_row(pid_b, bt_ptr, HAS_BT)
    h0 = load_acc(h0_ptr + row * stride_h0b + offs_d * stride_h0d, dmask, 0.0)
    if HAS_CU:
        head = is_seg_head(offs_t, cs_ptr, n_seq)
        h_prev = tl.where(
            head[:, None],
            gather_h0_heads(
                h0_ptr,
                offs_t,
                offs_d,
                dmask,
                cs_ptr,
                n_seq,
                stride_h0b,
                stride_h0d,
                bt_ptr,
                BLOCK_T,
                BLOCK_D,
                HAS_BT,
            ),
            h_prev,
        )
    else:
        h_prev = tl.where((offs_t == 0)[:, None], h0[None, :], h_prev)
    alpha = load_acc(
        alpha_ptr + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_d[None, :] * stride_ad,
        mask,
        0.0,
    )
    theta = load_acc(
        theta_ptr
        + pid_b * stride_thb
        + offs_t[:, None] * stride_tht
        + offs_d[None, :] * stride_thd,
        mask,
        0.0,
    )
    k, v, n_pre = _load_kvn(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    u = load_acc(u_ptr + offs_d, dmask, 0.0)
    r = load_acc(r_ptr + offs_d, dmask, 0.0)
    h_new, j = _titans_pred_j(h_prev, alpha, theta, k, v, n_pre, u, r)
    residual = h_new - h
    if HAS_CU:
        j = tl.where(head[:, None], 0.0, j)
    j = tl.where(mask, j, 1.0)
    residual = tl.where(mask, residual, 0.0)
    j_s, r_s = tl.associative_scan((j, residual), 0, _compose_diag)
    return j_s, r_s, residual, h


@triton.jit
def _titans_cell_agg_kernel(
    h_ptr,
    wx_ptr,
    alpha_ptr,
    theta_ptr,
    u_ptr,
    r_ptr,
    h0_ptr,
    agg_j_ptr,
    agg_r_ptr,
    res_max_ptr,
    time,
    d_h,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_h0b,
    stride_h0d,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_ab,
    stride_at,
    stride_ad,
    stride_thb,
    stride_tht,
    stride_thd,
    stride_ajb,
    stride_ajc,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ard,
    stride_rmb,
    stride_rmc,
    stride_rmd,
    cs_ptr,
    n_seq,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_BT: tl.constexpr,
    WRITE_RES_MAX: tl.constexpr,
):
    """Newton linearization → chunk aggregates only (no full-T ``j_loc``/``r_loc``)."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    j_s, r_s, residual, _ = _titans_chunk_jr(
        h_ptr,
        wx_ptr,
        alpha_ptr,
        theta_ptr,
        u_ptr,
        r_ptr,
        h0_ptr,
        pid_b,
        offs_t,
        offs_d,
        time,
        d_h,
        mask,
        dmask,
        stride_hb,
        stride_ht,
        stride_hd,
        stride_h0b,
        stride_h0d,
        stride_wb,
        stride_wt,
        stride_wd,
        stride_ab,
        stride_at,
        stride_ad,
        stride_thb,
        stride_tht,
        stride_thd,
        cs_ptr,
        n_seq,
        bt_ptr,
        BLOCK_T,
        BLOCK_D,
        HAS_CU,
        HAS_BT,
    )
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    agg_j = tl.sum(tl.where(last, j_s, 0.0), axis=0)
    agg_r = tl.sum(tl.where(last, r_s, 0.0), axis=0)
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + offs_d * stride_ajd,
        agg_j,
        dmask,
    )
    store_acc(
        agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + offs_d * stride_ard,
        agg_r,
        dmask,
    )
    if WRITE_RES_MAX:
        # Per-tile max|F| (raw residual); host ``amax`` for early-exit.
        tile_max = tl.max(tl.abs(residual))
        tl.store(
            res_max_ptr + pid_b * stride_rmb + pid_c * stride_rmc + pid_d * stride_rmd,
            tile_max,
        )


@triton.jit
def _titans_recompute_apply_kernel(
    h_ptr,
    wx_ptr,
    alpha_ptr,
    theta_ptr,
    u_ptr,
    r_ptr,
    h0_ptr,
    incl_r_ptr,
    time,
    d_h,
    omega,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_h0b,
    stride_h0d,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_ab,
    stride_at,
    stride_ad,
    stride_thb,
    stride_tht,
    stride_thd,
    stride_ib,
    stride_ic,
    stride_id,
    cs_ptr,
    n_seq,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_BT: tl.constexpr,
    HAS_CARRY: tl.constexpr,
):
    """Recompute local ``(j,r)`` scan and apply ``h += ω (J_loc @ carry + r_loc)``.

    Cell ALU runs twice per Newton iter when ``n_chunks>1`` (once for
    aggregates, once here); full-T ``j_loc``/``r_loc`` stay in registers.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    j_s, r_s, _, h = _titans_chunk_jr(
        h_ptr,
        wx_ptr,
        alpha_ptr,
        theta_ptr,
        u_ptr,
        r_ptr,
        h0_ptr,
        pid_b,
        offs_t,
        offs_d,
        time,
        d_h,
        mask,
        dmask,
        stride_hb,
        stride_ht,
        stride_hd,
        stride_h0b,
        stride_h0d,
        stride_wb,
        stride_wt,
        stride_wd,
        stride_ab,
        stride_at,
        stride_ad,
        stride_thb,
        stride_tht,
        stride_thd,
        cs_ptr,
        n_seq,
        bt_ptr,
        BLOCK_T,
        BLOCK_D,
        HAS_CU,
        HAS_BT,
    )
    if HAS_CARRY:
        idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
        carry = load_acc(
            incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + offs_d * stride_id,
            dmask,
            0.0,
        )
        carry = tl.where(pid_c > 0, carry, 0.0)
        delta = j_s * carry[None, :] + r_s
    else:
        delta = r_s
    store_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        h + omega * delta,
        mask,
    )


@triton.jit
def _titans_cell_update_kernel(
    h_ptr,
    wx_ptr,
    alpha_ptr,
    theta_ptr,
    u_ptr,
    r_ptr,
    h0_ptr,
    time,
    d_h,
    omega,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_h0b,
    stride_h0d,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_ab,
    stride_at,
    stride_ad,
    stride_thb,
    stride_tht,
    stride_thd,
    cs_ptr,
    n_seq,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """Single-chunk path: cell + local scan + ``ω`` update with no HBM ``j``/``r``."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    _, r_s, _, h = _titans_chunk_jr(
        h_ptr,
        wx_ptr,
        alpha_ptr,
        theta_ptr,
        u_ptr,
        r_ptr,
        h0_ptr,
        pid_b,
        offs_t,
        offs_d,
        time,
        d_h,
        mask,
        dmask,
        stride_hb,
        stride_ht,
        stride_hd,
        stride_h0b,
        stride_h0d,
        stride_wb,
        stride_wt,
        stride_wd,
        stride_ab,
        stride_at,
        stride_ad,
        stride_thb,
        stride_tht,
        stride_thd,
        cs_ptr,
        n_seq,
        bt_ptr,
        BLOCK_T,
        BLOCK_D,
        HAS_CU,
        HAS_BT,
    )
    store_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        h + omega * r_s,
        mask,
    )


def _newton_titans_fused_impl(
    wx: Tensor,
    u: Tensor,
    r: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    early_exit_atol: float | None = None,
    residual_history: list[float] | None = None,
    iters_done_out: list[int] | None = None,
) -> Tensor:
    """Alg. 1 for diagonal ParaTitans. Public entry: ``pararnn::newton_titans_fused``.

    Intermediate full-T ``j_loc``/``r_loc`` are never materialized: chunk
    aggregates stay in HBM (``T/BLOCK_T`` smaller); the apply kernel recomputes
    the local scan. Single-chunk solves fuse cell+scan+ω in one launch.
    """
    wx = wx.contiguous()
    u = u.contiguous()
    r = r.contiguous()
    batch, time, five_d = wx.shape
    d_h = u.numel()
    if r.numel() != d_h:
        raise ValueError(f"r numel {r.numel()} != u numel {d_h}")
    if five_d != 5 * d_h:
        raise ValueError(f"wx last dim {five_d} != 5 * d_h={5 * d_h}")
    has_cu = cu_seqlens is not None
    if has_cu:
        if batch != 1:
            raise ValueError(f"cu_seqlens packs wx with batch=1, got batch={batch}")
        cs = cu_seqlens.to(device=wx.device, dtype=torch.int32).contiguous()
        if cs.dim() != 1 or int(cs[0]) != 0 or int(cs[-1]) != time:
            raise ValueError(
                f"cu_seqlens must be (S+1,) with [0]=0 and [-1]=time={time}, got {tuple(cs.shape)}"
            )
        n_seq = int(cs.numel()) - 1
        h0, bt, has_bt = prepare_h0_block_table(wx, h0, n_seq, (d_h,), block_table)
    else:
        cs = u
        n_seq = 0
        h0, bt, has_bt = prepare_h0_block_table(wx, h0, batch, (d_h,), block_table)
    validate_cuda_tensors(wx, u, r, h0, name="newton_titans_fused")
    n_chunks, n_dtiles = time_tiles(time, d_h, _BLOCK_T, _BLOCK_D, _CHUNK_PAD, cap=False)
    h = wx.new_empty(batch, time, d_h)
    alpha_gate = wx.new_empty(batch, time, d_h)
    theta_gate = wx.new_empty(batch, time, d_h)
    grid_td = (batch, n_chunks, n_dtiles)
    _titans_init_kernel[grid_td](
        wx,
        h,
        alpha_gate,
        theta_gate,
        u,
        r,
        h0,
        d_h,
        time,
        *wx.stride(),
        *h.stride(),
        *alpha_gate.stride(),
        *theta_gate.stride(),
        h0.stride(0),
        h0.stride(1),
        cs,
        n_seq,
        bt,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        HAS_CU=has_cu,
        HAS_BT=has_bt,
    )
    if max_iters <= 0:
        return h

    need_res_max = early_exit_atol is not None or residual_history is not None
    omega_f = float(omega)
    # Single-chunk + fixed K: one launch, zero intermediate HBM for (j,r).
    fuse_one = n_chunks == 1 and not need_res_max
    if fuse_one:
        for it in range(max_iters):
            _titans_cell_update_kernel[grid_td](
                h,
                wx,
                alpha_gate,
                theta_gate,
                u,
                r,
                h0,
                time,
                d_h,
                omega_f,
                *h.stride(),
                h0.stride(0),
                h0.stride(1),
                *wx.stride(),
                *alpha_gate.stride(),
                *theta_gate.stride(),
                cs,
                n_seq,
                bt,
                BLOCK_T=_BLOCK_T,
                BLOCK_D=_BLOCK_D,
                HAS_CU=has_cu,
                HAS_BT=has_bt,
            )
            log_fused_iter(
                log,
                "newton_titans_fused_iter",
                it=it,
                time=time,
                batch=batch,
                d_h=d_h,
                n_chunks=n_chunks,
            )
        mark_fused_iters_done(iters_done_out, max_iters)
        log_fused_done(
            log,
            "newton_titans_fused",
            time=time,
            batch=batch,
            d_h=d_h,
            max_iters=max_iters,
            n_chunks=n_chunks,
            recompute_jr=False,
            fuse_one=True,
        )
        return h

    agg_j = h.new_empty(batch, n_chunks, d_h)
    agg_r = h.new_empty(batch, n_chunks, d_h)
    incl_r = h.new_empty(batch, n_chunks, d_h) if n_chunks > 1 else h  # dummy when no carry
    res_max = (
        h.new_empty(batch, n_chunks, n_dtiles, dtype=torch.float32) if need_res_max else h
    )
    has_carry = n_chunks > 1

    for it in range(max_iters):
        _titans_cell_agg_kernel[grid_td](
            h,
            wx,
            alpha_gate,
            theta_gate,
            u,
            r,
            h0,
            agg_j,
            agg_r,
            res_max,
            time,
            d_h,
            *h.stride(),
            h0.stride(0),
            h0.stride(1),
            *wx.stride(),
            *alpha_gate.stride(),
            *theta_gate.stride(),
            *agg_j.stride(),
            *agg_r.stride(),
            res_max.stride(0) if need_res_max else 0,
            res_max.stride(1) if need_res_max else 0,
            res_max.stride(2) if need_res_max else 0,
            cs,
            n_seq,
            bt,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            HAS_CU=has_cu,
            HAS_BT=has_bt,
            WRITE_RES_MAX=need_res_max,
        )
        log_fused_iter(
            log,
            "newton_titans_fused_iter",
            it=it,
            time=time,
            batch=batch,
            d_h=d_h,
            n_chunks=n_chunks,
        )
        if need_res_max and fused_early_exit_hit(
            res_max,
            early_exit_atol,
            iters_done_out=iters_done_out,
            it=it,
            residual_history=residual_history,
        ):
            log_fused_done(
                log,
                "newton_titans_fused",
                time=time,
                batch=batch,
                d_h=d_h,
                max_iters=it,
                n_chunks=n_chunks,
                early_exit=True,
                recompute_jr=True,
            )
            return h
        if has_carry:
            incl_r = scan_chunk_aggregates(agg_j, agg_r, out=incl_r)
        _titans_recompute_apply_kernel[grid_td](
            h,
            wx,
            alpha_gate,
            theta_gate,
            u,
            r,
            h0,
            incl_r,
            time,
            d_h,
            omega_f,
            *h.stride(),
            h0.stride(0),
            h0.stride(1),
            *wx.stride(),
            *alpha_gate.stride(),
            *theta_gate.stride(),
            *incl_r.stride(),
            cs,
            n_seq,
            bt,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            HAS_CU=has_cu,
            HAS_BT=has_bt,
            HAS_CARRY=has_carry,
        )
    mark_fused_iters_done(iters_done_out, max_iters)
    log_fused_done(
        log,
        "newton_titans_fused",
        time=time,
        batch=batch,
        d_h=d_h,
        max_iters=max_iters,
        n_chunks=n_chunks,
        recompute_jr=True,
        fuse_one=False,
    )
    return h

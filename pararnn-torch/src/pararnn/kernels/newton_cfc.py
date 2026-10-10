"""Fused ParaCfC Newton: cell + diagonal J + scan (Alg. 1).

``project_wx(x)`` is ``(B, T, 3 d_h)`` = ``(f_pre, c_x, b)`` where ``b`` is
the input-conditioned time bias ``W_b(feat)``. Per-step ``Δt`` is a separate
``(B, T)`` tensor (broadcast inside the kernel). Recurrent mix is diagonal
``u``. Gate ``a=σ(-(softplus(f)·Δt+b))`` is independent of ``h`` for
``gate_mix='input'``: written once in the App. A init kernel; Newton iters
reload ``a`` and evaluate only ``tanh``. Full-T ``j_loc``/``r_loc`` stay off
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

# Same tiles as scan_diag: 128 × 32 × 2 × 4 B = 32 KiB for (J, r).
_BLOCK_T = 128
_BLOCK_D = 32
# Leaf pad matches scan_diag._CHUNK_PAD (third-level superchunks past 8192).
_CHUNK_PAD = 64


@triton.jit
def _compose_diag(j_left, r_left, j_right, r_right):
    return j_right * j_left, j_right * r_left + r_right


@triton.jit
def _cfc_gate_a(f_pre, b, dt):
    """``a = σ(-(softplus(f)·Δt + b))`` — independent of ``h`` when gate_mix=input."""
    return tl.sigmoid(-(_softplus(f_pre) * dt + b))


@triton.jit
def _cfc_pred(h_prev, a, cx, u, NO_GATE: tl.constexpr):
    """App. A / step only — no Jacobian (init path)."""
    n = _tanh(cx + u * h_prev)
    if NO_GATE:
        return a * h_prev + n
    return a * h_prev + (1.0 - a) * n


@triton.jit
def _cfc_pred_j(h_prev, a, cx, u, NO_GATE: tl.constexpr):
    """CfC step+J from precomputed gate ``a``. Hot Newton path: tanh only."""
    n = _tanh(cx + u * h_prev)
    n_p = 1.0 - n * n
    if NO_GATE:
        h_new = a * h_prev + n
        j = a + n_p * u
    else:
        h_new = a * h_prev + (1.0 - a) * n
        j = a + (1.0 - a) * n_p * u
    return h_new, j


@triton.jit
def _load_wx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, sb, st, sd):
    """Load ``(f_pre, c_x, b)`` from 3-wide pack."""
    base = wx_ptr + pid_b * sb + offs_t[:, None] * st
    f_pre = load_acc(base + offs_d[None, :] * sd, mask, 0.0)
    cx = load_acc(base + (offs_d[None, :] + d_h) * sd, mask, 0.0)
    b = load_acc(base + (offs_d[None, :] + 2 * d_h) * sd, mask, 0.0)
    return f_pre, cx, b


@triton.jit
def _load_cx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, sb, st, sd):
    """Load candidate pre-activation ``c_x`` only (Newton iters reuse gate ``a``)."""
    base = wx_ptr + pid_b * sb + offs_t[:, None] * st
    return load_acc(base + (offs_d[None, :] + d_h) * sd, mask, 0.0)


@triton.jit
def _load_dt(dt_ptr, pid_b, offs_t, time, sb, st):
    """Load per-step Δt once per time row; broadcasts over ``BLOCK_D``."""
    mask_t = offs_t < time
    return load_acc(dt_ptr + pid_b * sb + offs_t[:, None] * st, mask_t[:, None], 1.0)


@triton.jit
def _cfc_init_kernel(
    wx_ptr,
    dt_ptr,
    h_ptr,
    a_ptr,
    u_ptr,
    h0_ptr,
    d_h,
    time,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_dtb,
    stride_dtt,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_ab,
    stride_at,
    stride_ad,
    stride_h0b,
    stride_h0d,
    cs_ptr,
    n_seq,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_BT: tl.constexpr,
    NO_GATE: tl.constexpr,
):
    """App. A guess + write gate ``a`` once for the Newton K-loop."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    f_pre, cx, b = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    dt = _load_dt(dt_ptr, pid_b, offs_t, time, stride_dtb, stride_dtt)
    u = load_acc(u_ptr + offs_d, dmask, 0.0)
    a = _cfc_gate_a(f_pre, b, dt)
    store_acc(
        a_ptr + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_d[None, :] * stride_ad,
        a,
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
    h_new = _cfc_pred(h_prev, a, cx, u, NO_GATE)
    store_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        h_new,
        mask,
    )


@triton.jit
def _cfc_chunk_jr(
    h_ptr,
    wx_ptr,
    a_ptr,
    u_ptr,
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
    cs_ptr,
    n_seq,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_BT: tl.constexpr,
    NO_GATE: tl.constexpr,
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
    a = load_acc(
        a_ptr + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_d[None, :] * stride_ad,
        mask,
        0.0,
    )
    cx = _load_cx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd)
    u = load_acc(u_ptr + offs_d, dmask, 0.0)
    h_new, j = _cfc_pred_j(h_prev, a, cx, u, NO_GATE)
    residual = h_new - h
    if HAS_CU:
        j = tl.where(head[:, None], 0.0, j)
    j = tl.where(mask, j, 1.0)
    residual = tl.where(mask, residual, 0.0)
    j_s, r_s = tl.associative_scan((j, residual), 0, _compose_diag)
    return j_s, r_s, residual, h


@triton.jit
def _cfc_cell_agg_kernel(
    h_ptr,
    wx_ptr,
    a_ptr,
    u_ptr,
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
    NO_GATE: tl.constexpr,
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
    j_s, r_s, residual, _ = _cfc_chunk_jr(
        h_ptr,
        wx_ptr,
        a_ptr,
        u_ptr,
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
        cs_ptr,
        n_seq,
        bt_ptr,
        BLOCK_T,
        BLOCK_D,
        HAS_CU,
        HAS_BT,
        NO_GATE,
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
def _cfc_recompute_apply_kernel(
    h_ptr,
    wx_ptr,
    a_ptr,
    u_ptr,
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
    NO_GATE: tl.constexpr,
    HAS_CARRY: tl.constexpr,
):
    """Recompute local ``(j,r)`` scan and apply ``h += ω (J_loc @ carry + r_loc)``.

    Avoids reading full-T ``j_loc``/``r_loc`` from HBM; cell ALU runs twice per
    Newton iter when ``n_chunks>1`` (once for aggregates, once here).
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
    j_s, r_s, _, h = _cfc_chunk_jr(
        h_ptr,
        wx_ptr,
        a_ptr,
        u_ptr,
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
        cs_ptr,
        n_seq,
        bt_ptr,
        BLOCK_T,
        BLOCK_D,
        HAS_CU,
        HAS_BT,
        NO_GATE,
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
def _cfc_cell_update_kernel(
    h_ptr,
    wx_ptr,
    a_ptr,
    u_ptr,
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
    cs_ptr,
    n_seq,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_BT: tl.constexpr,
    NO_GATE: tl.constexpr,
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
    _, r_s, _, h = _cfc_chunk_jr(
        h_ptr,
        wx_ptr,
        a_ptr,
        u_ptr,
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
        cs_ptr,
        n_seq,
        bt_ptr,
        BLOCK_T,
        BLOCK_D,
        HAS_CU,
        HAS_BT,
        NO_GATE,
    )
    store_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        h + omega * r_s,
        mask,
    )


@triton.jit
def _cfc_residual_max_kernel(
    h_ptr,
    wx_ptr,
    dt_ptr,
    a_ptr,
    u_ptr,
    h0_ptr,
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
    stride_dtb,
    stride_dtt,
    stride_ab,
    stride_at,
    stride_ad,
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
    HAS_A: tl.constexpr,
    NO_GATE: tl.constexpr,
):
    """Post-solve tile ``max |F(H*)|`` for ``residual_fail`` / ``NewtonStats``.

    ``HAS_A``: reload gate from ``a_ptr`` (end of fused Newton, ``a_gate`` live).
    Else recompute ``a`` from ``wx``+``Δt`` (after ``custom_op`` returns).
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
    if HAS_A:
        a = load_acc(
            a_ptr + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_d[None, :] * stride_ad,
            mask,
            0.0,
        )
        cx = _load_cx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd)
    else:
        f_pre, cx, b = _load_wx(
            wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
        )
        dt = _load_dt(dt_ptr, pid_b, offs_t, time, stride_dtb, stride_dtt)
        a = _cfc_gate_a(f_pre, b, dt)
    u = load_acc(u_ptr + offs_d, dmask, 0.0)
    h_new = _cfc_pred(h_prev, a, cx, u, NO_GATE)
    residual = tl.where(mask, h_new - h, 0.0)
    tl.store(
        res_max_ptr + pid_b * stride_rmb + pid_c * stride_rmc + pid_d * stride_rmd,
        tl.max(tl.abs(residual)),
    )


def cfc_residual_amax(
    h: Tensor,
    wx: Tensor,
    u: Tensor,
    dt: Tensor,
    *,
    h0: Tensor | None = None,
    a: Tensor | None = None,
    no_gate: bool = False,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    """``max |F(H*)|`` as a 0-d CUDA tensor (no host sync).

    Prefer ``a=`` (precomputed gate) when still live inside the Newton impl.
    After ``custom_op``, omit ``a`` and recompute the gate from ``wx``+``Δt``.
    Callers that need a Python ``float`` take ``.item()`` in ``_fill_stats``.
    """
    wx = wx.contiguous()
    u = u.contiguous()
    h = h.contiguous()
    if dt.ndim == 3 and dt.shape[-1] == 1:
        dt = dt.squeeze(-1)
    if dt.ndim != 2:
        raise ValueError(f"dt must be (B, T) or (B, T, 1), got {tuple(dt.shape)}")
    dt = dt.contiguous()
    batch, time, three_d = wx.shape
    d_h = u.numel()
    if three_d != 3 * d_h:
        raise ValueError(f"wx last dim {three_d} != 3 * d_h={3 * d_h}")
    if h.shape != (batch, time, d_h):
        raise ValueError(f"h shape {tuple(h.shape)} != {(batch, time, d_h)}")
    if dt.shape != (batch, time):
        raise ValueError(f"dt shape {tuple(dt.shape)} != wx batch/time {(batch, time)}")
    has_a = a is not None
    if has_a:
        a = a.contiguous()
        if a.shape != (batch, time, d_h):
            raise ValueError(f"a shape {tuple(a.shape)} != {(batch, time, d_h)}")
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
    validate_cuda_tensors(wx, u, dt, h, h0, name="cfc_residual_amax")
    n_chunks, n_dtiles = time_tiles(time, d_h, _BLOCK_T, _BLOCK_D, _CHUNK_PAD, cap=False)
    res_max = h.new_empty(batch, n_chunks, n_dtiles, dtype=torch.float32)
    a_ptr = a if has_a else h
    grid_td = (batch, n_chunks, n_dtiles)
    _cfc_residual_max_kernel[grid_td](
        h,
        wx,
        dt,
        a_ptr,
        u,
        h0,
        res_max,
        time,
        d_h,
        *h.stride(),
        h0.stride(0),
        h0.stride(1),
        *wx.stride(),
        dt.stride(0),
        dt.stride(1),
        *a_ptr.stride(),
        *res_max.stride(),
        cs,
        n_seq,
        bt,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        HAS_CU=has_cu,
        HAS_BT=has_bt,
        HAS_A=has_a,
        NO_GATE=bool(no_gate),
    )
    return res_max.amax()


def _cfc_fill_residual_max_out(
    residual_max_out: list[float | Tensor] | None,
    h: Tensor,
    wx: Tensor,
    u: Tensor,
    dt: Tensor,
    h0: Tensor,
    a_gate: Tensor,
    no_gate: bool,
    cu_seqlens: Tensor | None,
) -> Tensor:
    """Append final ``max |F|`` (0-d tensor); ``.item()`` waits in ``_fill_stats``."""
    if residual_max_out is not None:
        residual_max_out.clear()
        residual_max_out.append(
            cfc_residual_amax(
                h,
                wx,
                u,
                dt,
                h0=h0,
                a=a_gate,
                no_gate=no_gate,
                cu_seqlens=cu_seqlens,
                block_table=None,
            )
        )
    return h


def _newton_cfc_fused_impl(
    wx: Tensor,
    u: Tensor,
    dt: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    early_exit_atol: float | None = None,
    residual_history: list[float] | None = None,
    iters_done_out: list[int] | None = None,
    no_gate: bool = False,
    h_guess: Tensor | None = None,
    residual_max_out: list[float | Tensor] | None = None,
) -> Tensor:
    """Alg. 1 for diagonal ParaCfC. Public entry: ``pararnn::newton_cfc_fused``.

    Intermediate full-T ``j_loc``/``r_loc`` are never materialized: chunk
    aggregates stay in HBM (``T/BLOCK_T`` smaller); the apply kernel recomputes
    the local scan. Single-chunk solves fuse cell+scan+ω in one launch.

    ``h_guess`` (lab/experimental): after App. A / gate init, overwrite ``h``
    with a precomputed warm start of shape ``(B, T, d_h)`` before Newton.

    ``residual_max_out``: when set, append final ``max |F(H*)|`` using live
    ``a_gate`` (cheap fail-rail for callers that bypass ``custom_op``).
    """
    wx = wx.contiguous()
    u = u.contiguous()
    if dt.ndim == 3 and dt.shape[-1] == 1:
        dt = dt.squeeze(-1)
    if dt.ndim != 2:
        raise ValueError(f"dt must be (B, T) or (B, T, 1), got {tuple(dt.shape)}")
    dt = dt.contiguous()
    batch, time, three_d = wx.shape
    d_h = u.numel()
    if three_d != 3 * d_h:
        raise ValueError(f"wx last dim {three_d} != 3 * d_h={3 * d_h}")
    if dt.shape != (batch, time):
        raise ValueError(f"dt shape {tuple(dt.shape)} != wx batch/time {(batch, time)}")
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
    validate_cuda_tensors(wx, u, dt, h0, name="newton_cfc_fused")
    n_chunks, n_dtiles = time_tiles(time, d_h, _BLOCK_T, _BLOCK_D, _CHUNK_PAD, cap=False)
    h = wx.new_empty(batch, time, d_h)
    # gate_mix=input: a_t = σ(-(softplus(f)·Δt+b)) is independent of h — once per solve.
    a_gate = wx.new_empty(batch, time, d_h)
    grid_td = (batch, n_chunks, n_dtiles)
    no_gate_c = bool(no_gate)
    _cfc_init_kernel[grid_td](
        wx,
        dt,
        h,
        a_gate,
        u,
        h0,
        d_h,
        time,
        *wx.stride(),
        dt.stride(0),
        dt.stride(1),
        *h.stride(),
        *a_gate.stride(),
        h0.stride(0),
        h0.stride(1),
        cs,
        n_seq,
        bt,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        HAS_CU=has_cu,
        HAS_BT=has_bt,
        NO_GATE=no_gate_c,
    )
    if h_guess is not None:
        if h_guess.shape != h.shape:
            raise ValueError(f"h_guess shape {tuple(h_guess.shape)} != {tuple(h.shape)}")
        h.copy_(h_guess.contiguous())
    if max_iters <= 0:
        return _cfc_fill_residual_max_out(
            residual_max_out, h, wx, u, dt, h0, a_gate, no_gate_c, cu_seqlens
        )

    need_res_max = early_exit_atol is not None or residual_history is not None
    omega_f = float(omega)
    # Single-chunk + fixed K: one launch, zero intermediate HBM for (j,r).
    fuse_one = n_chunks == 1 and not need_res_max
    if fuse_one:
        for it in range(max_iters):
            _cfc_cell_update_kernel[grid_td](
                h,
                wx,
                a_gate,
                u,
                h0,
                time,
                d_h,
                omega_f,
                *h.stride(),
                h0.stride(0),
                h0.stride(1),
                *wx.stride(),
                *a_gate.stride(),
                cs,
                n_seq,
                bt,
                BLOCK_T=_BLOCK_T,
                BLOCK_D=_BLOCK_D,
                HAS_CU=has_cu,
                HAS_BT=has_bt,
                NO_GATE=no_gate_c,
            )
            log_fused_iter(
                log,
                "newton_cfc_fused_iter",
                it=it,
                time=time,
                batch=batch,
                d_h=d_h,
                n_chunks=n_chunks,
            )
        mark_fused_iters_done(iters_done_out, max_iters)
        log_fused_done(
            log,
            "newton_cfc_fused",
            time=time,
            batch=batch,
            d_h=d_h,
            max_iters=max_iters,
            n_chunks=n_chunks,
            recompute_jr=False,
            fuse_one=True,
        )
        return _cfc_fill_residual_max_out(
            residual_max_out, h, wx, u, dt, h0, a_gate, no_gate_c, cu_seqlens
        )

    agg_j = h.new_empty(batch, n_chunks, d_h)
    agg_r = h.new_empty(batch, n_chunks, d_h)
    incl_r = h.new_empty(batch, n_chunks, d_h) if n_chunks > 1 else h  # dummy when no carry
    res_max = (
        h.new_empty(batch, n_chunks, n_dtiles, dtype=torch.float32) if need_res_max else h
    )
    has_carry = n_chunks > 1

    for it in range(max_iters):
        _cfc_cell_agg_kernel[grid_td](
            h,
            wx,
            a_gate,
            u,
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
            *a_gate.stride(),
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
            NO_GATE=no_gate_c,
            WRITE_RES_MAX=need_res_max,
        )
        log_fused_iter(
            log,
            "newton_cfc_fused_iter",
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
                "newton_cfc_fused",
                time=time,
                batch=batch,
                d_h=d_h,
                max_iters=it,
                n_chunks=n_chunks,
                early_exit=True,
                recompute_jr=True,
            )
            return _cfc_fill_residual_max_out(
                residual_max_out, h, wx, u, dt, h0, a_gate, no_gate_c, cu_seqlens
            )
        if has_carry:
            incl_r = scan_chunk_aggregates(agg_j, agg_r, out=incl_r)
        _cfc_recompute_apply_kernel[grid_td](
            h,
            wx,
            a_gate,
            u,
            h0,
            incl_r,
            time,
            d_h,
            omega_f,
            *h.stride(),
            h0.stride(0),
            h0.stride(1),
            *wx.stride(),
            *a_gate.stride(),
            *incl_r.stride(),
            cs,
            n_seq,
            bt,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            HAS_CU=has_cu,
            HAS_BT=has_bt,
            NO_GATE=no_gate_c,
            HAS_CARRY=has_carry,
        )
    mark_fused_iters_done(iters_done_out, max_iters)
    log_fused_done(
        log,
        "newton_cfc_fused",
        time=time,
        batch=batch,
        d_h=d_h,
        max_iters=max_iters,
        n_chunks=n_chunks,
        recompute_jr=True,
        fuse_one=False,
    )
    return _cfc_fill_residual_max_out(
        residual_max_out, h, wx, u, dt, h0, a_gate, no_gate_c, cu_seqlens
    )

"""Triton VJP of the ParaSLSTM recurrence (Beck et al. 2024).

``state_prev`` detached (eq. 2.6 already scanned ``J^T``). ``W_x`` GEMM stays
in PyTorch. ``mix='diag'``: channelwise Triton. ``mix='head'``: factorized
``∇R_head`` (SRAM loop-``T`` for ``d_head ≤ 32``, eager otherwise; zero
atomics — partial ``∇R`` per ``(B, head)`` then ``.sum(0)``). Dtype gate is
``validate_cuda_tensors`` (bf16 if CC ≥ 8.0).

Window lengths and chunk packs: ``slstm_eq26_window_fused`` /
``slstm_eq26_states_fused`` remat ``J``, run the eq. 2.6 reverse, and pack
the cell VJP in **one** launch (no ``(B,T,4,4,d)`` Jacobian HBM). Packs use
``H*`` + ``h0`` strides (``PREPENDED=0``); no ``prepend_state`` / ``clone``.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors
from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_SLOTS,
    SLSTM_STABILIZER,
)
from pararnn.solvers.newton.config import FUSED_WINDOW_LENS

log = logging.getLogger(__name__)

_BLOCK_T = 64
_BLOCK_D = 32
_EQ26_BLOCK_D = 32
SLOT_C = tl.constexpr(0)
SLOT_N = tl.constexpr(1)
SLOT_M = tl.constexpr(2)
SLOT_H = tl.constexpr(3)


def _vjp_block_t(time: int) -> int:
    """Single time tile when ``T`` is a fused window length."""
    if int(time) in FUSED_WINDOW_LENS:
        return int(time)
    return _BLOCK_T


@triton.jit
def _tanh(x):
    return _nv_tanh(x)


@triton.jit
def _load_state(s_ptr, pid_b, offs_t, offs_d, slot, mask, sb, st, ss, sd):
    return load_acc(
        s_ptr + pid_b * sb + offs_t[:, None] * st + slot * ss + offs_d[None, :] * sd,
        mask,
        0.0,
    )


@triton.jit
def _load_state_t(s_ptr, pid_b, t, offs_d, slot, dmask, sb, st, ss, sd):
    return load_acc(
        s_ptr + pid_b * sb + t * st + slot * ss + offs_d * sd,
        dmask,
        0.0,
    )


@triton.jit
def _slstm_vjp_kernel(
    s_ptr,
    wx_ptr,
    ri_ptr,
    rf_ptr,
    rz_ptr,
    ro_ptr,
    mu_ptr,
    gwx_ptr,
    gri_ptr,
    grf_ptr,
    grz_ptr,
    gro_ptr,
    time,
    d_h,
    eps,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_mb,
    stride_mt,
    stride_ms,
    stride_md,
    stride_gb,
    stride_gt,
    stride_gd,
    stride_ab,
    stride_ac,
    stride_ad,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h

    c_prev = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    n_prev = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    m_prev = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    h_prev = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    mu_c = _load_state(
        mu_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_mb, stride_mt, stride_ms, stride_md
    )
    mu_n = _load_state(
        mu_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_mb, stride_mt, stride_ms, stride_md
    )
    mu_m = _load_state(
        mu_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_mb, stride_mt, stride_ms, stride_md
    )
    mu_h = _load_state(
        mu_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_mb, stride_mt, stride_ms, stride_md
    )

    base = wx_ptr + pid_b * stride_wb + offs_t[:, None] * stride_wt
    wx_i = load_acc(base + offs_d[None, :] * stride_wd, mask, 0.0)
    wx_f = load_acc(base + (offs_d[None, :] + d_h) * stride_wd, mask, 0.0)
    wx_z = load_acc(base + (offs_d[None, :] + 2 * d_h) * stride_wd, mask, 0.0)
    wx_o = load_acc(base + (offs_d[None, :] + 3 * d_h) * stride_wd, mask, 0.0)
    r_i = load_acc(ri_ptr + offs_d, dmask, 0.0)
    r_f = load_acc(rf_ptr + offs_d, dmask, 0.0)
    r_z = load_acc(rz_ptr + offs_d, dmask, 0.0)
    r_o = load_acc(ro_ptr + offs_d, dmask, 0.0)

    z_i = wx_i + r_i * h_prev
    z_f = wx_f + r_f * h_prev
    z_z = wx_z + r_z * h_prev
    z_o = wx_o + r_o * h_prev
    left = z_f + m_prev
    m_new = tl.maximum(left, z_i)
    # ∂maximum(left, z_i)/∂left; ties 0.5/0.5 like torch.maximum.
    alpha = tl.where(left > z_i, 1.0, 0.0) + 0.5 * tl.where(left == z_i, 1.0, 0.0)
    i_t = tl.exp(z_i - m_new)
    f_t = tl.exp(z_f + m_prev - m_new)
    z = _tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c_prev + i_t * z
    o = 1.0 / (1.0 + tl.exp(-z_o))
    denom = n_new + eps

    d_o = mu_h * (c_new / denom)
    d_cnew = mu_c + mu_h * (o / denom)
    d_nnew = mu_n + mu_h * (-o * c_new / (denom * denom))
    d_f = d_cnew * c_prev + d_nnew * n_prev
    d_i = d_cnew * z + d_nnew
    d_z = d_cnew * i_t
    d_zo = d_o * o * (1.0 - o)
    d_zz = d_z * (1.0 - z * z)
    d_mnew = mu_m - d_i * i_t - d_f * f_t
    d_zi = d_i * i_t + d_mnew * (1.0 - alpha)
    d_zf = d_f * f_t + d_mnew * alpha

    gout = gwx_ptr + pid_b * stride_gb + offs_t[:, None] * stride_gt
    store_acc(gout + offs_d[None, :] * stride_gd, d_zi, mask)
    store_acc(gout + (offs_d[None, :] + d_h) * stride_gd, d_zf, mask)
    store_acc(gout + (offs_d[None, :] + 2 * d_h) * stride_gd, d_zz, mask)
    store_acc(gout + (offs_d[None, :] + 3 * d_h) * stride_gd, d_zo, mask)

    off = pid_b * stride_ab + pid_c * stride_ac + offs_d * stride_ad
    store_acc(gri_ptr + off, tl.sum(tl.where(mask, d_zi * h_prev, 0.0), 0), dmask)
    store_acc(grf_ptr + off, tl.sum(tl.where(mask, d_zf * h_prev, 0.0), 0), dmask)
    store_acc(grz_ptr + off, tl.sum(tl.where(mask, d_zz * h_prev, 0.0), 0), dmask)
    store_acc(gro_ptr + off, tl.sum(tl.where(mask, d_zo * h_prev, 0.0), 0), dmask)


@triton.jit
def _slstm_eq26_window_kernel(
    s_ptr,
    wx_ptr,
    ri_ptr,
    rf_ptr,
    rz_ptr,
    ro_ptr,
    part_ptr,
    gwx_ptr,
    gri_ptr,
    grf_ptr,
    grz_ptr,
    gro_ptr,
    gh0_ptr,
    h0_in_ptr,
    mu_carry_ptr,
    time,
    d_h,
    eps,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_pb,
    stride_pt,
    stride_ps,
    stride_pd,
    stride_gb,
    stride_gt,
    stride_gd,
    stride_rb,
    stride_rd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    stride_h0in_b,
    stride_h0in_s,
    stride_h0in_d,
    stride_muc_b,
    stride_muc_s,
    stride_muc_d,
    BLOCK_D: tl.constexpr,
    PREPENDED: tl.constexpr,
    HAS_H0: tl.constexpr,
    HAS_MU_CARRY: tl.constexpr,
):
    """One launch: remat ``J`` + eq. 2.6 reverse + packed cell VJP.

    ``PREPENDED``: ``s`` is ``h_prev``. Else ``s`` is ``H*``; step ``t`` reads
    ``H*[t-1]`` or ``h0_in``. ``HAS_MU_CARRY`` adds into ``partial`` at ``T-1``.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = offs_d < d_h
    r_i = load_acc(ri_ptr + offs_d, dmask, 0.0)
    r_f = load_acc(rf_ptr + offs_d, dmask, 0.0)
    r_z = load_acc(rz_ptr + offs_d, dmask, 0.0)
    r_o = load_acc(ro_ptr + offs_d, dmask, 0.0)
    mu0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    mu1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    mu2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    mu3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j00 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j01 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j02 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j03 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j10 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j11 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j12 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j13 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j20 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j21 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j22 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j23 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j30 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j31 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j32 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    j33 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    g_ri = tl.zeros((BLOCK_D,), dtype=tl.float32)
    g_rf = tl.zeros((BLOCK_D,), dtype=tl.float32)
    g_rz = tl.zeros((BLOCK_D,), dtype=tl.float32)
    g_ro = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t_rev in range(0, time):
        t = time - 1 - t_rev
        if PREPENDED:
            c_prev = _load_state_t(
                s_ptr, pid_b, t, offs_d, SLOT_C, dmask, stride_sb, stride_st, stride_ss, stride_sd
            )
            n_prev = _load_state_t(
                s_ptr, pid_b, t, offs_d, SLOT_N, dmask, stride_sb, stride_st, stride_ss, stride_sd
            )
            m_prev = _load_state_t(
                s_ptr, pid_b, t, offs_d, SLOT_M, dmask, stride_sb, stride_st, stride_ss, stride_sd
            )
            h_prev = _load_state_t(
                s_ptr, pid_b, t, offs_d, SLOT_H, dmask, stride_sb, stride_st, stride_ss, stride_sd
            )
        else:
            if t == 0:
                if HAS_H0:
                    base_h = h0_in_ptr + pid_b * stride_h0in_b + offs_d * stride_h0in_d
                    c_prev = load_acc(base_h + 0 * stride_h0in_s, dmask, 0.0)
                    n_prev = load_acc(base_h + 1 * stride_h0in_s, dmask, 0.0)
                    m_prev = load_acc(base_h + 2 * stride_h0in_s, dmask, 0.0)
                    h_prev = load_acc(base_h + 3 * stride_h0in_s, dmask, 0.0)
                else:
                    c_prev = tl.zeros((BLOCK_D,), dtype=tl.float32)
                    n_prev = tl.zeros((BLOCK_D,), dtype=tl.float32)
                    m_prev = tl.zeros((BLOCK_D,), dtype=tl.float32)
                    h_prev = tl.zeros((BLOCK_D,), dtype=tl.float32)
            else:
                tp = t - 1
                c_prev = _load_state_t(
                    s_ptr, pid_b, tp, offs_d, SLOT_C, dmask, stride_sb, stride_st, stride_ss, stride_sd
                )
                n_prev = _load_state_t(
                    s_ptr, pid_b, tp, offs_d, SLOT_N, dmask, stride_sb, stride_st, stride_ss, stride_sd
                )
                m_prev = _load_state_t(
                    s_ptr, pid_b, tp, offs_d, SLOT_M, dmask, stride_sb, stride_st, stride_ss, stride_sd
                )
                h_prev = _load_state_t(
                    s_ptr, pid_b, tp, offs_d, SLOT_H, dmask, stride_sb, stride_st, stride_ss, stride_sd
                )
        base_w = wx_ptr + pid_b * stride_wb + t * stride_wt
        wx_i = load_acc(base_w + offs_d * stride_wd, dmask, 0.0)
        wx_f = load_acc(base_w + (offs_d + d_h) * stride_wd, dmask, 0.0)
        wx_z = load_acc(base_w + (offs_d + 2 * d_h) * stride_wd, dmask, 0.0)
        wx_o = load_acc(base_w + (offs_d + 3 * d_h) * stride_wd, dmask, 0.0)
        z_i = wx_i + r_i * h_prev
        z_f = wx_f + r_f * h_prev
        z_z = wx_z + r_z * h_prev
        z_o = wx_o + r_o * h_prev
        left = z_f + m_prev
        m_new = tl.maximum(left, z_i)
        alpha = tl.where(left > z_i, 1.0, 0.0) + 0.5 * tl.where(left == z_i, 1.0, 0.0)
        beta = 1.0 - alpha
        i_t = tl.exp(z_i - m_new)
        f_t = tl.exp(z_f + m_prev - m_new)
        z = _tanh(z_z)
        n_new = f_t * n_prev + i_t
        c_new = f_t * c_prev + i_t * z
        o = 1.0 / (1.0 + tl.exp(-z_o))
        denom = n_new + eps
        dm_dh = alpha * r_f + beta * r_i
        di_dm = -i_t * alpha
        df_dm = f_t * beta
        di_dh = i_t * (r_i - dm_dh)
        df_dh = f_t * (r_f - dm_dh)
        dz_dh = (1.0 - z * z) * r_z
        do_dh = o * (1.0 - o) * r_o
        nj00 = f_t
        nj01 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        nj02 = df_dm * c_prev + di_dm * z
        nj03 = df_dh * c_prev + di_dh * z + i_t * dz_dh
        nj10 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        nj11 = f_t
        nj12 = df_dm * n_prev + di_dm
        nj13 = df_dh * n_prev + di_dh
        nj20 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        nj21 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        nj22 = alpha
        nj23 = dm_dh
        inv = o / denom
        dn = -o * c_new / (denom * denom)
        du = c_new / denom
        nj30 = inv * nj00
        nj31 = dn * nj11
        nj32 = inv * nj02 + dn * nj12
        nj33 = inv * nj03 + dn * nj13 + du * do_dh
        base_p = part_ptr + pid_b * stride_pb + t * stride_pt + offs_d * stride_pd
        g0 = load_acc(base_p + 0 * stride_ps, dmask, 0.0)
        g1 = load_acc(base_p + 1 * stride_ps, dmask, 0.0)
        g2 = load_acc(base_p + 2 * stride_ps, dmask, 0.0)
        g3 = load_acc(base_p + 3 * stride_ps, dmask, 0.0)
        if t_rev == 0 and HAS_MU_CARRY:
            base_m = mu_carry_ptr + pid_b * stride_muc_b + offs_d * stride_muc_d
            g0 = g0 + load_acc(base_m + 0 * stride_muc_s, dmask, 0.0)
            g1 = g1 + load_acc(base_m + 1 * stride_muc_s, dmask, 0.0)
            g2 = g2 + load_acc(base_m + 2 * stride_muc_s, dmask, 0.0)
            g3 = g3 + load_acc(base_m + 3 * stride_muc_s, dmask, 0.0)
        if t_rev > 0:
            g0 = g0 + j00 * mu0 + j10 * mu1 + j20 * mu2 + j30 * mu3
            g1 = g1 + j01 * mu0 + j11 * mu1 + j21 * mu2 + j31 * mu3
            g2 = g2 + j02 * mu0 + j12 * mu1 + j22 * mu2 + j32 * mu3
            g3 = g3 + j03 * mu0 + j13 * mu1 + j23 * mu2 + j33 * mu3
        mu0 = g0
        mu1 = g1
        mu2 = g2
        mu3 = g3
        d_o = mu3 * (c_new / denom)
        d_cnew = mu0 + mu3 * (o / denom)
        d_nnew = mu1 + mu3 * (-o * c_new / (denom * denom))
        d_f = d_cnew * c_prev + d_nnew * n_prev
        d_i = d_cnew * z + d_nnew
        d_z = d_cnew * i_t
        d_zo = d_o * o * (1.0 - o)
        d_zz = d_z * (1.0 - z * z)
        d_mnew = mu2 - d_i * i_t - d_f * f_t
        d_zi = d_i * i_t + d_mnew * (1.0 - alpha)
        d_zf = d_f * f_t + d_mnew * alpha
        gout = gwx_ptr + pid_b * stride_gb + t * stride_gt
        store_acc(gout + offs_d * stride_gd, d_zi, dmask)
        store_acc(gout + (offs_d + d_h) * stride_gd, d_zf, dmask)
        store_acc(gout + (offs_d + 2 * d_h) * stride_gd, d_zz, dmask)
        store_acc(gout + (offs_d + 3 * d_h) * stride_gd, d_zo, dmask)
        g_ri += tl.where(dmask, d_zi * h_prev, 0.0)
        g_rf += tl.where(dmask, d_zf * h_prev, 0.0)
        g_rz += tl.where(dmask, d_zz * h_prev, 0.0)
        g_ro += tl.where(dmask, d_zo * h_prev, 0.0)
        j00 = nj00
        j01 = nj01
        j02 = nj02
        j03 = nj03
        j10 = nj10
        j11 = nj11
        j12 = nj12
        j13 = nj13
        j20 = nj20
        j21 = nj21
        j22 = nj22
        j23 = nj23
        j30 = nj30
        j31 = nj31
        j32 = nj32
        j33 = nj33
    h0_c = j00 * mu0 + j10 * mu1 + j20 * mu2 + j30 * mu3
    h0_n = j01 * mu0 + j11 * mu1 + j21 * mu2 + j31 * mu3
    h0_m = j02 * mu0 + j12 * mu1 + j22 * mu2 + j32 * mu3
    h0_h = j03 * mu0 + j13 * mu1 + j23 * mu2 + j33 * mu3
    base_h0 = gh0_ptr + pid_b * stride_h0b + offs_d * stride_h0d
    store_acc(base_h0 + 0 * stride_h0s, h0_c, dmask)
    store_acc(base_h0 + 1 * stride_h0s, h0_n, dmask)
    store_acc(base_h0 + 2 * stride_h0s, h0_m, dmask)
    store_acc(base_h0 + 3 * stride_h0s, h0_h, dmask)
    off_r = pid_b * stride_rb + offs_d * stride_rd
    store_acc(gri_ptr + off_r, g_ri, dmask)
    store_acc(grf_ptr + off_r, g_rf, dmask)
    store_acc(grz_ptr + off_r, g_rz, dmask)
    store_acc(gro_ptr + off_r, g_ro, dmask)


def _eq26_launch(
    state_buf: Tensor,
    wx: Tensor,
    r: Tensor,
    partial: Tensor,
    eps: float,
    *,
    prepended: bool,
    h0_in: Tensor | None = None,
    mu_carry: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Launch eq. 2.6; strided views stay strided (no forced ``.contiguous()``)."""
    batch, time, slots, d_h = state_buf.shape
    if slots != SLSTM_SLOTS:
        raise ValueError(f"state slots {slots} != {SLSTM_SLOTS}")
    if time < 1:
        raise ValueError(f"eq26 needs T >= 1, got {time}")
    if wx.shape != (batch, time, 4 * d_h):
        raise ValueError(f"wx shape {tuple(wx.shape)} != {(batch, time, 4 * d_h)}")
    if r.shape != (4, d_h):
        raise ValueError(f"r shape {tuple(r.shape)} != {(4, d_h)}")
    if partial.shape != (batch, time, SLSTM_SLOTS, d_h):
        raise ValueError(
            f"partial shape {tuple(partial.shape)} != {(batch, time, SLSTM_SLOTS, d_h)}"
        )
    validate_cuda_tensors(state_buf, wx, r, partial, name="slstm_eq26")
    r_i, r_f, r_z, r_o = r.unbind(0)
    g_wx = wx.new_empty(wx.shape)
    g_ri = torch.zeros(batch, d_h, device=wx.device, dtype=torch.float32)
    g_rf = torch.zeros(batch, d_h, device=wx.device, dtype=torch.float32)
    g_rz = torch.zeros(batch, d_h, device=wx.device, dtype=torch.float32)
    g_ro = torch.zeros(batch, d_h, device=wx.device, dtype=torch.float32)
    h0_vjp = state_buf.new_empty(batch, SLSTM_SLOTS, d_h)
    has_h0 = h0_in is not None
    has_mu = mu_carry is not None
    if has_h0:
        assert h0_in is not None
        if h0_in.shape != (batch, SLSTM_SLOTS, d_h):
            raise ValueError(f"h0_in shape {tuple(h0_in.shape)} != {(batch, SLSTM_SLOTS, d_h)}")
        h0_use = h0_in
    else:
        h0_use = h0_vjp
    if has_mu:
        assert mu_carry is not None
        if mu_carry.shape != (batch, SLSTM_SLOTS, d_h):
            raise ValueError(
                f"mu_carry shape {tuple(mu_carry.shape)} != {(batch, SLSTM_SLOTS, d_h)}"
            )
        mu_use = mu_carry
    else:
        mu_use = h0_vjp
    n_dtiles = triton.cdiv(d_h, _EQ26_BLOCK_D)
    _slstm_eq26_window_kernel[(batch, n_dtiles)](
        state_buf,
        wx,
        r_i,
        r_f,
        r_z,
        r_o,
        partial,
        g_wx,
        g_ri,
        g_rf,
        g_rz,
        g_ro,
        h0_vjp,
        h0_use,
        mu_use,
        time,
        d_h,
        float(eps),
        *state_buf.stride(),
        *wx.stride(),
        *partial.stride(),
        *g_wx.stride(),
        *g_ri.stride(),
        *h0_vjp.stride(),
        *h0_use.stride(),
        *mu_use.stride(),
        BLOCK_D=_EQ26_BLOCK_D,
        PREPENDED=prepended,
        HAS_H0=has_h0,
        HAS_MU_CARRY=has_mu,
    )
    dt = r.dtype
    g_r = torch.stack(
        (g_ri.sum(0), g_rf.sum(0), g_rz.sum(0), g_ro.sum(0)),
        dim=0,
    ).to(dt)
    return g_wx, g_r, h0_vjp.to(dt)


def slstm_eq26_window_fused(
    state_prev: Tensor,
    wx: Tensor,
    r: Tensor,
    partial: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused remat ``J`` + reverse + VJP for prepended ``h_prev``."""
    return _eq26_launch(state_prev, wx, r, partial, eps, prepended=True)


def slstm_eq26_states_fused(
    states: Tensor,
    h0: Tensor | None,
    wx: Tensor,
    r: Tensor,
    partial: Tensor,
    eps: float,
    *,
    mu_carry: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Eq. 2.6 on solved ``H*`` — no ``prepend_state`` / ``partial.clone``."""
    return _eq26_launch(
        states,
        wx,
        r,
        partial,
        eps,
        prepended=False,
        h0_in=h0,
        mu_carry=mu_carry,
    )



def slstm_recurrence_vjp_eager(
    state_prev: Tensor,
    wx: Tensor,
    r: Tensor,
    mu: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Channelwise sLSTM VJP. fp32 algebra, fp32 ``(B, T)`` reduce, cast to ``wx.dtype``."""
    dt = wx.dtype
    state_prev = state_prev.float()
    wx = wx.float()
    r = r.float()
    mu = mu.float()
    c_prev = state_prev[..., SLSTM_CELL, :]
    n_prev = state_prev[..., SLSTM_NORMALIZER, :]
    m_prev = state_prev[..., SLSTM_STABILIZER, :]
    h_prev = state_prev[..., SLSTM_HIDDEN, :]
    mu_c = mu[..., SLSTM_CELL, :]
    mu_n = mu[..., SLSTM_NORMALIZER, :]
    mu_m = mu[..., SLSTM_STABILIZER, :]
    mu_h = mu[..., SLSTM_HIDDEN, :]
    wx_i, wx_f, wx_z, wx_o = wx.chunk(4, dim=-1)
    r_i, r_f, r_z, r_o = r.unbind(0)
    z_i = wx_i + r_i * h_prev
    z_f = wx_f + r_f * h_prev
    z_z = wx_z + r_z * h_prev
    z_o = wx_o + r_o * h_prev
    left = z_f + m_prev
    m_new = torch.maximum(left, z_i)
    gt = (left > z_i).to(dtype=left.dtype)
    eq = (left == z_i).to(dtype=left.dtype)
    alpha = gt + 0.5 * eq
    i_t = torch.exp(z_i - m_new)
    f_t = torch.exp(z_f + m_prev - m_new)
    z = torch.tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c_prev + i_t * z
    o = torch.sigmoid(z_o)
    denom = n_new + eps
    d_o = mu_h * (c_new / denom)
    d_cnew = mu_c + mu_h * (o / denom)
    d_nnew = mu_n + mu_h * (-o * c_new / denom.square())
    d_f = d_cnew * c_prev + d_nnew * n_prev
    d_i = d_cnew * z + d_nnew
    d_z = d_cnew * i_t
    d_zo = d_o * o * (1.0 - o)
    d_zz = d_z * (1.0 - z.square())
    d_mnew = mu_m - d_i * i_t - d_f * f_t
    d_zi = d_i * i_t + d_mnew * (1.0 - alpha)
    d_zf = d_f * f_t + d_mnew * alpha
    g_wx = torch.cat((d_zi, d_zf, d_zz, d_zo), dim=-1)
    dims = (0, 1)
    g_r = torch.stack(
        (
            (d_zi * h_prev).sum(dim=dims),
            (d_zf * h_prev).sum(dim=dims),
            (d_zz * h_prev).sum(dim=dims),
            (d_zo * h_prev).sum(dim=dims),
        ),
        dim=0,
    )
    return g_wx.to(dt), g_r.to(dt)


def slstm_recurrence_vjp(
    state_prev: Tensor,
    wx: Tensor,
    r: Tensor,
    mu: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    if not state_prev.is_cuda:
        return slstm_recurrence_vjp_eager(state_prev, wx, r, mu, eps)
    validate_cuda_tensors(state_prev, wx, r, mu, name="slstm_recurrence_vjp")
    state_prev = state_prev.contiguous()
    wx = wx.contiguous()
    mu = mu.contiguous()
    r = r.contiguous()
    r_i, r_f, r_z, r_o = r.unbind(0)
    batch, time, _, d_h = state_prev.shape
    block_t = _vjp_block_t(time)
    n_chunks = (time + block_t - 1) // block_t
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    g_wx = wx.new_empty(wx.shape)
    acc_shape = (batch, n_chunks, d_h)
    g_ri = torch.zeros(acc_shape, device=state_prev.device, dtype=torch.float32)
    g_rf = torch.zeros(acc_shape, device=state_prev.device, dtype=torch.float32)
    g_rz = torch.zeros(acc_shape, device=state_prev.device, dtype=torch.float32)
    g_ro = torch.zeros(acc_shape, device=state_prev.device, dtype=torch.float32)
    _slstm_vjp_kernel[(batch, n_chunks, n_dtiles)](
        state_prev,
        wx,
        r_i,
        r_f,
        r_z,
        r_o,
        mu,
        g_wx,
        g_ri,
        g_rf,
        g_rz,
        g_ro,
        time,
        d_h,
        float(eps),
        *state_prev.stride(),
        *wx.stride(),
        *mu.stride(),
        *g_wx.stride(),
        *g_ri.stride(),
        BLOCK_T=block_t,
        BLOCK_D=_BLOCK_D,
    )
    dt = r.dtype
    dims = (0, 1)
    g_r = torch.stack(
        (
            g_ri.sum(dim=dims),
            g_rf.sum(dim=dims),
            g_rz.sum(dim=dims),
            g_ro.sum(dim=dims),
        ),
        dim=0,
    )
    return g_wx, g_r.to(dt)


def slstm_head_recurrence_vjp_eager(
    state_prev: Tensor,
    wx: Tensor,
    r_head: Tensor,
    mu: Tensor,
    *,
    n_heads: int,
    d_head: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Head-mix sLSTM VJP. ``r_head`` is ``(4, H, d_in, d_out)``; ``∇R`` via outer ``h⊗d_z``.

    fp32 algebra; reduce over ``(B, T)``; cast to ``wx.dtype``.
    """
    dt = wx.dtype
    state_prev = state_prev.float()
    wx = wx.float()
    r_head = r_head.float()
    mu = mu.float()
    prefix = state_prev.shape[:-2]
    d_h = n_heads * d_head
    c_prev = state_prev[..., SLSTM_CELL, :].reshape(*prefix, n_heads, d_head)
    n_prev = state_prev[..., SLSTM_NORMALIZER, :].reshape(*prefix, n_heads, d_head)
    m_prev = state_prev[..., SLSTM_STABILIZER, :].reshape(*prefix, n_heads, d_head)
    h_prev = state_prev[..., SLSTM_HIDDEN, :].reshape(*prefix, n_heads, d_head)
    mu_c = mu[..., SLSTM_CELL, :].reshape(*prefix, n_heads, d_head)
    mu_n = mu[..., SLSTM_NORMALIZER, :].reshape(*prefix, n_heads, d_head)
    mu_m = mu[..., SLSTM_STABILIZER, :].reshape(*prefix, n_heads, d_head)
    mu_h = mu[..., SLSTM_HIDDEN, :].reshape(*prefix, n_heads, d_head)
    wx_h = wx.reshape(*prefix, 4, n_heads, d_head)
    r_i, r_f, r_z, r_o = r_head.unbind(0)

    def mix(h: Tensor, r: Tensor) -> Tensor:
        return torch.matmul(h.unsqueeze(-2), r).squeeze(-2)

    z_i = mix(h_prev, r_i) + wx_h[..., 0, :, :]
    z_f = mix(h_prev, r_f) + wx_h[..., 1, :, :]
    z_z = mix(h_prev, r_z) + wx_h[..., 2, :, :]
    z_o = mix(h_prev, r_o) + wx_h[..., 3, :, :]
    left = z_f + m_prev
    m_new = torch.maximum(left, z_i)
    gt = (left > z_i).to(dtype=left.dtype)
    eq = (left == z_i).to(dtype=left.dtype)
    alpha = gt + 0.5 * eq
    i_t = torch.exp(z_i - m_new)
    f_t = torch.exp(z_f + m_prev - m_new)
    z = torch.tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c_prev + i_t * z
    o = torch.sigmoid(z_o)
    denom = n_new + eps
    d_o = mu_h * (c_new / denom)
    d_cnew = mu_c + mu_h * (o / denom)
    d_nnew = mu_n + mu_h * (-o * c_new / denom.square())
    d_f = d_cnew * c_prev + d_nnew * n_prev
    d_i = d_cnew * z + d_nnew
    d_z = d_cnew * i_t
    d_zo = d_o * o * (1.0 - o)
    d_zz = d_z * (1.0 - z.square())
    d_mnew = mu_m - d_i * i_t - d_f * f_t
    d_zi = d_i * i_t + d_mnew * (1.0 - alpha)
    d_zf = d_f * f_t + d_mnew * alpha
    g_wx = torch.stack((d_zi, d_zf, d_zz, d_zo), dim=-3).reshape(*prefix, 4 * d_h)
    # ∇R_g[h,i,o] = Σ_{b,t} h[i] * d_zg[o]
    if h_prev.dim() == 4:
        g_r = torch.stack(
            [torch.einsum("bthi,btho->hio", h_prev, d) for d in (d_zi, d_zf, d_zz, d_zo)],
            dim=0,
        )
    else:
        red = tuple(range(h_prev.dim() - 2))
        g_r = torch.stack(
            [
                torch.einsum("...hi,...ho->hio", h_prev, d).sum(dim=red)
                for d in (d_zi, d_zf, d_zz, d_zo)
            ],
            dim=0,
        )
    return g_wx.to(dt), g_r.to(dt)


def slstm_head_recurrence_vjp(
    state_prev: Tensor,
    wx: Tensor,
    r_head: Tensor,
    mu: Tensor,
    *,
    n_heads: int,
    d_head: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """CUDA when available; SRAM loop-``T`` Triton for ``d_head ≤ 32``, else eager."""
    if not state_prev.is_cuda or d_head > 32:
        return slstm_head_recurrence_vjp_eager(
            state_prev, wx, r_head, mu, n_heads=n_heads, d_head=d_head, eps=eps
        )
    return _slstm_head_vjp_triton(
        state_prev, wx, r_head, mu, n_heads=n_heads, d_head=d_head, eps=eps
    )


def _slstm_head_vjp_triton(
    state_prev: Tensor,
    wx: Tensor,
    r_head: Tensor,
    mu: Tensor,
    *,
    n_heads: int,
    d_head: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """One program per ``(B, head)``: loop ``T``, ``∇R`` in regs (GRU head SRAM pattern)."""
    validate_cuda_tensors(state_prev, wx, r_head, mu, name="slstm_head_recurrence_vjp")
    batch, time, _, d_h = state_prev.shape
    if d_h != n_heads * d_head:
        raise ValueError(f"d_h={d_h} != n_heads*d_head={n_heads * d_head}")
    state_prev = state_prev.contiguous()
    wx = wx.contiguous()
    mu = mu.contiguous()
    r_head = r_head.contiguous()
    g_wx = wx.new_empty(wx.shape)
    # Partial ∇R per batch row — host ``.sum(0)`` (zero ``tl.atomic*``).
    g_r_b = torch.zeros(
        batch, 4, n_heads, d_head, d_head, device=r_head.device, dtype=torch.float32
    )
    block = 1 << (d_head - 1).bit_length()
    block = min(max(block, 16), 32)
    _slstm_head_vjp_sram_kernel[(batch * n_heads,)](
        state_prev,
        wx,
        r_head,
        mu,
        g_wx,
        g_r_b,
        n_heads,
        time,
        d_head,
        d_h,
        float(eps),
        *state_prev.stride(),
        *wx.stride(),
        *mu.stride(),
        *g_wx.stride(),
        *g_r_b.stride(),
        *r_head.stride(),
        BLOCK_D=block,
    )
    return g_wx, g_r_b.sum(0).to(r_head.dtype)


@triton.jit
def _slstm_head_vjp_sram_kernel(
    s_ptr,
    wx_ptr,
    r_ptr,
    mu_ptr,
    gwx_ptr,
    gr_ptr,
    n_heads,
    time,
    d,
    d_h,
    eps,
    stride_s_b,
    stride_s_t,
    stride_s_s,
    stride_s_d,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_mu_b,
    stride_mu_t,
    stride_mu_s,
    stride_mu_d,
    stride_gwx_b,
    stride_gwx_t,
    stride_gwx_d,
    stride_gr_b,
    stride_gr_g,
    stride_gr_h,
    stride_gr_in,
    stride_gr_out,
    stride_r_g,
    stride_r_h,
    stride_r_in,
    stride_r_out,
    BLOCK_D: tl.constexpr,
):
    """One ``(batch, head)``: serial ``T``, accumulate ``∇R_g += h ⊗ ∂z_g`` in regs."""
    pid = tl.program_id(0)
    b = pid // n_heads
    head = pid % n_heads
    offs = tl.arange(0, BLOCK_D)
    mask = offs < d
    mask_ij = mask[:, None] & mask[None, :]
    head_off = head * d

    ri = load_acc(
        r_ptr
        + 0 * stride_r_g
        + head * stride_r_h
        + offs[:, None] * stride_r_in
        + offs[None, :] * stride_r_out,
        mask_ij,
        0.0,
    )
    rf = load_acc(
        r_ptr
        + 1 * stride_r_g
        + head * stride_r_h
        + offs[:, None] * stride_r_in
        + offs[None, :] * stride_r_out,
        mask_ij,
        0.0,
    )
    rz = load_acc(
        r_ptr
        + 2 * stride_r_g
        + head * stride_r_h
        + offs[:, None] * stride_r_in
        + offs[None, :] * stride_r_out,
        mask_ij,
        0.0,
    )
    ro = load_acc(
        r_ptr
        + 3 * stride_r_g
        + head * stride_r_h
        + offs[:, None] * stride_r_in
        + offs[None, :] * stride_r_out,
        mask_ij,
        0.0,
    )

    gri = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    grf = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    grz = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    gro = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

    for t in range(0, time):
        c = load_acc(
            s_ptr
            + b * stride_s_b
            + t * stride_s_t
            + SLOT_C * stride_s_s
            + (head_off + offs) * stride_s_d,
            mask,
            0.0,
        )
        n = load_acc(
            s_ptr
            + b * stride_s_b
            + t * stride_s_t
            + SLOT_N * stride_s_s
            + (head_off + offs) * stride_s_d,
            mask,
            0.0,
        )
        m = load_acc(
            s_ptr
            + b * stride_s_b
            + t * stride_s_t
            + SLOT_M * stride_s_s
            + (head_off + offs) * stride_s_d,
            mask,
            0.0,
        )
        h = load_acc(
            s_ptr
            + b * stride_s_b
            + t * stride_s_t
            + SLOT_H * stride_s_s
            + (head_off + offs) * stride_s_d,
            mask,
            0.0,
        )
        mu_c = load_acc(
            mu_ptr
            + b * stride_mu_b
            + t * stride_mu_t
            + SLOT_C * stride_mu_s
            + (head_off + offs) * stride_mu_d,
            mask,
            0.0,
        )
        mu_n = load_acc(
            mu_ptr
            + b * stride_mu_b
            + t * stride_mu_t
            + SLOT_N * stride_mu_s
            + (head_off + offs) * stride_mu_d,
            mask,
            0.0,
        )
        mu_m = load_acc(
            mu_ptr
            + b * stride_mu_b
            + t * stride_mu_t
            + SLOT_M * stride_mu_s
            + (head_off + offs) * stride_mu_d,
            mask,
            0.0,
        )
        mu_h = load_acc(
            mu_ptr
            + b * stride_mu_b
            + t * stride_mu_t
            + SLOT_H * stride_mu_s
            + (head_off + offs) * stride_mu_d,
            mask,
            0.0,
        )
        wxi = load_acc(
            wx_ptr
            + b * stride_wx_b
            + t * stride_wx_t
            + (0 * d_h + head_off + offs) * stride_wx_d,
            mask,
            0.0,
        )
        wxf = load_acc(
            wx_ptr
            + b * stride_wx_b
            + t * stride_wx_t
            + (1 * d_h + head_off + offs) * stride_wx_d,
            mask,
            0.0,
        )
        wxz = load_acc(
            wx_ptr
            + b * stride_wx_b
            + t * stride_wx_t
            + (2 * d_h + head_off + offs) * stride_wx_d,
            mask,
            0.0,
        )
        wxo = load_acc(
            wx_ptr
            + b * stride_wx_b
            + t * stride_wx_t
            + (3 * d_h + head_off + offs) * stride_wx_d,
            mask,
            0.0,
        )
        zi = tl.sum(h[:, None] * ri, axis=0) + wxi
        zf = tl.sum(h[:, None] * rf, axis=0) + wxf
        zz = tl.sum(h[:, None] * rz, axis=0) + wxz
        zo = tl.sum(h[:, None] * ro, axis=0) + wxo
        left = zf + m
        m_new = tl.maximum(left, zi)
        alpha = tl.where(left > zi, 1.0, 0.0) + 0.5 * tl.where(left == zi, 1.0, 0.0)
        i_t = tl.exp(zi - m_new)
        f_t = tl.exp(zf + m - m_new)
        z = _tanh(zz)
        n_new = f_t * n + i_t
        c_new = f_t * c + i_t * z
        o = tl.sigmoid(zo)
        denom = n_new + eps
        d_o = mu_h * (c_new / denom)
        d_cnew = mu_c + mu_h * (o / denom)
        d_nnew = mu_n + mu_h * (-o * c_new / (denom * denom))
        d_f = d_cnew * c + d_nnew * n
        d_i = d_cnew * z + d_nnew
        d_z = d_cnew * i_t
        d_zo = d_o * o * (1.0 - o)
        d_zz = d_z * (1.0 - z * z)
        d_mnew = mu_m - d_i * i_t - d_f * f_t
        d_zi = d_i * i_t + d_mnew * (1.0 - alpha)
        d_zf = d_f * f_t + d_mnew * alpha
        store_acc(
            gwx_ptr
            + b * stride_gwx_b
            + t * stride_gwx_t
            + (0 * d_h + head_off + offs) * stride_gwx_d,
            d_zi,
            mask,
        )
        store_acc(
            gwx_ptr
            + b * stride_gwx_b
            + t * stride_gwx_t
            + (1 * d_h + head_off + offs) * stride_gwx_d,
            d_zf,
            mask,
        )
        store_acc(
            gwx_ptr
            + b * stride_gwx_b
            + t * stride_gwx_t
            + (2 * d_h + head_off + offs) * stride_gwx_d,
            d_zz,
            mask,
        )
        store_acc(
            gwx_ptr
            + b * stride_gwx_b
            + t * stride_gwx_t
            + (3 * d_h + head_off + offs) * stride_gwx_d,
            d_zo,
            mask,
        )
        # ∇R[in,out] += h[in] * ∂z[out]
        gri += h[:, None] * d_zi[None, :]
        grf += h[:, None] * d_zf[None, :]
        grz += h[:, None] * d_zz[None, :]
        gro += h[:, None] * d_zo[None, :]

    gr_base = b * stride_gr_b + head * stride_gr_h
    store_acc(
        gr_ptr + gr_base + 0 * stride_gr_g + offs[:, None] * stride_gr_in + offs[None, :] * stride_gr_out,
        gri,
        mask_ij,
    )
    store_acc(
        gr_ptr + gr_base + 1 * stride_gr_g + offs[:, None] * stride_gr_in + offs[None, :] * stride_gr_out,
        grf,
        mask_ij,
    )
    store_acc(
        gr_ptr + gr_base + 2 * stride_gr_g + offs[:, None] * stride_gr_in + offs[None, :] * stride_gr_out,
        grz,
        mask_ij,
    )
    store_acc(
        gr_ptr + gr_base + 3 * stride_gr_g + offs[:, None] * stride_gr_in + offs[None, :] * stride_gr_out,
        gro,
        mask_ij,
    )

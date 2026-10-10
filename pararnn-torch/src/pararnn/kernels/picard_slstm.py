"""Fused frozen-gate sLSTM scan (Picard guess).

Gates are frozen in ``pre``: ``m`` is max-plus, ``n``/``c`` are two 1D
``ax+b`` scans, ``h`` is the readout. Same contract as
``slstm_frozen_gate_scan``. Span is two-level prefix scans (tile + chunk).

Newton still needs the 4×4 ``J`` of ``(c, n, m, h)`` because ``R h``
couples the next gates.

After the max-plus apply, phase-m temps ``a_loc``/``b_loc``/``agg_*``/
``incl_m`` are dead and alias the ``n``/``c`` scan lanes. ``m`` stays live.
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

log = logging.getLogger(__name__)

# Legacy two-level tiles (kept for the parallel fallback path).
_BLOCK_T = 128
_BLOCK_D = 16
_CHUNK_PAD = 128  # 128 * 128 = 16384. 3 lanes × 128 × 16 × 4 B = 24 KiB.
_NEG_INF = -1.0e30
# Serial mega-fuse: one program per (batch, d-tile) walks T in registers.
_SERIAL_BLOCK_D = 32
# Windowed Picard: tile length (paper C≤128). Scratch h is O(C), not O(T).
_PICARD_TILE = 128
_PICARD_TILE_D = 32


@triton.jit
def _picard_diag_all_tiles_kernel(
    wx_ptr,
    r_ptr,
    h0_ptr,
    h_prev_ptr,
    h_curr_ptr,
    out_ptr,
    time,
    d_h,
    eps,
    n_picard,
    tile,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_rd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    stride_hpb,
    stride_hpt,
    stride_hpd,
    stride_hcb,
    stride_hct,
    stride_hcd,
    stride_ob,
    stride_ot,
    stride_os,
    stride_od,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One launch: all tiles × ``1+n_picard`` passes; Rh fused; carry in regs."""
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = offs_d < d_h
    r_i = load_acc(r_ptr + 0 * stride_rd + offs_d, dmask, 0.0)
    r_f = load_acc(r_ptr + 1 * stride_rd + offs_d, dmask, 0.0)
    r_z = load_acc(r_ptr + 2 * stride_rd + offs_d, dmask, 0.0)
    r_o = load_acc(r_ptr + 3 * stride_rd + offs_d, dmask, 0.0)
    c = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_C * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    n = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_N * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    m = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_M * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    h_carry = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_H * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    n_passes = n_picard + 1
    for t0 in range(0, time, tile):
        rem = time - t0
        tile_len = tl.minimum(tile, rem)
        c_b = c
        n_b = n
        m_b = m
        h_b = h_carry
        h = h_carry
        c = c_b
        n = n_b
        m = m_b
        for ti in range(0, tile_len):
            t = t0 + ti
            base = wx_ptr + pid_b * stride_wb + t * stride_wt
            z_i = load_acc(base + offs_d * stride_wd, dmask, 0.0)
            z_f = load_acc(base + (offs_d + d_h) * stride_wd, dmask, 0.0)
            z_z = load_acc(base + (offs_d + 2 * d_h) * stride_wd, dmask, 0.0)
            z_o = load_acc(base + (offs_d + 3 * d_h) * stride_wd, dmask, 0.0)
            m_new = tl.maximum(z_f + m, z_i)
            i_t = tl.exp(z_i - m_new)
            f_t = tl.exp(z_f + m - m_new)
            z = _nv_tanh(z_z)
            n_new = f_t * n + i_t
            c_new = f_t * c + i_t * z
            o = tl.sigmoid(z_o)
            h = o * (c_new / (n_new + eps))
            c = c_new
            n = n_new
            m = m_new
            store_acc(
                h_prev_ptr + pid_b * stride_hpb + ti * stride_hpt + offs_d * stride_hpd,
                h,
                dmask,
            )
            if n_passes == 1:
                base_o = out_ptr + pid_b * stride_ob + t * stride_ot + offs_d * stride_od
                store_acc(base_o + SLOT_C * stride_os, c, dmask)
                store_acc(base_o + SLOT_N * stride_os, n, dmask)
                store_acc(base_o + SLOT_M * stride_os, m, dmask)
                store_acc(base_o + SLOT_H * stride_os, h, dmask)
        for p in range(1, n_passes):
            c = c_b
            n = n_b
            m = m_b
            last = p == n_picard
            # Odd p: read h_prev → write h_curr; even p: swap roles (no copy).
            src_is_prev = (p & 1) == 1
            for ti in range(0, tile_len):
                t = t0 + ti
                if ti == 0:
                    h_prev = h_b
                elif src_is_prev:
                    h_prev = load_acc(
                        h_prev_ptr
                        + pid_b * stride_hpb
                        + (ti - 1) * stride_hpt
                        + offs_d * stride_hpd,
                        dmask,
                        0.0,
                    )
                else:
                    h_prev = load_acc(
                        h_curr_ptr
                        + pid_b * stride_hcb
                        + (ti - 1) * stride_hct
                        + offs_d * stride_hcd,
                        dmask,
                        0.0,
                    )
                base = wx_ptr + pid_b * stride_wb + t * stride_wt
                z_i = load_acc(base + offs_d * stride_wd, dmask, 0.0) + r_i * h_prev
                z_f = load_acc(base + (offs_d + d_h) * stride_wd, dmask, 0.0) + r_f * h_prev
                z_z = load_acc(base + (offs_d + 2 * d_h) * stride_wd, dmask, 0.0) + r_z * h_prev
                z_o = load_acc(base + (offs_d + 3 * d_h) * stride_wd, dmask, 0.0) + r_o * h_prev
                m_new = tl.maximum(z_f + m, z_i)
                i_t = tl.exp(z_i - m_new)
                f_t = tl.exp(z_f + m - m_new)
                z = _nv_tanh(z_z)
                n_new = f_t * n + i_t
                c_new = f_t * c + i_t * z
                o = tl.sigmoid(z_o)
                h = o * (c_new / (n_new + eps))
                c = c_new
                n = n_new
                m = m_new
                if src_is_prev:
                    store_acc(
                        h_curr_ptr
                        + pid_b * stride_hcb
                        + ti * stride_hct
                        + offs_d * stride_hcd,
                        h,
                        dmask,
                    )
                else:
                    store_acc(
                        h_prev_ptr
                        + pid_b * stride_hpb
                        + ti * stride_hpt
                        + offs_d * stride_hpd,
                        h,
                        dmask,
                    )
                if last:
                    base_o = out_ptr + pid_b * stride_ob + t * stride_ot + offs_d * stride_od
                    store_acc(base_o + SLOT_C * stride_os, c, dmask)
                    store_acc(base_o + SLOT_N * stride_os, n, dmask)
                    store_acc(base_o + SLOT_M * stride_os, m, dmask)
                    store_acc(base_o + SLOT_H * stride_os, h, dmask)
        h_carry = h


@triton.jit
def _frozen_gate_serial_kernel(
    pre_ptr,
    h0_ptr,
    out_ptr,
    h_prev_ptr,
    r_ptr,
    time,
    d_h,
    eps,
    stride_pb,
    stride_pt,
    stride_pd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    stride_ob,
    stride_ot,
    stride_os,
    stride_od,
    stride_hpb,
    stride_hpt,
    stride_hpd,
    stride_rd,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_RH: tl.constexpr,
):
    """Frozen-gate ``(c,n,m,h)`` in one launch: serial ``t``, write final state only.

    ``USE_RH``: ``pre`` is raw ``wx``; gates get ``+ R ⊙ h_prev[t]`` (shifted hidden).
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = offs_d < d_h
    m = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_M * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    n = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_N * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    c = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_C * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    if USE_RH:
        r_i = load_acc(r_ptr + 0 * stride_rd + offs_d, dmask, 0.0)
        r_f = load_acc(r_ptr + 1 * stride_rd + offs_d, dmask, 0.0)
        r_z = load_acc(r_ptr + 2 * stride_rd + offs_d, dmask, 0.0)
        r_o = load_acc(r_ptr + 3 * stride_rd + offs_d, dmask, 0.0)
    for t in range(0, time):
        base = pre_ptr + pid_b * stride_pb + t * stride_pt
        z_i = load_acc(base + offs_d * stride_pd, dmask, 0.0)
        z_f = load_acc(base + (offs_d + d_h) * stride_pd, dmask, 0.0)
        z_z = load_acc(base + (offs_d + 2 * d_h) * stride_pd, dmask, 0.0)
        z_o = load_acc(base + (offs_d + 3 * d_h) * stride_pd, dmask, 0.0)
        if USE_RH:
            hp = load_acc(
                h_prev_ptr + pid_b * stride_hpb + t * stride_hpt + offs_d * stride_hpd,
                dmask,
                0.0,
            )
            z_i = z_i + r_i * hp
            z_f = z_f + r_f * hp
            z_z = z_z + r_z * hp
            z_o = z_o + r_o * hp
        m_new = tl.maximum(z_f + m, z_i)
        i_t = tl.exp(z_i - m_new)
        f_t = tl.exp(z_f + m - m_new)
        z = _nv_tanh(z_z)
        n_new = f_t * n + i_t
        c_new = f_t * c + i_t * z
        o = tl.sigmoid(z_o)
        h = o * (c_new / (n_new + eps))
        base_o = out_ptr + pid_b * stride_ob + t * stride_ot + offs_d * stride_od
        store_acc(base_o + SLOT_C * stride_os, c_new, dmask)
        store_acc(base_o + SLOT_N * stride_os, n_new, dmask)
        store_acc(base_o + SLOT_M * stride_os, m_new, dmask)
        store_acc(base_o + SLOT_H * stride_os, h, dmask)
        m = m_new
        n = n_new
        c = c_new


@triton.jit
def _compose_maxplus(a_l, b_l, a_r, b_r):
    """``m |-> max(a + m, b)``. left = earlier, right = later."""
    return a_r + a_l, tl.maximum(a_r + b_l, b_r)


@triton.jit
def _compose_diag2(j_l, n_l, c_l, j_r, n_r, c_r):
    """Two ``δ = j δ_prev + r`` scans sharing ``j`` (forget gate)."""
    return j_r * j_l, j_r * n_l + n_r, j_r * c_l + c_r


@triton.jit
def _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, gate, mask, sb, st, sd):
    idx = offs_d + gate * d_h
    return load_acc(
        pre_ptr + pid_b * sb + offs_t[:, None] * st + idx[None, :] * sd,
        mask,
        0.0,
    )


@triton.jit
def _m_local_kernel(
    pre_ptr,
    a_ptr,
    b_ptr,
    agg_a_ptr,
    agg_b_ptr,
    time,
    d_h,
    stride_pb,
    stride_pt,
    stride_pd,
    stride_ab,
    stride_at,
    stride_ad,
    stride_bb,
    stride_bt,
    stride_bd,
    stride_aab,
    stride_aac,
    stride_aad,
    stride_bab,
    stride_bac,
    stride_bad,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    z_i = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 0, mask, stride_pb, stride_pt, stride_pd)
    z_f = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 1, mask, stride_pb, stride_pt, stride_pd)
    a = tl.where(mask, z_f, 0.0)
    b = tl.where(mask, z_i, NEG_INF)
    a_s, b_s = tl.associative_scan((a, b), 0, _compose_maxplus)
    store_acc(
        a_ptr + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_d[None, :] * stride_ad,
        a_s,
        mask,
    )
    store_acc(
        b_ptr + pid_b * stride_bb + offs_t[:, None] * stride_bt + offs_d[None, :] * stride_bd,
        b_s,
        mask,
    )
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    dmask = offs_d < d_h
    store_acc(
        agg_a_ptr + pid_b * stride_aab + pid_c * stride_aac + offs_d * stride_aad,
        tl.sum(tl.where(last, a_s, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_b_ptr + pid_b * stride_bab + pid_c * stride_bac + offs_d * stride_bad,
        tl.sum(tl.where(last, b_s, 0.0), axis=0),
        dmask,
    )


@triton.jit
def _m_chunk_kernel(
    agg_a_ptr,
    agg_b_ptr,
    m0_ptr,
    incl_m_ptr,
    n_chunks,
    d_h,
    stride_aab,
    stride_aac,
    stride_aad,
    stride_bab,
    stride_bac,
    stride_bad,
    stride_m0b,
    stride_m0d,
    stride_ib,
    stride_ic,
    stride_id,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    a = load_acc(
        agg_a_ptr
        + pid_b * stride_aab
        + offs_c[:, None] * stride_aac
        + offs_d[None, :] * stride_aad,
        mask,
        0.0,
    )
    b = load_acc(
        agg_b_ptr
        + pid_b * stride_bab
        + offs_c[:, None] * stride_bac
        + offs_d[None, :] * stride_bad,
        mask,
        NEG_INF,
    )
    a_s, b_s = tl.associative_scan((a, b), 0, _compose_maxplus)
    m0 = load_acc(m0_ptr + pid_b * stride_m0b + offs_d * stride_m0d, dmask, 0.0)
    m_end = tl.maximum(a_s + m0[None, :], b_s)
    store_acc(
        incl_m_ptr + pid_b * stride_ib + offs_c[:, None] * stride_ic + offs_d[None, :] * stride_id,
        m_end,
        mask,
    )


@triton.jit
def _m_apply_kernel(
    a_ptr,
    b_ptr,
    m0_ptr,
    incl_m_ptr,
    m_ptr,
    time,
    d_h,
    stride_ab,
    stride_at,
    stride_ad,
    stride_bb,
    stride_bt,
    stride_bd,
    stride_m0b,
    stride_m0d,
    stride_ib,
    stride_ic,
    stride_id,
    stride_mb,
    stride_mt,
    stride_md,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NEG_INF: tl.constexpr,
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
    a = load_acc(
        a_ptr + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_d[None, :] * stride_ad,
        mask,
        0.0,
    )
    b = load_acc(
        b_ptr + pid_b * stride_bb + offs_t[:, None] * stride_bt + offs_d[None, :] * stride_bd,
        mask,
        NEG_INF,
    )
    m0 = load_acc(m0_ptr + pid_b * stride_m0b + offs_d * stride_m0d, dmask, 0.0)
    idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
    prev = load_acc(
        incl_m_ptr + pid_b * stride_ib + idx_c * stride_ic + offs_d * stride_id,
        dmask,
        0.0,
    )
    carry = tl.where(pid_c > 0, prev, m0)
    m = tl.maximum(a + carry[None, :], b)
    store_acc(
        m_ptr + pid_b * stride_mb + offs_t[:, None] * stride_mt + offs_d[None, :] * stride_md,
        m,
        mask,
    )


@triton.jit
def _nc_local_kernel(
    pre_ptr,
    m_ptr,
    h0_ptr,
    j_ptr,
    n_ptr,
    c_ptr,
    agg_j_ptr,
    agg_n_ptr,
    agg_c_ptr,
    time,
    d_h,
    stride_pb,
    stride_pt,
    stride_pd,
    stride_mb,
    stride_mt,
    stride_md,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    stride_jb,
    stride_jt,
    stride_jd,
    stride_nb,
    stride_nt,
    stride_nd,
    stride_cb,
    stride_ct,
    stride_cd,
    stride_ajb,
    stride_ajc,
    stride_ajd,
    stride_anb,
    stride_anc,
    stride_and,
    stride_acb,
    stride_acc,
    stride_acd,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
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
    z_i = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 0, mask, stride_pb, stride_pt, stride_pd)
    z_f = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 1, mask, stride_pb, stride_pt, stride_pd)
    z_z = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 2, mask, stride_pb, stride_pt, stride_pd)
    m = load_acc(
        m_ptr + pid_b * stride_mb + offs_t[:, None] * stride_mt + offs_d[None, :] * stride_md,
        mask,
        0.0,
    )
    offs_tm1 = offs_t - 1
    mask_prev = (offs_tm1[:, None] >= 0) & (offs_tm1[:, None] < time) & (offs_d[None, :] < d_h)
    m_prev = load_acc(
        m_ptr + pid_b * stride_mb + offs_tm1[:, None] * stride_mt + offs_d[None, :] * stride_md,
        mask_prev,
        0.0,
    )
    m0 = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_M * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    n0 = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_N * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    c0 = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_C * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    is_t0 = (offs_t == 0)[:, None]
    m_prev = tl.where(is_t0, m0[None, :], m_prev)
    i_t = tl.exp(z_i - m)
    f_t = tl.exp(z_f + m_prev - m)
    z = _nv_tanh(z_z)
    res_n = i_t
    res_c = i_t * z
    res_n = tl.where(is_t0, f_t * n0[None, :] + i_t, res_n)
    res_c = tl.where(is_t0, f_t * c0[None, :] + i_t * z, res_c)
    j = tl.where(mask, f_t, 1.0)
    res_n = tl.where(mask, res_n, 0.0)
    res_c = tl.where(mask, res_c, 0.0)
    j_s, n_s, c_s = tl.associative_scan((j, res_n, res_c), 0, _compose_diag2)
    store_acc(
        j_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + offs_d[None, :] * stride_jd,
        j_s,
        mask,
    )
    store_acc(
        n_ptr + pid_b * stride_nb + offs_t[:, None] * stride_nt + offs_d[None, :] * stride_nd,
        n_s,
        mask,
    )
    store_acc(
        c_ptr + pid_b * stride_cb + offs_t[:, None] * stride_ct + offs_d[None, :] * stride_cd,
        c_s,
        mask,
    )
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + offs_d * stride_ajd,
        tl.sum(tl.where(last, j_s, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_n_ptr + pid_b * stride_anb + pid_c * stride_anc + offs_d * stride_and,
        tl.sum(tl.where(last, n_s, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_c_ptr + pid_b * stride_acb + pid_c * stride_acc + offs_d * stride_acd,
        tl.sum(tl.where(last, c_s, 0.0), axis=0),
        dmask,
    )


@triton.jit
def _nc_chunk_kernel(
    agg_j_ptr,
    agg_n_ptr,
    agg_c_ptr,
    incl_n_ptr,
    incl_c_ptr,
    n_chunks,
    d_h,
    stride_ajb,
    stride_ajc,
    stride_ajd,
    stride_anb,
    stride_anc,
    stride_and,
    stride_acb,
    stride_acc,
    stride_acd,
    stride_inb,
    stride_inc,
    stride_ind,
    stride_icb,
    stride_icc,
    stride_icd,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    j = load_acc(
        agg_j_ptr
        + pid_b * stride_ajb
        + offs_c[:, None] * stride_ajc
        + offs_d[None, :] * stride_ajd,
        mask,
        1.0,
    )
    n = load_acc(
        agg_n_ptr
        + pid_b * stride_anb
        + offs_c[:, None] * stride_anc
        + offs_d[None, :] * stride_and,
        mask,
        0.0,
    )
    c = load_acc(
        agg_c_ptr
        + pid_b * stride_acb
        + offs_c[:, None] * stride_acc
        + offs_d[None, :] * stride_acd,
        mask,
        0.0,
    )
    _, n_s, c_s = tl.associative_scan((j, n, c), 0, _compose_diag2)
    store_acc(
        incl_n_ptr
        + pid_b * stride_inb
        + offs_c[:, None] * stride_inc
        + offs_d[None, :] * stride_ind,
        n_s,
        mask,
    )
    store_acc(
        incl_c_ptr
        + pid_b * stride_icb
        + offs_c[:, None] * stride_icc
        + offs_d[None, :] * stride_icd,
        c_s,
        mask,
    )


@triton.jit
def _nc_apply_h_kernel(
    pre_ptr,
    j_ptr,
    n_loc_ptr,
    c_loc_ptr,
    incl_n_ptr,
    incl_c_ptr,
    out_ptr,
    time,
    d_h,
    eps,
    stride_pb,
    stride_pt,
    stride_pd,
    stride_jb,
    stride_jt,
    stride_jd,
    stride_nb,
    stride_nt,
    stride_nd,
    stride_cb,
    stride_ct,
    stride_cd,
    stride_inb,
    stride_inc,
    stride_ind,
    stride_icb,
    stride_icc,
    stride_icd,
    stride_ob,
    stride_ot,
    stride_os,
    stride_od,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Downsweep with exclusive chunk carries stored in ``incl_*[c]``."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    j = load_acc(
        j_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + offs_d[None, :] * stride_jd,
        mask,
        1.0,
    )
    n_loc = load_acc(
        n_loc_ptr + pid_b * stride_nb + offs_t[:, None] * stride_nt + offs_d[None, :] * stride_nd,
        mask,
        0.0,
    )
    c_loc = load_acc(
        c_loc_ptr + pid_b * stride_cb + offs_t[:, None] * stride_ct + offs_d[None, :] * stride_cd,
        mask,
        0.0,
    )
    # ``incl_*[c]`` holds the exclusive carry into chunk ``c`` (rank + prior
    # windows). Legacy single-GPU callers write zeros at c=0 and inclusive
    # window-ends shifted so idx_c=c-1; CP callers write excl directly at c.
    cn = load_acc(
        incl_n_ptr + pid_b * stride_inb + pid_c * stride_inc + offs_d * stride_ind,
        dmask,
        0.0,
    )
    cc = load_acc(
        incl_c_ptr + pid_b * stride_icb + pid_c * stride_icc + offs_d * stride_icd,
        dmask,
        0.0,
    )
    n = j * cn[None, :] + n_loc
    c = j * cc[None, :] + c_loc
    z_o = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 3, mask, stride_pb, stride_pt, stride_pd)
    h = tl.sigmoid(z_o) * (c / (n + eps))
    store_acc(
        out_ptr
        + pid_b * stride_ob
        + offs_t[:, None] * stride_ot
        + SLOT_C * stride_os
        + offs_d[None, :] * stride_od,
        c,
        mask,
    )
    store_acc(
        out_ptr
        + pid_b * stride_ob
        + offs_t[:, None] * stride_ot
        + SLOT_N * stride_os
        + offs_d[None, :] * stride_od,
        n,
        mask,
    )
    store_acc(
        out_ptr
        + pid_b * stride_ob
        + offs_t[:, None] * stride_ot
        + SLOT_H * stride_os
        + offs_d[None, :] * stride_od,
        h,
        mask,
    )


def frozen_gate_scan_triton(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
    out: Tensor | None = None,
    r: Tensor | None = None,
    h_prev: Tensor | None = None,
) -> Tensor:
    """CUDA twin of ``slstm_frozen_gate_scan``. ``pre`` is ``(B, T, 4 d_h)``.

    Default path is a single serial-in-``t`` kernel (mega-fuse): ``(m,n,c,h)``
    stay in registers; only the final state hits HBM. The older two-level
    ``_m_*`` / ``_nc_*`` parallel scan remains as ``frozen_gate_scan_triton_parallel``.

    ``out`` : optional ``(B, T, 4, d_h)`` written in-place (Picard buffer reuse).
    ``r`` / ``h_prev`` : when both set, ``pre`` is raw ``wx`` and gates get
    ``+ R ⊙ h_prev`` in-kernel (global Picard; no host ``work`` buffer).
    """
    pre = pre.contiguous()
    batch, time, four_d = pre.shape
    if four_d % 4 != 0:
        raise ValueError(f"pre last dim {four_d} is not 4 * d_h")
    d_h = four_d // 4
    if h0 is None:
        h0 = pre.new_zeros(batch, SLSTM_SLOTS, d_h)
    else:
        h0 = h0.contiguous()
        if h0.shape != (batch, SLSTM_SLOTS, d_h):
            raise ValueError(f"h0 shape {tuple(h0.shape)} != {(batch, SLSTM_SLOTS, d_h)}")
        if h0.dtype != pre.dtype:
            h0 = h0.to(dtype=pre.dtype)
    use_rh = r is not None or h_prev is not None
    if use_rh:
        if r is None or h_prev is None:
            raise ValueError("frozen_gate_scan_triton: r and h_prev must be set together")
        r = r.contiguous()
        h_prev = h_prev.contiguous()
        if r.shape != (SLSTM_SLOTS, d_h):
            raise ValueError(f"r shape {tuple(r.shape)} != {(SLSTM_SLOTS, d_h)}")
        if h_prev.shape != (batch, time, d_h):
            raise ValueError(
                f"h_prev shape {tuple(h_prev.shape)} != {(batch, time, d_h)}"
            )
        if h_prev.dtype != pre.dtype:
            raise ValueError(f"h_prev dtype {h_prev.dtype} != pre dtype {pre.dtype}")
        validate_cuda_tensors(pre, h0, r, h_prev, name="frozen_gate_scan_triton")
        rh_hpb, rh_hpt, rh_hpd = h_prev.stride()
        rh_rd = r.stride(0)
    else:
        validate_cuda_tensors(pre, h0, name="frozen_gate_scan_triton")
        # Unused when ``USE_RH=False`` (DCE); no extra alloc on the hot path.
        r = pre
        h_prev = pre
        rh_hpb = rh_hpt = rh_hpd = rh_rd = 0
    if out is None:
        out = pre.new_empty(batch, time, SLSTM_SLOTS, d_h)
    else:
        if out.shape != (batch, time, SLSTM_SLOTS, d_h):
            raise ValueError(
                f"out shape {tuple(out.shape)} != {(batch, time, SLSTM_SLOTS, d_h)}"
            )
        if out.dtype != pre.dtype:
            raise ValueError(f"out dtype {out.dtype} != pre dtype {pre.dtype}")
        # Keep the caller's storage (windowed Picard writes into ``states[:, t0:t1]``).
    n_dtiles = triton.cdiv(d_h, _SERIAL_BLOCK_D)
    _frozen_gate_serial_kernel[(batch, n_dtiles)](
        pre,
        h0,
        out,
        h_prev,
        r,
        time,
        d_h,
        float(eps),
        *pre.stride(),
        *h0.stride(),
        *out.stride(),
        rh_hpb,
        rh_hpt,
        rh_hpd,
        rh_rd,
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_D=_SERIAL_BLOCK_D,
        USE_RH=use_rh,
    )
    if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
        log.debug(
            "frozen_gate_scan_triton",
            extra={
                "seq_len": time,
                "batch": batch,
                "d_h": d_h,
                "path": "serial_rh" if use_rh else "serial",
            },
        )
    return out


def picard_diag_windowed_triton(
    wx: Tensor,
    r: Tensor,
    *,
    eps: float,
    n_picard: int,
    h0: Tensor | None = None,
    chunk_len: int = _PICARD_TILE,
    out: Tensor | None = None,
) -> Tensor:
    """Windowed Picard for ``mix='diag'``: **one** Triton launch over all tiles.

    Each tile runs ``1 + n_picard`` fused Rh passes; ``h`` scratch is ``O(C)``;
    full state is written on the last pass per tile. Carry stays in registers.
    """
    wx = wx.contiguous()
    batch, time, four_d = wx.shape
    if four_d % 4 != 0:
        raise ValueError(f"wx last dim {four_d} is not 4 * d_h")
    d_h = four_d // 4
    if n_picard < 0:
        raise ValueError(f"n_picard must be >= 0, got {n_picard!r}")
    if chunk_len < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len!r}")
    r = r.contiguous()
    if r.shape != (SLSTM_SLOTS, d_h):
        raise ValueError(f"r shape {tuple(r.shape)} != {(SLSTM_SLOTS, d_h)}")
    if h0 is None:
        h0_buf = wx.new_zeros(batch, SLSTM_SLOTS, d_h)
    else:
        h0_buf = h0.contiguous()
        if h0_buf.shape != (batch, SLSTM_SLOTS, d_h):
            raise ValueError(f"h0 shape {tuple(h0_buf.shape)} != {(batch, SLSTM_SLOTS, d_h)}")
        if h0_buf.dtype != wx.dtype:
            h0_buf = h0_buf.to(dtype=wx.dtype)
    validate_cuda_tensors(wx, r, h0_buf, name="picard_diag_windowed_triton")
    if out is None:
        out = wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
    else:
        if out.shape != (batch, time, SLSTM_SLOTS, d_h):
            raise ValueError(
                f"out shape {tuple(out.shape)} != {(batch, time, SLSTM_SLOTS, d_h)}"
            )
        if out.dtype != wx.dtype:
            raise ValueError(f"out dtype {out.dtype} != wx dtype {wx.dtype}")
    tile = min(int(chunk_len), time)
    h_prev = wx.new_empty(batch, tile, d_h)
    h_curr = wx.new_empty(batch, tile, d_h)
    n_dtiles = triton.cdiv(d_h, _PICARD_TILE_D)
    _picard_diag_all_tiles_kernel[(batch, n_dtiles)](
        wx,
        r,
        h0_buf,
        h_prev,
        h_curr,
        out,
        time,
        d_h,
        float(eps),
        int(n_picard),
        tile,
        *wx.stride(),
        r.stride(0),
        *h0_buf.stride(),
        *h_prev.stride(),
        *h_curr.stride(),
        *out.stride(),
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_D=_PICARD_TILE_D,
    )
    if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
        log.debug(
            "picard_diag_windowed_triton",
            extra={
                "seq_len": time,
                "batch": batch,
                "d_h": d_h,
                "n_picard": n_picard,
                "chunk_len": tile,
                "path": "all_tiles_one_launch",
            },
        )
    return out


def picard_diag_global_triton(
    wx: Tensor,
    r: Tensor,
    *,
    eps: float,
    n_picard: int,
    h0: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Global Picard for ``mix='diag'``: serial scans with Rh fused in-kernel.

    Same semantics as repeated ``frozen_gate_scan(wx + R⊙h_prev)`` with fixed
    ``h0``. Scratch is one ``(B, T, d_h)`` shifted-hidden strip (no ``(B,T,4 d_h)``
    host ``work`` buffer). Uses the tuned serial mega-fuse kernel per pass.

    When ``T ≤ _PICARD_TILE``, delegates to the one-launch windowed kernel
    (tile ``=T``): same algebra, fewer launches on chunk windows.
    """
    wx = wx.contiguous()
    batch, time, four_d = wx.shape
    if four_d % 4 != 0:
        raise ValueError(f"wx last dim {four_d} is not 4 * d_h")
    if time <= _PICARD_TILE:
        return picard_diag_windowed_triton(
            wx,
            r,
            eps=eps,
            n_picard=n_picard,
            h0=h0,
            chunk_len=time,
            out=out,
        )
    d_h = four_d // 4
    if n_picard < 0:
        raise ValueError(f"n_picard must be >= 0, got {n_picard!r}")
    r = r.contiguous()
    if r.shape != (SLSTM_SLOTS, d_h):
        raise ValueError(f"r shape {tuple(r.shape)} != {(SLSTM_SLOTS, d_h)}")
    if h0 is None:
        h0_buf = wx.new_zeros(batch, SLSTM_SLOTS, d_h)
    else:
        h0_buf = h0.contiguous()
        if h0_buf.shape != (batch, SLSTM_SLOTS, d_h):
            raise ValueError(f"h0 shape {tuple(h0_buf.shape)} != {(batch, SLSTM_SLOTS, d_h)}")
        if h0_buf.dtype != wx.dtype:
            h0_buf = h0_buf.to(dtype=wx.dtype)
    validate_cuda_tensors(wx, r, h0_buf, name="picard_diag_global_triton")
    if out is None:
        out = wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
    else:
        if out.shape != (batch, time, SLSTM_SLOTS, d_h):
            raise ValueError(
                f"out shape {tuple(out.shape)} != {(batch, time, SLSTM_SLOTS, d_h)}"
            )
        if out.dtype != wx.dtype:
            raise ValueError(f"out dtype {out.dtype} != wx dtype {wx.dtype}")
    frozen_gate_scan_triton(wx, eps=eps, h0=h0_buf, out=out)
    if n_picard == 0:
        return out
    h_prev = wx.new_empty(batch, time, d_h)
    for _ in range(n_picard):
        h_prev[:, 0].copy_(h0_buf[:, SLSTM_HIDDEN])
        h_prev[:, 1:].copy_(out[:, :-1, SLSTM_HIDDEN, :])
        frozen_gate_scan_triton(
            wx, eps=eps, h0=h0_buf, out=out, r=r, h_prev=h_prev
        )
    if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
        log.debug(
            "picard_diag_global_triton",
            extra={
                "seq_len": time,
                "batch": batch,
                "d_h": d_h,
                "n_picard": n_picard,
                "path": "serial_fused_rh",
            },
        )
    return out


def frozen_gate_scan_triton_parallel(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
) -> Tensor:
    """Two-level max-plus + ``ax+b`` Picard (legacy; kept for A/B benches)."""
    pre = pre.contiguous()
    batch, time, four_d = pre.shape
    if four_d % 4 != 0:
        raise ValueError(f"pre last dim {four_d} is not 4 * d_h")
    d_h = four_d // 4
    if h0 is None:
        h0 = pre.new_zeros(batch, SLSTM_SLOTS, d_h)
    else:
        h0 = h0.contiguous()
        if h0.shape != (batch, SLSTM_SLOTS, d_h):
            raise ValueError(f"h0 shape {tuple(h0.shape)} != {(batch, SLSTM_SLOTS, d_h)}")
        if h0.dtype != pre.dtype:
            h0 = h0.to(dtype=pre.dtype)
    validate_cuda_tensors(pre, h0, name="frozen_gate_scan_triton_parallel")
    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    if n_chunks > _CHUNK_PAD:
        raise ValueError(
            f"T={time} needs {n_chunks} tiles of {_BLOCK_T}; cap is {_CHUNK_PAD} "
            f"(T≤{_BLOCK_T * _CHUNK_PAD}). Tile count cap for this kernel."
        )
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    grid = (batch, n_chunks, n_dtiles)
    grid_d = (batch, n_dtiles)
    m0 = h0[:, SLSTM_STABILIZER, :].contiguous()

    a_loc = pre.new_empty(batch, time, d_h)
    b_loc = pre.new_empty(batch, time, d_h)
    agg_a = pre.new_empty(batch, n_chunks, d_h)
    agg_b = pre.new_empty(batch, n_chunks, d_h)
    m = pre.new_empty(batch, time, d_h)
    incl_m = m.new_empty(batch, n_chunks, d_h)

    _m_local_kernel[grid](
        pre,
        a_loc,
        b_loc,
        agg_a,
        agg_b,
        time,
        d_h,
        *pre.stride(),
        *a_loc.stride(),
        *b_loc.stride(),
        *agg_a.stride(),
        *agg_b.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        NEG_INF=_NEG_INF,
    )
    if n_chunks == 1:
        incl_m.zero_()
    else:
        _m_chunk_kernel[grid_d](
            agg_a,
            agg_b,
            m0,
            incl_m,
            n_chunks,
            d_h,
            *agg_a.stride(),
            *agg_b.stride(),
            m0.stride(0),
            m0.stride(1),
            *incl_m.stride(),
            CHUNK_PAD=_CHUNK_PAD,
            BLOCK_D=_BLOCK_D,
            NEG_INF=_NEG_INF,
        )
    _m_apply_kernel[grid](
        a_loc,
        b_loc,
        m0,
        incl_m,
        m,
        time,
        d_h,
        *a_loc.stride(),
        *b_loc.stride(),
        m0.stride(0),
        m0.stride(1),
        *incl_m.stride(),
        *m.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        NEG_INF=_NEG_INF,
    )

    # Phase-m temps are dead after apply. Reuse as nc scan lanes;
    # ``m`` is the stabilizer copied into ``out``.
    j_loc = a_loc
    n_loc = b_loc
    c_loc = pre.new_empty(batch, time, d_h)
    agg_j = agg_a
    agg_n = agg_b
    agg_c = pre.new_empty(batch, n_chunks, d_h)
    _nc_local_kernel[grid](
        pre,
        m,
        h0,
        j_loc,
        n_loc,
        c_loc,
        agg_j,
        agg_n,
        agg_c,
        time,
        d_h,
        *pre.stride(),
        *m.stride(),
        *h0.stride(),
        *j_loc.stride(),
        *n_loc.stride(),
        *c_loc.stride(),
        *agg_j.stride(),
        *agg_n.stride(),
        *agg_c.stride(),
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    out = pre.new_empty(batch, time, SLSTM_SLOTS, d_h)
    out[:, :, SLSTM_STABILIZER, :] = m
    incl_n = incl_m
    incl_c = c_loc.new_empty(batch, n_chunks, d_h)
    if n_chunks > 1:
        _nc_chunk_kernel[grid_d](
            agg_j,
            agg_n,
            agg_c,
            incl_n,
            incl_c,
            n_chunks,
            d_h,
            *agg_j.stride(),
            *agg_n.stride(),
            *agg_c.stride(),
            *incl_n.stride(),
            *incl_c.stride(),
            CHUNK_PAD=_CHUNK_PAD,
            BLOCK_D=_BLOCK_D,
        )
        # Inclusive window-ends → exclusive carries into each window.
        excl_n = incl_n.new_empty(batch, n_chunks, d_h)
        excl_c = incl_c.new_empty(batch, n_chunks, d_h)
        excl_n[:, 0].zero_()
        excl_c[:, 0].zero_()
        excl_n[:, 1:] = incl_n[:, :-1]
        excl_c[:, 1:] = incl_c[:, :-1]
        incl_n = excl_n
        incl_c = excl_c
    else:
        incl_n.zero_()
        incl_c.zero_()
    _nc_apply_h_kernel[grid](
        pre,
        j_loc,
        n_loc,
        c_loc,
        incl_n,
        incl_c,
        out,
        time,
        d_h,
        float(eps),
        *pre.stride(),
        *j_loc.stride(),
        *n_loc.stride(),
        *c_loc.stride(),
        *incl_n.stride(),
        *incl_c.stride(),
        *out.stride(),
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
        log.debug(
            "frozen_gate_scan_triton_parallel",
            extra={"seq_len": time, "batch": batch, "d_h": d_h, "n_chunks": n_chunks},
        )
    return out


def _compose_maxplus_torch(a_l: Tensor, b_l: Tensor, a_r: Tensor, b_r: Tensor) -> tuple[Tensor, Tensor]:
    """Host twin of Triton ``_compose_maxplus``: ``m ↦ max(a+m, b)``."""
    return a_r + a_l, torch.maximum(a_r + b_l, b_r)


def _apply_maxplus_torch(m: Tensor, a: Tensor, b: Tensor) -> Tensor:
    return torch.maximum(a + m, b)


def _tile_maxplus_end(agg_a: Tensor, agg_b: Tensor) -> tuple[Tensor, Tensor]:
    """Compose window Triton ``(a,b)`` monoids → tile end."""
    batch, n_chunks, dim = agg_a.shape
    a = agg_a.new_zeros(batch, dim)
    b = agg_a.new_full((batch, dim), _NEG_INF)
    for c in range(n_chunks):
        a, b = _compose_maxplus_torch(a, b, agg_a[:, c], agg_b[:, c])
    return a, b


def _maxplus_incl_ends(agg_a: Tensor, agg_b: Tensor, m0: Tensor) -> Tensor:
    """Inclusive ``m`` at each window end given tile-start ``m0``."""
    batch, n_chunks, dim = agg_a.shape
    incl = agg_a.new_empty(batch, n_chunks, dim)
    m = m0
    for c in range(n_chunks):
        m = _apply_maxplus_torch(m, agg_a[:, c], agg_b[:, c])
        incl[:, c] = m
    return incl


def _maxplus_rank_carry(
    a_end: Tensor,
    b_end: Tensor,
    m0: Tensor,
    *,
    group,
) -> Tensor:
    """Exclusive Triton-``(max,+)`` prefix → ``m_carry`` for this rank."""
    import torch.distributed as dist

    from pararnn.solvers.seq_parallel import _all_gather_monoid

    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    if world <= 1:
        return m0
    rank = dist.get_rank(group=group)
    a_list, b_list = _all_gather_monoid(a_end, b_end, world, group, needs_grad=False)
    acc_a = a_end.new_zeros(a_end.shape)
    acc_b = a_end.new_full(a_end.shape, _NEG_INF)
    m_carry = m0
    for k in range(world):
        if k == rank:
            if rank == 0:
                m_carry = m0
            else:
                m_carry = _apply_maxplus_torch(m0, acc_a, acc_b)
        acc_a, acc_b = _compose_maxplus_torch(acc_a, acc_b, a_list[k], b_list[k])
    return m_carry


def _compose_diag_torch(
    j_l: Tensor, n_l: Tensor, c_l: Tensor, j_r: Tensor, n_r: Tensor, c_r: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Host twin of Triton ``_compose_diag2``."""
    return j_r * j_l, j_r * n_l + n_r, j_r * c_l + c_r


def _tile_diag_end(
    agg_j: Tensor, agg_n: Tensor, agg_c: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Compose window affine monoids → tile end ``(J, B_n, B_c)``."""
    batch, n_chunks, dim = agg_j.shape
    j = agg_j.new_ones(batch, dim)
    n = agg_j.new_zeros(batch, dim)
    c = agg_j.new_zeros(batch, dim)
    for i in range(n_chunks):
        j, n, c = _compose_diag_torch(j, n, c, agg_j[:, i], agg_n[:, i], agg_c[:, i])
    return j, n, c


def _diag_excl_carries(
    agg_j: Tensor,
    agg_n: Tensor,
    agg_c: Tensor,
    n_carry: Tensor,
    c_carry: Tensor,
) -> tuple[Tensor, Tensor]:
    """Exclusive ``(n,c)`` into each window given rank carries."""
    batch, n_chunks, dim = agg_j.shape
    excl_n = agg_j.new_empty(batch, n_chunks, dim)
    excl_c = agg_j.new_empty(batch, n_chunks, dim)
    n_acc = n_carry
    c_acc = c_carry
    for i in range(n_chunks):
        excl_n[:, i] = n_acc
        excl_c[:, i] = c_acc
        n_acc = agg_j[:, i] * n_acc + agg_n[:, i]
        c_acc = agg_j[:, i] * c_acc + agg_c[:, i]
    return excl_n, excl_c


def _diag_rank_carry(
    j_end: Tensor,
    n_end: Tensor,
    c_end: Tensor,
    *,
    group,
    rank: int,
) -> tuple[Tensor, Tensor]:
    """Exclusive diag prefix → ``(n_carry, c_carry)`` for this rank.

    Rank 0's ``n0``/``c0`` are baked into the local residual at ``t=0``;
    the monoid exclusive prefix therefore starts from zero on every world.
    """
    import torch.distributed as dist

    from pararnn.solvers.seq_parallel import _all_gather_monoid

    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    zeros = n_end.new_zeros(n_end.shape)
    if world <= 1:
        return zeros, zeros
    j_list, n_list = _all_gather_monoid(j_end, n_end, world, group, needs_grad=False)
    _, c_list = _all_gather_monoid(j_end, c_end, world, group, needs_grad=False)
    acc_j = j_end.new_ones(j_end.shape)
    acc_n = zeros.clone()
    acc_c = zeros.clone()
    n_carry = zeros
    c_carry = zeros
    for k in range(world):
        if k == rank:
            n_carry = acc_n.clone()
            c_carry = acc_c.clone()
        acc_j, acc_n, acc_c = _compose_diag_torch(
            acc_j, acc_n, acc_c, j_list[k], n_list[k], c_list[k]
        )
    return n_carry, c_carry


def frozen_gate_scan_fused_cp(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
    group=None,
    out: Tensor | None = None,
) -> Tensor:
    """Two-stage concurrent CP frozen-gate: ``(max,+)`` then affine ``n/c/h``.

    Stage 1: Triton ``C=128`` ``(Q,M)`` upsweep → monoid AllGather → exact ``m``.
    Stage 2: Triton affine upsweep from exact ``m`` → monoid AllGather → ``n,c,h``.
    All ranks run both stages from ``t=0`` (no serial P2P).
    """
    import torch.distributed as dist

    pre = pre.contiguous()
    batch, time, four_d = pre.shape
    if four_d % 4 != 0:
        raise ValueError(f"pre last dim {four_d} is not 4 * d_h")
    d_h = four_d // 4
    if h0 is None:
        h0 = pre.new_zeros(batch, SLSTM_SLOTS, d_h)
    else:
        h0 = h0.contiguous()
        if h0.shape != (batch, SLSTM_SLOTS, d_h):
            raise ValueError(f"h0 shape {tuple(h0.shape)} != {(batch, SLSTM_SLOTS, d_h)}")
        if h0.dtype != pre.dtype:
            h0 = h0.to(dtype=pre.dtype)
    validate_cuda_tensors(pre, h0, name="frozen_gate_scan_fused_cp")

    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    rank = dist.get_rank(group=group) if dist.is_initialized() else 0
    n_chunks = max(1, (time + _BLOCK_T - 1) // _BLOCK_T)
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    grid = (batch, n_chunks, n_dtiles)
    grid_d = (batch, n_dtiles)

    m0 = h0[:, SLSTM_STABILIZER, :].contiguous().float()
    # --- Stage 1: stabilizer m ---
    a_loc = pre.new_empty(batch, time, d_h)
    b_loc = pre.new_empty(batch, time, d_h)
    agg_a = pre.new_empty(batch, n_chunks, d_h)
    agg_b = pre.new_empty(batch, n_chunks, d_h)
    _m_local_kernel[grid](
        pre,
        a_loc,
        b_loc,
        agg_a,
        agg_b,
        time,
        d_h,
        *pre.stride(),
        *a_loc.stride(),
        *b_loc.stride(),
        *agg_a.stride(),
        *agg_b.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        NEG_INF=_NEG_INF,
    )
    q_end, m_end = _tile_maxplus_end(agg_a.float(), agg_b.float())
    m_carry = _maxplus_rank_carry(q_end, m_end, m0, group=group)
    m = pre.new_empty(batch, time, d_h)
    if n_chunks <= _CHUNK_PAD:
        incl_m = m.new_empty(batch, n_chunks, d_h)
        m_carry_t = m_carry.to(dtype=pre.dtype)
        _m_chunk_kernel[grid_d](
            agg_a,
            agg_b,
            m_carry_t,
            incl_m,
            n_chunks,
            d_h,
            *agg_a.stride(),
            *agg_b.stride(),
            m_carry_t.stride(0),
            m_carry_t.stride(1),
            *incl_m.stride(),
            CHUNK_PAD=_CHUNK_PAD,
            BLOCK_D=_BLOCK_D,
            NEG_INF=_NEG_INF,
        )
        _m_apply_kernel[grid](
            a_loc,
            b_loc,
            m_carry_t,
            incl_m,
            m,
            time,
            d_h,
            *a_loc.stride(),
            *b_loc.stride(),
            m_carry_t.stride(0),
            m_carry_t.stride(1),
            *incl_m.stride(),
            *m.stride(),
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            NEG_INF=_NEG_INF,
        )
    else:
        incl_m = _maxplus_incl_ends(agg_a.float(), agg_b.float(), m_carry)
        m_f = m.float()
        for c in range(n_chunks):
            t0 = c * _BLOCK_T
            t1 = min(t0 + _BLOCK_T, time)
            carry = m_carry if c == 0 else incl_m[:, c - 1]
            m_f[:, t0:t1] = torch.maximum(
                a_loc[:, t0:t1].float() + carry.unsqueeze(1),
                b_loc[:, t0:t1].float(),
            )
        m.copy_(m_f.to(dtype=pre.dtype))

    # --- Stage 2: affine n, c + h from exact m ---
    # h0_eff: m-slot = m_carry for m_prev at local t=0; rank>0 skips n/c inject.
    h0_eff = h0.clone()
    h0_eff[:, SLSTM_STABILIZER, :] = m_carry.to(dtype=pre.dtype)
    if rank > 0:
        h0_eff[:, SLSTM_NORMALIZER, :].zero_()
        h0_eff[:, SLSTM_CELL, :].zero_()

    j_loc = a_loc  # reuse Stage-1 scratch
    n_loc = b_loc
    c_loc = pre.new_empty(batch, time, d_h)
    agg_j = agg_a
    agg_n = agg_b
    agg_c = pre.new_empty(batch, n_chunks, d_h)
    _nc_local_kernel[grid](
        pre,
        m,
        h0_eff,
        j_loc,
        n_loc,
        c_loc,
        agg_j,
        agg_n,
        agg_c,
        time,
        d_h,
        *pre.stride(),
        *m.stride(),
        *h0_eff.stride(),
        *j_loc.stride(),
        *n_loc.stride(),
        *c_loc.stride(),
        *agg_j.stride(),
        *agg_n.stride(),
        *agg_c.stride(),
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    j_end, n_end, c_end = _tile_diag_end(agg_j.float(), agg_n.float(), agg_c.float())
    n_carry, c_carry = _diag_rank_carry(
        j_end, n_end, c_end, group=group, rank=rank
    )
    excl_n, excl_c = _diag_excl_carries(
        agg_j.float(), agg_n.float(), agg_c.float(), n_carry, c_carry
    )
    excl_n = excl_n.to(dtype=pre.dtype)
    excl_c = excl_c.to(dtype=pre.dtype)

    if out is None:
        stacked = pre.new_empty(batch, time, SLSTM_SLOTS, d_h)
    else:
        if out.shape != (batch, time, SLSTM_SLOTS, d_h):
            raise ValueError(f"out shape {tuple(out.shape)} != {(batch, time, SLSTM_SLOTS, d_h)}")
        stacked = out
    stacked[:, :, SLSTM_STABILIZER, :] = m
    _nc_apply_h_kernel[grid](
        pre,
        j_loc,
        n_loc,
        c_loc,
        excl_n,
        excl_c,
        stacked,
        time,
        d_h,
        float(eps),
        *pre.stride(),
        *j_loc.stride(),
        *n_loc.stride(),
        *c_loc.stride(),
        *excl_n.stride(),
        *excl_c.stride(),
        *stacked.stride(),
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
        log.debug(
            "frozen_gate_scan_fused_cp",
            extra={
                "seq_len": time,
                "batch": batch,
                "d_h": d_h,
                "n_chunks": n_chunks,
                "world": world,
                "rank": rank,
            },
        )
    return stacked


def frozen_gate_scan_fused_cp_windowed(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
    group=None,
    out: Tensor | None = None,
) -> Tensor:
    """Long-T alias of ``frozen_gate_scan_fused_cp`` (hierarchical tile compose).

    Local upsweep uses ``C=128`` ``(max,+)`` / affine tiles; rank aggregates
    AllGather; downsweep applies the exclusive prefix. For ``n_chunks`` past
    ``_CHUNK_PAD`` the compose falls back to a host loop over tile ends —
    same algebra, peak still ∝ local ``T``. Picard wraps this with windowed
    ``R·h`` in ``_picard_sharded_windowed_monoid``.
    """
    return frozen_gate_scan_fused_cp(
        pre, eps=eps, h0=h0, group=group, out=out
    )

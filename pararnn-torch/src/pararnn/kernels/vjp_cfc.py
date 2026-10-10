"""Triton / eager VJP of the ParaCfC recurrence (σ(-(soft·Δt+b)) + diag ``u``).

``h_prev`` is detached (eq. 2.6 already applied ``J^T``). ``∇u`` reduces with
per-batch fp32 tiles then ``.sum`` — no atomics.

``wx`` is ``(B, T, 3 d_h) = (f_pre, c_x, b)``; ``dt`` is ``(B, T)`` (or
``(B, T, 1)``). Gate chain matches Autograd on
``a = σ(-(softplus(f)·dt + b))``: ``∂a/∂soft = a(1-a)·(-dt)``,
``∂softplus/∂f = σ(f)``, ``∂a/∂b = a(1-a)·(-1)``.

``no_gate``: ``h = a⊙h + n`` so ``∂h/∂a = h``, ``∂h/∂n = 1`` (convex uses
``∂h/∂a = h-n``, ``∂h/∂n = 1-a``).

``cfc_diag_jacobian`` writes the Newton ``J`` tile for eq. 2.6 reverse without
an eager ``step_with_jacobian`` rematerialize.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels._fused_common import _softplus
from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors

_BLOCK_T = 128
_BLOCK_D = 32


def _normalize_dt(dt: Tensor, batch: int, time: int) -> Tensor:
    if dt.ndim == 3 and dt.shape[-1] == 1:
        dt = dt.squeeze(-1)
    if dt.shape != (batch, time):
        raise ValueError(f"dt shape {tuple(dt.shape)} != {(batch, time)}")
    return dt


@triton.jit
def _cfc_jac_kernel(
    h_ptr,
    wx_ptr,
    dt_ptr,
    u_ptr,
    j_ptr,
    time,
    d_h,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_dtb,
    stride_dtt,
    stride_jb,
    stride_jt,
    stride_jd,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_GATE: tl.constexpr,
):
    """Write diagonal ``∂h_t/∂h_{t-1}`` for eq. 2.6 reverse (gate_mix=input)."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    mask_t = offs_t < time

    h = load_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        mask,
        0.0,
    )
    base = wx_ptr + pid_b * stride_wb + offs_t[:, None] * stride_wt
    f_pre = load_acc(base + offs_d[None, :] * stride_wd, mask, 0.0)
    cx = load_acc(base + (offs_d[None, :] + d_h) * stride_wd, mask, 0.0)
    b = load_acc(base + (offs_d[None, :] + 2 * d_h) * stride_wd, mask, 0.0)
    dt = load_acc(
        dt_ptr + pid_b * stride_dtb + offs_t[:, None] * stride_dtt, mask_t[:, None], 1.0
    )
    u = load_acc(u_ptr + offs_d[None, :], dmask[None, :], 0.0)

    soft = _softplus(f_pre)
    a = tl.sigmoid(-(soft * dt + b))
    n = _nv_tanh(cx + u * h)
    n_p = 1.0 - n * n
    if NO_GATE:
        j = a + n_p * u
    else:
        j = a + (1.0 - a) * n_p * u
    store_acc(
        j_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + offs_d[None, :] * stride_jd,
        j,
        mask,
    )


@triton.jit
def _cfc_vjp_kernel(
    h_ptr,
    wx_ptr,
    dt_ptr,
    raw_dt_ptr,
    u_ptr,
    mu_ptr,
    gwx_ptr,
    gdt_ptr,
    gu_ptr,
    time,
    d_h,
    dt_eps,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_dtb,
    stride_dtt,
    stride_rdb,
    stride_rdt,
    stride_mb,
    stride_mt,
    stride_md,
    stride_gb,
    stride_gt,
    stride_gd,
    stride_gdtb,
    stride_gdtc,
    stride_gdtt,
    stride_ub,
    stride_uc,
    stride_ud,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NO_GATE: tl.constexpr,
    HAS_DT_MASK: tl.constexpr,
    GDT_DIRECT: tl.constexpr,
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
    mask_t = offs_t < time

    h = load_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        mask,
        0.0,
    )
    mu = load_acc(
        mu_ptr + pid_b * stride_mb + offs_t[:, None] * stride_mt + offs_d[None, :] * stride_md,
        mask,
        0.0,
    )
    base = wx_ptr + pid_b * stride_wb + offs_t[:, None] * stride_wt
    f_pre = load_acc(base + offs_d[None, :] * stride_wd, mask, 0.0)
    cx = load_acc(base + (offs_d[None, :] + d_h) * stride_wd, mask, 0.0)
    b = load_acc(base + (offs_d[None, :] + 2 * d_h) * stride_wd, mask, 0.0)
    dt = load_acc(
        dt_ptr + pid_b * stride_dtb + offs_t[:, None] * stride_dtt, mask_t[:, None], 1.0
    )
    u = load_acc(u_ptr + offs_d[None, :], dmask[None, :], 0.0)

    soft = _softplus(f_pre)
    a = tl.sigmoid(-(soft * dt + b))
    n = _nv_tanh(cx + u * h)
    if NO_GATE:
        d_h_da = mu * h
        d_npre = mu * (1.0 - n * n)
    else:
        d_h_da = mu * (h - n)
        d_npre = mu * (1.0 - a) * (1.0 - n * n)
    a1a = a * (1.0 - a)
    g_f = d_h_da * a1a * (-dt) * tl.sigmoid(f_pre)
    g_dt_chan = d_h_da * a1a * (-soft)
    g_b = d_h_da * a1a * (-1.0)

    gout = gwx_ptr + pid_b * stride_gb + offs_t[:, None] * stride_gt
    store_acc(gout + offs_d[None, :] * stride_gd, g_f, mask)
    store_acc(gout + (offs_d[None, :] + d_h) * stride_gd, d_npre, mask)
    store_acc(gout + (offs_d[None, :] + 2 * d_h) * stride_gd, g_b, mask)

    # Reduce ∂L/∂dt over channels; optional clamp_min mask on raw Δt.
    g_dt_row = tl.sum(tl.where(mask, g_dt_chan, 0.0), 1)
    if HAS_DT_MASK:
        raw_dt = load_acc(
            raw_dt_ptr + pid_b * stride_rdb + offs_t * stride_rdt,
            mask_t,
            0.0,
        )
        g_dt_row = tl.where(raw_dt >= dt_eps, g_dt_row, 0.0)
    if GDT_DIRECT:
        # Single d-tile: write ``(B, T)`` directly (no host tile ``sum``).
        store_acc(
            gdt_ptr + pid_b * stride_gdtb + offs_t * stride_gdtt,
            g_dt_row,
            mask_t,
        )
    else:
        store_acc(
            gdt_ptr
            + pid_b * stride_gdtb
            + pid_d * stride_gdtc
            + offs_t * stride_gdtt,
            g_dt_row,
            mask_t,
        )

    off_u = pid_b * stride_ub + pid_c * stride_uc + offs_d * stride_ud
    store_acc(gu_ptr + off_u, tl.sum(tl.where(mask, d_npre * h, 0.0), 0), dmask)


def cfc_diag_jacobian(
    h_prev: Tensor,
    wx: Tensor,
    dt: Tensor,
    u: Tensor,
    *,
    no_gate: bool = False,
) -> Tensor:
    """Diagonal ``J_t = ∂h_t/∂h_{t-1}`` for fused eq. 2.6 reverse (CUDA)."""
    if h_prev.device.type != "cuda":
        return cfc_diag_jacobian_eager(h_prev, wx, dt, u, no_gate=no_gate)
    batch, time, d_h = h_prev.shape
    dt = _normalize_dt(dt, batch, time)
    validate_cuda_tensors(h_prev, wx, dt, u, name="cfc_diag_jacobian")
    h_prev = h_prev.contiguous()
    wx = wx.contiguous()
    dt = dt.contiguous()
    u = u.contiguous()
    if wx.shape[-1] != 3 * d_h:
        raise ValueError(f"wx last dim {wx.shape[-1]} != 3*d_h={3 * d_h}")
    jac = h_prev.new_empty(batch, time, d_h)
    n_chunks = triton.cdiv(time, _BLOCK_T)
    n_dtiles = triton.cdiv(d_h, _BLOCK_D)
    _cfc_jac_kernel[(batch, n_chunks, n_dtiles)](
        h_prev,
        wx,
        dt,
        u,
        jac,
        time,
        d_h,
        *h_prev.stride(),
        *wx.stride(),
        dt.stride(0),
        dt.stride(1),
        *jac.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        NO_GATE=bool(no_gate),
    )
    return jac


def cfc_diag_jacobian_eager(
    h_prev: Tensor,
    wx: Tensor,
    dt: Tensor,
    u: Tensor,
    *,
    no_gate: bool = False,
) -> Tensor:
    """Eager diagonal ``J`` matching ``step_with_jacobian`` (gate_mix=input)."""
    out_dtype = h_prev.dtype
    h_prev = h_prev.float()
    wx = wx.float()
    u = u.float()
    batch, time, _ = h_prev.shape
    dt = _normalize_dt(dt.float(), batch, time)
    f_pre, cx, b = wx.chunk(3, dim=-1)
    soft = F.softplus(f_pre)
    a = torch.sigmoid(-(soft * dt.unsqueeze(-1) + b))
    n = torch.tanh(cx + u * h_prev)
    n_p = 1.0 - n.square()
    if no_gate:
        j = a + n_p * u
    else:
        j = a + (1.0 - a) * n_p * u
    return j.to(out_dtype)


def cfc_recurrence_vjp_eager(
    h_prev: Tensor,
    wx: Tensor,
    dt: Tensor,
    u: Tensor,
    mu: Tensor,
    *,
    no_gate: bool = False,
    raw_dt: Tensor | None = None,
    dt_eps: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor]:
    """CfC VJP. Returns ``(∇wx, ∇dt, ∇u)``; ``wx`` is 3-wide, ``dt`` is ``(B,T)``."""
    out_dtype = h_prev.dtype
    h_prev = h_prev.float()
    wx = wx.float()
    u = u.float()
    mu = mu.float()
    batch, time, _ = h_prev.shape
    dt = _normalize_dt(dt.float(), batch, time)
    f_pre, cx, b = wx.chunk(3, dim=-1)
    soft = F.softplus(f_pre)
    a = torch.sigmoid(-(soft * dt.unsqueeze(-1) + b))
    n = torch.tanh(cx + u * h_prev)
    if no_gate:
        d_h_da = mu * h_prev
        d_npre = mu * (1.0 - n.square())
    else:
        d_h_da = mu * (h_prev - n)
        d_npre = mu * (1.0 - a) * (1.0 - n.square())
    a1a = a * (1.0 - a)
    g_f = d_h_da * a1a * (-dt.unsqueeze(-1)) * torch.sigmoid(f_pre)
    g_dt = (d_h_da * a1a * (-soft)).sum(dim=-1)
    if raw_dt is not None:
        raw = _normalize_dt(raw_dt.float(), batch, time)
        g_dt = g_dt * (raw >= dt_eps).to(dtype=g_dt.dtype)
    g_b = d_h_da * a1a * (-1.0)
    g_wx = torch.cat((g_f, d_npre, g_b), dim=-1)
    g_u = (d_npre * h_prev).sum(dim=(0, 1))
    return g_wx.to(out_dtype), g_dt.to(out_dtype), g_u.to(out_dtype)


def cfc_recurrence_vjp(
    h_prev: Tensor,
    wx: Tensor,
    dt: Tensor,
    u: Tensor,
    mu: Tensor,
    *,
    no_gate: bool = False,
    raw_dt: Tensor | None = None,
    dt_eps: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor]:
    """CUDA tile VJP when available; eager otherwise. Returns ``(∇wx, ∇dt, ∇u)``.

    ``raw_dt`` (optional): pre-clamp Δt; when set, ``∇dt`` is zeroed where
    ``raw_dt < dt_eps`` inside the kernel (skips a host ``mul`` / ``clamp_min``).
    """
    if h_prev.device.type != "cuda":
        return cfc_recurrence_vjp_eager(
            h_prev, wx, dt, u, mu, no_gate=no_gate, raw_dt=raw_dt, dt_eps=dt_eps
        )
    batch, time, d_h = h_prev.shape
    dt = _normalize_dt(dt, batch, time)
    validate_cuda_tensors(h_prev, wx, dt, u, mu, name="cfc_recurrence_vjp")
    h_prev = h_prev.contiguous()
    wx = wx.contiguous()
    dt = dt.contiguous()
    u = u.contiguous()
    mu = mu.contiguous()
    if wx.shape[-1] != 3 * d_h:
        raise ValueError(f"wx last dim {wx.shape[-1]} != 3*d_h={3 * d_h}")
    has_dt_mask = raw_dt is not None
    if has_dt_mask:
        raw_dt = _normalize_dt(raw_dt, batch, time).contiguous()
        validate_cuda_tensors(raw_dt, name="cfc_recurrence_vjp.raw_dt")
    else:
        raw_dt = dt
    n_chunks = triton.cdiv(time, _BLOCK_T)
    n_dtiles = triton.cdiv(d_h, _BLOCK_D)
    g_wx = wx.new_empty(batch, time, 3 * d_h)
    gdt_direct = n_dtiles == 1
    if gdt_direct:
        g_dt_buf = wx.new_empty(batch, time)
        stride_gdt = (g_dt_buf.stride(0), 0, g_dt_buf.stride(1))
    else:
        g_dt_buf = wx.new_zeros(batch, n_dtiles, time)
        stride_gdt = g_dt_buf.stride()
    gu_tiles = wx.new_empty(batch, n_chunks, d_h)
    _cfc_vjp_kernel[(batch, n_chunks, n_dtiles)](
        h_prev,
        wx,
        dt,
        raw_dt,
        u,
        mu,
        g_wx,
        g_dt_buf,
        gu_tiles,
        time,
        d_h,
        float(dt_eps),
        *h_prev.stride(),
        *wx.stride(),
        dt.stride(0),
        dt.stride(1),
        raw_dt.stride(0),
        raw_dt.stride(1),
        *mu.stride(),
        *g_wx.stride(),
        *stride_gdt,
        *gu_tiles.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        NO_GATE=bool(no_gate),
        HAS_DT_MASK=has_dt_mask,
        GDT_DIRECT=gdt_direct,
    )
    if gdt_direct:
        g_dt = g_dt_buf
    else:
        g_dt = g_dt_buf.float().sum(dim=1).to(dtype=h_prev.dtype)
    g_u = gu_tiles.float().sum(dim=(0, 1)).to(dtype=h_prev.dtype)
    return g_wx, g_dt, g_u

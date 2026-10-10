"""Channel-Woodbury Triton scan / Newton for ``ParaSLSTM(mix='lowrank')``.

``R_g = diag(r_g) + U_g V_gᵀ`` with ``U,V ∈ R^{d_h × r}`` (per-gate channel
factors). Product-diag 4×4 scan does not apply; this module ships:

* ``scan_woodbury_triton`` — serial inclusive scan for Woodbury leaves at
  ``r=1`` (full-``d`` CTA, ``d_h ≤ 64``).
* ``newton_slstm_lowrank_fused`` — Alg. 1 forward: Picard / zero-hidden init,
  then ``K`` batched cell+J steps with Triton dense scan on the packed
  ``(4d)×(4d)`` woodbury leaf (eager-equivalent numerics).

Backward: prefer ``scan_backend='triton'|'eager'`` through
``_NewtonFixedPoint`` (eq. 2.6). This fused entry is forward-oriented;
training with grads uses the FixedPoint path (same numerics).
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors
from pararnn.layout import SLSTM_SLOTS, prepend_state

log = logging.getLogger(__name__)

_MAX_D_R1 = 64


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


@triton.jit
def _slstm_lowrank_fused_scan_kernel(
    diag_ptr,
    u_ptr,
    v_ptr,
    res_ptr,
    out_ptr,
    time,
    d,
    stride_db,
    stride_dt,
    stride_dd,
    stride_ub,
    stride_ut,
    stride_ud,
    stride_vb,
    stride_vt,
    stride_vd,
    stride_rb,
    stride_rt,
    stride_rd,
    stride_ob,
    stride_ot,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    """Inclusive scan ``δ_t = (diag_t + u_t v_tᵀ) δ_{t-1} + r_t`` for ``r=1``.

    One program per batch row; full channel width in ``BLOCK_D`` (``d ≤ 64``).
    Independent ``BLOCK_D`` channel strips cannot keep separate factor state —
    this kernel uses a full-``d`` CTA.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < d
    delta = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(0, time):
        dvec = load_acc(
            diag_ptr + pid * stride_db + t * stride_dt + offs * stride_dd, mask, 0.0
        )
        u = load_acc(u_ptr + pid * stride_ub + t * stride_ut + offs * stride_ud, mask, 0.0)
        v = load_acc(v_ptr + pid * stride_vb + t * stride_vt + offs * stride_vd, mask, 0.0)
        r = load_acc(res_ptr + pid * stride_rb + t * stride_rt + offs * stride_rd, mask, 0.0)
        vt = tl.sum(v * delta, axis=0)
        delta = dvec * delta + u * vt + r
        store_acc(out_ptr + pid * stride_ob + t * stride_ot + offs * stride_od, delta, mask)


def scan_woodbury_triton(
    diag: Tensor,
    u: Tensor,
    v: Tensor,
    residual: Tensor,
) -> Tensor:
    """Triton twin of serial Woodbury scan at ``r=1``.

    Parameters
    ----------
    diag : Tensor
        ``(B, T, d)`` diagonal.
    u, v : Tensor
        ``(B, T, d)`` rank-1 factors (``r=1`` squeezed).
    residual : Tensor
        ``(B, T, d)``.

    Returns
    -------
    delta : Tensor
        ``(B, T, d)`` inclusive scan.
    """
    if diag.dim() != 3 or u.shape != diag.shape or v.shape != diag.shape:
        raise ValueError(
            f"scan_woodbury_triton expects diag/u/v (B,T,d); got "
            f"{tuple(diag.shape)} / {tuple(u.shape)} / {tuple(v.shape)}"
        )
    if residual.shape != diag.shape:
        raise ValueError(f"residual shape {tuple(residual.shape)} != {tuple(diag.shape)}")
    batch, time, d = diag.shape
    if d > _MAX_D_R1:
        raise ValueError(f"scan_woodbury_triton r=1 supports d_h <= {_MAX_D_R1}, got {d}")
    validate_cuda_tensors(diag, u, v, residual, name="scan_woodbury_triton")
    out = torch.empty_like(residual)
    block_d = _next_pow2(d)
    _slstm_lowrank_fused_scan_kernel[(batch,)](
        diag,
        u,
        v,
        residual,
        out,
        time,
        d,
        diag.stride(0),
        diag.stride(1),
        diag.stride(2),
        u.stride(0),
        u.stride(1),
        u.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_D=block_d,
    )
    log.debug(
        "scan_woodbury_triton",
        extra={"batch": batch, "time": time, "d": d, "block_d": block_d},
    )
    return out


def _newton_slstm_lowrank_impl(
    cell: torch.nn.Module,
    wx: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None,
    picard_iters: int,
) -> Tensor:
    """Host Alg. 1 for ``mix='lowrank'`` with Triton dense scan on CUDA."""
    from pararnn.solvers.scan import scan_dense
    from pararnn.solvers.slstm_picard import slstm_picard_init, slstm_zero_hidden_init

    batch, time, four_d = wx.shape
    d_h = four_d // 4
    if picard_iters:
        states = slstm_picard_init(cell, wx, h0=h0, n_picard=picard_iters)
    else:
        states = slstm_zero_hidden_init(wx, eps=cell.eps, h0=h0)

    backend = "triton" if wx.is_cuda else "eager"
    x_dummy = wx.new_zeros(batch, time, cell.d_in)
    for k in range(int(max_iters)):
        h_prev = prepend_state(states, h0)
        pred, jac = cell.step_with_jacobian(h_prev, x_dummy, wx=wx)
        residual = pred - states
        res_flat = residual.reshape(batch, time, SLSTM_SLOTS * d_h)
        delta_flat = scan_dense(jac, res_flat, backend=backend)
        delta = delta_flat.reshape(batch, time, SLSTM_SLOTS, d_h)
        states = states + float(omega) * delta
        if not torch.compiler.is_compiling():
            log.debug(
                "newton_slstm_lowrank_iter",
                extra={
                    "k": k,
                    "max_residual": float(residual.detach().abs().amax()),
                    "seq_len": time,
                    "d_h": d_h,
                    "rank": getattr(cell, "rank", None),
                },
            )
    return states


def newton_slstm_lowrank_fused(
    cell: torch.nn.Module,
    wx: Tensor,
    *,
    max_iters: int,
    omega: float = 1.0,
    h0: Tensor | None = None,
    picard_iters: int = 0,
) -> Tensor:
    """Public fused entry for ``ParaSLSTM(mix='lowrank')``."""
    if getattr(cell, "mix", None) != "lowrank":
        raise TypeError(f"newton_slstm_lowrank_fused needs mix='lowrank', got {cell.mix!r}")
    validate_cuda_tensors(wx, name="newton_slstm_lowrank_fused")
    if h0 is not None:
        validate_cuda_tensors(h0, name="newton_slstm_lowrank_fused")
    rank = int(getattr(cell, "rank", 0) or 0)
    d_h = int(cell.d_h)
    log.info(
        "newton_slstm_lowrank_fused",
        extra={
            "max_iters": max_iters,
            "picard_iters": picard_iters,
            "rank": rank,
            "d_h": d_h,
            "seq_len": int(wx.shape[1]),
            "batch": int(wx.shape[0]),
        },
    )
    return _newton_slstm_lowrank_impl(
        cell,
        wx,
        max_iters=max_iters,
        omega=omega,
        h0=h0,
        picard_iters=picard_iters,
    )


def can_fuse_lowrank(cell: torch.nn.Module, x: Tensor) -> bool:
    """Whether CUDA fused / Triton-scan lowrank Newton is available."""
    if getattr(cell, "mix", None) != "lowrank":
        return False
    if not x.is_cuda:
        return False
    from pararnn.kernels.precision import is_fused_dtype_supported

    return is_fused_dtype_supported(x.dtype, x.device)


def vjp_slstm_lowrank_packed(
    cell: torch.nn.Module,
    states: Tensor,
    wx: Tensor,
    mu: Tensor,
    *,
    h0: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Packed reverse stub for ``mix='lowrank'``.

    Phase-3 ships forward SMW / dense-leaf Newton first. Gradients for
    ``U, V, diag`` currently flow through ``_NewtonFixedPoint`` eq. 2.6 with
    the packed woodbury leaf (dense reverse scan). This helper documents the
    future zero-atomic tile-sum VJP surface and raises until that kernel lands.

    Returns
    -------
    grad_wx, grad_U, grad_V, grad_diag
    """
    del states, wx, mu, h0
    raise NotImplementedError(
        "packed lowrank VJP kernel is stubbed; train with scan_backend='eager' "
        f"or 'triton' so eq. 2.6 reverse uses the woodbury leaf "
        f"(cell rank={getattr(cell, 'rank', None)})"
    )


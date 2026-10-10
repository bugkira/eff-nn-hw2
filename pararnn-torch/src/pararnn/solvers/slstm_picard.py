"""Picard / frozen-gate init for ParaSLSTM Newton."""

from __future__ import annotations

import logging
import os

import torch
from torch import Tensor, nn

from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_SLOTS,
    SLSTM_STABILIZER,
    prepend_state_ragged,
    segment_start_flags,
    validate_cu_seqlens,
)

log = logging.getLogger(__name__)

# Long-T concurrent CP: prefer 2-pass windowed monoid Picard. Set to ``1`` to
# restore the Θ(world) boundary-chained windowed warm-start (legacy gate).
_ENV_BOUNDARY_CHAIN = "PARARNN_PICARD_BOUNDARY_CHAIN"


def slstm_frozen_gate_scan(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """sLSTM ``(c, n, m, h)`` with gates frozen in ``pre`` ``(B, T, 4 d_h)``.

    ``m`` is the max-plus prefix ``m_t = max(z_f + m_{t-1}, z_i)``, then
    ``n`` and ``c`` are 1D scans. ``h`` is readout. Algebra in fp32; DRAM
    dtype preserved. CUDA fp16/fp32/bf16 (bf16: CC≥8.0) uses the Triton
    tiled scan; algebra stays fp32 via ``load_acc`` / ``store_acc``.

    ``out`` : optional ``(B, T, 4, d_h)`` buffer written in-place (Picard reuse).
    """
    from pararnn.kernels.precision import is_fused_dtype_supported

    if (
        cu_seqlens is None
        and pre.is_cuda
        and is_fused_dtype_supported(pre.dtype, pre.device)
        and (h0 is None or h0.is_cuda)
    ):
        from pararnn.kernels.picard_slstm import frozen_gate_scan_triton

        return frozen_gate_scan_triton(pre, eps=eps, h0=h0, out=out)
    return slstm_frozen_gate_scan_eager(
        pre, eps=eps, h0=h0, cu_seqlens=cu_seqlens, out=out
    )


def slstm_frozen_gate_scan_eager(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """CPU / fallback twin of ``slstm_frozen_gate_scan`` (eager Blelloch)."""
    from pararnn.solvers.scan import scan_diag

    batch, time, four_d = pre.shape
    d_h = four_d // 4
    orig = pre.dtype
    pre32 = pre.float()
    z_i, z_f, z_z, z_o = pre32.chunk(4, dim=-1)
    if h0 is None:
        c0 = n0 = m0 = pre32.new_zeros(batch, d_h)
        if cu_seqlens is not None:
            n_seq = int(validate_cu_seqlens(cu_seqlens, time).numel()) - 1
            c0 = n0 = m0 = pre32.new_zeros(n_seq, d_h)
    else:
        h0f = h0.float()
        c0 = h0f[:, SLSTM_CELL]
        n0 = h0f[:, SLSTM_NORMALIZER]
        m0 = h0f[:, SLSTM_STABILIZER]
    if cu_seqlens is None:
        p = z_f.cumsum(dim=1)
        m = p + torch.maximum((z_i - p).cummax(dim=1).values, m0.unsqueeze(1))
        m_prev = torch.cat((m0.unsqueeze(1), m[:, :-1]), dim=1)
        i_t = torch.exp(z_i - m)
        f_t = torch.exp(z_f + m_prev - m)
        z = torch.tanh(z_z)
        res_n = i_t.clone()
        res_n[:, 0] = f_t[:, 0] * n0 + i_t[:, 0]
        res_c = i_t * z
        res_c[:, 0] = f_t[:, 0] * c0 + i_t[:, 0] * z[:, 0]
    else:
        cs = validate_cu_seqlens(cu_seqlens, time)
        starts = cs[:-1]
        p = _seg_cumsum(z_f, cu_seqlens)
        zi_p = z_i - p
        zi_p_adj = zi_p.clone()
        zi_p_adj[0, starts] = torch.maximum(zi_p[0, starts], m0)
        m = p + _seg_cummax(zi_p_adj, cu_seqlens)
        i_t = torch.exp(z_i - m)
        m_prev = _prepend_feat(m, m0, cu_seqlens)
        f_t = torch.exp(z_f + m_prev - m)
        z = torch.tanh(z_z)
        res_n = i_t.clone()
        res_n[0, starts] = f_t[0, starts] * n0 + i_t[0, starts]
        res_c = i_t * z
        res_c[0, starts] = f_t[0, starts] * c0 + i_t[0, starts] * z[0, starts]
    n = scan_diag(f_t, res_n, cu_seqlens=cu_seqlens)
    c = scan_diag(f_t, res_c, cu_seqlens=cu_seqlens)
    o = torch.sigmoid(z_o)
    h = o * (c / (n + eps))
    stacked = torch.stack((c, n, m, h), dim=-2).to(dtype=orig)
    if out is not None:
        if out.shape != stacked.shape:
            raise ValueError(f"out shape {tuple(out.shape)} != {tuple(stacked.shape)}")
        out.copy_(stacked)
        return out
    return stacked


def slstm_frozen_gate_scan_sharded(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
    group: dist.ProcessGroup | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Stage-3 frozen-gate on a local tile: 2-stage ``(max,+)`` then affine CP.

    CUDA: Triton ``C=128`` Stage-1 upsweep + monoid AllGather + downsweep for
    ``m``, then diag monoid CP for ``n``/``c`` (Stage 2). CPU: eager twin.
    """
    import torch.distributed as dist

    from pararnn.kernels.precision import is_fused_dtype_supported

    if (
        pre.is_cuda
        and is_fused_dtype_supported(pre.dtype, pre.device)
        and (h0 is None or h0.is_cuda)
    ):
        from pararnn.kernels.picard_slstm import frozen_gate_scan_fused_cp

        return frozen_gate_scan_fused_cp(
            pre, eps=eps, h0=h0, group=group, out=out
        )

    from pararnn.solvers.seq_parallel import (
        _exchange_left_boundary,
        scan_diag_context_parallel,
        scan_maxplus_context_parallel,
    )

    batch, time, four_d = pre.shape
    d_h = four_d // 4
    orig = pre.dtype
    pre32 = pre.float()
    z_i, z_f, z_z, z_o = pre32.chunk(4, dim=-1)

    if h0 is None:
        c0 = n0 = m0 = pre32.new_zeros(batch, d_h)
    else:
        h0f = h0.float()
        c0 = h0f[:, SLSTM_CELL]
        n0 = h0f[:, SLSTM_NORMALIZER]
        m0 = h0f[:, SLSTM_STABILIZER]

    rank = dist.get_rank(group=group) if dist.is_initialized() else 0

    m, _ = scan_maxplus_context_parallel(z_f, z_i, m0=m0, group=group)
    m32 = m.float()
    left_m = _exchange_left_boundary(m32, m0, group, rank)
    first_m = m0 if rank == 0 else left_m
    m_prev = torch.cat((first_m.unsqueeze(1), m32[:, :-1]), dim=1)

    i_t = torch.exp(z_i - m32)
    f_t = torch.exp(z_f + m_prev - m32)
    z = torch.tanh(z_z)

    res_n = i_t.clone()
    res_c = i_t * z
    if rank == 0 and time > 0:
        res_n[:, 0] = f_t[:, 0] * n0 + i_t[:, 0]
        res_c[:, 0] = f_t[:, 0] * c0 + i_t[:, 0] * z[:, 0]

    n = scan_diag_context_parallel(f_t, res_n, group=group, backend="auto")
    c = scan_diag_context_parallel(f_t, res_c, group=group, backend="auto")
    o = torch.sigmoid(z_o)
    h = o * (c / (n + eps))
    stacked = torch.stack((c, n, m32, h), dim=-2).to(dtype=orig)
    if out is not None:
        if out.shape != stacked.shape:
            raise ValueError(f"out shape {tuple(out.shape)} != {tuple(stacked.shape)}")
        out.copy_(stacked)
        return out
    return stacked


def slstm_picard_init_sharded(
    cell: nn.Module,
    wx: Tensor,
    *,
    h0: Tensor | None = None,
    n_picard: int = 1,
    group: dist.ProcessGroup | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Concurrent Stage-3 Picard: sharded frozen-gate + boundary ``h`` exchange.

    ``wx`` is the **local** tile ``(B, T/p, 4 d_h)``. No full-``T`` gather.
    Each frozen-gate pass runs ``scan_maxplus_context_parallel`` and
    ``scan_diag_context_parallel`` (two-pass monoid AllGather) so every
    rank stays busy — sequence-parallel warm-start, peak ∝ ``T/p``.
    """
    import torch.distributed as dist

    from pararnn.solvers.seq_parallel import _exchange_left_boundary, _prepend_sharded

    if n_picard < 0:
        raise ValueError(f"n_picard must be >= 0, got {n_picard!r}")
    batch, time, four_d = wx.shape
    d_h = four_d // 4
    with torch.no_grad():
        states = out if out is not None else wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
        slstm_frozen_gate_scan_sharded(
            wx, eps=cell.eps, h0=h0, group=group, out=states
        )
        if n_picard == 0:
            return states

        rank = dist.get_rank(group=group) if dist.is_initialized() else 0
        work = wx.new_empty(batch, time, four_d)
        for _ in range(n_picard):
            left = _exchange_left_boundary(states, h0, group, rank)
            h_prev_states = _prepend_sharded(states, h0, left, rank)
            h_prev = h_prev_states[..., SLSTM_HIDDEN, :]
            _fill_pre_wx_plus_recurrent(cell, wx, h_prev, work)
            slstm_frozen_gate_scan_sharded(
                work, eps=cell.eps, h0=h0, group=group, out=states
            )
        return states


def slstm_picard_init_sharded_windowed(
    cell: nn.Module,
    wx: Tensor,
    *,
    h0: Tensor | None = None,
    n_picard: int = 3,
    chunk_len: int = 128,
    group: dist.ProcessGroup | None = None,
    out: Tensor | None = None,
    boundary_chain: bool | None = None,
) -> Tensor:
    """Concurrent hierarchical Picard with continuous cross-rank carry.

    world=1: windowed Triton Picard (``P=n_picard``, ``C=chunk_len``).

    world>1, short global ``T`` (≤4096): **2-stage monoid** Picard
    (``slstm_picard_init_sharded``) — O(1) AllGather / stage.

    world>1, long global ``T``: **windowed monoid CP** Picard (default):

      * rank-local windowed Picard (``P=2``, ``C=chunk_len``);
      * one true 2-pass ``frozen_gate_scan_fused_cp`` monoid stitch;
      * ``O(B·d)`` left exchange + windowed refine (``P=2``).

    Pass count is independent of ``world``. Legacy ``world × win2`` boundary
    chain: ``boundary_chain=True`` or ``PARARNN_PICARD_BOUNDARY_CHAIN=1``.
    """
    import torch.distributed as dist

    from pararnn.solvers.seq_parallel import _exchange_left_boundary

    c = int(chunk_len)
    if c < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len!r}")
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    if world <= 1:
        return slstm_picard_init(
            cell,
            wx,
            h0=h0,
            n_picard=n_picard,
            chunk_len=c,
            out=out,
        )

    batch, time, four_d = wx.shape
    d_h = four_d // 4
    if dist.is_initialized() and world > 1:
        local = torch.tensor([time], device=wx.device, dtype=torch.int64)
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local, group=group)
        global_t = int(sum(int(t.item()) for t in gathered))
    else:
        global_t = int(time)
    if global_t <= 4096:
        return slstm_picard_init_sharded(
            cell, wx, h0=h0, n_picard=n_picard, group=group, out=out
        )

    use_chain = (
        bool(boundary_chain)
        if boundary_chain is not None
        else os.environ.get(_ENV_BOUNDARY_CHAIN, "").strip() in ("1", "true", "True")
    )
    if use_chain:
        return _picard_sharded_boundary_chain(
            cell,
            wx,
            h0=h0,
            n_picard=n_picard,
            chunk_len=c,
            group=group,
            out=out,
        )
    return _picard_sharded_windowed_monoid(
        cell,
        wx,
        h0=h0,
        n_picard=n_picard,
        chunk_len=c,
        group=group,
        out=out,
    )


def _picard_sharded_boundary_chain(
    cell: nn.Module,
    wx: Tensor,
    *,
    h0: Tensor | None,
    n_picard: int,
    chunk_len: int,
    group,
    out: Tensor | None,
) -> Tensor:
    """Legacy long-T warm-start: ``world`` local windowed rounds + left exchange."""
    import torch.distributed as dist

    from pararnn.solvers.seq_parallel import _exchange_left_boundary

    batch, time, four_d = wx.shape
    d_h = four_d // 4
    world = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    p_local = 2 if int(n_picard) >= 2 else max(1, int(n_picard))
    if not torch.compiler.is_compiling():
        log.info(
            "slstm_picard_boundary_chain",
            extra={
                "world": world,
                "rank": rank,
                "local_t": time,
                "p_local": p_local,
                "chunk_len": int(chunk_len),
            },
        )
    with torch.no_grad():
        h0_buf = h0 if h0 is not None else wx.new_zeros(batch, SLSTM_SLOTS, d_h)
        left = h0_buf if rank == 0 else wx.new_zeros(batch, SLSTM_SLOTS, d_h)
        states = out
        for _ in range(world):
            states = slstm_picard_init(
                cell, wx, h0=left, n_picard=p_local, chunk_len=chunk_len, out=states
            )
            left = _exchange_left_boundary(states, h0, group, rank)
        return states


def _picard_sharded_windowed_monoid(
    cell: nn.Module,
    wx: Tensor,
    *,
    h0: Tensor | None,
    n_picard: int,
    chunk_len: int,
    group,
    out: Tensor | None,
) -> Tensor:
    """Long-T: windowed Picard + one 2-pass monoid stitch + boundary refine.

    Unchunked monoid ``P=3`` misses the Newton basin on long ``T``. Algorithm
    (fixed 2 windowed ``P=2`` passes — independent of ``world``):

      1. Rank-local windowed Picard (``P=2``, ``C=chunk_len``; zeros left
         on rank>0).
      2. One ``frozen_gate_scan_fused_cp`` monoid stitch (AllGather of tile
         aggregates) so the ``(max,+)`` / affine prefix is continuous.
      3. ``O(B·d)`` left-boundary exchange + windowed refine (``P=2``).

    Peak monoid AllGather count is O(1) per warm-start. Legacy ``Θ(world)``
    chain remains opt-in via ``PARARNN_PICARD_BOUNDARY_CHAIN``.
    """
    import torch.distributed as dist

    from pararnn.kernels.picard_slstm import frozen_gate_scan_fused_cp_windowed
    from pararnn.kernels.precision import is_fused_dtype_supported
    from pararnn.solvers.seq_parallel import _exchange_left_boundary, _prepend_sharded

    if n_picard < 0:
        raise ValueError(f"n_picard must be >= 0, got {n_picard!r}")
    batch, time, four_d = wx.shape
    d_h = four_d // 4
    rank = dist.get_rank(group=group) if dist.is_initialized() else 0
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    n_p = int(n_picard)
    c = int(chunk_len)
    if not torch.compiler.is_compiling():
        log.info(
            "slstm_picard_windowed_monoid_cp",
            extra={
                "world": world,
                "rank": rank,
                "local_t": time,
                "n_picard": n_p,
                "chunk_len": c,
            },
        )
    use_fused = (
        wx.is_cuda
        and is_fused_dtype_supported(wx.dtype, wx.device)
        and (h0 is None or h0.is_cuda)
    )

    def _monoid_stitch(states: Tensor, work: Tensor) -> Tensor:
        left = _exchange_left_boundary(states, h0, group, rank)
        h_prev_states = _prepend_sharded(states, h0, left, rank)
        h_prev = h_prev_states[..., SLSTM_HIDDEN, :]
        _fill_pre_wx_plus_recurrent(cell, wx, h_prev, work)
        if use_fused:
            return frozen_gate_scan_fused_cp_windowed(
                work, eps=cell.eps, h0=h0, group=group, out=states
            )
        return slstm_frozen_gate_scan_sharded(
            work, eps=cell.eps, h0=h0, group=group, out=states
        )

    with torch.no_grad():
        h0_buf = h0 if h0 is not None else wx.new_zeros(batch, SLSTM_SLOTS, d_h)
        left = h0_buf if rank == 0 else wx.new_zeros(batch, SLSTM_SLOTS, d_h)
        states = out
        work = wx.new_empty(batch, time, four_d)
        # Rank-local windowed seed (zeros left on rank>0), then one monoid
        # stitch so the next left exchange carries a continuous prefix.
        # P_local=2 matches the v7.2 win2 leaf (cheaper than full envelope P=3
        # twice); the monoid stitch replaces further world rounds.
        p_local = 2 if n_p >= 2 else max(1, n_p)
        states = slstm_picard_init(
            cell, wx, h0=left, n_picard=p_local, chunk_len=c, out=states
        )
        if n_p == 0:
            return states
        states = _monoid_stitch(states, work)
        left = _exchange_left_boundary(states, h0, group, rank)
        # Boundary-seeded windowed refine (plan risk fallback): re-enter the
        # world=1 basin with a continuous left carry. Measured K=3-in-basin
        # at p=2 T=64k d_h=256 (gate default).
        states = slstm_picard_init(
            cell, wx, h0=left, n_picard=p_local, chunk_len=c, out=states
        )
        return states


def _fill_pre_windowed_rh(
    cell: nn.Module,
    wx: Tensor,
    states: Tensor,
    left: Tensor,
    work: Tensor,
    *,
    chunk_len: int,
) -> None:
    """Write ``work ← wx + R·h_prev`` with window-local shifted ``h``.

    Within each ``chunk_len`` tile the left edge uses the previous window's
    last hidden (or ``left`` for the first tile); interior steps use
    ``states``. Matches the Rh pattern of windowed Picard while leaving a
    full-tile ``pre`` for a subsequent monoid frozen-gate scan.
    """
    batch, time, _four_d = wx.shape
    d_h = states.shape[-1]
    h_prev = wx.new_empty(batch, time, d_h)
    for t0 in range(0, time, chunk_len):
        t1 = min(t0 + chunk_len, time)
        if t0 == 0:
            h_prev[:, 0].copy_(left[:, SLSTM_HIDDEN])
        else:
            h_prev[:, t0].copy_(states[:, t0 - 1, SLSTM_HIDDEN, :])
        if t1 - t0 > 1:
            h_prev[:, t0 + 1 : t1].copy_(states[:, t0 : t1 - 1, SLSTM_HIDDEN, :])
    _fill_pre_wx_plus_recurrent(cell, wx, h_prev, work)


def _seg_cumsum(x: Tensor, cu_seqlens: Tensor | None) -> Tensor:
    from pararnn.solvers.scan import scan_diag

    ones = torch.ones_like(x)
    return scan_diag(ones, x, cu_seqlens=cu_seqlens)


def _seg_cummax(x: Tensor, cu_seqlens: Tensor) -> Tensor:
    flags = segment_start_flags(cu_seqlens, x.shape[1], batch=x.shape[0]).to(device=x.device)
    out = x.clone()
    f = flags.clone()
    step = 1
    time = x.shape[1]
    while step < time:
        left, f_l = out[:, :-step], f[:, :-step]
        right, f_r = out[:, step:], f[:, step:]
        take = f_r.reshape(*f_r.shape, *([1] * (right.dim() - f_r.dim())))
        combined = torch.maximum(left, right)
        out = out.clone()
        f = f.clone()
        out[:, step:] = torch.where(take, right, combined)
        f[:, step:] = f_r | f_l
        step *= 2
    return out


def _prepend_feat(feat: Tensor, h0_feat: Tensor, cu_seqlens: Tensor) -> Tensor:
    """Shift a ``(1, N, d)`` feature right; ``h0_feat`` is ``(S, d)`` at starts."""
    dummy = feat.unsqueeze(-2)
    h0 = h0_feat.unsqueeze(-2)
    return prepend_state_ragged(dummy, h0, cu_seqlens).squeeze(-2)


def slstm_zero_hidden_init(
    wx: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Newton guess: sLSTM with ``R h = 0`` (keep running ``m`` / ``n``).

    App. A ``f(0, x_t)`` zeros ``n`` independently at each t. Mixing ``R h``
    is left to Newton (or to ``slstm_picard_init``).
    """
    return slstm_frozen_gate_scan(wx, eps=eps, h0=h0, cu_seqlens=cu_seqlens)


def _fill_h_prev(
    h_prev: Tensor,
    states: Tensor,
    h0: Tensor | None,
    cu_seqlens: Tensor | None,
) -> None:
    """Write shifted hidden into ``h_prev`` ``(B, T, d_h)`` (Picard workspace)."""
    if cu_seqlens is not None:
        # Ragged: one full prepend then take the hidden slot (rare path).
        shifted = prepend_state_ragged(states, h0, cu_seqlens)
        h_prev.copy_(shifted[..., SLSTM_HIDDEN, :])
        return
    if h0 is None:
        h_prev[:, 0].zero_()
    else:
        h_prev[:, 0].copy_(h0[:, SLSTM_HIDDEN])
    h_prev[:, 1:].copy_(states[:, :-1, SLSTM_HIDDEN, :])


def _fill_pre_wx_plus_recurrent(
    cell: nn.Module,
    wx: Tensor,
    h_prev: Tensor,
    work: Tensor,
) -> None:
    """``work ← wx + R·h_prev`` without a second full-size ``R h`` alloc (diag)."""
    work.copy_(wx)
    r = getattr(cell, "R", None)
    if r is not None:
        # mix='diag': (4, d_h) ⊙ h → add into work viewed as (…, 4, d_h).
        clipped = cell.clipped_r()
        wv = work.view(*h_prev.shape[:-1], SLSTM_SLOTS, h_prev.shape[-1])
        for g in range(SLSTM_SLOTS):
            wv[..., g, :].addcmul_(h_prev, clipped[g])
        return
    work.add_(cell._recurrent(h_prev))


def _picard_global(
    cell: nn.Module,
    wx: Tensor,
    *,
    h0: Tensor | None,
    n_picard: int,
    cu_seqlens: Tensor | None,
    out: Tensor | None,
) -> Tensor:
    """Full-T Picard (1 zero-Rh scan + ``n_picard`` Rh scans)."""
    batch, time, four_d = wx.shape
    d_h = four_d // 4
    states = out if out is not None else wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
    slstm_frozen_gate_scan(
        wx, eps=cell.eps, h0=h0, cu_seqlens=cu_seqlens, out=states
    )
    if n_picard == 0:
        return states
    work = wx.new_empty(batch, time, four_d)
    h_prev = wx.new_empty(batch, time, d_h)
    for _ in range(n_picard):
        _fill_h_prev(h_prev, states, h0, cu_seqlens)
        _fill_pre_wx_plus_recurrent(cell, wx, h_prev, work)
        slstm_frozen_gate_scan(
            work, eps=cell.eps, h0=h0, cu_seqlens=cu_seqlens, out=states
        )
    return states


def _picard_windowed_python(
    cell: nn.Module,
    wx: Tensor,
    *,
    h0: Tensor | None,
    n_picard: int,
    chunk_len: int,
    out: Tensor | None,
) -> Tensor:
    """Per-tile Picard with ``O(C)`` work buffers (CPU / head / fallback)."""
    batch, time, four_d = wx.shape
    d_h = four_d // 4
    states = out if out is not None else wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
    carry = h0
    work = wx.new_empty(batch, chunk_len, four_d)
    h_prev = wx.new_empty(batch, chunk_len, d_h)
    for t0 in range(0, time, chunk_len):
        t1 = min(t0 + chunk_len, time)
        c_len = t1 - t0
        wx_c = wx[:, t0:t1]
        st_c = states[:, t0:t1]
        work_c = work[:, :c_len]
        h_c = h_prev[:, :c_len]
        slstm_frozen_gate_scan(wx_c, eps=cell.eps, h0=carry, out=st_c)
        for _ in range(n_picard):
            _fill_h_prev(h_c, st_c, carry, None)
            _fill_pre_wx_plus_recurrent(cell, wx_c, h_c, work_c)
            slstm_frozen_gate_scan(work_c, eps=cell.eps, h0=carry, out=st_c)
        carry = st_c[:, -1].contiguous()
    return states


def slstm_picard_init(
    cell: nn.Module,
    wx: Tensor,
    *,
    h0: Tensor | None = None,
    n_picard: int = 1,
    cu_seqlens: Tensor | None = None,
    chunk_len: int | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Zero-hidden scan, then ``n_picard`` frozen-gate scans using ``R h``.

    Each pass freezes mixing from the previous trajectory and rescans
    ``(c, n, m)``. Cap ``n_picard``: P ~ T is sequential.

    ``chunk_len`` : windowed Picard (paper §4.2 / App. D). Each tile of length
    ``C`` runs ``1+n_picard`` passes with left carry; peak scratch is ``O(C)``.
    On CUDA ``mix='diag'``: fused Rh (windowed = one launch over all tiles;
    global = serial mega-fuse per pass + shifted ``h`` scratch). Host
    ``wx+R⊙h`` path remains for CPU / head / ragged.

    Runs under ``torch.no_grad``: eq. 2.6 differentiates only the final root
    ``S*`` (Picard is a warm start).
    """
    if n_picard < 0:
        raise ValueError(f"n_picard must be >= 0, got {n_picard!r}")
    batch, time, four_d = wx.shape
    if four_d % 4 != 0:
        raise ValueError(f"wx last dim {four_d} is not 4 * d_h")
    if chunk_len is not None and int(chunk_len) < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len!r}")
    if chunk_len is not None and cu_seqlens is not None:
        raise ValueError("chunk_len cannot be combined with cu_seqlens")
    with torch.no_grad():
        # CUDA diag: fused Rh (no host wx+R⊙h). Windowed if chunk_len set;
        # else serial mega-fuse per pass. bf16 needs Ampere+ (same as fused Newton).
        from pararnn.kernels.precision import is_fused_dtype_supported

        use_fused = (
            cu_seqlens is None
            and wx.is_cuda
            and is_fused_dtype_supported(wx.dtype, wx.device)
            and getattr(cell, "mix", None) == "diag"
            and hasattr(cell, "clipped_r")
        )
        if use_fused and chunk_len is not None:
            from pararnn.kernels.picard_slstm import picard_diag_windowed_triton

            return picard_diag_windowed_triton(
                wx,
                cell.clipped_r(),
                eps=cell.eps,
                n_picard=n_picard,
                h0=h0,
                chunk_len=int(chunk_len),
                out=out,
            )
        if use_fused and chunk_len is None:
            from pararnn.kernels.picard_slstm import picard_diag_global_triton

            return picard_diag_global_triton(
                wx,
                cell.clipped_r(),
                eps=cell.eps,
                n_picard=n_picard,
                h0=h0,
                out=out,
            )
        if chunk_len is None:
            return _picard_global(
                cell, wx, h0=h0, n_picard=n_picard, cu_seqlens=cu_seqlens, out=out
            )
        return _picard_windowed_python(
            cell, wx, h0=h0, n_picard=n_picard, chunk_len=int(chunk_len), out=out
        )

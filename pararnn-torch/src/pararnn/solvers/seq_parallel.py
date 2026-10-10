"""Sequence-parallel Newton scan: tile locally, then apply a carry.

Same monoid as ``scan_diag``. Two CUDA streams on one device are virtual
ranks. ``scan_diag_context_parallel`` splits time across a process group
(NCCL / gloo): each rank scans its tile, AllGathers the tile monoid
``(P_end, δ_end)``, and applies the exclusive prefix carry. Communication
is ``O(B d)`` per rank, independent of local ``T``.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from torch import Tensor

from pararnn.kernels.precision import is_fused_dtype_supported
from pararnn.solvers.scan import _mv4, scan_block4, scan_diag

log = logging.getLogger(__name__)


# ============================================================================
# Differentiable carry exchange autograd Functions
# ============================================================================

class _CarryRecv(torch.autograd.Function):
    """Receive carry from peer rank with differentiable backward.
    
    Forward: receive (mu, j) from peer via irecv/wait.
    Backward: send gradients back to peer via isend/wait.
    
    This is used by the LEFT tile to receive the carry from the RIGHT tile.
    Forward: left receives (mu_boundary, j_boundary) from right.
    Backward: left sends gradients w.r.t. the carry to right.
    """
    @staticmethod
    def forward(ctx, peer, template, mu_shape, j_shape, group):
        # `template` must be a tensor that requires grad.  A custom autograd
        # Function only receives a backward node when at least one INPUT
        # requires grad; without such an input the freshly allocated outputs
        # have requires_grad=False, no grad_fn edge is created, and
        # ``backward`` is never called -- which deadlocks the peer rank's
        # ``_CarrySend.backward`` irecv.  The received values do not depend on
        # ``template``, so its gradient is None.
        ctx.peer, ctx.group = peer, group
        mu = torch.empty(mu_shape, dtype=template.dtype, device=template.device)
        j = torch.empty(j_shape, dtype=template.dtype, device=template.device)
        mu_h = dist.irecv(mu, src=peer, group=group)
        j_h = dist.irecv(j, src=peer, group=group)
        mu_h.wait()
        j_h.wait()
        return mu, j

    @staticmethod
    def backward(ctx, gmu, gj):
        # Forward: right -> left (carry flows right to left)
        # Backward: left -> right (gradients flow left to right)
        # gmu, gj are gradients w.r.t. the received carry values
        # Send them to the peer (right tile)
        sh = dist.isend(gmu.contiguous(), dst=ctx.peer, group=ctx.group)
        sj = dist.isend(gj.contiguous(), dst=ctx.peer, group=ctx.group)
        sh.wait()
        sj.wait()
        # grads for: peer, template, mu_shape, j_shape, group
        return None, None, None, None, None


class _CarrySend(torch.autograd.Function):
    """Send carry to peer rank with differentiable backward.
    
    Forward: send (mu, j) to peer via isend/wait, return them for graph continuity.
    Backward: receive gradients from peer and add to local gradient computation.
    
    This is used by the RIGHT tile to send the carry to the LEFT tile.
    Forward: right sends (mu_boundary, j_boundary) to left.
    Backward: right receives gradients from left and incorporates them.
    """
    @staticmethod
    def forward(ctx, mu, j, peer, group):
        ctx.peer, ctx.group = peer, group
        ctx.save_for_backward(mu, j)
        h1 = dist.isend(mu.contiguous(), dst=peer, group=group)
        h2 = dist.isend(j.contiguous(), dst=peer, group=group)
        h1.wait()
        h2.wait()
        # Return the inputs so they can be attached to the graph
        return mu, j

    @staticmethod
    def backward(ctx, gmu_out, gj_out):
        # gmu_out, gj_out are gradients flowing through the returned values
        # Since we attach with * 0, these should be zero
        # The actual gradients come from the peer via irecv
        mu, j = ctx.saved_tensors
        rmu = torch.empty_like(mu)
        rj = torch.empty_like(j)
        h1 = dist.irecv(rmu, src=ctx.peer, group=ctx.group)
        h2 = dist.irecv(rj, src=ctx.peer, group=ctx.group)
        h1.wait()
        h2.wait()
        # The received gradients are the gradients w.r.t. the sent values
        # Return gradients for mu, j, peer, group
        return rmu, rj, None, None


# ============================================================================

def scan_diag_two_ranks(
    jac: Tensor,
    residual: Tensor,
    *,
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream] | None = None,
    backend: str = "eager",
) -> Tensor:
    """Inclusive diag scan by splitting time in half, then a carry.

    Rank 1's local scan runs concurrently with rank 0. The carry apply waits.
    ``streams``: virtual ranks on one GPU. ``None`` runs both tiles in one
    stream (correctness path / CPU).
    """
    _check_diag(jac, residual)
    time = residual.shape[1]
    if time < 2:
        return scan_diag(jac, residual, backend=backend)
    mid = time // 2
    j0, r0 = jac[:, :mid], residual[:, :mid]
    j1, r1 = jac[:, mid:], residual[:, mid:]
    if streams is None or residual.device.type != "cuda":
        left = scan_diag(j0, r0, backend=backend)
        right_local = scan_diag(j1, r1, backend=backend)
        prefix = _diag_prefix_products(j1, backend=backend)
        right = _compose_diag_carry(right_local, prefix, left[:, -1])
        return torch.cat((left, right), dim=1)

    s0, s1 = streams
    left = residual.new_empty(r0.shape)
    right_local = residual.new_empty(r1.shape)
    prefix = residual.new_empty(r1.shape)
    with torch.cuda.stream(s0):
        left.copy_(scan_diag(j0, r0, backend=backend))
    with torch.cuda.stream(s1):
        right_local.copy_(scan_diag(j1, r1, backend=backend))
        prefix.copy_(_diag_prefix_products(j1, backend=backend))
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.current_stream().wait_stream(s1)
    right = _compose_diag_carry(right_local, prefix, left[:, -1])
    return torch.cat((left, right), dim=1)


def sequential_prefix_two_ranks(jac: Tensor, residual: Tensor) -> Tensor:
    """Naive prefix split: rank 1 starts after rank 0's last state exists.

    Same numeric result as ``scan_diag``. The right tile is scanned from the
    carry (dummy identity step). That data dependence serializes the two
    CUDA streams.
    """
    _check_diag(jac, residual)
    time = residual.shape[1]
    if time < 2:
        return scan_diag(jac, residual)
    mid = time // 2
    left = scan_diag(jac[:, :mid], residual[:, :mid])
    right = _scan_diag_from_carry(jac[:, mid:], residual[:, mid:], left[:, -1])
    return torch.cat((left, right), dim=1)


def _scan_diag_from_carry(
    jac: Tensor, residual: Tensor, carry: Tensor, *, backend: str = "eager"
) -> Tensor:
    """Scan a tile whose ``δ`` before the first step is ``carry``.

    Dummy identity step: ``δ = 1·0 + carry``, then the real tile.
    """
    ones = jac.new_ones(jac.shape[0], 1, jac.shape[-1])
    jac_pad = torch.cat((ones, jac), dim=1)
    res_pad = torch.cat((carry.unsqueeze(1), residual), dim=1)
    return scan_diag(jac_pad, res_pad, backend=backend)[:, 1:]


def _diag_prefix_products(jac: Tensor, *, backend: str = "eager") -> Tensor:
    """Inclusive products ``P_t = J_t ⋯ J_0`` (elementwise) of a tile.

    Scan of residual ``(J_0, 0, …)``: ``δ_t = J_t δ_{t-1}`` with ``δ_0 = J_0``.
    """
    r_prod = torch.zeros_like(jac)
    r_prod[:, 0] = jac[:, 0]
    return scan_diag(jac, r_prod, backend=backend)


def _compose_diag_carry(local: Tensor, prefix: Tensor, carry: Tensor) -> Tensor:
    """``δ_t = P_t carry + δ^{local}_t`` after a zero-init scan of the tile."""
    return local + prefix * carry.unsqueeze(1)


def _check_diag(jac: Tensor, residual: Tensor) -> None:
    if jac.shape != residual.shape:
        raise ValueError(f"jac {tuple(jac.shape)} != residual {tuple(residual.shape)}")
    if jac.dim() != 3:
        raise ValueError("scan_diag_two_ranks is diagonal (B, T, D) only")


def _resolve_scan_backend(x: Tensor, backend: str) -> str:
    if backend == "auto":
        return "triton" if is_fused_dtype_supported(x.dtype, x.device) else "eager"
    if backend not in ("eager", "triton"):
        raise ValueError(f"unknown scan backend {backend!r}")
    return backend


def time_shard_bounds(time: int, rank: int, world_size: int) -> tuple[int, int]:
    """Contiguous ``[start, end)`` along time. The last rank takes the remainder."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} outside [0, {world_size})")
    if time < 0:
        raise ValueError(f"time must be >= 0, got {time}")
    base = time // world_size
    if rank < world_size - 1:
        start = rank * base
        return start, start + base
    return (world_size - 1) * base, time


def _exchange_left_boundary(
    states: Tensor,
    h0: Tensor | None,
    group: dist.ProcessGroup | None,
    rank: int,
) -> Tensor:
    """Exchange left boundary state for sharded CP.
    
    For rank k, gather the last state from rank k-1.
    
    Args:
        states: (B, T_local, ..., D) local states with arbitrary trailing dims
        h0: (B, ..., D) global initial state (only used for rank 0)
        group: process group (``None`` = default group)
        rank: current rank
        
    Returns:
        left_boundary: (B, ..., D) state to prepend (h0 for rank 0, else last state from rank-1)
    
    Note:
        All ranks must enter the same collectives. The gathered payload is
        states[:, -1] which is rank-uniform in shape.
        time_shard_bounds may give the last rank a larger time tile, but
        states[:, -1] always has the same shape across ranks.
    """
    batch = states.shape[0]
    trailing_shape = states.shape[2:]  # arbitrary trailing dims (e.g., (4, D) or (D,))

    def _rank0_left() -> Tensor:
        if h0 is not None:
            return h0
        return states.new_zeros((batch,) + trailing_shape)

    if not dist.is_initialized():
        return _rank0_left() if rank == 0 else states.new_zeros((batch,) + trailing_shape)

    world = dist.get_world_size(group=group)
    if world == 1:
        return _rank0_left()

    # ALL ranks must execute the all_gather — do not return early for rank 0.
    # group=None selects the default process group (same as other collectives).
    last_states = states[:, -1].contiguous()  # (B, ..., D)
    gathered = [torch.empty_like(last_states) for _ in range(world)]
    dist.all_gather(gathered, last_states, group=group)

    if rank == 0:
        return _rank0_left()
    return gathered[rank - 1]


def _prepend_sharded(
    states: Tensor,
    h0: Tensor | None,
    left_boundary: Tensor,
    rank: int,
) -> Tensor:
    """Prepend left boundary state to local states.
    
    For rank k: cat([left_boundary, states[:, :-1]])
    For rank 0: first element is h0 (or zeros)
    
    Args:
        states: (B, T_local, ..., D) local states with arbitrary trailing dims
        h0: (B, ..., D) global initial state
        left_boundary: (B, ..., D) state from rank-1 (or h0 for rank 0)
        rank: current rank
        
    Returns:
        states_prepended: (B, T_local, ..., D) with boundary prepended
    """
    batch, local_t = states.shape[0], states.shape[1]
    
    if rank == 0:
        # For rank 0, prepend h0 (or zeros)
        first = h0 if h0 is not None else states.new_zeros((batch,) + states.shape[2:])
        return torch.cat([first.unsqueeze(1), states[:, :-1]], dim=1)
    else:
        # For other ranks, prepend left_boundary
        return torch.cat([left_boundary.unsqueeze(1), states[:, :-1]], dim=1)


def scan_diag_context_parallel(
    jac: Tensor,
    residual: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    """Inclusive diag scan on a time tile owned by this rank.

    Rank 0 holds the left of the sequence. After a local scan, ranks
    AllGather ``(P_end, δ_end)`` of shape ``(B, d)`` and apply the exclusive
    prefix carry of that monoid. ``world_size==1`` (or no process group) is
    ``scan_diag``.

    When ``jac``/``residual`` require grad, the local scan is eager and the
    AllGather is ``torch.distributed.nn.functional.all_gather`` so the VJP
    of the carry reaches earlier ranks (Rank 1 → Rank 0). Triton has no
    scan autograd; it stays on the ``no_grad`` path.
    """
    _check_diag(jac, residual)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    needs_grad = torch.is_grad_enabled() and (jac.requires_grad or residual.requires_grad)
    backend = "eager" if needs_grad else _resolve_scan_backend(jac, backend)
    if world == 1:
        return scan_diag(jac, residual, backend=backend)

    rank = dist.get_rank(group=group)
    batch, local_t, dim = residual.shape
    if local_t == 0:
        local = residual
        prefix = residual
        p_end = residual.new_ones(batch, dim)
        d_end = residual.new_zeros(batch, dim)
    else:
        local = scan_diag(jac, residual, backend=backend)
        prefix = _diag_prefix_products(jac, backend=backend)
        p_end = prefix[:, -1].contiguous()
        d_end = local[:, -1].contiguous()
    p_list, d_list = _all_gather_monoid(p_end, d_end, world, group, needs_grad)
    # Exclusive prefix of the gathered monoid. Rank 0's carry is 0; later
    # ranks apply (P_k, δ_k) from the left. `acc` after the loop uses every
    # gathered tensor so AllGather stays in the autograd graph on Rank 0
    # (otherwise that rank skips the backward collective and the carry VJP
    # never reaches jac/residual on the left tile).
    acc = p_end.new_zeros(batch, dim)
    carry = acc
    for k in range(world):
        if k == rank:
            carry = acc
        acc = p_list[k] * acc + d_list[k]
    out = _compose_diag_carry(local, prefix, carry)
    if needs_grad:
        out = out + acc.unsqueeze(1) * 0
    if not torch.compiler.is_compiling():
        log.debug(
            "scan_diag_context_parallel",
            extra={
                "rank": rank,
                "world": world,
                "local_t": int(jac.shape[1]),
                "batch": batch,
                "d": dim,
                "backend": backend,
                "needs_grad": needs_grad,
            },
        )
    return out


def reverse_scan_diag_context_parallel(
    jac: Tensor,
    partial: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    """Eq. 2.6 reverse diag scan on a time tile. Carry travels Rank N-1 → 0.

    Last rank reverse-scans with zero right state, then sends ``μ`` at the
    tile head to the left neighbor. Rank 0 is the one that applies a
    nonzero carry.
    
    Uses differentiable _CarryRecv/_CarrySend for autograd support.
    """
    from pararnn.solvers.scan import reverse_scan_diag

    _check_diag(jac, partial)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    needs_grad = torch.is_grad_enabled() and (jac.requires_grad or partial.requires_grad)
    backend = "eager" if needs_grad else _resolve_scan_backend(jac, backend)
    if world == 1:
        return reverse_scan_diag(jac, partial, backend=backend)

    rank = dist.get_rank(group=group)
    batch, local_t, dim = partial.shape
    dtype, device = partial.dtype, partial.device
    
    # All ranks execute the same ordered sequence of collectives based only on world
    # Rank < world-1: receive from right
    # Rank > 0: send to left (after computing mu)
    
    incoming_mu, incoming_j = None, None
    sent_mu, sent_j = None, None
    
    if rank < world - 1:
        # Receive from right neighbor (rank+1)
        incoming_mu, incoming_j = _CarryRecv.apply(
            rank + 1, jac, (batch, dim), (batch, dim), group
        )
    
    # Now compute the local result
    if rank == world - 1:
        # Last rank: no incoming carry, reverse scan with zero right state
        if local_t == 0:
            mu = partial
        else:
            mu = reverse_scan_diag(jac, partial, backend=backend)
    else:
        # Not last rank: we received from right
        carry_mu, j_next = incoming_mu, incoming_j
        if local_t == 0:
            # Empty tile: return partial
            mu = partial
        else:
            # Reverse scan with carry from right
            mu = _reverse_scan_diag_from_carry(jac, partial, carry_mu, j_next, backend=backend)
    
    # Now send the boundary values to left neighbor (if rank > 0)
    # Payload is mu[:, 0] and jac[:, 0]
    if rank > 0:
        if local_t > 0:
            payload_mu = mu[:, 0]
            payload_j = jac[:, 0]
        else:
            payload_mu = partial.new_zeros(batch, dim)
            payload_j = partial.new_zeros(batch, dim)
        sent_mu, sent_j = _CarrySend.apply(payload_mu, payload_j, rank - 1, group)
        # Attach sent values to output so backward fires
        mu = mu + sent_mu.unsqueeze(1) * 0

    # Keep the recv node in the autograd graph even when the incoming carry is
    # not otherwise consumed (empty tile); otherwise its backward never fires
    # and the peer's _CarrySend.backward irecv blocks forever.
    if incoming_mu is not None:
        mu = mu + (incoming_mu.sum() + incoming_j.sum()) * 0

    return mu


def scan_diag_context_parallel_full(
    jac: Tensor,
    residual: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    """Full ``(B, T, d)`` on every rank: shard time, CP-scan, AllGather concat.

    This is the Newton hook: ``H`` stays replicated; the scan work is ``T/N``.
    """
    _check_diag(jac, residual)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    if world == 1:
        tile_backend = _resolve_scan_backend(jac, backend)
        return scan_diag(jac, residual, backend=tile_backend)
    rank = dist.get_rank(group=group)
    time = int(jac.shape[1])
    start, end = time_shard_bounds(time, rank, world)
    local = scan_diag_context_parallel(
        jac[:, start:end].contiguous(),
        residual[:, start:end].contiguous(),
        group=group,
        backend=backend,
    )
    return all_gather_time_tiles(local, time, rank, world, group)


def reverse_scan_diag_context_parallel_full(
    jac: Tensor,
    partial: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    from pararnn.solvers.scan import reverse_scan_diag

    _check_diag(jac, partial)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    if world == 1:
        tile_backend = _resolve_scan_backend(jac, backend)
        return reverse_scan_diag(jac, partial, backend=tile_backend)
    rank = dist.get_rank(group=group)
    time = int(jac.shape[1])
    start, end = time_shard_bounds(time, rank, world)
    local = reverse_scan_diag_context_parallel(
        jac[:, start:end].contiguous(),
        partial[:, start:end].contiguous(),
        group=group,
        backend=backend,
    )
    return all_gather_time_tiles(local, time, rank, world, group)


def all_gather_time_tiles(
    tile: Tensor,
    time: int,
    rank: int,
    world: int,
    group: dist.ProcessGroup | None,
) -> Tensor:
    """Rebuild ``(B, T, …)`` from this rank's ``(B, T_local, …)`` tile."""
    bounds = [time_shard_bounds(time, r, world) for r in range(world)]
    max_len = max(end - start for start, end in bounds)
    loc = int(tile.shape[1])
    if loc == max_len:
        pad = tile
    elif loc == 0:
        pad = tile.new_zeros(tile.shape[0], max_len, *tile.shape[2:])
    else:
        pad = torch.cat(
            (tile, tile.new_zeros(tile.shape[0], max_len - loc, *tile.shape[2:])),
            dim=1,
        )
    # NOTE: the reassembled ``(B, T, ...)`` output is REPLICATED on every rank and
    # every rank computes the same downstream loss.  Using
    # ``torch.distributed.nn.functional.all_gather`` here would backprop through a
    # *summing* reduce-scatter, multiplying the gradient of this rank's own tile by
    # ``world`` (measured: exactly 2x at world=2).  Instead gather detached copies
    # and splice only this rank's own (attached) tile back into its slot, so that
    # ``d(out)/d(tile) == d(out[:, my_slice])/d(tile)`` with no cross-rank sum.
    # Cross-rank gradient flow through the monoid carry is handled separately by
    # ``_all_gather_monoid`` (which deliberately keeps the summing backward).
    gathered = [torch.empty_like(pad) for _ in range(world)]
    dist.all_gather(gathered, pad.detach().contiguous(), group=group)
    parts = [gathered[r][:, : bounds[r][1] - bounds[r][0]] for r in range(world)]
    if torch.is_grad_enabled() and tile.requires_grad:
        parts[rank] = tile
    return torch.cat(parts, dim=1)


def _all_gather_monoid(
    p_end: Tensor,
    d_end: Tensor,
    world: int,
    group: dist.ProcessGroup | None,
    needs_grad: bool,
) -> tuple[tuple[Tensor, ...] | list[Tensor], tuple[Tensor, ...] | list[Tensor]]:
    if needs_grad:
        from torch.distributed.nn.functional import all_gather

        return all_gather(p_end, group=group), all_gather(d_end, group=group)
    p_list = [torch.empty_like(p_end) for _ in range(world)]
    d_list = [torch.empty_like(d_end) for _ in range(world)]
    dist.all_gather(p_list, p_end.contiguous(), group=group)
    dist.all_gather(d_list, d_end.contiguous(), group=group)
    return p_list, d_list


def _reverse_scan_diag_from_carry(
    jac: Tensor,
    partial: Tensor,
    carry_mu: Tensor,
    j_next: Tensor,
    *,
    backend: str = "eager",
) -> Tensor:
    """Reverse diag scan; ``carry_mu`` is ``μ`` at the first time of the next tile.

    ``j_next`` is ``J`` at that same time (needed for ``μ_{e-1} = J_e μ_e + ∂_{e-1}``).
    """
    j_rev = jac.new_zeros(jac.shape)
    if jac.shape[1] > 1:
        j_rev[:, 1:] = jac.flip(1)[:, :-1]
    j_rev[:, 0] = j_next
    scanned = _scan_diag_from_carry(j_rev, partial.flip(1), carry_mu, backend=backend)
    return scanned.flip(1)


# ============================================================================
# Block4 (4x4) context parallel implementations
# ============================================================================


def _check_block4(jac: Tensor, residual: Tensor) -> None:
    """Validate block4 tensor shapes."""
    if jac.dim() != 5 or residual.dim() != 4:
        raise ValueError(f"block4 requires jac (B,T,4,4,D) and residual (B,T,4,D), got jac {tuple(jac.shape)} residual {tuple(residual.shape)}")
    if jac.shape[2:4] != (4, 4):
        raise ValueError(f"block4 jac must have (4,4) inner dims, got {jac.shape[2:4]}")
    if residual.shape[2] != 4:
        raise ValueError(f"block4 residual must have 4 in dim 2, got {residual.shape[2]}")
    if jac.shape[-1] != residual.shape[-1]:
        raise ValueError(f"block4 jac and residual must match in D: jac {jac.shape[-1]} vs residual {residual.shape[-1]}")


def _scan_block4_from_carry(
    jac: Tensor, residual: Tensor, carry: Tensor, *, backend: str = "eager"
) -> Tensor:
    """Scan a tile whose initial state is given by carry.

    Computes scan_block4(jac, residual') where residual'[0] = residual[0] + J_0 @ carry.
    This yields δ_local_t + P_t @ carry, replacing the old einsum-based carry apply.

    Args:
        jac: (B, T, 4, 4, D) Jacobian tile
        residual: (B, T, 4, D) residual tile
        carry: (B, 4, D) incoming carry from previous ranks
        backend: scan backend

    Returns:
        (B, T, 4, D) - scan result with carry applied
    """
    batch, time, _, d = residual.shape

    if time == 0:
        return residual.clone()

    # u = J_0 @ carry
    u = _mv4(jac[:, 0], carry)

    # r' = residual with r'[0] = r[0] + u
    r_prime = residual.clone()
    r_prime[:, 0] = r_prime[:, 0] + u

    return scan_block4(jac, r_prime, backend=backend)


def _apply_block4_carry_exclusive(
    p_list: list[Tensor],
    d_list: list[Tensor],
    rank: int,
    world: int,
    batch: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
    needs_grad: bool = False,
) -> Tensor:
    """Compute exclusive prefix of gathered monoid, return carry for this rank.
    
    Args:
        p_list: list of (B, 4, 4, D) prefix Jacobians from each rank
        d_list: list of (B, 4, D) deltas from each rank
        rank: this rank's ID
        world: total number of ranks
        batch, d: dimensions
        needs_grad: whether to keep all gathered tensors in autograd graph
    
    Returns:
        carry: (B, 4, D) - the exclusive prefix carry to apply to this rank's tile
    """
    # Exclusive prefix: for rank k, carry = ⊕_{i<k} (P_i, δ_i)
    # Use fp32 for precision when input is bf16/fp16
    acc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    
    acc_p = torch.eye(4, device=device, dtype=acc_dtype).unsqueeze(0).unsqueeze(-1).expand(batch, 4, 4, d).clone()
    acc_d = torch.zeros(batch, 4, d, device=device, dtype=acc_dtype)
    
    # Convert inputs to acc_dtype if needed
    p_list_dtype = [p.to(acc_dtype) for p in p_list] if acc_dtype != dtype else p_list
    d_list_dtype = [d_.to(acc_dtype) for d_ in d_list] if acc_dtype != dtype else d_list
    
    carry = acc_d.clone()
    for k in range(world):
        if k == rank:
            carry = acc_d.clone()
        # acc = (P_k, δ_k) ⊕ acc = (P_k @ P_acc, P_k @ δ_acc + δ_k)
        new_p = torch.einsum("bikd,bkjd->bijd", p_list_dtype[k], acc_p)
        new_d = torch.einsum('bikd,bkd->bid', p_list_dtype[k], acc_d) + d_list_dtype[k]
        acc_p, acc_d = new_p, new_d
    
    # Cast back to original dtype
    carry = carry.to(dtype)
    
    # For autograd: ensure all gathered tensors are in the graph
    # Add a dummy operation that references all p_list and d_list
    if needs_grad:
        dummy = carry
        for p, d_ in zip(p_list, d_list):
            dummy = dummy + p.sum() * 0 + d_.sum() * 0
        carry = carry + dummy * 0
    
    return carry


def scan_block4_context_parallel(
    jac: Tensor,
    residual: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    """Inclusive block4 scan on a time tile owned by this rank.

    Rank 0 holds the left of the sequence. After a local scan, ranks
    AllGather the tile monoid (P_end, δ_end) and apply the exclusive
    prefix carry. world_size==1 (or no process group) degenerates to
    plain scan_block4.

    When jac/residual require grad, the local scan is eager and the
    AllGather uses torch.distributed.nn.functional.all_gather for
    proper autograd support.
    """
    from pararnn.solvers.scan import scan_block4_monoid

    _check_block4(jac, residual)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    needs_grad = torch.is_grad_enabled() and (jac.requires_grad or residual.requires_grad)
    
    # Resolve backend - force eager if grad is needed (Triton has no autograd)
    if needs_grad:
        backend = "eager"
    elif backend == "auto":
        backend = "triton" if is_fused_dtype_supported(jac.dtype, jac.device) else "eager"
    
    if world == 1:
        return scan_block4(jac, residual, backend=backend)

    rank = dist.get_rank(group=group)
    batch, local_t, _, d = residual.shape
    
    if local_t == 0:
        # Empty tile: identity monoid
        local = residual
        p_end = jac.new_zeros(batch, 4, 4, d)
        for i in range(4):
            p_end[:, i, i, :] = 1.0
        d_end = jac.new_zeros(batch, 4, d)
    else:
        # Local scan and get tile-end monoid for free
        local, p_end = scan_block4_monoid(jac, residual, backend=backend)
        d_end = local[:, -1].contiguous()
    
    # AllGather monoid from all ranks
    p_list, d_list = _all_gather_monoid(p_end, d_end, world, group, needs_grad)
    
    # Compute exclusive prefix carry
    carry = _apply_block4_carry_exclusive(p_list, d_list, rank, world, batch, d, jac.device, jac.dtype, needs_grad)
    
    # Apply carry by re-running scan with modified residual[0]
    out = _scan_block4_from_carry(jac, residual, carry, backend=backend)
    
    if needs_grad:
        # Keep the carry in the graph
        out = out + carry.sum() * 0
    
    if not torch.compiler.is_compiling():
        log.debug(
            "scan_block4_context_parallel",
            extra={
                "rank": rank,
                "world": world,
                "local_t": int(jac.shape[1]),
                "batch": batch,
                "d": d,
                "backend": backend,
                "needs_grad": needs_grad,
            },
        )
    return out


def scan_block4_context_parallel_full(
    jac: Tensor,
    residual: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    """Full (B, T, 4, D) on every rank: shard time, CP-scan, AllGather concat.

    This is the Newton hook for ParaSLSTM(mix='diag'): H stays replicated;
    the scan work is T/N.
    """
    _check_block4(jac, residual)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    if world == 1:
        tile_backend = _resolve_scan_backend(jac, backend)
        return scan_block4(jac, residual, backend=tile_backend)
    
    rank = dist.get_rank(group=group)
    time = int(jac.shape[1])
    start, end = time_shard_bounds(time, rank, world)
    
    local = scan_block4_context_parallel(
        jac[:, start:end].contiguous(),
        residual[:, start:end].contiguous(),
        group=group,
        backend=backend,
    )
    return all_gather_time_tiles(local, time, rank, world, group)


def reverse_scan_block4_context_parallel(
    jac: Tensor,
    partial: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    """Eq. 2.6 reverse block4 scan on a time tile. Carry travels Rank N-1 → 0.

    Last rank reverse-scans with zero right state, then sends μ at the
    tile head to the left neighbor. The payload is (μ_boundary, J_boundary).
    
    Uses differentiable _CarryRecv/_CarrySend for autograd support.
    """
    from pararnn.solvers.scan import reverse_scan_block4

    _check_block4(jac, partial)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    needs_grad = torch.is_grad_enabled() and (jac.requires_grad or partial.requires_grad)
    backend = "eager" if needs_grad else _resolve_scan_backend(jac, backend)
    if world == 1:
        return reverse_scan_block4(jac, partial, backend=backend)

    rank = dist.get_rank(group=group)
    batch, local_t, _, d = partial.shape
    dtype, device = partial.dtype, partial.device
    
    # All ranks execute the same ordered sequence of collectives based only on world
    # Rank < world-1: receive from right
    # Rank > 0: send to left (after computing mu)
    
    incoming_mu, incoming_j = None, None
    sent_mu, sent_j = None, None
    
    if rank < world - 1:
        # Receive from right neighbor (rank+1)
        incoming_mu, incoming_j = _CarryRecv.apply(
            rank + 1, jac, (batch, 4, d), (batch, 4, 4, d), group
        )
    
    # Now compute the local result
    if rank == world - 1:
        # Last rank: no incoming carry, reverse scan with zero right state
        if local_t == 0:
            mu = partial.new_zeros(batch, 0, 4, d) if partial.shape[1] == 0 else partial.new_zeros(batch, 0, 4, d)
        else:
            mu = reverse_scan_block4(jac, partial, backend=backend)
    else:
        # Not last rank: we received from right
        carry_mu, j_next = incoming_mu, incoming_j
        if local_t == 0:
            # Empty tile: return zeros
            mu = partial.new_zeros(batch, 0, 4, d)
        else:
            # Reverse scan with carry from right
            mu = _reverse_scan_block4_from_carry(jac, partial, carry_mu, j_next, backend=backend)
    
    # Now send the boundary values to left neighbor (if rank > 0)
    # Payload is mu[:, 0] and jac[:, 0]
    if rank > 0:
        if local_t > 0:
            payload_mu = mu[:, 0]
            payload_j = jac[:, 0]
        else:
            payload_mu = partial.new_zeros(batch, 4, d)
            payload_j = jac.new_zeros(batch, 4, 4, d)
            for i in range(4):
                payload_j[:, i, i, :] = 1.0
        sent_mu, sent_j = _CarrySend.apply(payload_mu, payload_j, rank - 1, group)
        # Attach sent values to output so backward fires
        mu = mu + sent_mu.unsqueeze(1) * 0

    # Keep the recv node in the autograd graph even when the incoming carry is
    # not otherwise consumed (empty tile); otherwise its backward never fires
    # and the peer's _CarrySend.backward irecv blocks forever.
    if incoming_mu is not None:
        mu = mu + (incoming_mu.sum() + incoming_j.sum()) * 0

    return mu


def _reverse_scan_block4_from_carry(
    jac: Tensor,
    partial: Tensor,
    carry_mu: Tensor,
    j_next: Tensor,
    *,
    backend: str = "eager",
) -> Tensor:
    """Reverse block4 scan with carry from the next tile.
    
    Implements μ_t = J_{t+1}^T @ μ_{t+1} + g_t.
    
    Args:
        jac: (B, T, 4, 4, D) - Jacobian tile
        partial: (B, T, 4, D) - partial adjoints ∂L/∂h (g_t)
        carry_mu: (B, 4, D) - μ at the boundary (μ_{T} from the right)
        j_next: (B, 4, 4, D) - J at position T (the boundary Jacobian, i.e., J_T)
    
    Returns:
        mu: (B, T, 4, D) - reverse scan result [μ_{T-1}, μ_{T-2}, ..., μ_0]
    """
    from pararnn.solvers.scan import scan_block4

    batch, local_t, _, d = partial.shape

    if local_t == 0:
        return partial.new_zeros(batch, 0, 4, d)

    # Mirror ``reverse_scan_block4`` (scan.py): the 4x4 block lives in dims
    # (-3, -2), so transposing it is ``transpose(-3, -2)``. Using (-2, -1)
    # would swap the block dim with the feature dim D.
    # j_rev[k] = J_{T-k}^T for k = 0 .. T-1, with j_rev[0] = J_next^T taken
    # from the right neighbour (J at the first time of the next tile).
    j_t = jac.transpose(-3, -2)  # (B, T, 4, 4, D)
    j_rev = jac.new_zeros(batch, local_t, 4, 4, d)
    if local_t > 1:
        j_rev[:, 1:] = j_t.flip(1)[:, :-1]
    j_rev[:, 0] = j_next.transpose(-3, -2)

    # Dummy identity step: delta = I @ 0 + carry_mu, then the real tile.
    # Block4 analogue of ``_scan_diag_from_carry``'s ``ones`` pad.
    eye = jac.new_zeros(batch, 1, 4, 4, d)
    for i in range(4):
        eye[:, 0, i, i, :] = 1.0
    jac_pad = torch.cat((eye, j_rev), dim=1)  # (B, T+1, 4, 4, D)
    res_pad = torch.cat((carry_mu.unsqueeze(1), partial.flip(1)), dim=1)

    scanned = scan_block4(jac_pad, res_pad, backend=backend)  # (B, T+1, 4, D)
    return scanned[:, 1:].flip(1)


def reverse_scan_block4_context_parallel_full(
    jac: Tensor,
    partial: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    """Full (B, T, 4, D) on every rank: shard time, reverse CP-scan, AllGather concat."""
    from pararnn.solvers.scan import reverse_scan_block4

    _check_block4(jac, partial)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    if world == 1:
        tile_backend = _resolve_scan_backend(jac, backend)
        return reverse_scan_block4(jac, partial, backend=tile_backend)
    
    rank = dist.get_rank(group=group)
    time = int(jac.shape[1])
    start, end = time_shard_bounds(time, rank, world)
    
    local = reverse_scan_block4_context_parallel(
        jac[:, start:end].contiguous(),
        partial[:, start:end].contiguous(),
        group=group,
        backend=backend,
    )
    return all_gather_time_tiles(local, time, rank, world, group)


# ============================================================================
# Max-plus (stabilizer) context-parallel scan — Stage-3 Picard init
# ============================================================================

_MAXPLUS_NEG_INF = -1.0e20


def _maxplus_local(z_f: Tensor, z_i: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Local max-plus scan assuming ``m0 = -∞``.

    Returns ``(m_local, Q, Q_end, M_end)`` with
    ``Q_t = Σ_{k≤t} z_f``, ``M_end = max_t (z_i,t − Q_t)``,
    ``m_local = Q + cummax(z_i − Q)``.
    """
    q = z_f.cumsum(dim=1)
    m_run = (z_i - q).cummax(dim=1).values
    m_local = q + m_run
    return m_local, q, q[:, -1].contiguous(), m_run[:, -1].contiguous()


def _compose_maxplus(
    q_a: Tensor, m_a: Tensor, q_b: Tensor, m_b: Tensor
) -> tuple[Tensor, Tensor]:
    """Associative tile compose: B after A → ``(Q_A+Q_B, max(M_A, M_B−Q_A))``."""
    return q_a + q_b, torch.maximum(m_a, m_b - q_a)


def scan_maxplus_context_parallel(
    z_f: Tensor,
    z_i: Tensor,
    *,
    m0: Tensor | None = None,
    group: dist.ProcessGroup | None = None,
) -> tuple[Tensor, Tensor]:
    """Inclusive max-plus scan on a time tile: ``m_t = max(z_f + m_{t-1}, z_i)``.

    Returns ``(m, m_end)`` where ``m`` is ``(B, T_local, D)`` and ``m_end`` is
    the tile-end stabilizer. Boundary payload per rank is ``(Q_end, M_end)`` —
    ``2·B·D`` floats via AllGather (no full-``T`` buffers).
    """
    if z_f.shape != z_i.shape or z_f.dim() != 3:
        raise ValueError(
            f"maxplus expects z_f/z_i (B,T,D), got {tuple(z_f.shape)} / {tuple(z_i.shape)}"
        )
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    batch, local_t, dim = z_f.shape
    z_f32 = z_f.float()
    z_i32 = z_i.float()
    if m0 is None:
        m0_b = z_f32.new_zeros(batch, dim)
    else:
        m0_b = m0.float()

    if local_t == 0:
        empty = z_f32.new_zeros(batch, 0, dim)
        return empty.to(dtype=z_f.dtype), m0_b.to(dtype=z_f.dtype)

    m_local, q, q_end, m_end = _maxplus_local(z_f32, z_i32)
    if world == 1:
        m = q + torch.maximum((z_i32 - q).cummax(dim=1).values, m0_b.unsqueeze(1))
        return m.to(dtype=z_f.dtype), m[:, -1].contiguous().to(dtype=z_f.dtype)

    rank = dist.get_rank(group=group)
    q_list, m_list = _all_gather_monoid(q_end, m_end, world, group, needs_grad=False)

    acc_q = q_end.new_zeros(batch, dim)
    acc_m = q_end.new_full((batch, dim), _MAXPLUS_NEG_INF)
    m_carry = m0_b
    for k in range(world):
        if k == rank:
            if rank == 0:
                m_carry = m0_b
            else:
                m_carry = acc_q + torch.maximum(m0_b, acc_m)
        acc_q, acc_m = _compose_maxplus(acc_q, acc_m, q_list[k], m_list[k])

    m = torch.maximum(m_local, q + m_carry.unsqueeze(1))
    return m.to(dtype=z_f.dtype), m[:, -1].contiguous().to(dtype=z_f.dtype)

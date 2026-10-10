"""sLSTM Picard initial guess and adaptive P-rung retry (para-slstm.md)."""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import torch
from torch import Tensor, nn

from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.solvers.newton.config import NewtonConfig, NewtonStats
from pararnn.solvers.slstm_picard import slstm_picard_init, slstm_zero_hidden_init

log = logging.getLogger("pararnn.solvers.newton")

# Extra Picard passes above ``slstm_auto_picard(T)`` when ``picard_adapt``
# climbs after the envelope still misses ``picard_retry_atol``.
_PICARD_ADAPT_HEADROOM = 2

# Windowed / chunked solve: paper §4.2 / App. D — static P=3 for C≤128.
_WINDOWED_PICARD = 3


def slstm_auto_picard(
    seq_len: int,
    *,
    chunk_len: int | None = None,
    fused_time_loop: bool = False,
) -> int:
    """Library Picard ``P`` for ParaSLSTM by sequence length.

    Unchunked (global scan): measured SPMD envelope
    ``P(T) = max(1, ceil(0.75 log2 T - 2))`` at ``d_h=256``, K=3
    (phase grid through ``T=131072``; covers every finite ``max P*``).
    Windowed (``chunk_len`` or ``fused_time_loop``): static ``P=3``.

    Explicit ``picard_iters=0`` selects zero-hidden init only. When the
    envelope still misses ``residual_fail`` / ``picard_retry_atol``,
    ``slstm_picard_next`` raises ``P`` by one up to envelope + headroom.

    Parameters
    ----------
    seq_len : int
        Sequence length ``T`` (``x.shape[1]``).
    chunk_len : int or None
        ``NewtonConfig.chunk_len``; any windowed solve uses ``P=3``.
    fused_time_loop : bool
        ``NewtonConfig.fused_time_loop``; same static ``P=3``.

    Returns
    -------
    int
        Picard iteration count ``P >= 1``.
    """
    if chunk_len is not None or fused_time_loop:
        return _WINDOWED_PICARD
    t = max(1, int(seq_len))
    # T<8 → formula ≤0; floor at 1 (Dyck / smoke lengths).
    return max(1, int(math.ceil(0.75 * math.log2(t) - 2.0)))


def slstm_picard_next(
    picard_iters: int,
    seq_len: int | None = None,
    *,
    chunk_len: int | None = None,
    fused_time_loop: bool = False,
) -> int | None:
    """Next Picard depth after ``picard_iters`` for ``picard_adapt``.

    Climbs to ``slstm_auto_picard(...)`` when below the envelope, then
    ``+1`` up to envelope + ``_PICARD_ADAPT_HEADROOM``. Without ``seq_len``,
    climbs ``+1`` up to a fixed safety cap (envelope at ``T=2**17`` + headroom).

    Parameters
    ----------
    picard_iters : int
        Current Picard iteration count.
    seq_len : int or None
        Sequence length for the envelope target.
    chunk_len, fused_time_loop
        Forwarded to ``slstm_auto_picard`` when resolving the target.

    Returns
    -------
    int or None
        Next depth, or ``None`` when already at the adapt cap.
    """
    p = int(picard_iters)
    if seq_len is not None:
        target = slstm_auto_picard(
            seq_len, chunk_len=chunk_len, fused_time_loop=fused_time_loop
        )
    else:
        # Cap matches envelope(131072)=11 + headroom when T is unknown.
        target = slstm_auto_picard(1 << 17)
    cap = target + _PICARD_ADAPT_HEADROOM
    if p < target:
        return target
    if p < cap:
        return p + 1
    return None


def _resolve_picard(cell: nn.Module, x: Tensor, config: NewtonConfig) -> NewtonConfig:
    # Pipeline fused CP: static P=3 envelope (same as windowed fused_time_loop).
    if config.scan_backend == "fused_context_parallel":
        if isinstance(cell, ParaSLSTM) and config.picard_iters is None:
            chosen = slstm_auto_picard(
                x.shape[1],
                chunk_len=None,
                fused_time_loop=True,
            )
            if not torch.compiler.is_compiling():
                log.debug(
                    "slstm_picard_auto_fused_cp",
                    extra={
                        "chosen": chosen,
                        "local_seq_len": int(x.shape[1]),
                        "d_h": cell.d_h,
                        "scan_backend": config.scan_backend,
                    },
                )
            return replace(config, picard_iters=chosen, picard_adapt=False)
        return replace(config, picard_adapt=False)
    # Monoid fused CP: pin P=3 (windowed envelope) on all worlds.
    # world>1 warm-start is 2-pass windowed monoid Picard on long T
    # (local C-window Rh + frozen_gate_scan_fused_cp per outer step) and
    # unchunked 2-stage monoid on short T. Newton stays K=3 monoid CP.
    if config.scan_backend == "fused_context_parallel_monoid":
        if isinstance(cell, ParaSLSTM) and config.picard_iters is None:
            import torch.distributed as dist

            world = dist.get_world_size() if dist.is_initialized() else 1
            chosen = slstm_auto_picard(
                x.shape[1],
                chunk_len=None,
                fused_time_loop=True,
            )
            if not torch.compiler.is_compiling():
                log.info(
                    "slstm_picard_auto_fused_cp_monoid",
                    extra={
                        "chosen": chosen,
                        "world": world,
                        "local_seq_len": int(x.shape[1]),
                        "d_h": cell.d_h,
                        "picard": (
                            "windowed_monoid_cp" if world > 1 else "windowed"
                        ),
                    },
                )
            return replace(config, picard_iters=chosen, picard_adapt=False)
        return replace(config, picard_adapt=False)
    # Stage-3 monoid CP: sharded Picard (max-plus + affine). Keep
    # ``picard_adapt=False`` so early-exit / P-retry stays rank-symmetric.
    if config.scan_backend == "context_parallel_sharded":
        if isinstance(cell, ParaSLSTM) and config.picard_iters is None:
            global_t = _cp_global_seq_len(x)
            chosen = slstm_auto_picard(
                global_t,
                chunk_len=config.chunk_len,
                fused_time_loop=bool(config.fused_time_loop),
            )
            if not torch.compiler.is_compiling():
                log.debug(
                    "slstm_picard_auto_sharded",
                    extra={
                        "chosen": chosen,
                        "local_seq_len": int(x.shape[1]),
                        "global_seq_len": global_t,
                        "d_h": cell.d_h,
                    },
                )
            return replace(config, picard_iters=chosen, picard_adapt=False)
        return replace(config, picard_adapt=False)
    if config.picard_iters is None:
        if isinstance(cell, ParaSLSTM):
            chosen = slstm_auto_picard(
                x.shape[1],
                chunk_len=config.chunk_len,
                fused_time_loop=bool(config.fused_time_loop),
            )
            if not torch.compiler.is_compiling():
                log.debug(
                    "slstm_picard_auto",
                    extra={
                        "chosen": chosen,
                        "seq_len": x.shape[1],
                        "d_h": cell.d_h,
                        "chunk_len": config.chunk_len,
                        "fused_time_loop": bool(config.fused_time_loop),
                    },
                )
            return replace(config, picard_iters=chosen)
        return replace(config, picard_iters=0)
    if config.picard_iters and not isinstance(cell, ParaSLSTM):
        from pararnn.cells.para_m2rnn import ParaM2RNN

        if isinstance(cell, ParaM2RNN):
            # picard_iters>=1: frozen-W warm-start (one W=0 scan). Extra
            # values are treated the same (single warm-start).
            return config
        raise TypeError(
            f"NewtonConfig(picard_iters=) is ParaSLSTM / ParaM2RNN only (got {type(cell).__name__})"
        )
    return config


def _cp_global_seq_len(x: Tensor) -> int:
    """Sum of local time tiles across ranks (Stage-2). World=1 → ``x.shape[1]``.

    Uses an int-tensor AllGather (no pickle / ``all_gather_object``).
    """
    import torch.distributed as dist

    local_t = int(x.shape[1])
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return local_t
    device = x.device if x.is_cuda else torch.device("cpu")
    local = torch.tensor([local_t], device=device, dtype=torch.int64)
    world = dist.get_world_size()
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    return int(sum(int(t.item()) for t in gathered))


def _picard_retry_needed(res: float, config: NewtonConfig) -> bool:
    if not math.isfinite(res):
        return True
    cap = config.picard_retry_atol
    if cap is not None and res > cap:
        return True
    fail = config.residual_fail
    return fail is not None and res > fail


def _newton_forward_picard_adapt(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None,
    extra: Tensor | None = None,
    stats: NewtonStats | None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    """Re-run Alg. 1 at the next P rung when the guess was outside the basin."""
    from pararnn.solvers.newton import _fill_stats, _newton_solve

    cfg = config
    while True:
        st = stats if stats is not None else NewtonStats()
        quiet = replace(cfg, residual_fail=None)
        states = _newton_solve(
            cell,
            x,
            quiet,
            h0=h0,
            extra=extra,
            stats=st,
            cu_seqlens=cu_seqlens,
            block_table=block_table,
        )
        res = st.max_residual
        nxt = slstm_picard_next(
            int(cfg.picard_iters or 0),
            int(x.shape[1]),
            chunk_len=cfg.chunk_len,
            fused_time_loop=bool(cfg.fused_time_loop),
        )
        if _picard_retry_needed(res, config) and nxt is not None:
            if not torch.compiler.is_compiling():
                log.info(
                    "picard_adapt",
                    extra={
                        "from_p": int(cfg.picard_iters or 0),
                        "to_p": nxt,
                        "max_residual": res,
                        "seq_len": int(x.shape[1]),
                        "batch": int(x.shape[0]),
                        "d_h": cell.d_h,
                        "scan_backend": cfg.scan_backend,
                    },
                )
            cfg = replace(cfg, picard_iters=nxt)
            continue
        cap = config.residual_fail
        if cap is not None and (not math.isfinite(res) or res > cap):
            _fill_stats(
                cell,
                x,
                states,
                h0,
                config,
                iters=st.iters,
                stats=st,
                residual_history=st.residual_history,
                known_residual=res,
                cu_seqlens=cu_seqlens,
                block_table=block_table,
            )
        return states


def _slstm_newton_guess(
    cell: ParaSLSTM,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None,
    wx: Tensor | None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Zero-hidden, or extra frozen-gate Picard scans. Still O(log T)."""
    pre = wx if wx is not None else cell.W_x(x)
    # Monoid fused CP: world=1 windowed Picard; world>1 windowed monoid CP
    # (local C-window Rh + 2-pass frozen-gate AllGather) on long T.
    if config.scan_backend == "fused_context_parallel_monoid":
        n_p = int(config.picard_iters or 0)
        if n_p == 0:
            return slstm_zero_hidden_init(pre, eps=cell.eps, h0=h0, cu_seqlens=cu_seqlens)
        import torch.distributed as dist

        from pararnn.solvers.newton.config import FUSED_WINDOW_DEFAULT

        win = (
            int(config.fused_window_len)
            if config.fused_window_len is not None
            else FUSED_WINDOW_DEFAULT
        )
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world > 1:
            return _slstm_sharded_windowed_init(
                cell, pre, h0=h0, n_picard=n_p, chunk_len=win
            )
        return slstm_picard_init(
            cell, pre, h0=h0, n_picard=n_p, cu_seqlens=cu_seqlens, chunk_len=win
        )
    if config.scan_backend == "context_parallel_sharded":
        return _slstm_sharded_init(
            cell, pre, h0=h0, n_picard=int(config.picard_iters or 0)
        )
    if config.picard_iters:
        win = None
        if config.fused_time_loop:
            from pararnn.solvers.newton.config import FUSED_WINDOW_DEFAULT

            win = (
                int(config.fused_window_len)
                if config.fused_window_len is not None
                else FUSED_WINDOW_DEFAULT
            )
        elif config.chunk_len is not None:
            # Outer chunking already windows x; inner guess stays full-chunk.
            win = None
        if not torch.compiler.is_compiling():
            log.debug(
                "slstm_picard_init",
                extra={
                    "n_picard": config.picard_iters,
                    "seq_len": x.shape[1],
                    "batch": x.shape[0],
                    "d_h": cell.d_h,
                    "chunk_len": win,
                },
            )
        return slstm_picard_init(
            cell,
            pre,
            h0=h0,
            n_picard=config.picard_iters,
            cu_seqlens=cu_seqlens,
            chunk_len=win,
        )
    return slstm_zero_hidden_init(pre, eps=cell.eps, h0=h0, cu_seqlens=cu_seqlens)


def _slstm_sharded_init(
    cell: ParaSLSTM,
    pre: Tensor,
    *,
    h0: Tensor | None,
    n_picard: int,
) -> Tensor:
    """Stage-3 Picard / zero-hidden guess: local tile only (no full-``T`` gather)."""
    from pararnn.solvers.slstm_picard import slstm_picard_init_sharded

    return slstm_picard_init_sharded(
        cell, pre, h0=h0, n_picard=n_picard, group=None
    )


def _slstm_sharded_windowed_init(
    cell: ParaSLSTM,
    pre: Tensor,
    *,
    h0: Tensor | None,
    n_picard: int,
    chunk_len: int,
) -> Tensor:
    """Hierarchical continuous Picard warm-start (see ``sharded_windowed``)."""
    from pararnn.solvers.slstm_picard import slstm_picard_init_sharded_windowed

    return slstm_picard_init_sharded_windowed(
        cell,
        pre,
        h0=h0,
        n_picard=n_picard,
        chunk_len=chunk_len,
        group=None,
    )

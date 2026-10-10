"""Newton iterations wrapping a parallel scan (Danieli et al. 2025 Alg. 1).

K=3 (App. A) for ParaGRU/ParaLSTM. Init is eq. A.1 except ParaSLSTM, which
starts from the zero-hidden unroll. Backward is eq. 2.6 (one reverse scan).

``scan_backend='fused'`` is a handwritten Triton kernel for ParaGRU,
ParaLSTM, and ParaSLSTM ``mix='diag'``, and factorized Newton for
``ParaGRU(mix='head')`` / ``ParaSLSTM(mix='head')``. ``'auto'`` picks fused
on CUDA for those cells. ``ParaM2RNN`` uses a factorized Kronecker scan
(``newton_m2rnn_factorized``) on every backend alias; no dense ``(KV)²``.
``ParaRWKV7`` is linear in ``S``: ``newton_apply`` redirects to the
associative ``(G,U)`` monoid scan (or sequential when ``scan_backend='eager'``).
Packed ``cu_seqlens``: fused diag ParaGRU stays in-kernel; head ParaGRU
raises on explicit ``fused`` and remaps ``auto`` → ``eager``; LSTM/sLSTM
fused falls back to Triton scan with ``J=0`` at segment heads.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterator
from dataclasses import replace

import torch
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.kernels.device_recipe import resolve_slstm_launch
from pararnn.layout import prepend_state, prepend_state_ragged, validate_cu_seqlens
from pararnn.solvers.jacobian import step_and_jacobian
from pararnn.solvers.newton import config, dispatch, picard
from pararnn.solvers.newton.config import (
    _RESIDUAL_WARN,
    FUSED_WINDOW_DEFAULT,
    FUSED_WINDOW_LENS,
    NewtonConfig,
    NewtonDivergenceError,
    NewtonStats,
    _validate_config,
)
from pararnn.solvers.newton.config import (
    compile_safe_config as compile_safe_config,
)
from pararnn.solvers.newton.dispatch import (
    _resolve_backend,
    _reverse_scan,
    _scan,
    _state_vjp_at_times,
    _t0_state_vjp,
)
from pararnn.solvers.newton.picard import _newton_forward_picard_adapt, _slstm_newton_guess
from pararnn.solvers.slstm_log import (
    SLSTMLogCoords,
    slstm_clamp_log_coords,
    slstm_decode_log,
    slstm_encode_log,
)
from pararnn.solvers.vjp import cell_vjp_wx, uses_packed_vjp

# Diag fused ``chunk_len``: pack this many C-windows into one FTL walk so
# launch count is ``ceil(T/(pack·C))`` while ``W_x`` still streams at pack
# size. Default 128 (3060 T=131k); live value from ``resolve_slstm_launch``.
_SLSTM_CHUNK_PACK_TILES = 128


def _slstm_pack_tiles(device: torch.device | str | int | None, window_len: int) -> int:
    """Device-table pack size (CPU / non-CUDA → shape fallback inside resolve)."""
    return int(resolve_slstm_launch(device, window_len).pack_tiles)


LIBRARY_NEWTON_ITERS = config.LIBRARY_NEWTON_ITERS
_DEFAULT_RESIDUAL_ATOL = config._DEFAULT_RESIDUAL_ATOL
_DEFAULT_RESIDUAL_FAIL = config._DEFAULT_RESIDUAL_FAIL
slstm_auto_picard = picard.slstm_auto_picard
slstm_picard_next = picard.slstm_picard_next
_PICARD_ADAPT_HEADROOM = picard._PICARD_ADAPT_HEADROOM
_resolve_picard = picard._resolve_picard
_picard_retry_needed = picard._picard_retry_needed
_can_triton_scan = dispatch._can_triton_scan
_can_fuse = dispatch._can_fuse
_pick_auto = dispatch._pick_auto
_fused_error = dispatch._fused_error
_scan_named = dispatch._scan_named
_reverse_scan_named = dispatch._reverse_scan_named
_scan_infer = dispatch._scan_infer
_reverse_scan_infer = dispatch._reverse_scan_infer
_dense_slot_pack = dispatch._dense_slot_pack
_head_slot_pack = dispatch._head_slot_pack
_head_slot_unpack = dispatch._head_slot_unpack

log = logging.getLogger(__name__)

# Pack for the duration of ``_NewtonFixedPoint.apply`` only (forward thread).
# Backward reads ``ctx`` alone — never this pack (autograd may run on another
# worker thread). Overlapping concurrent ``newton_apply`` calls are unsupported;
# no ``threading.Lock`` here (Dynamo cannot trace that context manager).
_fwd_pack: dict[str, object] = {}


class _NewtonFixedPoint(torch.autograd.Function):
    """Eq. 2.6 backward. Module-level so Dynamo does not hit ``__build_class__``.

    Non-tensor state for backward lives on ``ctx`` (legal and thread-safe).
    ``_fwd_pack`` is only read inside ``forward``, then discarded.
    """

    @staticmethod
    def forward(
        ctx, x_in: Tensor, h0_in: Tensor, extra_in: Tensor, *param_tensors: Tensor
    ) -> Tensor:
        del param_tensors
        cell = _fwd_pack["cell"]
        config = _fwd_pack["config"]
        cu_seqlens = _fwd_pack["cu_seqlens"]
        stats = _fwd_pack["stats"]
        has_h0 = bool(_fwd_pack["has_h0"])
        has_extra = bool(_fwd_pack["has_extra"])
        block_table = _fwd_pack["block_table"]
        assert isinstance(cell, nn.Module)
        assert isinstance(config, NewtonConfig)
        h0_fwd = h0_in if has_h0 else None
        extra_fwd = extra_in if has_extra else None
        with torch.no_grad():
            states = _newton_forward(
                cell,
                x_in,
                config,
                h0=h0_fwd,
                extra=extra_fwd,
                stats=stats,  # type: ignore[arg-type]
                cu_seqlens=cu_seqlens,  # type: ignore[arg-type]
                block_table=block_table,  # type: ignore[arg-type]
            )
        # Everything backward needs — on ctx (autograd worker threads safe).
        ctx.cell = cell
        ctx.has_h0 = has_h0
        ctx.has_extra = has_extra
        ctx.recompute = bool(config.recompute)
        # Rematerialize with the same resolved config (incl. fused). Eq. 2.6
        # reverse scan maps fused → triton (no fused reverse kernel).
        ctx.fwd_config = config
        ctx.scan_backend = "triton" if config.scan_backend == "fused" else config.scan_backend
        ctx.jacobian = config.jacobian
        ctx.jac_structure = config.jac_structure
        # Window for eq. 2.6: ``chunk_len`` or ``fused_time_loop`` tile (same C).
        ctx.ift_window_len = _ift_window_len(config)
        ctx.has_cu_seqlens = cu_seqlens is not None
        saved_cs = (
            cu_seqlens if isinstance(cu_seqlens, Tensor) else x_in.new_zeros(0, dtype=torch.long)
        )
        if ctx.recompute:
            # Level 2: drop H* from the Function. With a window, rematerialize
            # per tile inside ``_eq26_vjp_chunked`` (peak O(C), not O(T)).
            ctx.save_for_backward(x_in, h0_in, extra_in, saved_cs)
        else:
            ctx.save_for_backward(states, x_in, h0_in, extra_in, saved_cs)
        return states

    @staticmethod
    def backward(ctx, grad_states: Tensor):
        ift_win = getattr(ctx, "ift_window_len", None)
        if ctx.recompute:
            x_in, h0_in, extra_in, saved_cs = ctx.saved_tensors
            h0_fwd = h0_in if ctx.has_h0 else None
            extra_fwd = extra_in if ctx.has_extra else None
            cs_bwd = saved_cs if ctx.has_cu_seqlens else None
            if ift_win is None:
                with torch.no_grad():
                    states = _newton_forward(
                        ctx.cell,
                        x_in,
                        ctx.fwd_config,
                        h0=h0_fwd,
                        extra=extra_fwd,
                        stats=None,
                        cu_seqlens=cs_bwd,
                    )
            else:
                # Windowed Level-2: no full-T H* remat up front.
                states = None
        else:
            states, x_in, h0_in, extra_in, saved_cs = ctx.saved_tensors
            h0_fwd = h0_in if ctx.has_h0 else None
            extra_fwd = extra_in if ctx.has_extra else None
            cs_bwd = saved_cs if ctx.has_cu_seqlens else None
        with _newton_precision_region(x_in.device):
            if ift_win is not None:
                grad_x, param_grads, grad_h0, g_wx = _eq26_vjp_chunked(
                    ctx.cell,
                    states,
                    x_in,
                    grad_states,
                    chunk_len=int(ift_win),
                    backend=ctx.scan_backend,
                    jacobian=ctx.jacobian,
                    jac_structure=ctx.jac_structure,
                    h0=h0_fwd,
                    extra=extra_fwd,
                    remat_config=ctx.fwd_config if states is None else None,
                )
            else:
                grad_x, param_grads, grad_h0, g_wx = _eq26_vjp(
                    ctx.cell,
                    states,
                    x_in,
                    grad_states,
                    backend=ctx.scan_backend,
                    jacobian=ctx.jacobian,
                    jac_structure=ctx.jac_structure,
                    h0=h0_fwd,
                    extra=extra_fwd,
                    cu_seqlens=cs_bwd,
                )
        if not x_in.requires_grad:
            grad_x = None
        if not ctx.has_h0 or not h0_in.requires_grad:
            grad_h0 = None
        if not ctx.has_extra or not extra_in.requires_grad:
            grad_extra = None
        else:
            grad_extra = g_wx
        return (grad_x, grad_h0, grad_extra, *param_grads)


@contextlib.contextmanager
def _newton_precision_region(device: torch.device) -> Iterator[None]:
    """Keep Newton / eq. 2.6 in the tensor dtype under outer ``autocast``.

    AMP would half ``W_x(x)`` while states stay in the module dtype and break
    the VJP dtype check. Disable autocast for the solve; callers that want
    fp16/bf16 Newton put the module and ``x`` in that dtype explicitly.
    """
    if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        with torch.autocast(device_type="cuda", enabled=False):
            yield
        return
    if device.type == "cpu" and torch.is_autocast_enabled("cpu"):
        with torch.autocast(device_type="cpu", enabled=False):
            yield
        return
    yield


def newton_apply(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig | None = None,
    *,
    h0: Tensor | None = None,
    extra: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    """Solve ``F(H)=0`` with Newton + associative scan (Danieli et al. Alg. 1).

    Inner solve is a scan on the local Jacobian monoid. With gradients,
    backward is one reverse scan (eq. 2.6). For train/eval switching use
    :class:`~pararnn.layers.ParaRNN`; call this for custom cells or benches.

    Parameters
    ----------
    cell : nn.Module
        Cell with ``step`` (prefer ``step_with_jacobian``). Fused: ParaGRU,
        ParaLSTM, ParaSLSTM ``mix='diag'|'head'``. Factorized Kronecker:
        ``ParaM2RNN`` (recipe ``max_iters=4``).
    x : Tensor of shape (batch, time, d_in)
        With ``cu_seqlens``, shape ``(1, N, d_in)``.
    config : NewtonConfig, optional
        Defaults to ``NewtonConfig()`` (``K=3``; use ``K=4`` for ``ParaM2RNN``).
    h0 : Tensor, optional
        Paper ``h_0`` (default zeros). Packed: ``(S, ...)``. With
        ``block_table``: pool ``(C, ...)``.
    extra : Tensor, optional
        Additive bias on the input affine, same shape as ``W_x(x)`` /
        ``project_wx(x)``. Forward uses ``pre = W_x(x) + extra``; Autograd
        returns ``∇_{extra} L = ∇_{wx} L``.
    stats : NewtonStats, optional
        Filled in-place with residual / backend info.
    cu_seqlens : Tensor of shape (S + 1,), optional
        Packed-batch offsets; segment heads use ``J=0``.
    block_table : Tensor of shape (batch,) or (S,), optional
        Slot ids into a paged ``h0`` pool (fused inference; no ``chunk_len``).

    Returns
    -------
    H : Tensor of shape (batch, time, *state)
        Solved trajectory (GRU ``d_h``; LSTM ``(2, d_h)``; sLSTM ``(4, d_h)``;
        M²RNN ``(K, V)``).

    Raises
    ------
    NewtonDivergenceError
        ``max |F|`` above ``config.residual_fail``.
    ValueError
        Bad ``block_table`` / backend / chunk combo.

    See Also
    --------
    sequential_apply, ParaRNN, NewtonConfig

    Notes
    -----
    Packed fused diag ParaGRU stays in-kernel; head + ``cu_seqlens`` needs
    ``eager`` (``fused`` raises; ``auto`` remaps). LSTM/sLSTM fused +
    ``cu_seqlens`` falls back to Triton scan. ``chunk_len`` windows forward
    and reverse. ``recompute=True`` rematerializes ``H*`` in backward (Level 2).
    """
    config = config or NewtonConfig()
    _validate_config(config)
    # RWKV-7 Goose is linear in S: no nonlinear fixed point. Redirect before
    # the IFT Autograd.Function so grads are ordinary BPTT through the scan.
    if getattr(cell, "jac_structure", None) == "rwkv7":
        if extra is not None:
            raise TypeError("ParaRWKV7 newton_apply does not accept extra=")
        return _newton_rwkv7(
            cell, x, config, h0=h0, stats=stats, cu_seqlens=cu_seqlens, block_table=block_table
        )
    # max_iters=None → measured K*(T) envelope (or newton_iters_by_t pin table).
    from pararnn.solvers.newton.k_star import resolve_max_iters

    config = resolve_max_iters(cell, x, config)
    assert config.max_iters is not None  # resolved: int pin or auto K*(T)
    config = _resolve_backend(cell, x, config, cu_seqlens=cu_seqlens)
    if block_table is not None:
        if config.chunk_len is not None:
            raise ValueError("block_table cannot be combined with chunk_len")
        if config.scan_backend != "fused":
            raise ValueError("block_table newton needs scan_backend='fused' (or auto on CUDA)")
    cs = None
    if cu_seqlens is not None:
        if config.scan_backend == "fused" and not isinstance(cell, ParaGRU):
            if block_table is not None:
                raise ValueError("block_table packed fused is ParaGRU only")
            config = replace(config, scan_backend="triton")
        cs = _prepare_ragged(x, h0, config, cu_seqlens, block_table=block_table)
    params = tuple(cell.parameters())
    has_h0 = h0 is not None
    has_extra = extra is not None
    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad
        or (has_h0 and h0.requires_grad)
        or (has_extra and extra.requires_grad)
        or any(p.requires_grad for p in params)
    )
    if block_table is not None and needs_grad:
        raise RuntimeError("block_table newton is inference-only")
    if needs_grad and config.scan_backend in (
        "fused_context_parallel",
        "fused_context_parallel_monoid",
    ):
        raise RuntimeError(
            f"{config.scan_backend} is forward-only for now "
            "(pipeline / monoid CP boundary); run under torch.no_grad()"
        )
    with _newton_precision_region(x.device):
        if not needs_grad:
            return _newton_forward(
                cell,
                x,
                config,
                h0=h0,
                extra=extra,
                stats=stats,
                cu_seqlens=cs,
                block_table=block_table,
            )

        h0_leaf = h0 if has_h0 else x.new_zeros(())
        extra_leaf = extra if has_extra else x.new_empty(0)
        _fwd_pack.clear()
        _fwd_pack.update(
            cell=cell,
            config=config,
            cu_seqlens=cs,
            stats=stats,
            has_h0=has_h0,
            has_extra=has_extra,
            block_table=block_table,
            extra=extra if has_extra else None,
        )
        try:
            return _NewtonFixedPoint.apply(x, h0_leaf, extra_leaf, *params)
        finally:
            _fwd_pack.clear()


def _prepare_ragged(
    x: Tensor,
    h0: Tensor | None,
    config: NewtonConfig,
    cu_seqlens: Tensor,
    *,
    block_table: Tensor | None = None,
) -> Tensor:
    if x.shape[0] != 1:
        raise ValueError(f"cu_seqlens packs x with batch=1, got batch={x.shape[0]}")
    if config.chunk_len is not None:
        raise ValueError("chunk_len cannot be combined with cu_seqlens")
    if config.fused_time_loop:
        raise ValueError("fused_time_loop cannot be combined with cu_seqlens")
    cs = validate_cu_seqlens(cu_seqlens, x.shape[1])
    n_seq = int(cs.numel()) - 1
    if h0 is not None and block_table is None and h0.shape[0] != n_seq:
        raise ValueError(f"h0 batch {h0.shape[0]} != n_seq {n_seq}")
    if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
        log.debug(
            "newton_ragged",
            extra={
                "n_seq": n_seq,
                "packed_len": int(x.shape[1]),
                "scan_backend": config.scan_backend,
                "requested_backend": config.scan_backend,
            },
        )
    return cs


def _newton_forward(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    extra: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    if config.chunk_len is not None and config.scan_backend == "context_parallel_sharded":
        raise ValueError(
            "chunk_len cannot be combined with scan_backend='context_parallel_sharded'"
        )
    if config.chunk_len is not None:
        return _newton_chunked(cell, x, config, h0=h0, extra=extra, stats=stats)
    if config.picard_adapt and isinstance(cell, ParaSLSTM) and not torch.compiler.is_compiling():
        return _newton_forward_picard_adapt(
            cell,
            x,
            config,
            h0=h0,
            extra=extra,
            stats=stats,
            cu_seqlens=cu_seqlens,
            block_table=block_table,
        )
    return _newton_solve(
        cell,
        x,
        config,
        h0=h0,
        extra=extra,
        stats=stats,
        cu_seqlens=cu_seqlens,
        block_table=block_table,
    )


def _newton_solve(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    extra: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    if config.scan_backend == "fused_context_parallel":
        return _newton_fused_context_parallel(
            cell, x, config, h0=h0, extra=extra, stats=stats, cu_seqlens=cu_seqlens
        )
    if config.scan_backend == "fused_context_parallel_monoid":
        return _newton_fused_context_parallel_monoid(
            cell, x, config, h0=h0, extra=extra, stats=stats, cu_seqlens=cu_seqlens
        )
    if getattr(cell, "jac_structure", None) == "m2rnn":
        if extra is not None:
            raise TypeError("ParaM2RNN factorized Newton does not accept extra=")
        return _newton_m2rnn(
            cell, x, config, h0=h0, stats=stats, cu_seqlens=cu_seqlens, block_table=block_table
        )
    if config.chunk_len is not None and config.scan_backend == "context_parallel_sharded":
        raise ValueError(
            "chunk_len cannot be combined with scan_backend='context_parallel_sharded'"
        )
    if config.chunk_len is not None:
        return _newton_chunked(cell, x, config, h0=h0, extra=extra, stats=stats)
    if config.fused_time_loop and config.scan_backend != "fused":
        raise ValueError(
            "fused_time_loop requires scan_backend='fused' (or auto on CUDA ParaSLSTM)"
        )
    compiling = torch.compiler.is_compiling()
    if config.scan_backend == "fused":
        use_early = bool(config.fused_early_exit) and not compiling
        if use_early:
            states, iters_done, history, last_res, wx_ee = _newton_fused_early_exit(
                cell,
                x,
                config,
                h0=h0,
                extra=extra,
                cu_seqlens=cu_seqlens,
                block_table=block_table,
            )
            if last_res is None and (stats is not None or config.residual_fail is not None):
                last_res = _fused_known_residual(
                    cell,
                    x,
                    states,
                    wx_ee,
                    h0=h0,
                    cu_seqlens=cu_seqlens,
                    block_table=block_table,
                )
            _fill_stats(
                cell,
                x,
                states,
                h0,
                config,
                iters=iters_done,
                stats=stats,
                residual_history=history,
                known_residual=last_res,
                cu_seqlens=cu_seqlens,
                block_table=block_table,
                wx=wx_ee,
            )
            return states
        residual_buf: list[float | Tensor] = []
        need_fill = not compiling and (stats is not None or config.residual_fail is not None)
        from pararnn.cells.para_cfc import ParaCfC

        want_res = need_fill and (
            isinstance(cell, ParaCfC)
            or (isinstance(cell, ParaSLSTM) and getattr(cell, "mix", None) == "diag")
        )
        states, wx_fused = _newton_fused(
            cell,
            x,
            config,
            h0=h0,
            extra=extra,
            cu_seqlens=cu_seqlens,
            block_table=block_table,
            residual_max_out=residual_buf if want_res else None,
        )
        if need_fill:
            known = residual_buf[0] if residual_buf else _fused_known_residual(
                cell,
                x,
                states,
                wx_fused,
                h0=h0,
                cu_seqlens=cu_seqlens,
                block_table=block_table,
            )
            _fill_stats(
                cell,
                x,
                states,
                h0,
                config,
                iters=config.max_iters,
                stats=stats,
                known_residual=known,
                cu_seqlens=cu_seqlens,
                block_table=block_table,
                wx=wx_fused,
            )
        return states

    states, iters_done, history, last_res, residual_is_current = _newton_forward_pure(
        cell,
        x,
        config,
        h0=h0,
        extra=extra,
        cu_seqlens=cu_seqlens,
        early_stop=bool(config.residual_atol is not None and not compiling),
    )
    if not compiling:
        _fill_stats(
            cell,
            x,
            states,
            h0,
            config,
            iters=iters_done,
            stats=stats,
            residual_history=history,
            known_residual=last_res if residual_is_current else None,
            cu_seqlens=cu_seqlens,
        )
    return states


def _newton_forward_pure(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    extra: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    early_stop: bool = False,
) -> tuple[Tensor, int, list[float], float, bool]:
    """Tensor Newton loop. No logging, no ``NewtonStats``, no fused path.

    ``early_stop=False`` (compile / ``residual_atol=None``): fixed ``max_iters``.
    ``early_stop=True``: host ``float(amax)`` + break — eager debug only.
    """
    if config.scan_backend == "context_parallel_sharded" and cu_seqlens is not None:
        raise ValueError(
            "scan_backend='context_parallel_sharded' cannot combine with cu_seqlens"
        )

    wx = _merge_extra(_wx_if_analytic(cell, x, config.jacobian), extra)
    if isinstance(cell, ParaSLSTM):
        states = _slstm_newton_guess(cell, x, config, h0=h0, wx=wx, cu_seqlens=cu_seqlens)
    else:
        h_prev0 = _init_h_prev(cell, x, h0, cu_seqlens=cu_seqlens)
        states, _ = step_and_jacobian(
            cell,
            h_prev0,
            x,
            wx=wx,
            jacobian=config.jacobian,
            jac_structure=config.jac_structure,
        )

    native = cell
    h0_loop = h0
    if config.coords == "log":
        eps = native.eps
        states = slstm_encode_log(states, eps=eps)
        if h0 is None:
            n_h0 = int(cu_seqlens.numel()) - 1 if cu_seqlens is not None else x.shape[0]
            h0_loop = slstm_encode_log(
                x.new_zeros(n_h0, native.state_slots, native.d_h),
                eps=eps,
            )
        else:
            h0_loop = slstm_encode_log(h0, eps=eps)
        cell = SLSTMLogCoords(native)

    structure = config.jac_structure or getattr(cell, "jac_structure", None)
    iters_done = 0
    last_res = float("nan")
    history: list[float] = []
    residual_is_current = False
    atol = config.residual_atol if early_stop else None
    sharded = config.scan_backend == "context_parallel_sharded"
    cp_group = None
    cp_rank = 0
    if sharded:
        import torch.distributed as dist

        from pararnn.solvers.seq_parallel import _exchange_left_boundary, _prepend_sharded

        if dist.is_initialized():
            cp_rank = dist.get_rank()
    for it in range(config.max_iters):
        if sharded:
            left = _exchange_left_boundary(states, h0_loop, cp_group, cp_rank)
            h_prev = _prepend_sharded(states, h0_loop, left, cp_rank)
        else:
            h_prev = _prepend(states, h0_loop, cu_seqlens)
        pred, jac = step_and_jacobian(
            cell,
            h_prev,
            x,
            wx=wx,
            jacobian=config.jacobian,
            jac_structure=config.jac_structure,
        )
        residual = pred - states
        iters_done = it + 1
        if atol is not None:
            last_res = float(residual.detach().abs().amax())
            if sharded:
                last_res = _all_reduce_max_residual(last_res, residual.device)
            history.append(last_res)
            residual_is_current = True
            if last_res < atol:
                break
        delta = _scan(
            jac, residual, backend=config.scan_backend, structure=structure, cu_seqlens=cu_seqlens
        )
        if states.dtype in (torch.float16, torch.bfloat16):
            states = (states.float() + config.omega * delta.float()).to(states.dtype)
        else:
            states = states + config.omega * delta
        if config.coords == "log":
            states = slstm_clamp_log_coords(states)
        residual_is_current = False
    if config.coords == "log":
        states = slstm_decode_log(states, eps=native.eps)
        residual_is_current = False
    if not early_stop:
        iters_done = config.max_iters
    return states, iters_done, history, last_res, residual_is_current


def _all_reduce_max_residual(res: float, device: torch.device) -> float:
    """Rank-symmetric ``max |F|`` so early-exit collectives stay matched."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return res
    t = torch.tensor([res], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _slstm_chunk_pack_ok(cell: nn.Module, x: Tensor, config: NewtonConfig, length: int) -> bool:
    """Whether ``chunk_len`` can pack N windows into one fused time-loop walk."""
    if not isinstance(cell, ParaSLSTM) or getattr(cell, "mix", None) != "diag":
        return False
    if length not in FUSED_WINDOW_LENS:
        return False
    if not x.is_cuda:
        return False
    if config.scan_backend not in ("auto", "fused"):
        return False
    if config.coords == "log":
        return False
    if config.fused_early_exit:
        return False
    return True


def _slstm_eq26_pack_ok(
    cell: nn.Module, x: Tensor, length: int, backend: str
) -> bool:
    """Whether eq. 2.6 can reverse a chunk pack in one ``slstm_eq26_window_fused``."""
    if not isinstance(cell, ParaSLSTM) or getattr(cell, "mix", None) != "diag":
        return False
    if length not in FUSED_WINDOW_LENS:
        return False
    if not x.is_cuda:
        return False
    if backend not in ("auto", "triton", "fused"):
        return False
    return True


def _newton_chunked(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None,
    extra: Tensor | None = None,
    stats: NewtonStats | None,
) -> Tensor:
    """Newton on windows of ``chunk_len``; last state of a chunk is the next ``h0``.

    Span is linear in the number of chunks. Each window is the usual Alg. 1
    (eager / triton / fused). ``chunk_len=64`` from T=64 K=3 at ``d_h=256``.

    Peak discipline: one full ``S*`` buffer written in place (no ``cat`` of
    per-window clones); ``W_x`` GEMM only on the live window; fail-rail
    residual is ``max`` over window residuals (no second full-T ``wx``).

    Diag fused CUDA: packs device-recipe ``pack_tiles`` windows into one
    ``fused_time_loop`` walk (streamed ``W_x`` of size ``pack·C``) so host
    launch count drops from ``T/C`` to ``ceil(T/(pack·C))``.
    """
    length = int(config.chunk_len)
    batch, time, _ = x.shape
    states = _zero_state_like_input(cell, x)
    carry = h0
    iters_done = 0
    max_res = float("-inf")
    have_res = False

    if _slstm_chunk_pack_ok(cell, x, config, length):
        pack_tiles = _slstm_pack_tiles(x.device, length)
        pack_len = pack_tiles * length
        inner = replace(
            config,
            chunk_len=None,
            fused_time_loop=True,
            fused_window_len=length,
        )
        if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "newton_chunked_slstm_pack",
                extra={
                    "seq_len": time,
                    "batch": batch,
                    "chunk_len": length,
                    "pack_tiles": pack_tiles,
                    "pack_len": pack_len,
                    "n_packs": (time + pack_len - 1) // pack_len,
                },
            )
        for t0 in range(0, time, pack_len):
            t1 = min(t0 + pack_len, time)
            chunk_stats = (
                NewtonStats() if (stats is not None or config.residual_fail is not None) else None
            )
            piece = _newton_forward(
                cell,
                x[:, t0:t1],
                inner,
                h0=carry,
                extra=None if extra is None else extra[:, t0:t1],
                stats=chunk_stats,
            )
            states[:, t0:t1].copy_(piece)
            carry = states[:, t1 - 1]
            del piece
            if chunk_stats is not None:
                iters_done += chunk_stats.iters
                r = float(chunk_stats.max_residual)
                if math.isfinite(r):
                    max_res = max(max_res, r)
                    have_res = True
    else:
        inner = replace(config, chunk_len=None)
        for t0 in range(0, time, length):
            t1 = min(t0 + length, time)
            chunk_stats = (
                NewtonStats() if (stats is not None or config.residual_fail is not None) else None
            )
            piece = _newton_forward(
                cell,
                x[:, t0:t1],
                inner,
                h0=carry,
                extra=None if extra is None else extra[:, t0:t1],
                stats=chunk_stats,
            )
            states[:, t0:t1].copy_(piece)
            carry = states[:, t1 - 1]
            del piece
            if chunk_stats is not None:
                iters_done += chunk_stats.iters
                r = float(chunk_stats.max_residual)
                if math.isfinite(r):
                    max_res = max(max_res, r)
                    have_res = True

    known = max_res if have_res else None
    _fill_stats(
        cell,
        x,
        states,
        h0,
        config,
        iters=iters_done,
        stats=stats,
        known_residual=known,
    )
    return states


def _fill_stats(
    cell: nn.Module,
    x: Tensor,
    states: Tensor,
    h0: Tensor | None,
    config: NewtonConfig,
    *,
    iters: int,
    stats: NewtonStats | None,
    residual_history: list[float] | tuple[float, ...] = (),
    known_residual: float | Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    wx: Tensor | None = None,
) -> None:
    if torch.compiler.is_compiling():
        return
    if stats is None and config.residual_fail is None:
        return
    h0_use = h0
    if block_table is not None and h0 is not None:
        h0_use = h0.index_select(0, block_table.to(device=h0.device, dtype=torch.long))
    if known_residual is None:
        # Default residual_fail re-evals F(H*) once. Fused ParaCfC / diag
        # ParaSLSTM prefer Triton ``*_residual_amax`` via ``known_residual``.
        # Other cells reuse the fused GEMM when the caller passes ``wx``.
        if config.scan_backend == "context_parallel_sharded":
            import torch.distributed as dist

            from pararnn.solvers.seq_parallel import _exchange_left_boundary, _prepend_sharded

            rank = dist.get_rank() if dist.is_initialized() else 0
            left = _exchange_left_boundary(states, h0_use, None, rank)
            h_prev = _prepend_sharded(states, h0_use, left, rank)
        else:
            h_prev = _prepend(states, h0_use, cu_seqlens)
        if wx is not None:
            pred = cell.step(h_prev, x, wx=wx)
        else:
            pred = cell.step(h_prev, x)
        res = float((pred - states).detach().abs().amax())
        if config.scan_backend == "context_parallel_sharded":
            res = _all_reduce_max_residual(res, states.device)
    elif isinstance(known_residual, Tensor):
        # 0-d (or reduce) CUDA amax — single D2H at the fail/log boundary.
        res = float(known_residual.detach().float().amax().item())
        if config.scan_backend == "context_parallel_sharded":
            res = _all_reduce_max_residual(res, states.device)
    else:
        res = float(known_residual)
        if config.scan_backend == "context_parallel_sharded":
            res = _all_reduce_max_residual(res, states.device)
    hist = tuple(residual_history)
    if stats is not None:
        stats.max_residual = res
        stats.iters = iters
        stats.scan_backend = config.scan_backend
        stats.picard_iters = int(config.picard_iters or 0)
        stats.residual_history = hist
    cap = config.residual_fail
    extra = {
        "max_residual": res,
        "residual_fail": cap,
        "iters": iters,
        "seq_len": int(x.shape[1]),
        "batch": int(x.shape[0]),
        "cell": type(cell).__name__,
        "picard_iters": int(config.picard_iters or 0),
        "scan_backend": config.scan_backend,
        "residual_history": hist[-8:],
    }
    diverged = not math.isfinite(res) or (cap is not None and res > cap)
    if diverged and cap is not None:
        log.error("newton_diverged", extra=extra)
        raise NewtonDivergenceError(
            f"Newton residual {res:.3e} after {iters} iters exceeds residual_fail="
            f"{cap:g} (seq_len={x.shape[1]}, picard={int(config.picard_iters or 0)}, "
            f"history={hist[-8:]!r}). For ParaSLSTM raise picard_iters."
        )
    if res > _RESIDUAL_WARN and log.isEnabledFor(logging.WARNING):
        log.warning("newton_residual_high", extra=extra)


def _newton_fused_context_parallel(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    extra: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Paper fused windowed walk under time sharding (pipeline left boundary).

    Each rank runs ``fused_time_loop`` on its local ``T/world`` tile. Rank 0
    uses ``h0``; rank ``r>0`` waits for rank ``r-1``'s last solved state, then
    walks. Wall-clock ≈ single-GPU fused on full ``T`` (serial dependency);
    peak activation scales as ``O(T/world)``. Same residual class as
    ``scan_backend='fused'`` + ``fused_time_loop``.
    """
    import torch.distributed as dist

    if cu_seqlens is not None:
        raise ValueError("fused_context_parallel cannot combine with cu_seqlens")
    if not isinstance(cell, ParaSLSTM) or getattr(cell, "mix", None) != "diag":
        raise TypeError(
            "fused_context_parallel is ParaSLSTM(mix='diag') only "
            f"(got {type(cell).__name__} mix={getattr(cell, 'mix', None)!r})"
        )

    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    window = config.fused_window_len
    if window is None:
        window = FUSED_WINDOW_DEFAULT
    inner = replace(
        config,
        scan_backend="fused",
        fused_time_loop=True,
        fused_window_len=int(window),
        picard_adapt=False,
    )

    left = h0
    if world > 1:
        slots = int(getattr(cell, "state_slots", 4))
        if rank > 0:
            if h0 is not None:
                buf = torch.empty_like(h0)
            else:
                buf = x.new_zeros((x.shape[0], slots, cell.d_h))
            dist.recv(buf, src=rank - 1)
            left = buf

    residual_buf: list[float | Tensor] = []
    need_fill = not torch.compiler.is_compiling() and (
        stats is not None or config.residual_fail is not None
    )
    states, wx_fused = _newton_fused(
        cell,
        x,
        inner,
        h0=left,
        extra=extra,
        residual_max_out=residual_buf if need_fill else None,
    )

    if world > 1:
        last = states[:, -1].contiguous()
        if rank < world - 1:
            dist.send(last, dst=rank + 1)
        dist.barrier()

    if need_fill:
        known = residual_buf[0] if residual_buf else _fused_known_residual(
            cell, x, states, wx_fused, h0=left
        )
        if world > 1 and dist.is_initialized():
            if isinstance(known, Tensor):
                kt = known.detach().float().reshape(())
            else:
                kt = x.new_tensor(float(known))
            dist.all_reduce(kt, op=dist.ReduceOp.MAX)
            known = kt
        _fill_stats(
            cell,
            x,
            states,
            left,
            config,
            iters=int(config.max_iters or 0),
            stats=stats,
            known_residual=known,
            wx=wx_fused,
        )
    return states


def _newton_fused_context_parallel_monoid(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    extra: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Two-pass fused CP: concurrent local Newton + monoid AllGather.

    Per Newton iter (all ranks from t=0):
      1. Exchange left state boundary (nonlinear ``h_{t-1}``).
      2. Triton cell+J+local scan → tile monoid ``(P_end, δ_end)``.
      3. AllGather monoids; exclusive prefix → rank carry.
      4. Apply ``δ = P_within @ carry + u`` and ω-update.

    Peak ∝ ``T/world``; wall ≈ fused cost on ``T/world``. Forward-only.
    Diag ParaSLSTM only. Warm-start is boundary-chained windowed Picard on
    long ``T`` (``P=2`` × ``world`` rounds with left-boundary exchange) or
    2-stage monoid on short ``T``; Newton stays at requested ``max_iters``
    (``K=3``).
    """
    import torch.distributed as dist

    from pararnn.kernels.newton_slstm import (
        slstm_fused_cp_downsweep,
        slstm_fused_cp_upsweep,
    )
    from pararnn.solvers.seq_parallel import (
        _all_gather_monoid,
        _apply_block4_carry_exclusive,
        _exchange_left_boundary,
    )

    if cu_seqlens is not None:
        raise ValueError("fused_context_parallel_monoid cannot combine with cu_seqlens")
    if not isinstance(cell, ParaSLSTM) or getattr(cell, "mix", None) != "diag":
        raise TypeError(
            "fused_context_parallel_monoid is ParaSLSTM(mix='diag') only "
            f"(got {type(cell).__name__} mix={getattr(cell, 'mix', None)!r})"
        )
    if not x.is_cuda:
        raise TypeError("fused_context_parallel_monoid needs CUDA")

    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    wx = _merge_extra(_input_affine(cell, x), extra)
    if wx is None:
        raise TypeError("fused_context_parallel_monoid needs cell.W_x")
    r_w = cell.clipped_r()
    eps = float(cell.eps)
    omega = float(config.omega)
    max_iters = int(config.max_iters or 0)
    scan_tile = str(config.scan_tile)
    batch = int(x.shape[0])
    slots = int(getattr(cell, "state_slots", 4))
    d_h = int(cell.d_h)

    states = _slstm_newton_guess(cell, x, config, h0=h0, wx=wx)
    for _it in range(max_iters):
        left = _exchange_left_boundary(states, h0, None, rank)
        pack = slstm_fused_cp_upsweep(
            states, wx, r_w, left, eps=eps, scan_tile=scan_tile
        )
        if world == 1:
            carry = None
        else:
            p_list, d_list = _all_gather_monoid(
                pack.p_end, pack.d_end, world, None, needs_grad=False
            )
            carry = _apply_block4_carry_exclusive(
                p_list,
                d_list,
                rank,
                world,
                batch,
                d_h,
                states.device,
                states.dtype,
                needs_grad=False,
            )
        states = slstm_fused_cp_downsweep(states, pack, carry, omega=omega)

    if not torch.compiler.is_compiling() and (
        stats is not None or config.residual_fail is not None
    ):
        left = _exchange_left_boundary(states, h0, None, rank)
        # Residual via one cell.step on the boundary-prepended tile (stats only).
        from pararnn.solvers.seq_parallel import _prepend_sharded

        h_prev = _prepend_sharded(states, h0, left, rank)
        pred = cell.step(h_prev, x, wx=wx)
        known = (pred - states).detach().abs().amax()
        if world > 1 and dist.is_initialized():
            kt = known.detach().float().reshape(())
            dist.all_reduce(kt, op=dist.ReduceOp.MAX)
            known = kt
        _fill_stats(
            cell,
            x,
            states,
            h0,
            config,
            iters=max_iters,
            stats=stats,
            known_residual=known,
            wx=wx,
        )
    return states


def _newton_fused(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    extra: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    residual_max_out: list[float | Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Alg. 1 with cell+J+scan in Triton. ``W_x(x)`` is still one PyTorch GEMM.

    Fused kernels are ``pararnn::newton_*_fused`` custom ops (``register_fake``)
    so Dynamo does not trace into Triton. Eq. 2.6 stays on ``_NewtonFixedPoint``.

    ``residual_max_out``: ParaCfC / diag ParaSLSTM — append final ``max |F(H*)|``
    (0-d tensor for SLSTM) so the fail rail / ``NewtonStats`` skip a host
    ``cell.step``.

    Returns
    -------
    states, wx
        Solved trajectory and the input affine reused by ``_fill_stats``.
    """
    wx = _merge_extra(_input_affine(cell, x), extra)
    if wx is None:
        raise TypeError(f"fused Newton needs cell.W_x; got {type(cell).__name__}")
    window = config.fused_window_len
    if config.fused_time_loop and window is None:
        window = FUSED_WINDOW_DEFAULT
    if not torch.compiler.is_compiling():
        log.debug(
            "newton_fused",
            extra={
                "cell": type(cell).__name__,
                "seq_len": x.shape[1],
                "batch": x.shape[0],
                "d_h": cell.d_h,
                "max_iters": config.max_iters,
                "device": str(x.device),
                "h0": h0 is not None,
                "picard_iters": int(config.picard_iters or 0),
                "scan_tile": config.scan_tile,
                "fused_time_loop": config.fused_time_loop,
                "fused_window_len": window,
                "block_table": block_table is not None,
                "fused_early_exit": config.fused_early_exit,
                "residual_max_out": residual_max_out is not None,
            },
        )
    from pararnn.kernels.fused_newton import fused_newton

    states = fused_newton(
        cell,
        wx,
        max_iters=config.max_iters,
        omega=config.omega,
        h0=h0,
        log_coords=config.coords == "log",
        picard_iters=int(config.picard_iters or 0),
        scan_tile=config.scan_tile,
        fused_time_loop=config.fused_time_loop,
        fused_window_len=window,
        cu_seqlens=cu_seqlens,
        block_table=block_table,
        dt=_cfc_dt(cell, x),
        residual_max_out=residual_max_out,
    )
    return states, wx


def _newton_fused_early_exit(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    extra: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> tuple[Tensor, int, list[float], float | None, Tensor]:
    """Experimental fused path: GPU ``max|r_loc|`` after each cell kernel.

    Stops before the scan/ω update when the Newton residual is already below
    ``residual_atol``. One D2H sync per checked iter (Python ``break``);
    no second ``cell.step``. Variable ``iters`` — train/DDP/compile should
    keep ``fused_early_exit=False``.
    """
    wx = _merge_extra(_input_affine(cell, x), extra)
    if wx is None:
        raise TypeError(f"fused Newton needs cell.W_x; got {type(cell).__name__}")
    atol = config.residual_atol
    if atol is None:
        raise ValueError("fused_early_exit requires residual_atol")
    history: list[float] = []
    iters_done_out: list[int] = []
    from pararnn.kernels.fused_newton import fused_newton

    window = config.fused_window_len
    if config.fused_time_loop and window is None:
        window = FUSED_WINDOW_DEFAULT
    log.info(
        "newton_fused_early_exit",
        extra={
            "cell": type(cell).__name__,
            "seq_len": int(x.shape[1]),
            "batch": int(x.shape[0]),
            "max_iters": config.max_iters,
            "residual_atol": float(atol),
        },
    )
    states = fused_newton(
        cell,
        wx,
        max_iters=config.max_iters,
        omega=config.omega,
        h0=h0,
        log_coords=config.coords == "log",
        picard_iters=int(config.picard_iters or 0),
        scan_tile=config.scan_tile,
        fused_time_loop=config.fused_time_loop,
        fused_window_len=window,
        cu_seqlens=cu_seqlens,
        block_table=block_table,
        early_exit_atol=float(atol),
        residual_history=history,
        iters_done_out=iters_done_out,
        dt=_cfc_dt(cell, x),
    )
    iters_done = int(iters_done_out[0]) if iters_done_out else int(config.max_iters)
    # Only trust history residual when we stopped early (it is max|F| of H*).
    last_res = history[-1] if history and iters_done < int(config.max_iters) else None
    return states, iters_done, history, last_res, wx


def _fused_known_residual(
    cell: nn.Module,
    x: Tensor,
    states: Tensor,
    wx: Tensor | None,
    *,
    h0: Tensor | None,
    cu_seqlens: Tensor | None,
    block_table: Tensor | None,
) -> float | Tensor | None:
    """Cheap fused ``max |F(H*)|`` for ParaCfC / diag ParaSLSTM fail rails.

    Skips the post-solve ``cell.step`` in ``_fill_stats`` when the fail rail
    or ``NewtonStats`` needs a number. Other cells return ``None`` (eager
    residual path keeps reusing ``wx`` when provided). CfC / SLSTM amax
    helpers return a 0-d CUDA tensor; ``_fill_stats`` syncs once via ``.item()``.
    """
    from pararnn.cells.para_cfc import ParaCfC

    if wx is None:
        return None
    if isinstance(cell, ParaCfC):
        from pararnn.kernels.newton_cfc import cfc_residual_amax

        if getattr(cell, "gate_mix", "input") != "input":
            return None
        dt = _cfc_dt(cell, x)
        if dt is None:
            return None
        return cfc_residual_amax(
            states,
            wx,
            cell.clipped_u(),
            dt,
            h0=h0,
            no_gate=getattr(cell, "update", "convex") == "no_gate",
            cu_seqlens=cu_seqlens,
            block_table=block_table,
        )
    if isinstance(cell, ParaSLSTM) and getattr(cell, "mix", None) == "diag":
        from pararnn.kernels.newton_slstm import slstm_residual_amax

        return slstm_residual_amax(
            states,
            wx,
            cell.clipped_r(),
            h0=h0,
            eps=float(cell.eps),
            block_table=block_table,
        )
    return None


def _cfc_fused_known_residual(
    cell: nn.Module,
    x: Tensor,
    states: Tensor,
    wx: Tensor | None,
    *,
    h0: Tensor | None,
    cu_seqlens: Tensor | None,
    block_table: Tensor | None,
) -> float | None:
    """Alias for ``_fused_known_residual`` (older CfC call sites)."""
    return _fused_known_residual(
        cell,
        x,
        states,
        wx,
        h0=h0,
        cu_seqlens=cu_seqlens,
        block_table=block_table,
    )


def _eq26_vjp(
    cell: nn.Module,
    states: Tensor,
    x: Tensor,
    partial: Tensor,
    *,
    backend: str = "eager",
    jacobian: str = "auto",
    jac_structure: str | None = None,
    h0: Tensor | None = None,
    extra: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
) -> tuple[Tensor | None, tuple[Tensor | None, ...], Tensor | None, Tensor | None]:
    """``∇_x L``, per-parameter grads, ``∇_{h0} L``, and ``∇_{wx} L`` (eq. 2.6).

    ``∇_{h0} L = J_0^T μ_0``. Reverse scan over ``H`` starts at t=0; ``J_0``
    is the h0 adjoint. Ragged: one adjoint per sequence start. ``∇_{wx} L``
    matches ``∂L/∂extra`` when the forward used ``pre = W_x(x) + extra``.
    """
    sharded = backend == "context_parallel_sharded"
    if sharded and cu_seqlens is not None:
        raise ValueError(
            "scan_backend='context_parallel_sharded' cannot combine with cu_seqlens"
        )
    if sharded:
        import torch.distributed as dist

        from pararnn.solvers.seq_parallel import _exchange_left_boundary, _prepend_sharded

        rank = dist.get_rank() if dist.is_initialized() else 0
        left = _exchange_left_boundary(states, h0, None, rank)
        h_prev = _prepend_sharded(states, h0, left, rank)
    else:
        h_prev = _prepend(states, h0, cu_seqlens)
    if getattr(cell, "jac_structure", None) == "m2rnn" and cu_seqlens is None:
        from pararnn.kernels.newton_m2rnn import m2rnn_t0_vjp, reverse_factor_scan_m2rnn

        wx = _merge_extra(_input_affine(cell, x), extra)
        if wx is None:
            raise TypeError("ParaM2RNN factorized VJP needs cell.W_x")
        with torch.no_grad():
            mu = reverse_factor_scan_m2rnn(cell, h_prev, partial, wx=wx)
            h0_vjp = m2rnn_t0_vjp(cell, h_prev, mu, wx=wx)
        packed = uses_packed_vjp(cell)
        grad_x, param_grads, g_wx = cell_vjp_wx(
            cell, h_prev, x, mu, packed=packed, wx=wx
        )
        grad_h0 = None if h0 is None else h0_vjp
        return grad_x, param_grads, grad_h0, g_wx

    # Factorized reverse for ParaGRU / ParaSLSTM head on CUDA (matches fused forward).
    if (
        isinstance(cell, ParaGRU)
        and cell.mix == "head"
        and backend in ("triton", "fused")
        and cu_seqlens is None
        and jacobian in ("auto", "analytic")
    ):
        wx = _merge_extra(_input_affine(cell, x), extra)
        if wx is None:
            raise TypeError("ParaGRU head factorized VJP needs cell.W_x")
        from pararnn.kernels.newton_gru_head import (
            gru_head_t0_vjp,
            reverse_factor_scan_gru_head,
        )

        with torch.no_grad():
            mu = reverse_factor_scan_gru_head(cell, h_prev, partial, wx=wx)
            h0_vjp = gru_head_t0_vjp(cell, h_prev, mu, wx=wx)
        packed = uses_packed_vjp(cell)
        grad_x, param_grads, g_wx = cell_vjp_wx(
            cell, h_prev, x, mu, packed=packed, wx=wx
        )
        grad_h0 = None if h0 is None else h0_vjp
        return grad_x, param_grads, grad_h0, g_wx

    if (
        isinstance(cell, ParaSLSTM)
        and cell.mix == "head"
        and backend in ("triton", "fused")
        and cu_seqlens is None
        and jacobian in ("auto", "analytic")
    ):
        wx = _merge_extra(_input_affine(cell, x), extra)
        if wx is None:
            raise TypeError("ParaSLSTM head factorized VJP needs cell.W_x")
        from pararnn.kernels.newton_slstm_head import (
            reverse_factor_scan_slstm_head,
            slstm_head_t0_vjp,
        )

        with torch.no_grad():
            mu = reverse_factor_scan_slstm_head(cell, h_prev, partial, wx=wx)
            h0_vjp = slstm_head_t0_vjp(cell, h_prev, mu, wx=wx)
        packed = uses_packed_vjp(cell)
        grad_x, param_grads, g_wx = cell_vjp_wx(
            cell, h_prev, x, mu, packed=packed, wx=wx
        )
        grad_h0 = None if h0 is None else h0_vjp
        return grad_x, param_grads, grad_h0, g_wx

    # ParaSLSTM mix=diag: Triton block4 J rematerialize (skips eager step_and_jacobian).
    if (
        isinstance(cell, ParaSLSTM)
        and cell.mix == "diag"
        and backend in ("triton", "fused")
        and jacobian in ("auto", "analytic")
        and x.is_cuda
        and cu_seqlens is None
    ):
        from pararnn.solvers.vjp import _align_grads, _clip_mask, _linear_vjp

        wx = _merge_extra(_input_affine(cell, x), extra)
        if wx is None:
            raise TypeError("ParaSLSTM fused reverse needs cell.W_x")
        time = int(x.shape[1])
        # Window lengths: remat J + reverse + packed VJP in one launch.
        if time in FUSED_WINDOW_LENS:
            from pararnn.kernels.vjp_slstm import slstm_eq26_window_fused

            with torch.no_grad():
                g_wx, g_r, h0_vjp = slstm_eq26_window_fused(
                    h_prev,
                    wx,
                    cell.clipped_r(),
                    partial,
                    float(cell.eps),
                )
            g_r = g_r * _clip_mask(cell.R, cell.max_recurrent_norm)
            grad_x, grad_w, grad_b = _linear_vjp(cell.W_x, x, g_wx)
            grad_x, param_grads = _align_grads(
                cell,
                grad_x,
                {
                    "R": g_r,
                    "W_x.weight": grad_w,
                    "W_x.bias": grad_b,
                },
            )
            grad_h0 = None if h0 is None else h0_vjp
            return grad_x, param_grads, grad_h0, g_wx
        from pararnn.kernels.newton_slstm import slstm_block4_jacobian

        with torch.no_grad():
            jac = slstm_block4_jacobian(
                h_prev,
                wx,
                cell.clipped_r(),
                eps=float(cell.eps),
            )
            mu = _reverse_scan(
                jac, partial, backend=backend, structure="block4", cu_seqlens=cu_seqlens
            )
        packed = uses_packed_vjp(cell)
        grad_x, param_grads, g_wx = cell_vjp_wx(
            cell, h_prev, x, mu, packed=packed, wx=wx
        )
        h0_vjp = _t0_state_vjp(jac, mu)
        grad_h0 = None if h0 is None else h0_vjp
        return grad_x, param_grads, grad_h0, g_wx

    # ParaCfC gate_mix=input: Triton J rematerialize (skips eager step_with_jacobian).
    from pararnn.cells.para_cfc import ParaCfC

    if (
        isinstance(cell, ParaCfC)
        and getattr(cell, "gate_mix", "input") == "input"
        and getattr(cell, "update", "convex") in ("convex", "no_gate")
        and backend in ("triton", "fused")
        and jacobian in ("auto", "analytic")
        and x.is_cuda
    ):
        from pararnn.cells.para_cfc import split_cfc_input
        from pararnn.kernels.vjp_cfc import cfc_diag_jacobian

        wx = _merge_extra(_input_affine(cell, x), extra)
        if wx is None:
            raise TypeError("ParaCfC fused reverse needs project_wx / W_xb")
        _feat, dt = split_cfc_input(x)
        no_gate = getattr(cell, "update", "convex") == "no_gate"
        with torch.no_grad():
            jac = cfc_diag_jacobian(
                h_prev,
                wx,
                dt.squeeze(-1),
                cell.clipped_u(),
                no_gate=no_gate,
            )
            mu = _reverse_scan(
                jac, partial, backend=backend, structure="diag", cu_seqlens=cu_seqlens
            )
        packed = uses_packed_vjp(cell)
        grad_x, param_grads, g_wx = cell_vjp_wx(
            cell, h_prev, x, mu, packed=packed, wx=wx
        )
        if cu_seqlens is None:
            h0_vjp = _t0_state_vjp(jac, mu)
        else:
            starts = validate_cu_seqlens(cu_seqlens, states.shape[1])[:-1]
            h0_vjp = _state_vjp_at_times(jac, mu, starts.to(device=jac.device))[0]
        grad_h0 = None if h0 is None else h0_vjp
        return grad_x, param_grads, grad_h0, g_wx

    wx = _merge_extra(_wx_if_analytic(cell, x, jacobian), extra)
    structure = jac_structure or getattr(cell, "jac_structure", None)
    with torch.no_grad():
        _, jac = step_and_jacobian(
            cell,
            h_prev,
            x,
            wx=wx,
            jacobian=jacobian,
            jac_structure=jac_structure,
        )
        mu = _reverse_scan(
            jac, partial, backend=backend, structure=structure, cu_seqlens=cu_seqlens
        )
    packed = uses_packed_vjp(cell)
    grad_x, param_grads, g_wx = cell_vjp_wx(
        cell, h_prev, x, mu, packed=packed, wx=wx
    )
    if cu_seqlens is None:
        h0_vjp = _t0_state_vjp(jac, mu)
        if sharded:
            import torch.distributed as dist

            # Only global t=0 (rank 0) contributes to ∇_{h0}; other ranks zero.
            if dist.is_initialized() and dist.get_rank() != 0 and h0_vjp is not None:
                h0_vjp = torch.zeros_like(h0_vjp)
    else:
        starts = validate_cu_seqlens(cu_seqlens, states.shape[1])[:-1]
        h0_vjp = _state_vjp_at_times(jac, mu, starts.to(device=jac.device))[0]
    grad_h0 = None if h0 is None else h0_vjp
    return grad_x, param_grads, grad_h0, g_wx


def _ift_window_len(config: NewtonConfig) -> int | None:
    """Tile length for windowed eq. 2.6, or ``None`` for a global reverse scan.

    ``chunk_len`` and ``fused_time_loop`` use the same window; both must
    reverse with carry ``∇_{h0}`` so peak ``J`` stays ``O(C)`` on long ``T``.
    """
    if config.chunk_len is not None:
        return int(config.chunk_len)
    if config.fused_time_loop:
        if config.fused_window_len is not None:
            return int(config.fused_window_len)
        return int(FUSED_WINDOW_DEFAULT)
    return None


def _eq26_vjp_chunked(
    cell: nn.Module,
    states: Tensor | None,
    x: Tensor,
    partial: Tensor,
    *,
    chunk_len: int,
    backend: str = "eager",
    jacobian: str = "auto",
    jac_structure: str | None = None,
    h0: Tensor | None = None,
    extra: Tensor | None = None,
    remat_config: NewtonConfig | None = None,
) -> tuple[Tensor | None, tuple[Tensor | None, ...], Tensor | None, Tensor | None]:
    """Eq. 2.6 on the same windows as ``_newton_chunked`` / ``fused_time_loop``.

    Reverse order: ``∇_{h0}`` of window ``i+1`` adds to ``∂L/∂S`` at the last
    step of window ``i`` (that step is the next window's ``h0``).

    ``remat_config`` : Level-2 windowed path — ``states`` is ``None``; each
    tile's ``H*`` is rematerialized under ``no_grad``. Diag fused CUDA packs
    device-recipe ``pack_tiles`` windows per remat walk and per eq. 2.6 reverse
    (peak ``O(pack·C)``; one ``slstm_eq26_states_fused`` per pack).
    """
    length = int(chunk_len)
    time = int(x.shape[1])
    n_params = len(tuple(cell.parameters()))
    acc_x: Tensor | None = None
    acc_extra: Tensor | None = None
    acc_params: list[Tensor | None] = [None] * n_params
    carry_h0 = None
    starts = list(range(0, time, length))

    newton_pack = remat_config is not None and _slstm_chunk_pack_ok(
        cell, x, remat_config, length
    )
    eq26_pack = _slstm_eq26_pack_ok(cell, x, length, backend)
    pack_tiles = _slstm_pack_tiles(x.device, length) if (newton_pack or eq26_pack) else 1
    pack_len = pack_tiles * length if (newton_pack or eq26_pack) else length
    pack_inner: NewtonConfig | None = None
    leaf_inner: NewtonConfig | None = None
    carries: list[Tensor | None] | None = None

    if remat_config is not None:
        if states is not None:
            raise ValueError("remat_config requires states=None")
        leaf_inner = replace(
            remat_config,
            chunk_len=None,
            fused_time_loop=False,
            recompute=False,
            residual_fail=None,
            picard_adapt=False,
        )
        if newton_pack:
            pack_inner = replace(
                remat_config,
                chunk_len=None,
                fused_time_loop=True,
                fused_window_len=length,
                recompute=False,
                residual_fail=None,
                picard_adapt=False,
            )
        # Remat seeds: pack path stores one carry per pack boundary
        # (``carries[p0 // pack_len]``). Per-window ends were ~T/C host
        # ``aten::copy_`` launches on the fbwd critical path.
        carries = [h0]
        with torch.no_grad():
            if newton_pack:
                assert pack_inner is not None
                for p0 in range(0, time, pack_len):
                    p1 = min(p0 + pack_len, time)
                    piece = _newton_forward(
                        cell,
                        x[:, p0:p1],
                        pack_inner,
                        h0=carries[-1],
                        extra=None if extra is None else extra[:, p0:p1],
                        stats=None,
                    )
                    if p1 < time:
                        carries.append(piece[:, -1].contiguous())
                    del piece
            else:
                assert leaf_inner is not None
                for t0 in starts:
                    t1 = min(t0 + length, time)
                    piece = _newton_forward(
                        cell,
                        x[:, t0:t1],
                        leaf_inner,
                        h0=carries[-1],
                        extra=None if extra is None else extra[:, t0:t1],
                        stats=None,
                    )
                    carries.append(piece[:, -1].contiguous())
                    del piece

    def _accumulate_window(
        t0: int,
        t1: int,
        piece: Tensor,
        h0_w: Tensor | None,
        *,
        free_piece: bool,
    ) -> None:
        nonlocal acc_x, acc_extra, carry_h0
        xw = x[:, t0:t1]
        extra_w = None if extra is None else extra[:, t0:t1]
        partial_w = partial[:, t0:t1]
        if carry_h0 is not None:
            partial_w = partial_w.clone()
            partial_w[:, -1] = partial_w[:, -1] + carry_h0
        piece_c = piece if piece.is_contiguous() else piece.contiguous()
        xw_c = xw if xw.is_contiguous() else xw.contiguous()
        partial_c = partial_w if partial_w.is_contiguous() else partial_w.contiguous()
        extra_c = None
        if extra_w is not None:
            extra_c = extra_w if extra_w.is_contiguous() else extra_w.contiguous()
        gx, pgrads, gh0, g_wx = _eq26_vjp(
            cell,
            piece_c,
            xw_c,
            partial_c,
            backend=backend,
            jacobian=jacobian,
            jac_structure=jac_structure,
            h0=h0_w,
            extra=extra_c,
        )
        carry_h0 = gh0
        if free_piece:
            del piece, piece_c
        if gx is not None:
            if acc_x is None:
                acc_x = x.new_zeros(x.shape)
            acc_x[:, t0:t1] = gx
        if g_wx is not None and extra is not None:
            if acc_extra is None:
                acc_extra = extra.new_zeros(extra.shape)
            acc_extra[:, t0:t1] = g_wx
        for i, g in enumerate(pgrads):
            if g is None:
                continue
            acc_params[i] = g if acc_params[i] is None else acc_params[i] + g

    def _accumulate_eq26_pack(
        p0: int,
        p1: int,
        piece: Tensor,
        h0_pack: Tensor | None,
        *,
        free_piece: bool,
    ) -> None:
        """One ``slstm_eq26_states_fused`` over ``[p0, p1)`` + one ``W_x`` VJP."""
        nonlocal acc_x, acc_extra, carry_h0
        from pararnn.kernels.vjp_slstm import slstm_eq26_states_fused
        from pararnn.solvers.vjp import _align_grads, _clip_mask, _linear_vjp

        xw = x[:, p0:p1]
        extra_w = None if extra is None else extra[:, p0:p1]
        partial_w = partial[:, p0:p1]
        wx = _merge_extra(_input_affine(cell, xw), extra_w)
        if wx is None:
            raise TypeError("ParaSLSTM fused eq. 2.6 pack needs cell.W_x")
        with torch.no_grad():
            # Zero-copy: H* + h0 strides + in-kernel μ-carry (no prepend / clone).
            g_wx, g_r, h0_vjp = slstm_eq26_states_fused(
                piece,
                h0_pack,
                wx,
                cell.clipped_r(),
                partial_w,
                float(cell.eps),
                mu_carry=carry_h0,
            )
        g_r = g_r * _clip_mask(cell.R, cell.max_recurrent_norm)
        gx, grad_w, grad_b = _linear_vjp(cell.W_x, xw, g_wx)
        gx, pgrads = _align_grads(
            cell,
            gx,
            {
                "R": g_r,
                "W_x.weight": grad_w,
                "W_x.bias": grad_b,
            },
        )
        carry_h0 = h0_vjp
        if free_piece:
            del piece
        if gx is not None:
            if acc_x is None:
                acc_x = x.new_zeros(x.shape)
            acc_x[:, p0:p1] = gx
        if g_wx is not None and extra is not None:
            if acc_extra is None:
                acc_extra = extra.new_zeros(extra.shape)
            acc_extra[:, p0:p1] = g_wx
        for i, g in enumerate(pgrads):
            if g is None:
                continue
            acc_params[i] = g if acc_params[i] is None else acc_params[i] + g

    if eq26_pack:
        if remat_config is not None:
            assert carries is not None
            for p0 in reversed(list(range(0, time, pack_len))):
                p1 = min(p0 + pack_len, time)
                if newton_pack:
                    h0_pack = carries[p0 // pack_len]
                else:
                    # Leaf remat seeds are still per-window.
                    h0_pack = carries[p0 // length]
                remat_cfg = pack_inner if newton_pack else leaf_inner
                assert remat_cfg is not None
                with torch.no_grad():
                    if newton_pack:
                        pack_states = _newton_forward(
                            cell,
                            x[:, p0:p1],
                            remat_cfg,
                            h0=h0_pack,
                            extra=None if extra is None else extra[:, p0:p1],
                            stats=None,
                        )
                    else:
                        # Remat window-by-window into a pack buffer.
                        pack_states = _zero_state_like_input(cell, x[:, p0:p1])
                        carry = h0_pack
                        for w0 in range(p0, p1, length):
                            w1 = min(w0 + length, p1)
                            piece = _newton_forward(
                                cell,
                                x[:, w0:w1],
                                remat_cfg,
                                h0=carry,
                                extra=None if extra is None else extra[:, w0:w1],
                                stats=None,
                            )
                            pack_states[:, w0 - p0 : w1 - p0] = piece
                            carry = piece[:, -1]
                            del piece
                _accumulate_eq26_pack(p0, p1, pack_states, h0_pack, free_piece=True)
        else:
            assert states is not None
            for p0 in reversed(list(range(0, time, pack_len))):
                p1 = min(p0 + pack_len, time)
                piece = states[:, p0:p1]
                h0_pack = h0 if p0 == 0 else states[:, p0 - 1]
                _accumulate_eq26_pack(p0, p1, piece, h0_pack, free_piece=False)
    elif newton_pack and remat_config is not None:
        assert carries is not None and pack_inner is not None
        for p0 in reversed(list(range(0, time, pack_len))):
            p1 = min(p0 + pack_len, time)
            win_starts = list(range(p0, p1, length))
            h0_pack = carries[p0 // pack_len]
            with torch.no_grad():
                pack_states = _newton_forward(
                    cell,
                    x[:, p0:p1],
                    pack_inner,
                    h0=h0_pack,
                    extra=None if extra is None else extra[:, p0:p1],
                    stats=None,
                )
            for t0 in reversed(win_starts):
                t1 = min(t0 + length, p1)
                piece = pack_states[:, t0 - p0 : t1 - p0]
                # Window h0 from rematerialized pack (pack-boundary seed only).
                h0_w = (
                    h0_pack
                    if t0 == p0
                    else pack_states[:, t0 - p0 - 1]
                )
                _accumulate_window(t0, t1, piece, h0_w, free_piece=False)
            del pack_states
    else:
        for idx, t0 in enumerate(reversed(starts)):
            t1 = min(t0 + length, time)
            if remat_config is not None:
                assert carries is not None and leaf_inner is not None
                win_i = len(starts) - 1 - idx
                h0_w = carries[win_i]
                with torch.no_grad():
                    piece = _newton_forward(
                        cell,
                        x[:, t0:t1],
                        leaf_inner,
                        h0=h0_w,
                        extra=None if extra is None else extra[:, t0:t1],
                        stats=None,
                    )
                _accumulate_window(t0, t1, piece, h0_w, free_piece=True)
            else:
                assert states is not None
                piece = states[:, t0:t1]
                h0_w = h0 if t0 == 0 else states[:, t0 - 1]
                _accumulate_window(t0, t1, piece, h0_w, free_piece=False)

    grad_h0 = None if h0 is None else carry_h0
    return acc_x, tuple(acc_params), grad_h0, acc_extra


def _init_h_prev(
    cell: nn.Module,
    x: Tensor,
    h0: Tensor | None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """App. A: ``H^0_t = f(h_{t-1}, x_t)`` in parallel; only starts see ``h0``."""
    h_prev0 = _zero_state_like_input(cell, x)
    if cu_seqlens is None:
        if h0 is not None:
            h_prev0[:, 0] = h0
        return h_prev0
    starts = validate_cu_seqlens(cu_seqlens, x.shape[1])[:-1]
    if h0 is not None:
        h_prev0[0, starts] = h0
    return h_prev0


def _prepend(states: Tensor, h0: Tensor | None, cu_seqlens: Tensor | None) -> Tensor:
    if cu_seqlens is None:
        return prepend_state(states, h0)
    return prepend_state_ragged(states, h0, cu_seqlens)


def _merge_extra(wx: Tensor | None, extra: Tensor | None) -> Tensor | None:
    """``pre = W_x(x) + extra``; ``extra is None`` keeps ``wx`` unchanged."""
    if extra is None:
        return wx
    if wx is None:
        return extra
    return wx + extra


def _wx_if_analytic(cell: nn.Module, x: Tensor, jacobian: str) -> Tensor | None:
    """Reuse ``W_x(x)`` only on the analytic-J path (eq. 3.1)."""
    mode = jacobian
    if mode == "auto":
        mode = "analytic" if hasattr(cell, "step_with_jacobian") else "autograd"
    if mode != "analytic":
        return None
    return _input_affine(cell, x)


def _input_affine(cell: nn.Module, x: Tensor) -> Tensor | None:
    """Input affine for Newton; ``project_wx`` when present (CfC: 3-wide pack)."""
    project = getattr(cell, "project_wx", None)
    if callable(project):
        return project(x)
    lin = getattr(cell, "W_x", None)
    if lin is None:
        return None
    return lin(x)


def _cfc_dt(cell: nn.Module, x: Tensor) -> Tensor | None:
    """Per-step Δt ``(B, T)`` for fused ParaCfC; ``None`` for other cells."""
    from pararnn.cells.para_cfc import ParaCfC, split_cfc_input

    if not isinstance(cell, ParaCfC):
        return None
    _feat, dt = split_cfc_input(x)
    return dt.squeeze(-1).contiguous()


def _zero_state_like_input(cell: nn.Module, x: Tensor) -> Tensor:
    """Zeros with a time axis, matching ``step``'s previous-state layout."""
    batch, time, _ = x.shape
    tail = getattr(cell, "state_shape", None)
    if tail is not None:
        return x.new_zeros(batch, time, *tuple(tail))
    d_h = cell.d_h
    slots = getattr(cell, "state_slots", None)
    if slots is None:
        slots = 2 if isinstance(cell, ParaLSTM) else 1
    if slots == 1:
        return x.new_zeros(batch, time, d_h)
    return x.new_zeros(batch, time, slots, d_h)


def _newton_m2rnn(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    """Factorized Kronecker Newton for ``ParaM2RNN`` (no dense ``(KV)²``)."""
    from pararnn.cells.para_m2rnn import ParaM2RNN
    from pararnn.kernels.newton_m2rnn import newton_m2rnn_factorized

    if not isinstance(cell, ParaM2RNN):
        raise TypeError(f"_newton_m2rnn needs ParaM2RNN, got {type(cell).__name__}")
    if cu_seqlens is not None or block_table is not None:
        raise ValueError(
            "ParaM2RNN factorized Newton is rectangular-batch only "
            "(no cu_seqlens / block_table yet)"
        )
    if config.chunk_len is not None:
        raise ValueError("ParaM2RNN does not support chunk_len yet")
    if config.coords == "log":
        raise TypeError("NewtonConfig(coords='log') is ParaSLSTM only")
    # picard_iters>=1 → frozen-W (W=0) warm-start; see m2rnn_frozen_w_scan.
    frozen_w = int(config.picard_iters or 0) > 0

    compiling = torch.compiler.is_compiling()
    history: list[float] = []
    force_eager = config.scan_backend == "eager"
    backend_tag = "fused" if config.scan_backend == "fused" else "factor_m2rnn"
    # Default residual_atol early-stops past K*; fixed over-provisioned K loses
    # to sequential on large KV (wall ≈ K_used × scan).
    states = newton_m2rnn_factorized(
        cell,
        x,
        max_iters=config.max_iters,
        omega=config.omega,
        h0=h0,
        residual_history=None if (compiling or not force_eager) else history,
        force_eager=force_eager,
        residual_atol=None if compiling else config.residual_atol,
        frozen_w_init=frozen_w,
        backtrack=bool(getattr(config, "newton_backtrack", False)),
    )
    if not compiling:
        cfg = replace(config, scan_backend=backend_tag)
        last = history[-1] if history else None
        _fill_stats(
            cell,
            x,
            states,
            h0,
            cfg,
            iters=config.max_iters,
            stats=stats,
            residual_history=history,
            known_residual=last,
        )
    return states


def _newton_rwkv7(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    """Linear RWKV-7 monoid: associative scan (default) or sequential oracle.

    The map is affine in ``S``. ``scan_backend='eager'`` uses
    ``sequential_apply``; other aliases use ``rwkv7_associative_scan``.
    """
    from pararnn.cells.para_rwkv7 import ParaRWKV7
    from pararnn.kernels.rwkv7_scan import (
        rwkv7_associative_scan,
        rwkv7_build_g,
        rwkv7_outer_vk,
    )
    from pararnn.solvers.sequential import sequential_apply

    if not isinstance(cell, ParaRWKV7):
        raise TypeError(f"_newton_rwkv7 needs ParaRWKV7, got {type(cell).__name__}")
    if cu_seqlens is not None or block_table is not None:
        raise ValueError(
            "ParaRWKV7 scan is rectangular-batch only (no cu_seqlens / block_table yet)"
        )
    if config.chunk_len is not None:
        raise ValueError("ParaRWKV7 does not support chunk_len yet")
    if config.coords == "log":
        raise TypeError("NewtonConfig(coords='log') is ParaSLSTM only")

    use_seq = config.scan_backend == "eager"
    if use_seq:
        states = sequential_apply(cell, x, h0)
        backend_tag = "rwkv7_sequential"
    else:
        w, a, kappa, v, k, _r = cell.project(x)
        g = rwkv7_build_g(w, a, kappa)
        u = rwkv7_outer_vk(v, k)
        states = rwkv7_associative_scan(g, u, s0=h0)
        backend_tag = "rwkv7_scan"

    if not torch.compiler.is_compiling():
        if log.isEnabledFor(logging.INFO):
            log.info(
                "rwkv7_linear_redirect",
                extra={
                    "backend": backend_tag,
                    "seq_len": int(x.shape[1]),
                    "batch": int(x.shape[0]),
                    "n_heads": cell.n_heads,
                    "d_head": cell.d_head,
                },
            )
        cfg = replace(config, scan_backend=backend_tag)
        _fill_stats(
            cell,
            x,
            states,
            h0,
            cfg,
            iters=0,
            stats=stats,
            residual_history=[],
            known_residual=0.0,
        )
    return states

"""Dispatch fused Newton for ParaGRU, ParaLSTM, and ParaSLSTM.

``mix='diag'`` cells use handwritten Triton Alg. 1 custom ops.
``ParaGRU(mix='head')`` / ``ParaSLSTM(mix='head')`` use factorized-``J``
custom ops (no dense ``d×d`` / ``(4d)×(4d)``). ``mix='dense'`` sLSTM stays
on the PyTorch cell + scan path. ``log_coords`` / Picard are ParaSLSTM.
``chunk_len`` stays a Python loop.

Default path is ``pararnn::newton_*_fused`` custom ops (fixed ``max_iters``).
``early_exit_atol`` bypasses the custom op and calls the impl directly
(experimental ``NewtonConfig.fused_early_exit``): GPU ``max|F|`` (per-tile
reduce on the recompute path, or ``r_loc`` where still materialized) plus
one D2H per checked iter.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pararnn.cells.para_cfc import ParaCfC
from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_nlru import ParaNLRU
from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.cells.para_titans import ParaTitans
from pararnn.kernels._compat import require_fused_triton


def fused_newton(
    cell: nn.Module,
    wx: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None,
    log_coords: bool = False,
    picard_iters: int = 0,
    scan_tile: str = "assoc",
    fused_time_loop: bool = False,
    fused_window_len: int | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    early_exit_atol: float | None = None,
    residual_history: list[float] | None = None,
    iters_done_out: list[int] | None = None,
    dt: Tensor | None = None,
    residual_max_out: list[float | Tensor] | None = None,
) -> Tensor:
    """Run fused Alg. 1 for supported cells (diag GRU/LSTM/sLSTM, head GRU).

    ``ParaGRU(mix='head')`` dispatches to ``pararnn::newton_gru_head_fused``.
    Packed ``cu_seqlens`` raises :exc:`TypeError` on the head path; pad to a
    rectangular batch or use ``scan_backend='eager'``.

    Parameters
    ----------
    cell : nn.Module
        ``ParaGRU``, ``ParaLSTM``, or ``ParaSLSTM(mix='diag')``.
    wx : Tensor
        Precomputed input projection for the cell (shape depends on cell).
    max_iters : int
        Newton steps ``K``.
    omega : float
        Step damping.
    h0 : Tensor or None
        Optional initial state.
    log_coords, picard_iters, scan_tile, fused_time_loop, fused_window_len
        ParaSLSTM knobs; invalid for head GRU.
    cu_seqlens : Tensor or None, default=None
        Packed-time offsets (diag GRU / supported paths only).
    block_table : Tensor or None, default=None
        Optional sequence remap (diag paths).
    early_exit_atol, residual_history, iters_done_out
        Experimental early-exit (diag only): GPU ``max|F|`` from ``r_loc``.
    residual_max_out
        When set on ParaCfC / diag ParaSLSTM, append final ``max |F(H*)|``
        (0-d CUDA tensor for SLSTM) and bypass the custom op (same path as
        early-exit).

    Returns
    -------
    H : Tensor
        Parallel Newton states matching the cell state layout.

    Raises
    ------
    TypeError
        When the cell / knobs combination is unsupported (including head
        GRU with ``cu_seqlens`` or ``fused_early_exit``).
    """
    require_fused_triton()
    early = early_exit_atol is not None
    if isinstance(cell, (ParaNLRU, ParaCfC, ParaTitans)):
        if log_coords:
            raise TypeError("fused log coords is ParaSLSTM only")
        if picard_iters:
            raise TypeError("fused Picard is ParaSLSTM only")
        if scan_tile != "assoc":
            raise TypeError("fused scan_tile is ParaSLSTM only")
        if fused_time_loop:
            raise TypeError("fused_time_loop is ParaSLSTM only")
        u = cell.clipped_u()
        if isinstance(cell, ParaTitans):
            r = cell.clipped_r()
            if early:
                from pararnn.kernels.newton_titans import _newton_titans_fused_impl

                return _newton_titans_fused_impl(
                    wx,
                    u,
                    r,
                    max_iters=max_iters,
                    omega=omega,
                    h0=h0,
                    cu_seqlens=cu_seqlens,
                    block_table=block_table,
                    early_exit_atol=early_exit_atol,
                    residual_history=residual_history,
                    iters_done_out=iters_done_out,
                )
            from pararnn.kernels.custom_ops import newton_titans_fused

            return newton_titans_fused(
                wx,
                u,
                r,
                h0,
                cu_seqlens,
                block_table,
                max_iters=max_iters,
                omega=omega,
            )
        if isinstance(cell, ParaCfC):
            no_gate = getattr(cell, "update", "convex") == "no_gate"
            if dt is None:
                raise TypeError("fused ParaCfC needs dt=(B,T) separate from wx=(B,T,3*d_h)")
            if early or residual_max_out is not None:
                from pararnn.kernels.newton_cfc import _newton_cfc_fused_impl

                return _newton_cfc_fused_impl(
                    wx,
                    u,
                    dt,
                    max_iters=max_iters,
                    omega=omega,
                    h0=h0,
                    cu_seqlens=cu_seqlens,
                    block_table=block_table,
                    early_exit_atol=early_exit_atol,
                    residual_history=residual_history,
                    iters_done_out=iters_done_out,
                    no_gate=no_gate,
                    residual_max_out=residual_max_out,
                )
            from pararnn.kernels.custom_ops import newton_cfc_fused

            return newton_cfc_fused(
                wx,
                u,
                dt,
                h0,
                cu_seqlens,
                block_table,
                max_iters=max_iters,
                omega=omega,
                no_gate=no_gate,
            )
        if early:
            from pararnn.kernels.newton_nlru import _newton_nlru_fused_impl

            return _newton_nlru_fused_impl(
                wx,
                u,
                max_iters=max_iters,
                omega=omega,
                h0=h0,
                cu_seqlens=cu_seqlens,
                block_table=block_table,
                early_exit_atol=early_exit_atol,
                residual_history=residual_history,
                iters_done_out=iters_done_out,
            )
        from pararnn.kernels.custom_ops import newton_nlru_fused

        return newton_nlru_fused(
            wx,
            u,
            h0,
            cu_seqlens,
            block_table,
            max_iters=max_iters,
            omega=omega,
        )
    if isinstance(cell, ParaGRU):
        if cell.mix == "head":
            if log_coords:
                raise TypeError("fused log coords is ParaSLSTM only")
            if picard_iters:
                raise TypeError("fused Picard is ParaSLSTM only")
            if scan_tile != "assoc":
                raise TypeError("fused scan_tile is ParaSLSTM only")
            if fused_time_loop:
                raise TypeError("fused_time_loop is ParaSLSTM only")
            if block_table is not None:
                raise TypeError("block_table fused Newton is mix='diag' ParaGRU only")
            if cu_seqlens is not None:
                raise TypeError(
                    "ParaGRU(mix='head') fused Newton does not support cu_seqlens yet; "
                    "use scan_backend='eager' for ragged packs, or pad to a rectangular batch"
                )
            if early:
                raise TypeError(
                    "fused_early_exit is mix='diag' only; ParaGRU(mix='head') uses fixed K"
                )
            a_z, a_r, a_n = cell.clipped_a_head()
            from pararnn.kernels.custom_ops import newton_gru_head_fused

            return newton_gru_head_fused(
                wx,
                a_z,
                a_r,
                a_n,
                h0,
                None,
                max_iters=max_iters,
                omega=omega,
            )
        if cell.mix != "diag":
            raise TypeError(f"fused Newton is mix='diag'|'head' only; got mix={cell.mix!r}")
        if log_coords:
            raise TypeError("fused log coords is ParaSLSTM only")
        if picard_iters:
            raise TypeError("fused Picard is ParaSLSTM only")
        if scan_tile != "assoc":
            raise TypeError("fused scan_tile is ParaSLSTM only")
        if fused_time_loop:
            raise TypeError("fused_time_loop is ParaSLSTM only")
        a_z, a_r, a_n = cell.clipped_a()
        if early:
            from pararnn.kernels.newton_gru import _newton_gru_fused_impl

            return _newton_gru_fused_impl(
                wx,
                a_z,
                a_r,
                a_n,
                max_iters=max_iters,
                omega=omega,
                h0=h0,
                cu_seqlens=cu_seqlens,
                block_table=block_table,
                early_exit_atol=early_exit_atol,
                residual_history=residual_history,
                iters_done_out=iters_done_out,
            )
        from pararnn.kernels.custom_ops import newton_gru_fused

        return newton_gru_fused(
            wx,
            a_z,
            a_r,
            a_n,
            h0,
            cu_seqlens,
            block_table,
            max_iters=max_iters,
            omega=omega,
        )
    if isinstance(cell, ParaLSTM):
        if log_coords:
            raise TypeError("fused log coords is ParaSLSTM only")
        if picard_iters:
            raise TypeError("fused Picard is ParaSLSTM only")
        if scan_tile != "assoc":
            raise TypeError("fused scan_tile is ParaSLSTM only")
        if fused_time_loop:
            raise TypeError("fused_time_loop is ParaSLSTM only")
        a_f, a_z, a_o, c_f, c_o = cell.clipped_recurrent()
        if early:
            from pararnn.kernels.newton_lstm import _newton_lstm_fused_impl

            return _newton_lstm_fused_impl(
                wx,
                a_f,
                a_z,
                a_o,
                c_f,
                c_o,
                max_iters=max_iters,
                omega=omega,
                h0=h0,
                block_table=block_table,
                early_exit_atol=early_exit_atol,
                residual_history=residual_history,
                iters_done_out=iters_done_out,
            )
        from pararnn.kernels.custom_ops import newton_lstm_fused

        return newton_lstm_fused(
            wx,
            a_f,
            a_z,
            a_o,
            c_f,
            c_o,
            h0,
            block_table,
            max_iters=max_iters,
            omega=omega,
        )
    if isinstance(cell, ParaSLSTM):
        if cell.mix == "head":
            if log_coords:
                raise TypeError("fused log coords is mix='diag' ParaSLSTM only")
            if scan_tile != "assoc":
                raise TypeError("fused scan_tile is mix='diag' ParaSLSTM only")
            if fused_time_loop:
                raise TypeError("fused_time_loop is mix='diag' ParaSLSTM only")
            if block_table is not None:
                raise TypeError("block_table fused Newton is mix='diag' ParaSLSTM only")
            if cu_seqlens is not None:
                raise TypeError(
                    "ParaSLSTM(mix='head') fused Newton does not support cu_seqlens yet; "
                    "use scan_backend='eager' for ragged packs, or pad to a rectangular batch"
                )
            if early:
                raise TypeError(
                    "fused_early_exit is mix='diag' only; ParaSLSTM(mix='head') uses fixed K"
                )
            from pararnn.solvers.slstm_picard import (
                slstm_picard_init,
                slstm_zero_hidden_init,
            )

            if picard_iters:
                states = slstm_picard_init(cell, wx, h0=h0, n_picard=picard_iters)
            else:
                states = slstm_zero_hidden_init(wx, eps=cell.eps, h0=h0)
            from pararnn.kernels.custom_ops import newton_slstm_head_fused

            return newton_slstm_head_fused(
                wx,
                cell.clipped_r_head(),
                h0,
                states,
                max_iters=max_iters,
                omega=omega,
                eps=cell.eps,
            )
        if cell.mix == "lowrank":
            if log_coords:
                raise TypeError("fused log coords is mix='diag' ParaSLSTM only")
            if fused_time_loop:
                raise TypeError("fused_time_loop is mix='diag' ParaSLSTM only")
            if block_table is not None:
                raise TypeError("block_table fused Newton is mix='diag' ParaSLSTM only")
            if cu_seqlens is not None:
                raise TypeError(
                    "ParaSLSTM(mix='lowrank') fused Newton does not support cu_seqlens yet"
                )
            if early:
                raise TypeError("fused_early_exit is mix='diag' only for now")
            from pararnn.kernels.newton_slstm_lowrank import newton_slstm_lowrank_fused

            return newton_slstm_lowrank_fused(
                cell,
                wx,
                max_iters=max_iters,
                omega=omega,
                h0=h0,
                picard_iters=picard_iters,
            )
        if cell.mix != "diag":
            raise TypeError(
                f"fused Newton is mix='diag'|'head'|'lowrank' only; got mix={cell.mix!r}"
            )
        from pararnn.solvers.slstm_picard import (
            slstm_frozen_gate_scan,
            slstm_picard_init,
        )
        from pararnn.solvers.newton.config import FUSED_WINDOW_DEFAULT, FUSED_WINDOW_LENS

        h0_init = h0
        if block_table is not None and h0 is not None:
            h0_init = h0.index_select(0, block_table.long())
        # One trajectory buffer: Picard / zero-hidden write in place; Newton
        # overwrites the same storage (no second full-T clone).
        batch, time, four_d = wx.shape
        d_h = four_d // 4
        from pararnn.layout import SLSTM_SLOTS

        states = wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
        # Embed Picard inside the leaf / window walk when we can do one launch
        # (FTL, or a single window length). Early-exit / residual rails keep
        # host Picard + multi-launch Newton.
        win = None
        if fused_time_loop:
            win = (
                int(fused_window_len)
                if fused_window_len is not None
                else FUSED_WINDOW_DEFAULT
            )
        early = early_exit_atol is not None
        embed_picard = (
            not log_coords
            and not early
            and residual_max_out is None
            and residual_history is None
            and (
                fused_time_loop
                or (time in FUSED_WINDOW_LENS and int(picard_iters) >= 0)
            )
        )
        n_picard_walk: int | None
        if embed_picard:
            n_picard_walk = int(picard_iters)
        else:
            n_picard_walk = None
            if picard_iters:
                slstm_picard_init(
                    cell,
                    wx,
                    h0=h0_init,
                    n_picard=picard_iters,
                    chunk_len=win,
                    out=states,
                )
            else:
                slstm_frozen_gate_scan(wx, eps=cell.eps, h0=h0_init, out=states)
        if early or residual_max_out is not None or not torch.compiler.is_compiling():
            # Eager: in-place on the Picard buffer (custom_op forbids aliasing returns).
            from pararnn.kernels.newton_slstm import _newton_slstm_fused_impl

            return _newton_slstm_fused_impl(
                wx,
                cell.clipped_r(),
                h0=h0,
                states=states,
                block_table=block_table,
                max_iters=max_iters,
                omega=omega,
                eps=cell.eps,
                log_coords=log_coords,
                scan_tile=scan_tile,
                time_loop=fused_time_loop,
                window_len=0 if fused_window_len is None else int(fused_window_len),
                early_exit_atol=early_exit_atol,
                residual_history=residual_history,
                iters_done_out=iters_done_out,
                residual_max_out=residual_max_out,
                n_picard=n_picard_walk,
            )
        from pararnn.kernels.custom_ops import newton_slstm_fused

        # Compile path: host Picard still (custom_op has no n_picard yet).
        if embed_picard:
            if picard_iters:
                slstm_picard_init(
                    cell,
                    wx,
                    h0=h0_init,
                    n_picard=picard_iters,
                    chunk_len=win,
                    out=states,
                )
            else:
                slstm_frozen_gate_scan(wx, eps=cell.eps, h0=h0_init, out=states)
        return newton_slstm_fused(
            wx,
            cell.clipped_r(),
            h0,
            states,
            block_table,
            max_iters=max_iters,
            omega=omega,
            eps=cell.eps,
            log_coords=log_coords,
            scan_tile=scan_tile,
            time_loop=fused_time_loop,
            window_len=0 if fused_window_len is None else int(fused_window_len),
        )
    raise TypeError(
        f"fused Newton is ParaGRU/ParaLSTM/ParaSLSTM/ParaNLRU/ParaCfC/ParaTitans only; "
        f"got {type(cell).__name__}"
    )

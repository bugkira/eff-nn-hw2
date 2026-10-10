"""Fused ParaSLSTM Newton: diag mix, 4×4 J + scan (Beck sLSTM + paper Alg. 1).

``W_x(x)`` stays a cuBLAS GEMM. This kernel is diag mix, 4×4 SRAM.
CUDA float16/float32, and bf16 on compute capability ≥ 8.0; cell+scan
algebra in fp32. DRAM is the tensor dtype.

The 4×4 linearizes ``R h`` into the next gates. Picard (frozen ``R h``)
for window / leaf lengths runs inside ``_slstm_window_walk_kernel``
(``1+N_PICARD`` passes, then K Newton) with Newton-solved left carry.
Long-T host Picard remains the 1D max-plus path in ``picard_slstm.py``.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels._fused_common import (
    _load_h0,
    _load_state,
    _store_state,
    _tanh,
    alloc_fp32_update,
    fp32_newton_work,
    fp32_omega_add,
    fused_early_exit_hit,
    log_fused_done,
    log_fused_iter,
    mark_fused_iters_done,
    prepare_h0_block_table,
    time_tiles,
)
from pararnn.kernels._scan_common import (
    _load_j as _load_j_lane,
)
from pararnn.kernels._scan_common import (
    _store_agg_j,
    incl_block_aggregates,
)
from pararnn.kernels._scan_common import (
    _store_j as _store_j_lane,
)
from pararnn.kernels.device_recipe import resolve_slstm_launch
from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors
from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_SLOTS,
    SLSTM_STABILIZER,
)
from pararnn.solvers.newton.config import FUSED_WINDOW_DEFAULT, FUSED_WINDOW_LENS

log = logging.getLogger(__name__)

# Multi-chunk Newton: 20 lanes × 32 × 16 × 4 B = 40 KiB (fast at long T).
_BLOCK_T = 32
_BLOCK_D = 16
# Chunk-incl leaf: 20 × CHUNK_PAD × CHUNK_D × 4 B ≈ 160 KiB (regs + scan temps).
# CHUNK_D=16: warp-coalesced loads along contiguous d_h. PAD and walk
# launch knobs (warps/stages/block_d) come from ``device_recipe`` tables
# (3060/2080 measured; A100 row filled by lab sweep).
# Leaf: n_chunks≤pad → T≤BLOCK_T×pad; past that, eager Blelloch.
# When T ∈ FUSED_WINDOW_LENS (32/64/128), use T as BLOCK_T so chunked
# Newton is one leaf (no _chunk_incl inside the window).
_CHUNK_D = 16
# Fallback pad when device is unavailable (CPU paths / import-time docs).
_CHUNK_PAD = 128


def _leaf_block_t(time: int) -> int:
    """``BLOCK_T`` for fused Alg. 1: single leaf when ``T`` is a window length."""
    if int(time) in FUSED_WINDOW_LENS:
        return int(time)
    return _BLOCK_T


@triton.jit
def _compose_block4(
    a00,
    a01,
    a02,
    a03,
    a10,
    a11,
    a12,
    a13,
    a20,
    a21,
    a22,
    a23,
    a30,
    a31,
    a32,
    a33,
    u0,
    u1,
    u2,
    u3,
    b00,
    b01,
    b02,
    b03,
    b10,
    b11,
    b12,
    b13,
    b20,
    b21,
    b22,
    b23,
    b30,
    b31,
    b32,
    b33,
    v0,
    v1,
    v2,
    v3,
):
    o00 = b00 * a00 + b01 * a10 + b02 * a20 + b03 * a30
    o01 = b00 * a01 + b01 * a11 + b02 * a21 + b03 * a31
    o02 = b00 * a02 + b01 * a12 + b02 * a22 + b03 * a32
    o03 = b00 * a03 + b01 * a13 + b02 * a23 + b03 * a33
    o10 = b10 * a00 + b11 * a10 + b12 * a20 + b13 * a30
    o11 = b10 * a01 + b11 * a11 + b12 * a21 + b13 * a31
    o12 = b10 * a02 + b11 * a12 + b12 * a22 + b13 * a32
    o13 = b10 * a03 + b11 * a13 + b12 * a23 + b13 * a33
    o20 = b20 * a00 + b21 * a10 + b22 * a20 + b23 * a30
    o21 = b20 * a01 + b21 * a11 + b22 * a21 + b23 * a31
    o22 = b20 * a02 + b21 * a12 + b22 * a22 + b23 * a32
    o23 = b20 * a03 + b21 * a13 + b22 * a23 + b23 * a33
    o30 = b30 * a00 + b31 * a10 + b32 * a20 + b33 * a30
    o31 = b30 * a01 + b31 * a11 + b32 * a21 + b33 * a31
    o32 = b30 * a02 + b31 * a12 + b32 * a22 + b33 * a32
    o33 = b30 * a03 + b31 * a13 + b32 * a23 + b33 * a33
    w0 = b00 * u0 + b01 * u1 + b02 * u2 + b03 * u3 + v0
    w1 = b10 * u0 + b11 * u1 + b12 * u2 + b13 * u3 + v1
    w2 = b20 * u0 + b21 * u1 + b22 * u2 + b23 * u3 + v2
    w3 = b30 * u0 + b31 * u1 + b32 * u2 + b33 * u3 + v3
    return (
        o00,
        o01,
        o02,
        o03,
        o10,
        o11,
        o12,
        o13,
        o20,
        o21,
        o22,
        o23,
        o30,
        o31,
        o32,
        o33,
        w0,
        w1,
        w2,
        w3,
    )


@triton.jit
def _compose_block4_sparse_leaves(
    a00,
    a01,
    a02,
    a03,
    a10,
    a11,
    a12,
    a13,
    a20,
    a21,
    a22,
    a23,
    a30,
    a31,
    a32,
    a33,
    u0,
    u1,
    u2,
    u3,
    b00,
    b01,
    b02,
    b03,
    b10,
    b11,
    b12,
    b13,
    b20,
    b21,
    b22,
    b23,
    b30,
    b31,
    b32,
    b33,
    v0,
    v1,
    v2,
    v3,
):
    """``combine(earlier, later)`` when both sides are native leaf J.

    Structural zeros ``j_cn=j_nc=j_mc=j_mn=0`` (``a01,a10,a20,a21`` and the
    same on ``b``). Prefix products fill in — return value is dense. Unused
    leaf lanes are still in the signature so this matches ``associative_scan``
    arity; they are ignored.
    """
    o00 = b00 * a00 + b03 * a30
    o01 = b03 * a31
    o02 = b00 * a02 + b02 * a22 + b03 * a32
    o03 = b00 * a03 + b02 * a23 + b03 * a33
    o10 = b13 * a30
    o11 = b11 * a11 + b13 * a31
    o12 = b11 * a12 + b12 * a22 + b13 * a32
    o13 = b11 * a13 + b12 * a23 + b13 * a33
    o20 = b23 * a30
    o21 = b23 * a31
    o22 = b22 * a22 + b23 * a32
    o23 = b22 * a23 + b23 * a33
    o30 = b30 * a00 + b33 * a30
    o31 = b31 * a11 + b33 * a31
    o32 = b30 * a02 + b31 * a12 + b32 * a22 + b33 * a32
    o33 = b30 * a03 + b31 * a13 + b32 * a23 + b33 * a33
    w0 = b00 * u0 + b02 * u2 + b03 * u3 + v0
    w1 = b11 * u1 + b12 * u2 + b13 * u3 + v1
    w2 = b22 * u2 + b23 * u3 + v2
    w3 = b30 * u0 + b31 * u1 + b32 * u2 + b33 * u3 + v3
    return (
        o00,
        o01,
        o02,
        o03,
        o10,
        o11,
        o12,
        o13,
        o20,
        o21,
        o22,
        o23,
        o30,
        o31,
        o32,
        o33,
        w0,
        w1,
        w2,
        w3,
    )


@triton.jit
def _compose_block4_sparse_right(
    a00,
    a01,
    a02,
    a03,
    a10,
    a11,
    a12,
    a13,
    a20,
    a21,
    a22,
    a23,
    a30,
    a31,
    a32,
    a33,
    u0,
    u1,
    u2,
    u3,
    b00,
    b01,
    b02,
    b03,
    b10,
    b11,
    b12,
    b13,
    b20,
    b21,
    b22,
    b23,
    b30,
    b31,
    b32,
    b33,
    v0,
    v1,
    v2,
    v3,
):
    """Dense left × native leaf right (``b01=b10=b20=b21=0``)."""
    o00 = b00 * a00 + b02 * a20 + b03 * a30
    o01 = b00 * a01 + b02 * a21 + b03 * a31
    o02 = b00 * a02 + b02 * a22 + b03 * a32
    o03 = b00 * a03 + b02 * a23 + b03 * a33
    o10 = b11 * a10 + b12 * a20 + b13 * a30
    o11 = b11 * a11 + b12 * a21 + b13 * a31
    o12 = b11 * a12 + b12 * a22 + b13 * a32
    o13 = b11 * a13 + b12 * a23 + b13 * a33
    o20 = b22 * a20 + b23 * a30
    o21 = b22 * a21 + b23 * a31
    o22 = b22 * a22 + b23 * a32
    o23 = b22 * a23 + b23 * a33
    o30 = b30 * a00 + b31 * a10 + b32 * a20 + b33 * a30
    o31 = b30 * a01 + b31 * a11 + b32 * a21 + b33 * a31
    o32 = b30 * a02 + b31 * a12 + b32 * a22 + b33 * a32
    o33 = b30 * a03 + b31 * a13 + b32 * a23 + b33 * a33
    w0 = b00 * u0 + b02 * u2 + b03 * u3 + v0
    w1 = b11 * u1 + b12 * u2 + b13 * u3 + v1
    w2 = b22 * u2 + b23 * u3 + v2
    w3 = b30 * u0 + b31 * u1 + b32 * u2 + b33 * u3 + v3
    return (
        o00,
        o01,
        o02,
        o03,
        o10,
        o11,
        o12,
        o13,
        o20,
        o21,
        o22,
        o23,
        o30,
        o31,
        o32,
        o33,
        w0,
        w1,
        w2,
        w3,
    )


@triton.jit
def _gather_time_rows(x, idx_1d, BLOCK_D: tl.constexpr, N_OUT: tl.constexpr):
    """Gather time rows of ``x`` ``(T, D)`` at ``idx_1d`` ``(N_OUT,)`` → ``(N_OUT, D)``.

    Double-``tl.trans`` puts the gather axis last so Triton can lower a dense
    row permute; measured faster than mid-axis ``reshape``+``gather`` /
    ``broadcast_to`` interleave on 2080 Ti (see lab
    ``diag_leaf_pair_reshape.py``).
    """
    idx = idx_1d[None, :] + tl.zeros((BLOCK_D, 1), dtype=tl.int32)
    return tl.trans(tl.gather(tl.trans(x), idx, axis=1))


@triton.jit
def _expand_half_to_t(half, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, HALF: tl.constexpr):
    """Replicate each half-row onto ``[2i, 2i+1]`` via gather (``t // 2``)."""
    idx = (tl.arange(0, BLOCK_T) // 2)[None, :] + tl.zeros((BLOCK_D, 1), dtype=tl.int32)
    return tl.trans(tl.gather(tl.trans(half), idx, axis=1))


@triton.jit
def _split_even_odd(x, HALF: tl.constexpr, BLOCK_D: tl.constexpr):
    """``x`` ``(2*HALF, D)`` → even/odd ``(HALF, D)`` via time-row gather."""
    idx_even = tl.arange(0, HALF) * 2
    idx_odd = idx_even + 1
    return (
        _gather_time_rows(x, idx_even, BLOCK_D, HALF),
        _gather_time_rows(x, idx_odd, BLOCK_D, HALF),
    )


@triton.jit
def _interleave_rows(
    even, odd, HALF: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr
):
    """``out[2i]=even[i]``, ``out[2i+1]=odd[i]`` via expand + ``where``."""
    ev = _expand_half_to_t(even, BLOCK_T, BLOCK_D, HALF)
    od = _expand_half_to_t(odd, BLOCK_T, BLOCK_D, HALF)
    is_odd = ((tl.arange(0, BLOCK_T) % 2) == 1)[:, None]
    return tl.where(is_odd, od, ev)


@triton.jit
def _exclusive_shift_row(q, eye_val, HALF: tl.constexpr, BLOCK_D: tl.constexpr):
    """Inclusive ``q`` → exclusive: ``I`` at 0, ``q[i-1]`` thereafter (gather axis 0)."""
    offs = tl.arange(0, HALF)
    first = (offs == 0)[:, None]
    idx = tl.maximum(offs - 1, 0)[:, None] + tl.zeros((HALF, BLOCK_D), dtype=tl.int32)
    return tl.where(first, eye_val, tl.gather(q, idx, axis=0))


@triton.jit
def _assoc_scan_block4_leaf_pairs(
    j00,
    j01,
    j02,
    j03,
    j10,
    j11,
    j12,
    j13,
    j20,
    j21,
    j22,
    j23,
    j30,
    j31,
    j32,
    j33,
    r0,
    r1,
    r2,
    r3,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Inclusive scan: sparse×sparse pair reduce, then dense ``associative_scan``.

    Native leaf pattern only (``LOG=False``). ``BLOCK_T`` must be even.
    Even/odd split and re-interleave use time-row ``gather`` + expand/``where``
    (reshape/mid-axis gather and ``broadcast_to`` interleave measured slower
    and higher shared on 2080 Ti; lab ``diag_leaf_pair_reshape.py``).
    Exclusive pair shift keeps one axis-0 gather per lane.
    """
    HALF: tl.constexpr = BLOCK_T // 2
    a00, b00 = _split_even_odd(j00, HALF, BLOCK_D)
    a01, b01 = _split_even_odd(j01, HALF, BLOCK_D)
    a02, b02 = _split_even_odd(j02, HALF, BLOCK_D)
    a03, b03 = _split_even_odd(j03, HALF, BLOCK_D)
    a10, b10 = _split_even_odd(j10, HALF, BLOCK_D)
    a11, b11 = _split_even_odd(j11, HALF, BLOCK_D)
    a12, b12 = _split_even_odd(j12, HALF, BLOCK_D)
    a13, b13 = _split_even_odd(j13, HALF, BLOCK_D)
    a20, b20 = _split_even_odd(j20, HALF, BLOCK_D)
    a21, b21 = _split_even_odd(j21, HALF, BLOCK_D)
    a22, b22 = _split_even_odd(j22, HALF, BLOCK_D)
    a23, b23 = _split_even_odd(j23, HALF, BLOCK_D)
    a30, b30 = _split_even_odd(j30, HALF, BLOCK_D)
    a31, b31 = _split_even_odd(j31, HALF, BLOCK_D)
    a32, b32 = _split_even_odd(j32, HALF, BLOCK_D)
    a33, b33 = _split_even_odd(j33, HALF, BLOCK_D)
    u0, v0 = _split_even_odd(r0, HALF, BLOCK_D)
    u1, v1 = _split_even_odd(r1, HALF, BLOCK_D)
    u2, v2 = _split_even_odd(r2, HALF, BLOCK_D)
    u3, v3 = _split_even_odd(r3, HALF, BLOCK_D)
    (
        p00,
        p01,
        p02,
        p03,
        p10,
        p11,
        p12,
        p13,
        p20,
        p21,
        p22,
        p23,
        p30,
        p31,
        p32,
        p33,
        pu0,
        pu1,
        pu2,
        pu3,
    ) = _compose_block4_sparse_leaves(
        a00,
        a01,
        a02,
        a03,
        a10,
        a11,
        a12,
        a13,
        a20,
        a21,
        a22,
        a23,
        a30,
        a31,
        a32,
        a33,
        u0,
        u1,
        u2,
        u3,
        b00,
        b01,
        b02,
        b03,
        b10,
        b11,
        b12,
        b13,
        b20,
        b21,
        b22,
        b23,
        b30,
        b31,
        b32,
        b33,
        v0,
        v1,
        v2,
        v3,
    )
    (
        q00,
        q01,
        q02,
        q03,
        q10,
        q11,
        q12,
        q13,
        q20,
        q21,
        q22,
        q23,
        q30,
        q31,
        q32,
        q33,
        qu0,
        qu1,
        qu2,
        qu3,
    ) = tl.associative_scan(
        (
            p00,
            p01,
            p02,
            p03,
            p10,
            p11,
            p12,
            p13,
            p20,
            p21,
            p22,
            p23,
            p30,
            p31,
            p32,
            p33,
            pu0,
            pu1,
            pu2,
            pu3,
        ),
        0,
        _compose_block4,
    )
    e00 = _exclusive_shift_row(q00, 1.0, HALF, BLOCK_D)
    e01 = _exclusive_shift_row(q01, 0.0, HALF, BLOCK_D)
    e02 = _exclusive_shift_row(q02, 0.0, HALF, BLOCK_D)
    e03 = _exclusive_shift_row(q03, 0.0, HALF, BLOCK_D)
    e10 = _exclusive_shift_row(q10, 0.0, HALF, BLOCK_D)
    e11 = _exclusive_shift_row(q11, 1.0, HALF, BLOCK_D)
    e12 = _exclusive_shift_row(q12, 0.0, HALF, BLOCK_D)
    e13 = _exclusive_shift_row(q13, 0.0, HALF, BLOCK_D)
    e20 = _exclusive_shift_row(q20, 0.0, HALF, BLOCK_D)
    e21 = _exclusive_shift_row(q21, 0.0, HALF, BLOCK_D)
    e22 = _exclusive_shift_row(q22, 1.0, HALF, BLOCK_D)
    e23 = _exclusive_shift_row(q23, 0.0, HALF, BLOCK_D)
    e30 = _exclusive_shift_row(q30, 0.0, HALF, BLOCK_D)
    e31 = _exclusive_shift_row(q31, 0.0, HALF, BLOCK_D)
    e32 = _exclusive_shift_row(q32, 0.0, HALF, BLOCK_D)
    e33 = _exclusive_shift_row(q33, 1.0, HALF, BLOCK_D)
    eu0 = _exclusive_shift_row(qu0, 0.0, HALF, BLOCK_D)
    eu1 = _exclusive_shift_row(qu1, 0.0, HALF, BLOCK_D)
    eu2 = _exclusive_shift_row(qu2, 0.0, HALF, BLOCK_D)
    eu3 = _exclusive_shift_row(qu3, 0.0, HALF, BLOCK_D)
    (
        even00,
        even01,
        even02,
        even03,
        even10,
        even11,
        even12,
        even13,
        even20,
        even21,
        even22,
        even23,
        even30,
        even31,
        even32,
        even33,
        evenu0,
        evenu1,
        evenu2,
        evenu3,
    ) = _compose_block4_sparse_right(
        e00,
        e01,
        e02,
        e03,
        e10,
        e11,
        e12,
        e13,
        e20,
        e21,
        e22,
        e23,
        e30,
        e31,
        e32,
        e33,
        eu0,
        eu1,
        eu2,
        eu3,
        a00,
        a01,
        a02,
        a03,
        a10,
        a11,
        a12,
        a13,
        a20,
        a21,
        a22,
        a23,
        a30,
        a31,
        a32,
        a33,
        u0,
        u1,
        u2,
        u3,
    )
    return (
        _interleave_rows(even00, q00, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even01, q01, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even02, q02, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even03, q03, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even10, q10, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even11, q11, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even12, q12, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even13, q13, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even20, q20, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even21, q21, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even22, q22, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even23, q23, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even30, q30, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even31, q31, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even32, q32, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(even33, q33, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(evenu0, qu0, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(evenu1, qu1, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(evenu2, qu2, HALF, BLOCK_T, BLOCK_D),
        _interleave_rows(evenu3, qu3, HALF, BLOCK_T, BLOCK_D),
    )


@triton.jit
def _seq_scan_block4(
    j00,
    j01,
    j02,
    j03,
    j10,
    j11,
    j12,
    j13,
    j20,
    j21,
    j22,
    j23,
    j30,
    j31,
    j32,
    j33,
    r0,
    r1,
    r2,
    r3,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Inclusive serial prefix along time. Ablation vs ``tl.associative_scan``.

    Accumulators stay ``(BLOCK_D,)``. Each step gathers one time row with
    ``tl.gather`` — no ``tl.sum(tl.where(sel, …))`` reduction trees.
    """
    a00 = tl.full((BLOCK_D,), 1.0, tl.float32)
    a01 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a02 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a03 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a10 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a11 = tl.full((BLOCK_D,), 1.0, tl.float32)
    a12 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a13 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a20 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a21 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a22 = tl.full((BLOCK_D,), 1.0, tl.float32)
    a23 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a30 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a31 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a32 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a33 = tl.full((BLOCK_D,), 1.0, tl.float32)
    w0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    o00 = tl.zeros_like(j00)
    o01 = tl.zeros_like(j00)
    o02 = tl.zeros_like(j00)
    o03 = tl.zeros_like(j00)
    o10 = tl.zeros_like(j00)
    o11 = tl.zeros_like(j00)
    o12 = tl.zeros_like(j00)
    o13 = tl.zeros_like(j00)
    o20 = tl.zeros_like(j00)
    o21 = tl.zeros_like(j00)
    o22 = tl.zeros_like(j00)
    o23 = tl.zeros_like(j00)
    o30 = tl.zeros_like(j00)
    o31 = tl.zeros_like(j00)
    o32 = tl.zeros_like(j00)
    o33 = tl.zeros_like(j00)
    u0 = tl.zeros_like(r0)
    u1 = tl.zeros_like(r0)
    u2 = tl.zeros_like(r0)
    u3 = tl.zeros_like(r0)
    offs_t = tl.arange(0, BLOCK_T)
    # Transpose once; gather along the time axis inside the loop.
    t00, t01, t02, t03 = tl.trans(j00), tl.trans(j01), tl.trans(j02), tl.trans(j03)
    t10, t11, t12, t13 = tl.trans(j10), tl.trans(j11), tl.trans(j12), tl.trans(j13)
    t20, t21, t22, t23 = tl.trans(j20), tl.trans(j21), tl.trans(j22), tl.trans(j23)
    t30, t31, t32, t33 = tl.trans(j30), tl.trans(j31), tl.trans(j32), tl.trans(j33)
    tr0, tr1, tr2, tr3 = tl.trans(r0), tl.trans(r1), tl.trans(r2), tl.trans(r3)
    for t in tl.static_range(BLOCK_T):
        idx = tl.zeros((BLOCK_D, 1), dtype=tl.int32) + t
        e00 = tl.reshape(tl.gather(t00, idx, axis=1), (BLOCK_D,))
        e01 = tl.reshape(tl.gather(t01, idx, axis=1), (BLOCK_D,))
        e02 = tl.reshape(tl.gather(t02, idx, axis=1), (BLOCK_D,))
        e03 = tl.reshape(tl.gather(t03, idx, axis=1), (BLOCK_D,))
        e10 = tl.reshape(tl.gather(t10, idx, axis=1), (BLOCK_D,))
        e11 = tl.reshape(tl.gather(t11, idx, axis=1), (BLOCK_D,))
        e12 = tl.reshape(tl.gather(t12, idx, axis=1), (BLOCK_D,))
        e13 = tl.reshape(tl.gather(t13, idx, axis=1), (BLOCK_D,))
        e20 = tl.reshape(tl.gather(t20, idx, axis=1), (BLOCK_D,))
        e21 = tl.reshape(tl.gather(t21, idx, axis=1), (BLOCK_D,))
        e22 = tl.reshape(tl.gather(t22, idx, axis=1), (BLOCK_D,))
        e23 = tl.reshape(tl.gather(t23, idx, axis=1), (BLOCK_D,))
        e30 = tl.reshape(tl.gather(t30, idx, axis=1), (BLOCK_D,))
        e31 = tl.reshape(tl.gather(t31, idx, axis=1), (BLOCK_D,))
        e32 = tl.reshape(tl.gather(t32, idx, axis=1), (BLOCK_D,))
        e33 = tl.reshape(tl.gather(t33, idx, axis=1), (BLOCK_D,))
        er0 = tl.reshape(tl.gather(tr0, idx, axis=1), (BLOCK_D,))
        er1 = tl.reshape(tl.gather(tr1, idx, axis=1), (BLOCK_D,))
        er2 = tl.reshape(tl.gather(tr2, idx, axis=1), (BLOCK_D,))
        er3 = tl.reshape(tl.gather(tr3, idx, axis=1), (BLOCK_D,))
        (
            a00,
            a01,
            a02,
            a03,
            a10,
            a11,
            a12,
            a13,
            a20,
            a21,
            a22,
            a23,
            a30,
            a31,
            a32,
            a33,
            w0,
            w1,
            w2,
            w3,
        ) = _compose_block4(
            a00,
            a01,
            a02,
            a03,
            a10,
            a11,
            a12,
            a13,
            a20,
            a21,
            a22,
            a23,
            a30,
            a31,
            a32,
            a33,
            w0,
            w1,
            w2,
            w3,
            e00,
            e01,
            e02,
            e03,
            e10,
            e11,
            e12,
            e13,
            e20,
            e21,
            e22,
            e23,
            e30,
            e31,
            e32,
            e33,
            er0,
            er1,
            er2,
            er3,
        )
        sel = (offs_t == t)[:, None]
        o00 = tl.where(sel, a00[None, :], o00)
        o01 = tl.where(sel, a01[None, :], o01)
        o02 = tl.where(sel, a02[None, :], o02)
        o03 = tl.where(sel, a03[None, :], o03)
        o10 = tl.where(sel, a10[None, :], o10)
        o11 = tl.where(sel, a11[None, :], o11)
        o12 = tl.where(sel, a12[None, :], o12)
        o13 = tl.where(sel, a13[None, :], o13)
        o20 = tl.where(sel, a20[None, :], o20)
        o21 = tl.where(sel, a21[None, :], o21)
        o22 = tl.where(sel, a22[None, :], o22)
        o23 = tl.where(sel, a23[None, :], o23)
        o30 = tl.where(sel, a30[None, :], o30)
        o31 = tl.where(sel, a31[None, :], o31)
        o32 = tl.where(sel, a32[None, :], o32)
        o33 = tl.where(sel, a33[None, :], o33)
        u0 = tl.where(sel, w0[None, :], u0)
        u1 = tl.where(sel, w1[None, :], u1)
        u2 = tl.where(sel, w2[None, :], u2)
        u3 = tl.where(sel, w3[None, :], u3)
    return (
        o00,
        o01,
        o02,
        o03,
        o10,
        o11,
        o12,
        o13,
        o20,
        o21,
        o22,
        o23,
        o30,
        o31,
        o32,
        o33,
        u0,
        u1,
        u2,
        u3,
    )


@triton.jit
def _at_k(x3, k, C: tl.constexpr):
    """Slice ``x3[:, k, :]`` via gather (no ``sum(where)`` reduction)."""
    n: tl.constexpr = x3.shape[0]
    d: tl.constexpr = x3.shape[2]
    idx = tl.zeros((n, 1, d), dtype=tl.int32) + k
    return tl.reshape(tl.gather(x3, idx, axis=1), (n, d))


@triton.jit
def _put_k(dst3, val2, k, C: tl.constexpr):
    """Scatter ``val2`` into ``dst3[:, k, :]``."""
    sel = tl.arange(0, C)[None, :, None] == k
    return tl.where(sel, val2[:, None, :], dst3)


@triton.jit
def _row_n(x_nd, n, D: tl.constexpr):
    """Extract coarse-row ``n`` from ``(N, D)`` via gather."""
    xt = tl.trans(x_nd)
    idx = tl.zeros((D, 1), dtype=tl.int32) + n
    return tl.reshape(tl.gather(xt, idx, axis=1), (D,))


@triton.jit
def _put_row(x_nd, val_d, n, N: tl.constexpr):
    """Write ``val_d`` ``(D,)`` into row ``n`` of ``(N, D)``."""
    sel = (tl.arange(0, N) == n)[:, None]
    return tl.where(sel, val_d[None, :], x_nd)


@triton.jit
def _slstm_fwd_step(
    c_prev,
    n_prev,
    m_prev,
    h_mix,
    zi_x,
    zf_x,
    zz_x,
    zo_x,
    r_i,
    r_f,
    r_z,
    r_o,
    eps,
    rh_scale,
):
    """One sLSTM step (no J). ``rh_scale=0`` is the zero-hidden Picard pass."""
    z_i = rh_scale * r_i * h_mix + zi_x
    z_f = rh_scale * r_f * h_mix + zf_x
    z_z = rh_scale * r_z * h_mix + zz_x
    z_o = rh_scale * r_o * h_mix + zo_x
    left = z_f + m_prev
    m_new = tl.where(left > z_i, left, z_i)
    i_t = tl.exp(z_i - m_new)
    f_t = tl.exp(z_f + m_prev - m_new)
    z = _tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c_prev + i_t * z
    o = tl.sigmoid(z_o)
    h_new = o * (c_new / (n_new + eps))
    return c_new, n_new, m_new, h_new


@triton.jit
def _thomas_pcr_scan_block4(
    j00,
    j01,
    j02,
    j03,
    j10,
    j11,
    j12,
    j13,
    j20,
    j21,
    j22,
    j23,
    j30,
    j31,
    j32,
    j33,
    r0,
    r1,
    r2,
    r3,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    THOMAS_C: tl.constexpr,
):
    """Blocked 4×4 scan: sequential compose of ``THOMAS_C`` steps, then PCR.

    ``THOMAS_C`` must divide ``BLOCK_T``. Intra-group Thomas is a loop of
    length ``C`` on shape ``(T/C, D)``, not ``tl.range(BLOCK_T)``. Coarse
    inclusive scan is ``tl.associative_scan`` of ``T/C`` monoid elements.
    """
    n_coarse: tl.constexpr = BLOCK_T // THOMAS_C
    j00g = tl.reshape(j00, [n_coarse, THOMAS_C, BLOCK_D])
    j01g = tl.reshape(j01, [n_coarse, THOMAS_C, BLOCK_D])
    j02g = tl.reshape(j02, [n_coarse, THOMAS_C, BLOCK_D])
    j03g = tl.reshape(j03, [n_coarse, THOMAS_C, BLOCK_D])
    j10g = tl.reshape(j10, [n_coarse, THOMAS_C, BLOCK_D])
    j11g = tl.reshape(j11, [n_coarse, THOMAS_C, BLOCK_D])
    j12g = tl.reshape(j12, [n_coarse, THOMAS_C, BLOCK_D])
    j13g = tl.reshape(j13, [n_coarse, THOMAS_C, BLOCK_D])
    j20g = tl.reshape(j20, [n_coarse, THOMAS_C, BLOCK_D])
    j21g = tl.reshape(j21, [n_coarse, THOMAS_C, BLOCK_D])
    j22g = tl.reshape(j22, [n_coarse, THOMAS_C, BLOCK_D])
    j23g = tl.reshape(j23, [n_coarse, THOMAS_C, BLOCK_D])
    j30g = tl.reshape(j30, [n_coarse, THOMAS_C, BLOCK_D])
    j31g = tl.reshape(j31, [n_coarse, THOMAS_C, BLOCK_D])
    j32g = tl.reshape(j32, [n_coarse, THOMAS_C, BLOCK_D])
    j33g = tl.reshape(j33, [n_coarse, THOMAS_C, BLOCK_D])
    r0g = tl.reshape(r0, [n_coarse, THOMAS_C, BLOCK_D])
    r1g = tl.reshape(r1, [n_coarse, THOMAS_C, BLOCK_D])
    r2g = tl.reshape(r2, [n_coarse, THOMAS_C, BLOCK_D])
    r3g = tl.reshape(r3, [n_coarse, THOMAS_C, BLOCK_D])
    p00 = _at_k(j00g, 0, THOMAS_C)
    p01 = _at_k(j01g, 0, THOMAS_C)
    p02 = _at_k(j02g, 0, THOMAS_C)
    p03 = _at_k(j03g, 0, THOMAS_C)
    p10 = _at_k(j10g, 0, THOMAS_C)
    p11 = _at_k(j11g, 0, THOMAS_C)
    p12 = _at_k(j12g, 0, THOMAS_C)
    p13 = _at_k(j13g, 0, THOMAS_C)
    p20 = _at_k(j20g, 0, THOMAS_C)
    p21 = _at_k(j21g, 0, THOMAS_C)
    p22 = _at_k(j22g, 0, THOMAS_C)
    p23 = _at_k(j23g, 0, THOMAS_C)
    p30 = _at_k(j30g, 0, THOMAS_C)
    p31 = _at_k(j31g, 0, THOMAS_C)
    p32 = _at_k(j32g, 0, THOMAS_C)
    p33 = _at_k(j33g, 0, THOMAS_C)
    pu0 = _at_k(r0g, 0, THOMAS_C)
    pu1 = _at_k(r1g, 0, THOMAS_C)
    pu2 = _at_k(r2g, 0, THOMAS_C)
    pu3 = _at_k(r3g, 0, THOMAS_C)
    i00 = j00g * 0.0
    i01 = j01g * 0.0
    i02 = j02g * 0.0
    i03 = j03g * 0.0
    i10 = j10g * 0.0
    i11 = j11g * 0.0
    i12 = j12g * 0.0
    i13 = j13g * 0.0
    i20 = j20g * 0.0
    i21 = j21g * 0.0
    i22 = j22g * 0.0
    i23 = j23g * 0.0
    i30 = j30g * 0.0
    i31 = j31g * 0.0
    i32 = j32g * 0.0
    i33 = j33g * 0.0
    iu0 = r0g * 0.0
    iu1 = r1g * 0.0
    iu2 = r2g * 0.0
    iu3 = r3g * 0.0
    i00 = _put_k(i00, p00, 0, THOMAS_C)
    i01 = _put_k(i01, p01, 0, THOMAS_C)
    i02 = _put_k(i02, p02, 0, THOMAS_C)
    i03 = _put_k(i03, p03, 0, THOMAS_C)
    i10 = _put_k(i10, p10, 0, THOMAS_C)
    i11 = _put_k(i11, p11, 0, THOMAS_C)
    i12 = _put_k(i12, p12, 0, THOMAS_C)
    i13 = _put_k(i13, p13, 0, THOMAS_C)
    i20 = _put_k(i20, p20, 0, THOMAS_C)
    i21 = _put_k(i21, p21, 0, THOMAS_C)
    i22 = _put_k(i22, p22, 0, THOMAS_C)
    i23 = _put_k(i23, p23, 0, THOMAS_C)
    i30 = _put_k(i30, p30, 0, THOMAS_C)
    i31 = _put_k(i31, p31, 0, THOMAS_C)
    i32 = _put_k(i32, p32, 0, THOMAS_C)
    i33 = _put_k(i33, p33, 0, THOMAS_C)
    iu0 = _put_k(iu0, pu0, 0, THOMAS_C)
    iu1 = _put_k(iu1, pu1, 0, THOMAS_C)
    iu2 = _put_k(iu2, pu2, 0, THOMAS_C)
    iu3 = _put_k(iu3, pu3, 0, THOMAS_C)
    for k in tl.static_range(1, THOMAS_C):
        e00 = _at_k(j00g, k, THOMAS_C)
        e01 = _at_k(j01g, k, THOMAS_C)
        e02 = _at_k(j02g, k, THOMAS_C)
        e03 = _at_k(j03g, k, THOMAS_C)
        e10 = _at_k(j10g, k, THOMAS_C)
        e11 = _at_k(j11g, k, THOMAS_C)
        e12 = _at_k(j12g, k, THOMAS_C)
        e13 = _at_k(j13g, k, THOMAS_C)
        e20 = _at_k(j20g, k, THOMAS_C)
        e21 = _at_k(j21g, k, THOMAS_C)
        e22 = _at_k(j22g, k, THOMAS_C)
        e23 = _at_k(j23g, k, THOMAS_C)
        e30 = _at_k(j30g, k, THOMAS_C)
        e31 = _at_k(j31g, k, THOMAS_C)
        e32 = _at_k(j32g, k, THOMAS_C)
        e33 = _at_k(j33g, k, THOMAS_C)
        er0 = _at_k(r0g, k, THOMAS_C)
        er1 = _at_k(r1g, k, THOMAS_C)
        er2 = _at_k(r2g, k, THOMAS_C)
        er3 = _at_k(r3g, k, THOMAS_C)
        (
            p00,
            p01,
            p02,
            p03,
            p10,
            p11,
            p12,
            p13,
            p20,
            p21,
            p22,
            p23,
            p30,
            p31,
            p32,
            p33,
            pu0,
            pu1,
            pu2,
            pu3,
        ) = _compose_block4(
            p00,
            p01,
            p02,
            p03,
            p10,
            p11,
            p12,
            p13,
            p20,
            p21,
            p22,
            p23,
            p30,
            p31,
            p32,
            p33,
            pu0,
            pu1,
            pu2,
            pu3,
            e00,
            e01,
            e02,
            e03,
            e10,
            e11,
            e12,
            e13,
            e20,
            e21,
            e22,
            e23,
            e30,
            e31,
            e32,
            e33,
            er0,
            er1,
            er2,
            er3,
        )
        i00 = _put_k(i00, p00, k, THOMAS_C)
        i01 = _put_k(i01, p01, k, THOMAS_C)
        i02 = _put_k(i02, p02, k, THOMAS_C)
        i03 = _put_k(i03, p03, k, THOMAS_C)
        i10 = _put_k(i10, p10, k, THOMAS_C)
        i11 = _put_k(i11, p11, k, THOMAS_C)
        i12 = _put_k(i12, p12, k, THOMAS_C)
        i13 = _put_k(i13, p13, k, THOMAS_C)
        i20 = _put_k(i20, p20, k, THOMAS_C)
        i21 = _put_k(i21, p21, k, THOMAS_C)
        i22 = _put_k(i22, p22, k, THOMAS_C)
        i23 = _put_k(i23, p23, k, THOMAS_C)
        i30 = _put_k(i30, p30, k, THOMAS_C)
        i31 = _put_k(i31, p31, k, THOMAS_C)
        i32 = _put_k(i32, p32, k, THOMAS_C)
        i33 = _put_k(i33, p33, k, THOMAS_C)
        iu0 = _put_k(iu0, pu0, k, THOMAS_C)
        iu1 = _put_k(iu1, pu1, k, THOMAS_C)
        iu2 = _put_k(iu2, pu2, k, THOMAS_C)
        iu3 = _put_k(iu3, pu3, k, THOMAS_C)
    (
        g00,
        g01,
        g02,
        g03,
        g10,
        g11,
        g12,
        g13,
        g20,
        g21,
        g22,
        g23,
        g30,
        g31,
        g32,
        g33,
        gu0,
        gu1,
        gu2,
        gu3,
    ) = tl.associative_scan(
        (
            p00,
            p01,
            p02,
            p03,
            p10,
            p11,
            p12,
            p13,
            p20,
            p21,
            p22,
            p23,
            p30,
            p31,
            p32,
            p33,
            pu0,
            pu1,
            pu2,
            pu3,
        ),
        0,
        _compose_block4,
    )
    offs_n = tl.arange(0, n_coarse)
    first = offs_n[:, None] == 0
    x00 = tl.where(first, 1.0, 0.0)
    x01 = tl.zeros_like(p00)
    x02 = tl.zeros_like(p00)
    x03 = tl.zeros_like(p00)
    x10 = tl.zeros_like(p00)
    x11 = tl.where(first, 1.0, 0.0)
    x12 = tl.zeros_like(p00)
    x13 = tl.zeros_like(p00)
    x20 = tl.zeros_like(p00)
    x21 = tl.zeros_like(p00)
    x22 = tl.where(first, 1.0, 0.0)
    x23 = tl.zeros_like(p00)
    x30 = tl.zeros_like(p00)
    x31 = tl.zeros_like(p00)
    x32 = tl.zeros_like(p00)
    x33 = tl.where(first, 1.0, 0.0)
    xu0 = tl.zeros_like(p00)
    xu1 = tl.zeros_like(p00)
    xu2 = tl.zeros_like(p00)
    xu3 = tl.zeros_like(p00)
    for n in tl.static_range(1, n_coarse):
        sel_d = offs_n[:, None] == n
        x00 = tl.where(sel_d, _row_n(g00, n - 1, BLOCK_D)[None, :], x00)
        x01 = tl.where(sel_d, _row_n(g01, n - 1, BLOCK_D)[None, :], x01)
        x02 = tl.where(sel_d, _row_n(g02, n - 1, BLOCK_D)[None, :], x02)
        x03 = tl.where(sel_d, _row_n(g03, n - 1, BLOCK_D)[None, :], x03)
        x10 = tl.where(sel_d, _row_n(g10, n - 1, BLOCK_D)[None, :], x10)
        x11 = tl.where(sel_d, _row_n(g11, n - 1, BLOCK_D)[None, :], x11)
        x12 = tl.where(sel_d, _row_n(g12, n - 1, BLOCK_D)[None, :], x12)
        x13 = tl.where(sel_d, _row_n(g13, n - 1, BLOCK_D)[None, :], x13)
        x20 = tl.where(sel_d, _row_n(g20, n - 1, BLOCK_D)[None, :], x20)
        x21 = tl.where(sel_d, _row_n(g21, n - 1, BLOCK_D)[None, :], x21)
        x22 = tl.where(sel_d, _row_n(g22, n - 1, BLOCK_D)[None, :], x22)
        x23 = tl.where(sel_d, _row_n(g23, n - 1, BLOCK_D)[None, :], x23)
        x30 = tl.where(sel_d, _row_n(g30, n - 1, BLOCK_D)[None, :], x30)
        x31 = tl.where(sel_d, _row_n(g31, n - 1, BLOCK_D)[None, :], x31)
        x32 = tl.where(sel_d, _row_n(g32, n - 1, BLOCK_D)[None, :], x32)
        x33 = tl.where(sel_d, _row_n(g33, n - 1, BLOCK_D)[None, :], x33)
        xu0 = tl.where(sel_d, _row_n(gu0, n - 1, BLOCK_D)[None, :], xu0)
        xu1 = tl.where(sel_d, _row_n(gu1, n - 1, BLOCK_D)[None, :], xu1)
        xu2 = tl.where(sel_d, _row_n(gu2, n - 1, BLOCK_D)[None, :], xu2)
        xu3 = tl.where(sel_d, _row_n(gu3, n - 1, BLOCK_D)[None, :], xu3)
    xb00 = x00[:, None, :]
    xb01 = x01[:, None, :]
    xb02 = x02[:, None, :]
    xb03 = x03[:, None, :]
    xb10 = x10[:, None, :]
    xb11 = x11[:, None, :]
    xb12 = x12[:, None, :]
    xb13 = x13[:, None, :]
    xb20 = x20[:, None, :]
    xb21 = x21[:, None, :]
    xb22 = x22[:, None, :]
    xb23 = x23[:, None, :]
    xb30 = x30[:, None, :]
    xb31 = x31[:, None, :]
    xb32 = x32[:, None, :]
    xb33 = x33[:, None, :]
    xbu0 = xu0[:, None, :]
    xbu1 = xu1[:, None, :]
    xbu2 = xu2[:, None, :]
    xbu3 = xu3[:, None, :]
    (
        o00,
        o01,
        o02,
        o03,
        o10,
        o11,
        o12,
        o13,
        o20,
        o21,
        o22,
        o23,
        o30,
        o31,
        o32,
        o33,
        ou0,
        ou1,
        ou2,
        ou3,
    ) = _compose_block4(
        xb00,
        xb01,
        xb02,
        xb03,
        xb10,
        xb11,
        xb12,
        xb13,
        xb20,
        xb21,
        xb22,
        xb23,
        xb30,
        xb31,
        xb32,
        xb33,
        xbu0,
        xbu1,
        xbu2,
        xbu3,
        i00,
        i01,
        i02,
        i03,
        i10,
        i11,
        i12,
        i13,
        i20,
        i21,
        i22,
        i23,
        i30,
        i31,
        i32,
        i33,
        iu0,
        iu1,
        iu2,
        iu3,
    )
    return (
        tl.reshape(o00, [BLOCK_T, BLOCK_D]),
        tl.reshape(o01, [BLOCK_T, BLOCK_D]),
        tl.reshape(o02, [BLOCK_T, BLOCK_D]),
        tl.reshape(o03, [BLOCK_T, BLOCK_D]),
        tl.reshape(o10, [BLOCK_T, BLOCK_D]),
        tl.reshape(o11, [BLOCK_T, BLOCK_D]),
        tl.reshape(o12, [BLOCK_T, BLOCK_D]),
        tl.reshape(o13, [BLOCK_T, BLOCK_D]),
        tl.reshape(o20, [BLOCK_T, BLOCK_D]),
        tl.reshape(o21, [BLOCK_T, BLOCK_D]),
        tl.reshape(o22, [BLOCK_T, BLOCK_D]),
        tl.reshape(o23, [BLOCK_T, BLOCK_D]),
        tl.reshape(o30, [BLOCK_T, BLOCK_D]),
        tl.reshape(o31, [BLOCK_T, BLOCK_D]),
        tl.reshape(o32, [BLOCK_T, BLOCK_D]),
        tl.reshape(o33, [BLOCK_T, BLOCK_D]),
        tl.reshape(ou0, [BLOCK_T, BLOCK_D]),
        tl.reshape(ou1, [BLOCK_T, BLOCK_D]),
        tl.reshape(ou2, [BLOCK_T, BLOCK_D]),
        tl.reshape(ou3, [BLOCK_T, BLOCK_D]),
    )


@triton.jit
def _scan_select_block4(
    j00,
    j01,
    j02,
    j03,
    j10,
    j11,
    j12,
    j13,
    j20,
    j21,
    j22,
    j23,
    j30,
    j31,
    j32,
    j33,
    r0,
    r1,
    r2,
    r3,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SEQ: tl.constexpr,
    THOMAS_C: tl.constexpr,
    LEAF_SPARSE: tl.constexpr,
):
    """Tile scan: Thomas+PCR, serial prefix, or associative (optional leaf pairs)."""
    if THOMAS_C >= 2:
        return _thomas_pcr_scan_block4(
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
            r0,
            r1,
            r2,
            r3,
            BLOCK_T,
            BLOCK_D,
            THOMAS_C,
        )
    if SEQ:
        return _seq_scan_block4(
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
            r0,
            r1,
            r2,
            r3,
            BLOCK_T,
            BLOCK_D,
        )
    if LEAF_SPARSE:
        return _assoc_scan_block4_leaf_pairs(
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
            r0,
            r1,
            r2,
            r3,
            BLOCK_T,
            BLOCK_D,
        )
    return tl.associative_scan(
        (
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
            r0,
            r1,
            r2,
            r3,
        ),
        0,
        _compose_block4,
    )


@triton.jit
def _slstm_pred_j(
    c_prev,
    n_prev,
    m_prev,
    h_prev,
    zi_x,
    zf_x,
    zz_x,
    zo_x,
    r_i,
    r_f,
    r_z,
    r_o,
    eps,
):
    """sLSTM step + 4x4 channelwise J. Ties on max split 0.5/0.5 (PyTorch)."""
    z_i = r_i * h_prev + zi_x
    z_f = r_f * h_prev + zf_x
    z_z = r_z * h_prev + zz_x
    z_o = r_o * h_prev + zo_x
    left = z_f + m_prev
    m_new = tl.where(left > z_i, left, z_i)
    alpha = tl.where(left > z_i, 1.0, 0.0) + tl.where(left == z_i, 0.5, 0.0)
    beta = 1.0 - alpha
    i_t = tl.exp(z_i - m_new)
    f_t = tl.exp(z_f + m_prev - m_new)
    z = _tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c_prev + i_t * z
    o = tl.sigmoid(z_o)
    denom = n_new + eps
    h_new = o * (c_new / denom)
    dm_dh = alpha * r_f + beta * r_i
    di_dm = -i_t * alpha
    df_dm = f_t * beta
    di_dh = i_t * (r_i - dm_dh)
    df_dh = f_t * (r_f - dm_dh)
    dz_dh = (1.0 - z * z) * r_z
    do_dh = o * (1.0 - o) * r_o
    j_cc = f_t
    j_cn = 0.0
    j_cm = df_dm * c_prev + di_dm * z
    j_ch = df_dh * c_prev + di_dh * z + i_t * dz_dh
    j_nc = 0.0
    j_nn = f_t
    j_nm = df_dm * n_prev + di_dm
    j_nh = df_dh * n_prev + di_dh
    j_mc = 0.0
    j_mn = 0.0
    j_mm = alpha
    j_mh = dm_dh
    inv = o / denom
    dn = -o * c_new / (denom * denom)
    du = c_new / denom
    j_hc = inv * j_cc
    j_hn = dn * j_nn
    j_hm = inv * j_cm + dn * j_nm
    j_hh = inv * j_ch + dn * j_nh + du * do_dh
    return (
        c_new,
        n_new,
        m_new,
        h_new,
        j_cc,
        j_cn,
        j_cm,
        j_ch,
        j_nc,
        j_nn,
        j_nm,
        j_nh,
        j_mc,
        j_mn,
        j_mm,
        j_mh,
        j_hc,
        j_hn,
        j_hm,
        j_hh,
    )


@triton.jit
def _slstm_log_pred_j(
    u_prev,
    ln_prev,
    m_prev,
    h_prev,
    zi_x,
    zf_x,
    zz_x,
    zo_x,
    r_i,
    r_f,
    r_z,
    r_o,
):
    """Convex combo + LSE step and 4x4 J. Slots are ``(u, log n, m, h)``."""
    z_i = r_i * h_prev + zi_x
    z_f = r_f * h_prev + zf_x
    z_z = r_z * h_prev + zz_x
    z_o = r_o * h_prev + zo_x
    left = z_f + m_prev
    m_new = tl.where(left > z_i, left, z_i)
    alpha = tl.where(left > z_i, 1.0, 0.0) + tl.where(left == z_i, 0.5, 0.0)
    beta = 1.0 - alpha
    a = z_f + m_prev - m_new + ln_prev
    b = z_i - m_new
    mx = tl.maximum(a, b)
    ln_new = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))
    gamma = tl.exp(b - ln_new)
    z = _tanh(z_z)
    omg = 1.0 - gamma
    u_new = omg * u_prev + gamma * z
    o = tl.sigmoid(z_o)
    h_new = o * u_new
    dm_dh = alpha * r_f + beta * r_i
    da_dh = r_f - dm_dh
    db_dh = r_i - dm_dh
    da_dm = beta
    db_dm = -alpha
    dln_dln = omg
    dln_dm = omg * da_dm + gamma * db_dm
    dln_dh = omg * da_dh + gamma * db_dh
    dgamma_dln = gamma * (0.0 - dln_dln)
    dgamma_dm = gamma * (db_dm - dln_dm)
    dgamma_dh = gamma * (db_dh - dln_dh)
    dz_dh = (1.0 - z * z) * r_z
    uz = z - u_prev
    du_du = omg
    du_dln = uz * dgamma_dln
    du_dm = uz * dgamma_dm
    du_dh = uz * dgamma_dh + gamma * dz_dh
    do_dh = o * (1.0 - o) * r_o
    j_uu = du_du
    j_uln = du_dln
    j_um = du_dm
    j_uh = du_dh
    j_lnu = 0.0
    j_lnln = dln_dln
    j_lnm = dln_dm
    j_lnh = dln_dh
    j_mu = 0.0
    j_mn = 0.0
    j_mm = alpha
    j_mh = dm_dh
    j_hu = o * du_du
    j_hln = o * du_dln
    j_hm = o * du_dm
    j_hh = o * du_dh + u_new * do_dh
    return (
        u_new,
        ln_new,
        m_new,
        h_new,
        j_uu,
        j_uln,
        j_um,
        j_uh,
        j_lnu,
        j_lnln,
        j_lnm,
        j_lnh,
        j_mu,
        j_mn,
        j_mm,
        j_mh,
        j_hu,
        j_hln,
        j_hm,
        j_hh,
    )


@triton.jit
def _load_wx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, sb, st, sd):
    base = wx_ptr + pid_b * sb + offs_t[:, None] * st
    zi = load_acc(base + offs_d[None, :] * sd, mask, 0.0)
    zf = load_acc(base + (offs_d[None, :] + d_h) * sd, mask, 0.0)
    zz = load_acc(base + (offs_d[None, :] + 2 * d_h) * sd, mask, 0.0)
    zo = load_acc(base + (offs_d[None, :] + 3 * d_h) * sd, mask, 0.0)
    return zi, zf, zz, zo


@triton.jit
def _load_r_gate(r_ptr, gate, offs_d, dmask, sg, sd):
    return load_acc(r_ptr + gate * sg + offs_d * sd, dmask, 0.0)


@triton.jit
def _slstm_init_kernel(
    wx_ptr,
    s_ptr,
    r_ptr,
    h0_ptr,
    d_h,
    time,
    eps,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_rg,
    stride_rd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    bt_ptr,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """App. A: only t=0 sees ``h0``; later t still ``f(0, x_t)``."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    zi, zf, zz, zo = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    r_i = _load_r_gate(r_ptr, 0, offs_d, dmask, stride_rg, stride_rd)
    r_f = _load_r_gate(r_ptr, 1, offs_d, dmask, stride_rg, stride_rd)
    r_z = _load_r_gate(r_ptr, 2, offs_d, dmask, stride_rg, stride_rd)
    r_o = _load_r_gate(r_ptr, 3, offs_d, dmask, stride_rg, stride_rd)
    c_prev = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    n_prev = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    m_prev = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    h_prev = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    if pid_c == 0:
        c0 = _load_h0(
            h0_ptr, pid_b, offs_d, SLOT_C, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
        )
        n0 = _load_h0(
            h0_ptr, pid_b, offs_d, SLOT_N, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
        )
        m0 = _load_h0(
            h0_ptr, pid_b, offs_d, SLOT_M, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
        )
        h0 = _load_h0(
            h0_ptr, pid_b, offs_d, SLOT_H, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
        )
        is_t0 = (offs_t == 0)[:, None]
        c_prev = tl.where(is_t0, c0[None, :], c_prev)
        n_prev = tl.where(is_t0, n0[None, :], n_prev)
        m_prev = tl.where(is_t0, m0[None, :], m_prev)
        h_prev = tl.where(is_t0, h0[None, :], h_prev)
    c, n, m, h, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _slstm_pred_j(
        c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o, eps
    )
    _store_state(
        s_ptr, c, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    _store_state(
        s_ptr, n, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    _store_state(
        s_ptr, m, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    _store_state(
        s_ptr, h, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )


@triton.jit
def _slstm_tile_prev_from_loaded(
    c,
    n,
    m,
    h,
    s_ptr,
    h0_ptr,
    pid_b,
    pid_c,
    t0,
    offs_d,
    dmask,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    bt_ptr,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """Build ``S(t-1)`` from one tile load of ``S(t)``.

    Interior rows: register shift. Tile head: ``h0`` when ``pid_c==0``, else one
    HBM row at ``t0-1`` (later chunks skip the unused ``h0`` load).
    """
    local_t = tl.arange(0, BLOCK_T)
    idx_prev = tl.maximum(local_t - 1, 0)[None, :] + tl.zeros((BLOCK_D, 1), dtype=tl.int32)
    c_prev = tl.trans(tl.gather(tl.trans(c), idx_prev, axis=1))
    n_prev = tl.trans(tl.gather(tl.trans(n), idx_prev, axis=1))
    m_prev = tl.trans(tl.gather(tl.trans(m), idx_prev, axis=1))
    h_prev = tl.trans(tl.gather(tl.trans(h), idx_prev, axis=1))
    is_head = (local_t == 0)[:, None]
    if pid_c == 0:
        c0 = _load_h0(
            h0_ptr, pid_b, offs_d, SLOT_C, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
        )
        n0 = _load_h0(
            h0_ptr, pid_b, offs_d, SLOT_N, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
        )
        m0 = _load_h0(
            h0_ptr, pid_b, offs_d, SLOT_M, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
        )
        h0 = _load_h0(
            h0_ptr, pid_b, offs_d, SLOT_H, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
        )
        c_prev = tl.where(is_head, c0[None, :], c_prev)
        n_prev = tl.where(is_head, n0[None, :], n_prev)
        m_prev = tl.where(is_head, m0[None, :], m_prev)
        h_prev = tl.where(is_head, h0[None, :], h_prev)
    else:
        offs_b = tl.zeros((1,), dtype=tl.int32) + (t0 - 1)
        mask_b = dmask[None, :]
        c_prev = tl.where(
            is_head,
            _load_state(
                s_ptr,
                pid_b,
                offs_b,
                offs_d,
                SLOT_C,
                mask_b,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            ),
            c_prev,
        )
        n_prev = tl.where(
            is_head,
            _load_state(
                s_ptr,
                pid_b,
                offs_b,
                offs_d,
                SLOT_N,
                mask_b,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            ),
            n_prev,
        )
        m_prev = tl.where(
            is_head,
            _load_state(
                s_ptr,
                pid_b,
                offs_b,
                offs_d,
                SLOT_M,
                mask_b,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            ),
            m_prev,
        )
        h_prev = tl.where(
            is_head,
            _load_state(
                s_ptr,
                pid_b,
                offs_b,
                offs_d,
                SLOT_H,
                mask_b,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            ),
            h_prev,
        )
    return c_prev, n_prev, m_prev, h_prev


@triton.jit
def _slstm_cell_local_scan_kernel(
    s_ptr,
    wx_ptr,
    r_ptr,
    h0_ptr,
    j_loc_ptr,
    r_loc_ptr,
    agg_j_ptr,
    agg_r_ptr,
    incl_r_ptr,
    time,
    d_h,
    eps,
    omega,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_rg,
    stride_rd,
    stride_jb,
    stride_jt,
    stride_jk,
    stride_jd,
    stride_rb,
    stride_rt,
    stride_rs,
    stride_rd_s,
    stride_ajb,
    stride_ajc,
    stride_ajk,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ars,
    stride_ard,
    stride_ib,
    stride_ic,
    stride_is,
    stride_id,
    bt_ptr,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOG: tl.constexpr,
    SEQ: tl.constexpr,
    THOMAS_C: tl.constexpr,
    HAS_BT: tl.constexpr,
    APPLY_OMEGA: tl.constexpr,
    APPLY_CARRY: tl.constexpr,
    STORE_JR: tl.constexpr,
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
    c = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    n = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    m = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    h = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    c_prev, n_prev, m_prev, h_prev = _slstm_tile_prev_from_loaded(
        c,
        n,
        m,
        h,
        s_ptr,
        h0_ptr,
        pid_b,
        pid_c,
        t0,
        offs_d,
        dmask,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
        stride_h0b,
        stride_h0s,
        stride_h0d,
        bt_ptr,
        SLOT_C,
        SLOT_N,
        SLOT_M,
        SLOT_H,
        BLOCK_T,
        BLOCK_D,
        HAS_BT,
    )
    zi, zf, zz, zo = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    r_i = _load_r_gate(r_ptr, 0, offs_d, dmask, stride_rg, stride_rd)
    r_f = _load_r_gate(r_ptr, 1, offs_d, dmask, stride_rg, stride_rd)
    r_z = _load_r_gate(r_ptr, 2, offs_d, dmask, stride_rg, stride_rd)
    r_o = _load_r_gate(r_ptr, 3, offs_d, dmask, stride_rg, stride_rd)
    if LOG:
        (
            c_new,
            n_new,
            m_new,
            h_new,
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
        ) = _slstm_log_pred_j(c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o)
    else:
        (
            c_new,
            n_new,
            m_new,
            h_new,
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
        ) = _slstm_pred_j(c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o, eps)
    r0 = tl.where(mask, c_new - c, 0.0)
    r1 = tl.where(mask, n_new - n, 0.0)
    r2 = tl.where(mask, m_new - m, 0.0)
    r3 = tl.where(mask, h_new - h, 0.0)
    j00 = tl.where(mask, j00, 1.0)
    j01 = tl.where(mask, j01, 0.0)
    j02 = tl.where(mask, j02, 0.0)
    j03 = tl.where(mask, j03, 0.0)
    j10 = tl.where(mask, j10, 0.0)
    j11 = tl.where(mask, j11, 1.0)
    j12 = tl.where(mask, j12, 0.0)
    j13 = tl.where(mask, j13, 0.0)
    j20 = tl.where(mask, j20, 0.0)
    j21 = tl.where(mask, j21, 0.0)
    j22 = tl.where(mask, j22, 1.0)
    j23 = tl.where(mask, j23, 0.0)
    j30 = tl.where(mask, j30, 0.0)
    j31 = tl.where(mask, j31, 0.0)
    j32 = tl.where(mask, j32, 0.0)
    j33 = tl.where(mask, j33, 1.0)
    (
        s00,
        s01,
        s02,
        s03,
        s10,
        s11,
        s12,
        s13,
        s20,
        s21,
        s22,
        s23,
        s30,
        s31,
        s32,
        s33,
        u0,
        u1,
        u2,
        u3,
    ) = _scan_select_block4(
        j00,
        j01,
        j02,
        j03,
        j10,
        j11,
        j12,
        j13,
        j20,
        j21,
        j22,
        j23,
        j30,
        j31,
        j32,
        j33,
        r0,
        r1,
        r2,
        r3,
        BLOCK_T,
        BLOCK_D,
        SEQ,
        THOMAS_C,
        LEAF_SPARSE=(not LOG),
    )
    if APPLY_OMEGA:
        # Single-chunk: scanned residual is the Newton δ — apply ω in-kernel.
        _store_state(
            s_ptr,
            c + omega * u0,
            pid_b,
            offs_t,
            offs_d,
            SLOT_C,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
        _store_state(
            s_ptr,
            n + omega * u1,
            pid_b,
            offs_t,
            offs_d,
            SLOT_N,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
        _store_state(
            s_ptr,
            m + omega * u2,
            pid_b,
            offs_t,
            offs_d,
            SLOT_M,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
        _store_state(
            s_ptr,
            h + omega * u3,
            pid_b,
            offs_t,
            offs_d,
            SLOT_H,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
    elif APPLY_CARRY:
        # Multi-chunk: recompute local scan in-regs; δ = J_loc @ carry + u.
        idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
        c0 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + idx_c * stride_ic
            + SLOT_C * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c1 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + idx_c * stride_ic
            + SLOT_N * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c2 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + idx_c * stride_ic
            + SLOT_M * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c3 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + idx_c * stride_ic
            + SLOT_H * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c0 = tl.where(pid_c > 0, c0, 0.0)
        c1 = tl.where(pid_c > 0, c1, 0.0)
        c2 = tl.where(pid_c > 0, c2, 0.0)
        c3 = tl.where(pid_c > 0, c3, 0.0)
        delta_c = s00 * c0[None, :] + s01 * c1[None, :] + s02 * c2[None, :] + s03 * c3[None, :] + u0
        delta_n = s10 * c0[None, :] + s11 * c1[None, :] + s12 * c2[None, :] + s13 * c3[None, :] + u1
        delta_m = s20 * c0[None, :] + s21 * c1[None, :] + s22 * c2[None, :] + s23 * c3[None, :] + u2
        delta_h = s30 * c0[None, :] + s31 * c1[None, :] + s32 * c2[None, :] + s33 * c3[None, :] + u3
        _store_state(
            s_ptr,
            c + omega * delta_c,
            pid_b,
            offs_t,
            offs_d,
            SLOT_C,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
        _store_state(
            s_ptr,
            n + omega * delta_n,
            pid_b,
            offs_t,
            offs_d,
            SLOT_N,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
        _store_state(
            s_ptr,
            m + omega * delta_m,
            pid_b,
            offs_t,
            offs_d,
            SLOT_M,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
        _store_state(
            s_ptr,
            h + omega * delta_h,
            pid_b,
            offs_t,
            offs_d,
            SLOT_H,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
    else:
        if STORE_JR:
            _store_j_lane(
                j_loc_ptr, s00, pid_b, offs_t, offs_d, 0, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s01, pid_b, offs_t, offs_d, 1, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s02, pid_b, offs_t, offs_d, 2, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s03, pid_b, offs_t, offs_d, 3, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s10, pid_b, offs_t, offs_d, 4, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s11, pid_b, offs_t, offs_d, 5, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s12, pid_b, offs_t, offs_d, 6, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s13, pid_b, offs_t, offs_d, 7, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s20, pid_b, offs_t, offs_d, 8, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s21, pid_b, offs_t, offs_d, 9, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s22, pid_b, offs_t, offs_d, 10, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s23, pid_b, offs_t, offs_d, 11, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s30, pid_b, offs_t, offs_d, 12, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s31, pid_b, offs_t, offs_d, 13, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s32, pid_b, offs_t, offs_d, 14, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_j_lane(
                j_loc_ptr, s33, pid_b, offs_t, offs_d, 15, mask, stride_jb, stride_jt, stride_jk, stride_jd
            )
            _store_state(
                r_loc_ptr,
                u0,
                pid_b,
                offs_t,
                offs_d,
                SLOT_C,
                mask,
                stride_rb,
                stride_rt,
                stride_rs,
                stride_rd_s,
            )
            _store_state(
                r_loc_ptr,
                u1,
                pid_b,
                offs_t,
                offs_d,
                SLOT_N,
                mask,
                stride_rb,
                stride_rt,
                stride_rs,
                stride_rd_s,
            )
            _store_state(
                r_loc_ptr,
                u2,
                pid_b,
                offs_t,
                offs_d,
                SLOT_M,
                mask,
                stride_rb,
                stride_rt,
                stride_rs,
                stride_rd_s,
            )
            _store_state(
                r_loc_ptr,
                u3,
                pid_b,
                offs_t,
                offs_d,
                SLOT_H,
                mask,
                stride_rb,
                stride_rt,
                stride_rs,
                stride_rd_s,
            )
        last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s00, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            0,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s01, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            1,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s02, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            2,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s03, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            3,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s10, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            4,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s11, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            5,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s12, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            6,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s13, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            7,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s20, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            8,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s21, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            9,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s22, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            10,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s23, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            11,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s30, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            12,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s31, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            13,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s32, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            14,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        _store_agg_j(
            agg_j_ptr,
            tl.sum(tl.where(last, s33, 0.0), axis=0),
            pid_b,
            pid_c,
            offs_d,
            15,
            dmask,
            stride_ajb,
            stride_ajc,
            stride_ajk,
            stride_ajd,
        )
        store_acc(
            agg_r_ptr
            + pid_b * stride_arb
            + pid_c * stride_arc
            + SLOT_C * stride_ars
            + offs_d * stride_ard,
            tl.sum(tl.where(last, u0, 0.0), axis=0),
            dmask,
        )
        store_acc(
            agg_r_ptr
            + pid_b * stride_arb
            + pid_c * stride_arc
            + SLOT_N * stride_ars
            + offs_d * stride_ard,
            tl.sum(tl.where(last, u1, 0.0), axis=0),
            dmask,
        )
        store_acc(
            agg_r_ptr
            + pid_b * stride_arb
            + pid_c * stride_arc
            + SLOT_M * stride_ars
            + offs_d * stride_ard,
            tl.sum(tl.where(last, u2, 0.0), axis=0),
            dmask,
        )
        store_acc(
            agg_r_ptr
            + pid_b * stride_arb
            + pid_c * stride_arc
            + SLOT_H * stride_ars
            + offs_d * stride_ard,
            tl.sum(tl.where(last, u3, 0.0), axis=0),
            dmask,
        )


@triton.jit
def _chunk_incl_kernel(
    agg_j_ptr,
    agg_r_ptr,
    incl_r_ptr,
    n_chunks,
    d_h,
    stride_ajb,
    stride_ajc,
    stride_ajk,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ars,
    stride_ard,
    stride_ib,
    stride_ic,
    stride_is,
    stride_id,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    j00 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        0,
        1.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j01 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        1,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j02 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        2,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j03 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        3,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j10 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        4,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j11 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        5,
        1.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j12 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        6,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j13 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        7,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j20 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        8,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j21 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        9,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j22 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        10,
        1.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j23 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        11,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j30 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        12,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j31 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        13,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j32 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        14,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j33 = _load_j_lane(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        15,
        1.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    u0 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 0, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u1 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 1, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u2 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 2, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u3 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 3, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        s0,
        s1,
        s2,
        s3,
    ) = tl.associative_scan(
        (
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
            u0,
            u1,
            u2,
            u3,
        ),
        0,
        _compose_block4,
    )
    _store_state(
        incl_r_ptr, s0, pid_b, offs_c, offs_d, 0, mask, stride_ib, stride_ic, stride_is, stride_id
    )
    _store_state(
        incl_r_ptr, s1, pid_b, offs_c, offs_d, 1, mask, stride_ib, stride_ic, stride_is, stride_id
    )
    _store_state(
        incl_r_ptr, s2, pid_b, offs_c, offs_d, 2, mask, stride_ib, stride_ic, stride_is, stride_id
    )
    _store_state(
        incl_r_ptr, s3, pid_b, offs_c, offs_d, 3, mask, stride_ib, stride_ic, stride_is, stride_id
    )


@triton.jit
def _chunk_monoid_ends_kernel(
    agg_j_ptr,
    agg_r_ptr,
    p_end_ptr,
    d_end_ptr,
    n_chunks,
    d_h,
    stride_ajb,
    stride_ajc,
    stride_ajk,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ars,
    stride_ard,
    stride_pb,
    stride_pk,
    stride_pd,
    stride_db,
    stride_ds,
    stride_dd,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """SRAM chunk scan → tile-end monoid ``(P_end, δ_end)`` only."""
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    j00 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 0, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j01 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 1, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j02 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 2, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j03 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 3, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j10 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 4, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j11 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 5, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j12 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 6, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j13 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 7, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j20 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 8, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j21 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 9, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j22 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 10, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j23 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 11, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j30 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 12, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j31 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 13, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j32 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 14, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j33 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 15, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    u0 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 0, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u1 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 1, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u2 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 2, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u3 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 3, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    (
        p00,
        p01,
        p02,
        p03,
        p10,
        p11,
        p12,
        p13,
        p20,
        p21,
        p22,
        p23,
        p30,
        p31,
        p32,
        p33,
        s0,
        s1,
        s2,
        s3,
    ) = tl.associative_scan(
        (
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
            u0,
            u1,
            u2,
            u3,
        ),
        0,
        _compose_block4,
    )
    last = n_chunks - 1
    sel = offs_c[:, None] == last
    store_acc(
        p_end_ptr + pid_b * stride_pb + 0 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p00, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 1 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p01, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 2 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p02, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 3 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p03, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 4 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p10, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 5 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p11, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 6 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p12, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 7 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p13, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 8 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p20, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 9 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p21, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 10 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p22, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 11 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p23, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 12 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p30, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 13 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p31, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 14 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p32, 0.0), axis=0),
        dmask,
    )
    store_acc(
        p_end_ptr + pid_b * stride_pb + 15 * stride_pk + offs_d * stride_pd,
        tl.sum(tl.where(sel, p33, 0.0), axis=0),
        dmask,
    )
    store_acc(
        d_end_ptr + pid_b * stride_db + 0 * stride_ds + offs_d * stride_dd,
        tl.sum(tl.where(sel, s0, 0.0), axis=0),
        dmask,
    )
    store_acc(
        d_end_ptr + pid_b * stride_db + 1 * stride_ds + offs_d * stride_dd,
        tl.sum(tl.where(sel, s1, 0.0), axis=0),
        dmask,
    )
    store_acc(
        d_end_ptr + pid_b * stride_db + 2 * stride_ds + offs_d * stride_dd,
        tl.sum(tl.where(sel, s2, 0.0), axis=0),
        dmask,
    )
    store_acc(
        d_end_ptr + pid_b * stride_db + 3 * stride_ds + offs_d * stride_dd,
        tl.sum(tl.where(sel, s3, 0.0), axis=0),
        dmask,
    )


@triton.jit
def _chunk_carries_rank_kernel(
    agg_j_ptr,
    agg_r_ptr,
    rank_ptr,
    carry_ptr,
    n_chunks,
    d_h,
    stride_ajb,
    stride_ajc,
    stride_ajk,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ars,
    stride_ard,
    stride_rb,
    stride_rs,
    stride_rd,
    stride_cb,
    stride_cc,
    stride_cs,
    stride_cd,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Exclusive leaf carries: ``P_excl @ rank_carry + δ_excl``."""
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    j00 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 0, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j01 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 1, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j02 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 2, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j03 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 3, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j10 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 4, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j11 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 5, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j12 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 6, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j13 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 7, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j20 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 8, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j21 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 9, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j22 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 10, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j23 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 11, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j30 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 12, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j31 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 13, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j32 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 14, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    j33 = _load_j_lane(
        agg_j_ptr, pid_b, offs_c, offs_d, 15, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd
    )
    u0 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 0, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u1 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 1, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u2 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 2, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u3 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 3, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    (
        p00,
        p01,
        p02,
        p03,
        p10,
        p11,
        p12,
        p13,
        p20,
        p21,
        p22,
        p23,
        p30,
        p31,
        p32,
        p33,
        s0,
        s1,
        s2,
        s3,
    ) = tl.associative_scan(
        (
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
            u0,
            u1,
            u2,
            u3,
        ),
        0,
        _compose_block4,
    )
    e00 = _exclusive_shift_row(p00, 1.0, CHUNK_PAD, BLOCK_D)
    e01 = _exclusive_shift_row(p01, 0.0, CHUNK_PAD, BLOCK_D)
    e02 = _exclusive_shift_row(p02, 0.0, CHUNK_PAD, BLOCK_D)
    e03 = _exclusive_shift_row(p03, 0.0, CHUNK_PAD, BLOCK_D)
    e10 = _exclusive_shift_row(p10, 0.0, CHUNK_PAD, BLOCK_D)
    e11 = _exclusive_shift_row(p11, 1.0, CHUNK_PAD, BLOCK_D)
    e12 = _exclusive_shift_row(p12, 0.0, CHUNK_PAD, BLOCK_D)
    e13 = _exclusive_shift_row(p13, 0.0, CHUNK_PAD, BLOCK_D)
    e20 = _exclusive_shift_row(p20, 0.0, CHUNK_PAD, BLOCK_D)
    e21 = _exclusive_shift_row(p21, 0.0, CHUNK_PAD, BLOCK_D)
    e22 = _exclusive_shift_row(p22, 1.0, CHUNK_PAD, BLOCK_D)
    e23 = _exclusive_shift_row(p23, 0.0, CHUNK_PAD, BLOCK_D)
    e30 = _exclusive_shift_row(p30, 0.0, CHUNK_PAD, BLOCK_D)
    e31 = _exclusive_shift_row(p31, 0.0, CHUNK_PAD, BLOCK_D)
    e32 = _exclusive_shift_row(p32, 0.0, CHUNK_PAD, BLOCK_D)
    e33 = _exclusive_shift_row(p33, 1.0, CHUNK_PAD, BLOCK_D)
    eu0 = _exclusive_shift_row(s0, 0.0, CHUNK_PAD, BLOCK_D)
    eu1 = _exclusive_shift_row(s1, 0.0, CHUNK_PAD, BLOCK_D)
    eu2 = _exclusive_shift_row(s2, 0.0, CHUNK_PAD, BLOCK_D)
    eu3 = _exclusive_shift_row(s3, 0.0, CHUNK_PAD, BLOCK_D)
    rc0 = load_acc(rank_ptr + pid_b * stride_rb + 0 * stride_rs + offs_d * stride_rd, dmask, 0.0)
    rc1 = load_acc(rank_ptr + pid_b * stride_rb + 1 * stride_rs + offs_d * stride_rd, dmask, 0.0)
    rc2 = load_acc(rank_ptr + pid_b * stride_rb + 2 * stride_rs + offs_d * stride_rd, dmask, 0.0)
    rc3 = load_acc(rank_ptr + pid_b * stride_rb + 3 * stride_rs + offs_d * stride_rd, dmask, 0.0)
    c0 = e00 * rc0[None, :] + e01 * rc1[None, :] + e02 * rc2[None, :] + e03 * rc3[None, :] + eu0
    c1 = e10 * rc0[None, :] + e11 * rc1[None, :] + e12 * rc2[None, :] + e13 * rc3[None, :] + eu1
    c2 = e20 * rc0[None, :] + e21 * rc1[None, :] + e22 * rc2[None, :] + e23 * rc3[None, :] + eu2
    c3 = e30 * rc0[None, :] + e31 * rc1[None, :] + e32 * rc2[None, :] + e33 * rc3[None, :] + eu3
    _store_state(
        carry_ptr, c0, pid_b, offs_c, offs_d, 0, mask, stride_cb, stride_cc, stride_cs, stride_cd
    )
    _store_state(
        carry_ptr, c1, pid_b, offs_c, offs_d, 1, mask, stride_cb, stride_cc, stride_cs, stride_cd
    )
    _store_state(
        carry_ptr, c2, pid_b, offs_c, offs_d, 2, mask, stride_cb, stride_cc, stride_cs, stride_cd
    )
    _store_state(
        carry_ptr, c3, pid_b, offs_c, offs_d, 3, mask, stride_cb, stride_cc, stride_cs, stride_cd
    )


@triton.jit
def _slstm_apply_update_kernel(
    s_ptr,
    j_loc_ptr,
    r_loc_ptr,
    incl_r_ptr,
    time,
    d_h,
    omega,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_jb,
    stride_jt,
    stride_jk,
    stride_jd,
    stride_rb,
    stride_rt,
    stride_rs,
    stride_rd,
    stride_ib,
    stride_ic,
    stride_is,
    stride_id,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CARRY_BUF: tl.constexpr,
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
    j00 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 0, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j01 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 1, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j02 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 2, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j03 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 3, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j10 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 4, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j11 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 5, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j12 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 6, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j13 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 7, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j20 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 8, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j21 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 9, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j22 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 10, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j23 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 11, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j30 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 12, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j31 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 13, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j32 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 14, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j33 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 15, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    r0 = _load_state(
        r_loc_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    r1 = _load_state(
        r_loc_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    r2 = _load_state(
        r_loc_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    r3 = _load_state(
        r_loc_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    if HAS_CARRY_BUF:
        # Exclusive carry for this chunk (inter-rank + within-tile prefix).
        c0 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + pid_c * stride_ic
            + SLOT_C * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c1 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + pid_c * stride_ic
            + SLOT_N * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c2 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + pid_c * stride_ic
            + SLOT_M * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c3 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + pid_c * stride_ic
            + SLOT_H * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
    else:
        idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
        c0 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + idx_c * stride_ic
            + SLOT_C * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c1 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + idx_c * stride_ic
            + SLOT_N * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c2 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + idx_c * stride_ic
            + SLOT_M * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c3 = load_acc(
            incl_r_ptr
            + pid_b * stride_ib
            + idx_c * stride_ic
            + SLOT_H * stride_is
            + offs_d * stride_id,
            dmask,
            0.0,
        )
        c0 = tl.where(pid_c > 0, c0, 0.0)
        c1 = tl.where(pid_c > 0, c1, 0.0)
        c2 = tl.where(pid_c > 0, c2, 0.0)
        c3 = tl.where(pid_c > 0, c3, 0.0)
    delta_c = j00 * c0[None, :] + j01 * c1[None, :] + j02 * c2[None, :] + j03 * c3[None, :] + r0
    delta_n = j10 * c0[None, :] + j11 * c1[None, :] + j12 * c2[None, :] + j13 * c3[None, :] + r1
    delta_m = j20 * c0[None, :] + j21 * c1[None, :] + j22 * c2[None, :] + j23 * c3[None, :] + r2
    delta_h = j30 * c0[None, :] + j31 * c1[None, :] + j32 * c2[None, :] + j33 * c3[None, :] + r3
    c = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    n = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    m = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    h = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    _store_state(
        s_ptr,
        c + omega * delta_c,
        pid_b,
        offs_t,
        offs_d,
        SLOT_C,
        mask,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )
    _store_state(
        s_ptr,
        n + omega * delta_n,
        pid_b,
        offs_t,
        offs_d,
        SLOT_N,
        mask,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )
    _store_state(
        s_ptr,
        m + omega * delta_m,
        pid_b,
        offs_t,
        offs_d,
        SLOT_M,
        mask,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )
    _store_state(
        s_ptr,
        h + omega * delta_h,
        pid_b,
        offs_t,
        offs_d,
        SLOT_H,
        mask,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )


@triton.jit
def _slstm_window_walk_kernel(
    s_ptr,
    wx_ptr,
    r_ptr,
    h0_ptr,
    time,
    d_h,
    eps,
    omega,
    n_tiles,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_rg,
    stride_rd,
    bt_ptr,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SEQ: tl.constexpr,
    THOMAS_C: tl.constexpr,
    MAX_ITERS: tl.constexpr,
    N_PICARD: tl.constexpr,
    DO_PICARD: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """Windowed Newton: sequential ``BLOCK_T`` tiles, solved-state carry in regs.

    Grid is ``(B, n_dtiles)``. Each program walks time; diag mix is per-feature
    so tiles of different ``d`` do not share a carry. Same contract as
    ``chunk_len=BLOCK_T``: optional ``1+N_PICARD`` frozen-gate passes (Picard
    with Newton-solved left carry), then K Newton steps on a tile (state in
    registers — one HBM write after the K-loop), then the next tile uses the
    solved last state as ``h_{t-1}``.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    d0 = pid_d * BLOCK_D
    offs_d = d0 + tl.arange(0, BLOCK_D)
    dmask = offs_d < d_h
    r_i = _load_r_gate(r_ptr, 0, offs_d, dmask, stride_rg, stride_rd)
    r_f = _load_r_gate(r_ptr, 1, offs_d, dmask, stride_rg, stride_rd)
    r_z = _load_r_gate(r_ptr, 2, offs_d, dmask, stride_rg, stride_rd)
    r_o = _load_r_gate(r_ptr, 3, offs_d, dmask, stride_rg, stride_rd)
    carry_c = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_C, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    carry_n = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_N, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    carry_m = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_M, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    carry_h = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_H, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    local_t = tl.arange(0, BLOCK_T)
    for pid_c in tl.range(n_tiles):
        t0 = pid_c * BLOCK_T
        offs_t = t0 + local_t
        mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
        zi, zf, zz, zo = _load_wx(
            wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
        )
        rem = time - t0
        if DO_PICARD:
            # Picard on this tile with Newton-solved carry (chunk_len contract).
            # Intermediate passes keep only ``h`` (Rh frozen-gate). Full
            # ``(c,n,m,h)`` is materialized on the last pass for Newton —
            # static specialization drops the three trajectories from live
            # ranges on earlier passes (register / occupancy pressure).
            h = tl.zeros((BLOCK_T, BLOCK_D), dtype=zi.dtype)
            for p in tl.static_range(N_PICARD):
                rh_scale = 0.0 if p == 0 else 1.0
                c_run = carry_c
                n_run = carry_n
                m_run = carry_m
                h_pass = tl.zeros((BLOCK_T, BLOCK_D), dtype=zi.dtype)
                # Unrolled ``range``: Picard is step-serial (carry dep); loop
                # form (``tl.range``+``_row_sel``) was ~3× slower on 3060.
                for ti in range(BLOCK_T):
                    in_t = ti < rem
                    zi_t = _row_n(zi, ti, BLOCK_D)
                    zf_t = _row_n(zf, ti, BLOCK_D)
                    zz_t = _row_n(zz, ti, BLOCK_D)
                    zo_t = _row_n(zo, ti, BLOCK_D)
                    h_mix = carry_h
                    if ti > 0:
                        h_mix = _row_n(h, ti - 1, BLOCK_D)
                    c_new, n_new, m_new, h_new = _slstm_fwd_step(
                        c_run,
                        n_run,
                        m_run,
                        h_mix,
                        zi_t,
                        zf_t,
                        zz_t,
                        zo_t,
                        r_i,
                        r_f,
                        r_z,
                        r_o,
                        eps,
                        rh_scale,
                    )
                    c_run = tl.where(in_t, c_new, c_run)
                    n_run = tl.where(in_t, n_new, n_run)
                    m_run = tl.where(in_t, m_new, m_run)
                    h_pass = tl.where(in_t, _put_row(h_pass, h_new, ti, BLOCK_T), h_pass)
                h = h_pass
            # Last Picard pass: land full state for the Newton K-loop.
            rh_scale = 0.0 if N_PICARD == 0 else 1.0
            c = tl.zeros((BLOCK_T, BLOCK_D), dtype=zi.dtype)
            n = tl.zeros((BLOCK_T, BLOCK_D), dtype=zi.dtype)
            m = tl.zeros((BLOCK_T, BLOCK_D), dtype=zi.dtype)
            h_pass = tl.zeros((BLOCK_T, BLOCK_D), dtype=zi.dtype)
            c_run = carry_c
            n_run = carry_n
            m_run = carry_m
            for ti in range(BLOCK_T):
                in_t = ti < rem
                zi_t = _row_n(zi, ti, BLOCK_D)
                zf_t = _row_n(zf, ti, BLOCK_D)
                zz_t = _row_n(zz, ti, BLOCK_D)
                zo_t = _row_n(zo, ti, BLOCK_D)
                h_mix = carry_h
                if ti > 0:
                    h_mix = _row_n(h, ti - 1, BLOCK_D)
                c_new, n_new, m_new, h_new = _slstm_fwd_step(
                    c_run,
                    n_run,
                    m_run,
                    h_mix,
                    zi_t,
                    zf_t,
                    zz_t,
                    zo_t,
                    r_i,
                    r_f,
                    r_z,
                    r_o,
                    eps,
                    rh_scale,
                )
                c_run = tl.where(in_t, c_new, c_run)
                n_run = tl.where(in_t, n_new, n_run)
                m_run = tl.where(in_t, m_new, m_run)
                c = tl.where(in_t, _put_row(c, c_new, ti, BLOCK_T), c)
                n = tl.where(in_t, _put_row(n, n_new, ti, BLOCK_T), n)
                m = tl.where(in_t, _put_row(m, m_new, ti, BLOCK_T), m)
                h_pass = tl.where(in_t, _put_row(h_pass, h_new, ti, BLOCK_T), h_pass)
            h = h_pass
        else:
            # Tile guess once from DRAM; K Newton steps stay in registers.
            c = _load_state(
                s_ptr,
                pid_b,
                offs_t,
                offs_d,
                SLOT_C,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            n = _load_state(
                s_ptr,
                pid_b,
                offs_t,
                offs_d,
                SLOT_N,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            m = _load_state(
                s_ptr,
                pid_b,
                offs_t,
                offs_d,
                SLOT_M,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            h = _load_state(
                s_ptr,
                pid_b,
                offs_t,
                offs_d,
                SLOT_H,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
        for _it in tl.static_range(MAX_ITERS):
            # prev[t] = state[t-1] inside the tile; tile head uses solved carry.
            # Gather along T via transpose (same idiom as ``_row_n`` / seq scan).
            idx_prev = tl.maximum(local_t - 1, 0)[None, :] + tl.zeros(
                (BLOCK_D, 1), dtype=tl.int32
            )
            c_prev = tl.trans(tl.gather(tl.trans(c), idx_prev, axis=1))
            n_prev = tl.trans(tl.gather(tl.trans(n), idx_prev, axis=1))
            m_prev = tl.trans(tl.gather(tl.trans(m), idx_prev, axis=1))
            h_prev = tl.trans(tl.gather(tl.trans(h), idx_prev, axis=1))
            is_head = (local_t == 0)[:, None]
            c_prev = tl.where(is_head, carry_c[None, :], c_prev)
            n_prev = tl.where(is_head, carry_n[None, :], n_prev)
            m_prev = tl.where(is_head, carry_m[None, :], m_prev)
            h_prev = tl.where(is_head, carry_h[None, :], h_prev)
            (
                c_new,
                n_new,
                m_new,
                h_new,
                j00,
                j01,
                j02,
                j03,
                j10,
                j11,
                j12,
                j13,
                j20,
                j21,
                j22,
                j23,
                j30,
                j31,
                j32,
                j33,
            ) = _slstm_pred_j(
                c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o, eps
            )
            r0 = tl.where(mask, c_new - c, 0.0)
            r1 = tl.where(mask, n_new - n, 0.0)
            r2 = tl.where(mask, m_new - m, 0.0)
            r3 = tl.where(mask, h_new - h, 0.0)
            j00 = tl.where(mask, j00, 1.0)
            j01 = tl.where(mask, j01, 0.0)
            j02 = tl.where(mask, j02, 0.0)
            j03 = tl.where(mask, j03, 0.0)
            j10 = tl.where(mask, j10, 0.0)
            j11 = tl.where(mask, j11, 1.0)
            j12 = tl.where(mask, j12, 0.0)
            j13 = tl.where(mask, j13, 0.0)
            j20 = tl.where(mask, j20, 0.0)
            j21 = tl.where(mask, j21, 0.0)
            j22 = tl.where(mask, j22, 1.0)
            j23 = tl.where(mask, j23, 0.0)
            j30 = tl.where(mask, j30, 0.0)
            j31 = tl.where(mask, j31, 0.0)
            j32 = tl.where(mask, j32, 0.0)
            j33 = tl.where(mask, j33, 1.0)
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                u0,
                u1,
                u2,
                u3,
            ) = _scan_select_block4(
                j00,
                j01,
                j02,
                j03,
                j10,
                j11,
                j12,
                j13,
                j20,
                j21,
                j22,
                j23,
                j30,
                j31,
                j32,
                j33,
                r0,
                r1,
                r2,
                r3,
                BLOCK_T,
                BLOCK_D,
                SEQ,
                THOMAS_C,
                LEAF_SPARSE=True,
            )
            c = c + omega * u0
            n = n + omega * u1
            m = m + omega * u2
            h = h + omega * u3
        _store_state(
            s_ptr,
            c,
            pid_b,
            offs_t,
            offs_d,
            SLOT_C,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
        _store_state(
            s_ptr,
            n,
            pid_b,
            offs_t,
            offs_d,
            SLOT_N,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
        _store_state(
            s_ptr,
            m,
            pid_b,
            offs_t,
            offs_d,
            SLOT_M,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
        _store_state(
            s_ptr,
            h,
            pid_b,
            offs_t,
            offs_d,
            SLOT_H,
            mask,
            stride_sb,
            stride_st,
            stride_ss,
            stride_sd,
        )
        # Solved last step of this tile → left boundary for the next tile.
        last_local = tl.maximum(tl.minimum(BLOCK_T - 1, time - 1 - t0), 0)
        carry_c = _row_n(c, last_local, BLOCK_D)
        carry_n = _row_n(n, last_local, BLOCK_D)
        carry_m = _row_n(m, last_local, BLOCK_D)
        carry_h = _row_n(h, last_local, BLOCK_D)


def _parse_scan_tile(scan_tile: str) -> tuple[int, bool]:
    """Return ``(thomas_c, seq)``. ``thomas_c=0`` is assoc or seq."""
    if scan_tile == "assoc":
        return 0, False
    if scan_tile == "seq":
        return 0, True
    if scan_tile in ("thomas", "thomas4"):
        return 4, False
    if scan_tile == "thomas2":
        return 2, False
    raise ValueError(f"unknown scan_tile {scan_tile!r}; use assoc, seq, thomas, thomas2, thomas4")


def _slstm_fused_windows(
    wx: Tensor,
    r: Tensor,
    states: Tensor,
    h0: Tensor,
    *,
    max_iters: int,
    omega: float,
    eps: float,
    scan_tile: str,
    window_len: int,
    bt: Tensor,
    has_bt: bool,
    n_picard: int | None = None,
) -> Tensor:
    """In-kernel windowed Newton: ``window_len`` tiles, solved-state DRAM carry.

    Same contract as ``chunk_len=window_len``: each window is optional Picard
    (``1+n_picard`` frozen-gate passes with Newton-solved left carry) then a
    full K-step Newton; the next tile reads the solved last state as
    ``h_{t-1}``. One launch walks T. ``window_len`` is Triton ``BLOCK_T``
    (32 / 64 / 128). ``n_picard=None`` loads the tile guess from ``states``.
    """
    if window_len not in FUSED_WINDOW_LENS:
        raise ValueError(f"window_len must be one of {FUSED_WINDOW_LENS}, got {window_len}")
    batch, time, _four = wx.shape
    d_h = r.shape[-1]
    # Window walk carries solved state in regs; launch knobs from device table.
    recipe = resolve_slstm_launch(wx.device, window_len)
    block_d = recipe.block_d
    n_stages = recipe.num_stages
    chunk_pad = recipe.chunk_pad
    n_chunks, n_dtiles = time_tiles(
        time,
        d_h,
        window_len,
        block_d,
        chunk_pad,
        cap=False,
    )
    thomas_c, seq = _parse_scan_tile(scan_tile)
    do_picard = n_picard is not None
    n_p = int(n_picard) if do_picard else 0
    if do_picard and n_p < 0:
        raise ValueError(f"n_picard must be >= 0, got {n_picard!r}")
    if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
        log.debug(
            "newton_slstm_fused_windows",
            extra={
                "seq_len": time,
                "batch": batch,
                "d_h": d_h,
                "n_tiles": n_chunks,
                "window_len": window_len,
                "block_d": block_d,
                "num_warps": recipe.num_warps,
                "num_stages": n_stages,
                "chunk_pad": chunk_pad,
                "newton_iters": max_iters,
                "n_picard": n_p if do_picard else None,
                "do_picard": do_picard,
                "scan_tile": scan_tile,
                "device": str(wx.device),
                "dtype": str(wx.dtype),
            },
        )
    _slstm_window_walk_kernel[(batch, n_dtiles)](
        states,
        wx,
        r,
        h0,
        time,
        d_h,
        float(eps),
        float(omega),
        n_chunks,
        *states.stride(),
        *h0.stride(),
        *wx.stride(),
        *r.stride(),
        bt,
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_T=window_len,
        BLOCK_D=block_d,
        SEQ=seq,
        THOMAS_C=thomas_c,
        MAX_ITERS=max_iters,
        N_PICARD=n_p,
        DO_PICARD=do_picard,
        HAS_BT=has_bt,
        num_warps=recipe.num_warps,
        num_stages=n_stages,
    )
    return states


def _newton_slstm_fused_impl(
    wx: Tensor,
    r: Tensor,
    *,
    max_iters: int,
    omega: float,
    eps: float,
    h0: Tensor | None = None,
    states: Tensor | None = None,
    log_coords: bool = False,
    scan_tile: str = "assoc",
    time_loop: bool = False,
    window_len: int | None = None,
    block_table: Tensor | None = None,
    early_exit_atol: float | None = None,
    residual_history: list[float] | None = None,
    iters_done_out: list[int] | None = None,
    residual_max_out: list[float | Tensor] | None = None,
    n_picard: int | None = None,
) -> Tensor:
    """Alg. 1 for diag-mix ParaSLSTM. Public entry: ``pararnn::newton_slstm_fused``.

    ``n_picard``: when set, Picard (``1+n_picard`` passes) runs inside the
    window / leaf walk with Newton-solved carries — no host frozen-gate
    launches. ``residual_max_out``: when set, append a 0-d CUDA ``max |F(H*)|``
    (native coords after log decode). Host ``.item()`` stays in ``_fill_stats``.
    """
    from pararnn.solvers.slstm_log import (
        slstm_clamp_log_coords,
        slstm_decode_log,
        slstm_encode_log,
    )

    wx = wx.contiguous()
    r = r.contiguous()
    batch, time, four_d = wx.shape
    d_h = r.shape[-1]
    if r.shape != (4, d_h):
        raise ValueError(f"r shape {tuple(r.shape)} != {(4, d_h)}")
    if four_d != 4 * d_h:
        raise ValueError(f"wx last dim {four_d} != 4 * d_h={4 * d_h}")
    if log_coords and block_table is not None:
        raise TypeError("block_table fused sLSTM is native coords only")
    if n_picard is not None and int(n_picard) < 0:
        raise ValueError(f"n_picard must be >= 0, got {n_picard!r}")
    h0, bt, has_bt = prepare_h0_block_table(wx, h0, batch, (SLSTM_SLOTS, d_h), block_table)
    validate_cuda_tensors(wx, r, h0, name="newton_slstm_fused")
    if states is not None:
        validate_cuda_tensors(wx, states, name="newton_slstm_fused")
    leaf_t = _leaf_block_t(time)
    chunk_pad = resolve_slstm_launch(wx.device, leaf_t).chunk_pad
    n_chunks, n_dtiles = time_tiles(
        time,
        d_h,
        leaf_t,
        _BLOCK_D,
        chunk_pad,
        cap=False,
        cap_suffix=(f" (T≤{leaf_t * chunk_pad}). Shrink CHUNK_D or raise CHUNK_PAD."),
    )
    grid_td = (batch, n_chunks, n_dtiles)
    if states is None:
        states = wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
        if n_picard is None:
            _slstm_init_kernel[grid_td](
                wx,
                states,
                r,
                h0,
                d_h,
                time,
                float(eps),
                *wx.stride(),
                *states.stride(),
                *r.stride(),
                *h0.stride(),
                bt,
                SLOT_C=SLSTM_CELL,
                SLOT_N=SLSTM_NORMALIZER,
                SLOT_M=SLSTM_STABILIZER,
                SLOT_H=SLSTM_HIDDEN,
                BLOCK_T=leaf_t,
                BLOCK_D=_BLOCK_D,
                HAS_BT=has_bt,
            )
    else:
        if states.shape != (batch, time, SLSTM_SLOTS, d_h):
            raise ValueError(
                f"states shape {tuple(states.shape)} != {(batch, time, SLSTM_SLOTS, d_h)}"
            )
        if states.dtype != wx.dtype:
            states = states.to(dtype=wx.dtype)
        # In-place Newton on the Picard / caller buffer (one trajectory).
        # ``contiguous()`` only copies when storage is strided.
        states = states.contiguous()
    h0_native = h0
    if log_coords:
        if n_picard is not None:
            raise TypeError("in-kernel Picard is native coords only")
        states = slstm_encode_log(states, eps=float(eps))
        h0 = slstm_encode_log(h0, eps=float(eps))
    if max_iters <= 0:
        if log_coords:
            states = slstm_decode_log(states, eps=float(eps))
        return _slstm_fill_residual_max_out(
            residual_max_out,
            states,
            wx,
            r,
            h0_native,
            float(eps),
            block_table=None,
        )
    work, h0_work = fp32_newton_work(states, h0)
    if time_loop:
        if log_coords:
            raise TypeError("fused_time_loop is native coords only")
        work = _slstm_fused_windows(
            wx,
            r,
            work,
            h0_work,
            max_iters=max_iters,
            omega=omega,
            eps=eps,
            scan_tile=scan_tile,
            window_len=int(window_len) if window_len is not None else FUSED_WINDOW_DEFAULT,
            bt=bt,
            has_bt=has_bt,
            n_picard=n_picard,
        )
        if work.dtype != wx.dtype:
            work = work.to(dtype=wx.dtype)
        return _slstm_fill_residual_max_out(
            residual_max_out,
            work,
            wx,
            r,
            h0_native,
            float(eps),
            block_table=None,
        )

    need_res_max = early_exit_atol is not None or residual_history is not None
    # Single leaf when T≤leaf_t (incl. chunk_len ∈ {32,64,128}): cell+scan+ω
    # in one launch — no _chunk_incl. Multi-chunk: aggregates + recompute-apply.
    fuse_one = n_chunks == 1 and not need_res_max
    recompute = n_chunks > 1 and not need_res_max
    omega_f = float(omega)
    thomas_c, seq = _parse_scan_tile(scan_tile)
    dummy = work
    dummy_stride = work.stride()

    # Leaf / window length: Picard+K (or K-only) in one walk launch.
    # ``_slstm_fused_windows`` is native-only (it has no ``log_coords``), so log
    # coords must stay on the log-aware ``_cell_launch`` path below — the same
    # path used for ``T`` outside ``FUSED_WINDOW_LENS`` (T>=256).
    if fuse_one and time in FUSED_WINDOW_LENS and max_iters > 0 and not log_coords:
        assert not log_coords  # defensive: native-only window walk
        work = _slstm_fused_windows(
            wx,
            r,
            work,
            h0_work,
            max_iters=max_iters,
            omega=omega,
            eps=eps,
            scan_tile=scan_tile,
            window_len=time,
            bt=bt,
            has_bt=has_bt,
            n_picard=n_picard,
        )
        mark_fused_iters_done(iters_done_out, max_iters)
        log_fused_done(
            log,
            "newton_slstm_fused",
            time=time,
            batch=batch,
            d_h=d_h,
            max_iters=max_iters,
            n_chunks=n_chunks,
            log_coords=log_coords,
            scan_tile=scan_tile,
            acc_dtype=str(work.dtype),
            fuse_one=True,
            recompute_jr=False,
            n_picard=n_picard,
        )
        if log_coords:
            work = slstm_decode_log(work, eps=float(eps))
        if work.dtype != wx.dtype:
            work = work.to(dtype=wx.dtype)
        return _slstm_fill_residual_max_out(
            residual_max_out,
            work,
            wx,
            r,
            h0_native,
            float(eps),
            block_table=None,
        )

    def _cell_launch(
        *,
        grid,
        j_buf,
        r_buf,
        agg_j_buf,
        agg_r_buf,
        incl_buf,
        j_stride,
        r_stride,
        aj_stride,
        ar_stride,
        i_stride,
        apply_omega: bool,
        apply_carry: bool,
        store_jr: bool,
        block_t: int = leaf_t,
        block_d: int = _BLOCK_D,
    ) -> None:
        _slstm_cell_local_scan_kernel[grid](
            work,
            wx,
            r,
            h0_work,
            j_buf,
            r_buf,
            agg_j_buf,
            agg_r_buf,
            incl_buf,
            time,
            d_h,
            float(eps),
            omega_f,
            *work.stride(),
            *h0_work.stride(),
            *wx.stride(),
            *r.stride(),
            *j_stride,
            *r_stride,
            *aj_stride,
            *ar_stride,
            *i_stride,
            bt,
            SLOT_C=SLSTM_CELL,
            SLOT_N=SLSTM_NORMALIZER,
            SLOT_M=SLSTM_STABILIZER,
            SLOT_H=SLSTM_HIDDEN,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            LOG=log_coords,
            SEQ=seq,
            THOMAS_C=thomas_c,
            HAS_BT=has_bt,
            APPLY_OMEGA=apply_omega,
            APPLY_CARRY=apply_carry,
            STORE_JR=store_jr,
        )

    if fuse_one:
        for it in range(max_iters):
            _cell_launch(
                grid=grid_td,
                j_buf=dummy,
                r_buf=dummy,
                agg_j_buf=dummy,
                agg_r_buf=dummy,
                incl_buf=dummy,
                j_stride=dummy_stride,
                r_stride=dummy_stride,
                aj_stride=dummy_stride,
                ar_stride=dummy_stride,
                i_stride=dummy_stride,
                apply_omega=True,
                apply_carry=False,
                store_jr=False,
            )
            log_fused_iter(
                log,
                "newton_slstm_fused_iter",
                it=it,
                time=time,
                batch=batch,
                d_h=d_h,
                n_chunks=n_chunks,
            )
            if log_coords:
                work = slstm_clamp_log_coords(work)
        mark_fused_iters_done(iters_done_out, max_iters)
        log_fused_done(
            log,
            "newton_slstm_fused",
            time=time,
            batch=batch,
            d_h=d_h,
            max_iters=max_iters,
            n_chunks=n_chunks,
            log_coords=log_coords,
            scan_tile=scan_tile,
            acc_dtype=str(work.dtype),
            fuse_one=True,
            recompute_jr=False,
        )
        if log_coords:
            work = slstm_decode_log(work, eps=float(eps))
        if work.dtype != wx.dtype:
            work = work.to(dtype=wx.dtype)
        return _slstm_fill_residual_max_out(
            residual_max_out,
            work,
            wx,
            r,
            h0_native,
            float(eps),
            block_table=None,
        )

    agg_j = work.new_empty(batch, n_chunks, 16, d_h)
    agg_r = work.new_empty(batch, n_chunks, SLSTM_SLOTS, d_h)
    incl_r = work.new_empty(batch, n_chunks, SLSTM_SLOTS, d_h)
    if recompute:
        j_loc = dummy
        r_loc = dummy
        j_stride = dummy_stride
        r_stride = dummy_stride
        states32, r32 = None, None
    else:
        j_loc = work.new_empty(batch, time, 16, d_h)
        r_loc = work.new_empty(batch, time, SLSTM_SLOTS, d_h)
        j_stride = j_loc.stride()
        r_stride = r_loc.stride()
        states32, r32 = alloc_fp32_update(work, r_loc, n_chunks)

    for it in range(max_iters):
        _cell_launch(
            grid=grid_td,
            j_buf=j_loc,
            r_buf=r_loc,
            agg_j_buf=agg_j,
            agg_r_buf=agg_r,
            incl_buf=dummy,
            j_stride=j_stride,
            r_stride=r_stride,
            aj_stride=agg_j.stride(),
            ar_stride=agg_r.stride(),
            i_stride=dummy_stride,
            apply_omega=False,
            apply_carry=False,
            store_jr=not recompute,
        )
        log_fused_iter(
            log,
            "newton_slstm_fused_iter",
            it=it,
            time=time,
            batch=batch,
            d_h=d_h,
            n_chunks=n_chunks,
        )
        if need_res_max and fused_early_exit_hit(
            r_loc,
            early_exit_atol,
            iters_done_out=iters_done_out,
            it=it,
            residual_history=residual_history,
        ):
            out = work
            if log_coords:
                out = slstm_decode_log(work, eps=float(eps))
            if out.dtype != wx.dtype:
                out = out.to(dtype=wx.dtype)
            log_fused_done(
                log,
                "newton_slstm_fused",
                time=time,
                batch=batch,
                d_h=d_h,
                max_iters=it,
                n_chunks=n_chunks,
                log_coords=log_coords,
                scan_tile=scan_tile,
                acc_dtype=str(work.dtype),
                early_exit=True,
                recompute_jr=False,
            )
            return _slstm_fill_residual_max_out(
                residual_max_out,
                out,
                wx,
                r,
                h0_native,
                float(eps),
                block_table=None,
            )
        if recompute:
            incl_r = incl_block_aggregates(
                agg_j,
                agg_r,
                n_state=SLSTM_SLOTS,
                chunk_incl_kernel=_chunk_incl_kernel,
                chunk_pad=chunk_pad,
                block_d=_CHUNK_D,
                logger=log,
                out=incl_r,
            )
            _cell_launch(
                grid=grid_td,
                j_buf=dummy,
                r_buf=dummy,
                agg_j_buf=dummy,
                agg_r_buf=dummy,
                incl_buf=incl_r,
                j_stride=dummy_stride,
                r_stride=dummy_stride,
                aj_stride=dummy_stride,
                ar_stride=dummy_stride,
                i_stride=incl_r.stride(),
                apply_omega=False,
                apply_carry=True,
                store_jr=False,
            )
        elif states32 is not None and r32 is not None:
            fp32_omega_add(work, r_loc, omega_f, states32, r32)
        else:
            incl_r = incl_block_aggregates(
                agg_j,
                agg_r,
                n_state=SLSTM_SLOTS,
                chunk_incl_kernel=_chunk_incl_kernel,
                chunk_pad=chunk_pad,
                block_d=_CHUNK_D,
                logger=log,
                out=incl_r,
            )
            _slstm_apply_update_kernel[grid_td](
                work,
                j_loc,
                r_loc,
                incl_r,
                time,
                d_h,
                omega_f,
                *work.stride(),
                *j_loc.stride(),
                *r_loc.stride(),
                *incl_r.stride(),
                SLOT_C=SLSTM_CELL,
                SLOT_N=SLSTM_NORMALIZER,
                SLOT_M=SLSTM_STABILIZER,
                SLOT_H=SLSTM_HIDDEN,
                BLOCK_T=_BLOCK_T,
                BLOCK_D=_BLOCK_D,
                HAS_CARRY_BUF=False,
            )
        if log_coords:
            work = slstm_clamp_log_coords(work)
    mark_fused_iters_done(iters_done_out, max_iters)
    log_fused_done(
        log,
        "newton_slstm_fused",
        time=time,
        batch=batch,
        d_h=d_h,
        max_iters=max_iters,
        n_chunks=n_chunks,
        log_coords=log_coords,
        scan_tile=scan_tile,
        acc_dtype=str(work.dtype),
        fuse_one=False,
        recompute_jr=recompute,
    )
    if log_coords:
        work = slstm_decode_log(work, eps=float(eps))
    if work.dtype != wx.dtype:
        work = work.to(dtype=wx.dtype)
    return _slstm_fill_residual_max_out(
        residual_max_out,
        work,
        wx,
        r,
        h0_native,
        float(eps),
        block_table=None,
    )


@triton.jit
def _slstm_residual_max_kernel(
    s_ptr,
    wx_ptr,
    r_ptr,
    h0_ptr,
    res_max_ptr,
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
    stride_rg,
    stride_rd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    stride_rmb,
    stride_rmc,
    stride_rmd,
    bt_ptr,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """Post-solve tile ``max |F(H*)|`` for ``residual_fail`` / ``NewtonStats``."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    c = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    n = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    m = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    h = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    c_prev, n_prev, m_prev, h_prev = _slstm_tile_prev_from_loaded(
        c,
        n,
        m,
        h,
        s_ptr,
        h0_ptr,
        pid_b,
        pid_c,
        t0,
        offs_d,
        dmask,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
        stride_h0b,
        stride_h0s,
        stride_h0d,
        bt_ptr,
        SLOT_C,
        SLOT_N,
        SLOT_M,
        SLOT_H,
        BLOCK_T,
        BLOCK_D,
        HAS_BT,
    )
    zi, zf, zz, zo = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    r_i = _load_r_gate(r_ptr, 0, offs_d, dmask, stride_rg, stride_rd)
    r_f = _load_r_gate(r_ptr, 1, offs_d, dmask, stride_rg, stride_rd)
    r_z = _load_r_gate(r_ptr, 2, offs_d, dmask, stride_rg, stride_rd)
    r_o = _load_r_gate(r_ptr, 3, offs_d, dmask, stride_rg, stride_rd)
    c_new, n_new, m_new, h_new, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _slstm_pred_j(
        c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o, eps
    )
    r0 = tl.where(mask, c_new - c, 0.0)
    r1 = tl.where(mask, n_new - n, 0.0)
    r2 = tl.where(mask, m_new - m, 0.0)
    r3 = tl.where(mask, h_new - h, 0.0)
    tile = tl.maximum(tl.abs(r0), tl.abs(r1))
    tile = tl.maximum(tile, tl.abs(r2))
    tile = tl.maximum(tile, tl.abs(r3))
    tl.store(
        res_max_ptr + pid_b * stride_rmb + pid_c * stride_rmc + pid_d * stride_rmd,
        tl.max(tile),
    )


@triton.jit
def _slstm_block4_jac_kernel(
    h_prev_ptr,
    wx_ptr,
    r_ptr,
    jac_ptr,
    time,
    d_h,
    eps,
    stride_hb,
    stride_ht,
    stride_hs,
    stride_hd,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_rg,
    stride_rd,
    stride_jb,
    stride_jt,
    stride_jr,
    stride_jc,
    stride_jd,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """``J_t = ∂s_t/∂s_{t-1}`` as ``(B,T,4,4,d_h)`` for eq. 2.6 reverse."""
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
        h_prev_ptr,
        pid_b,
        offs_t,
        offs_d,
        SLOT_C,
        mask,
        stride_hb,
        stride_ht,
        stride_hs,
        stride_hd,
    )
    n_prev = _load_state(
        h_prev_ptr,
        pid_b,
        offs_t,
        offs_d,
        SLOT_N,
        mask,
        stride_hb,
        stride_ht,
        stride_hs,
        stride_hd,
    )
    m_prev = _load_state(
        h_prev_ptr,
        pid_b,
        offs_t,
        offs_d,
        SLOT_M,
        mask,
        stride_hb,
        stride_ht,
        stride_hs,
        stride_hd,
    )
    h_prev = _load_state(
        h_prev_ptr,
        pid_b,
        offs_t,
        offs_d,
        SLOT_H,
        mask,
        stride_hb,
        stride_ht,
        stride_hs,
        stride_hd,
    )
    zi, zf, zz, zo = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    r_i = _load_r_gate(r_ptr, 0, offs_d, dmask, stride_rg, stride_rd)
    r_f = _load_r_gate(r_ptr, 1, offs_d, dmask, stride_rg, stride_rd)
    r_z = _load_r_gate(r_ptr, 2, offs_d, dmask, stride_rg, stride_rd)
    r_o = _load_r_gate(r_ptr, 3, offs_d, dmask, stride_rg, stride_rd)
    (
        _,
        _,
        _,
        _,
        j00,
        j01,
        j02,
        j03,
        j10,
        j11,
        j12,
        j13,
        j20,
        j21,
        j22,
        j23,
        j30,
        j31,
        j32,
        j33,
    ) = _slstm_pred_j(c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o, eps)
    j00 = tl.where(mask, j00, 1.0)
    j01 = tl.where(mask, j01, 0.0)
    j02 = tl.where(mask, j02, 0.0)
    j03 = tl.where(mask, j03, 0.0)
    j10 = tl.where(mask, j10, 0.0)
    j11 = tl.where(mask, j11, 1.0)
    j12 = tl.where(mask, j12, 0.0)
    j13 = tl.where(mask, j13, 0.0)
    j20 = tl.where(mask, j20, 0.0)
    j21 = tl.where(mask, j21, 0.0)
    j22 = tl.where(mask, j22, 1.0)
    j23 = tl.where(mask, j23, 0.0)
    j30 = tl.where(mask, j30, 0.0)
    j31 = tl.where(mask, j31, 0.0)
    j32 = tl.where(mask, j32, 0.0)
    j33 = tl.where(mask, j33, 1.0)
    base = jac_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + offs_d[None, :] * stride_jd
    store_acc(base + 0 * stride_jr + 0 * stride_jc, j00, mask)
    store_acc(base + 0 * stride_jr + 1 * stride_jc, j01, mask)
    store_acc(base + 0 * stride_jr + 2 * stride_jc, j02, mask)
    store_acc(base + 0 * stride_jr + 3 * stride_jc, j03, mask)
    store_acc(base + 1 * stride_jr + 0 * stride_jc, j10, mask)
    store_acc(base + 1 * stride_jr + 1 * stride_jc, j11, mask)
    store_acc(base + 1 * stride_jr + 2 * stride_jc, j12, mask)
    store_acc(base + 1 * stride_jr + 3 * stride_jc, j13, mask)
    store_acc(base + 2 * stride_jr + 0 * stride_jc, j20, mask)
    store_acc(base + 2 * stride_jr + 1 * stride_jc, j21, mask)
    store_acc(base + 2 * stride_jr + 2 * stride_jc, j22, mask)
    store_acc(base + 2 * stride_jr + 3 * stride_jc, j23, mask)
    store_acc(base + 3 * stride_jr + 0 * stride_jc, j30, mask)
    store_acc(base + 3 * stride_jr + 1 * stride_jc, j31, mask)
    store_acc(base + 3 * stride_jr + 2 * stride_jc, j32, mask)
    store_acc(base + 3 * stride_jr + 3 * stride_jc, j33, mask)


def slstm_residual_amax(
    states: Tensor,
    wx: Tensor,
    r: Tensor,
    *,
    h0: Tensor | None = None,
    eps: float = 1e-6,
    block_table: Tensor | None = None,
) -> Tensor:
    """``max |F(H*)|`` as a 0-d CUDA tensor (no host sync).

    Callers that need a Python ``float`` (fail rail / logs) take ``.item()``
    at the comparison boundary — keep the GEMM→Newton pipeline async until then.
    """
    wx = wx.contiguous()
    r = r.contiguous()
    states = states.contiguous()
    batch, time, slots, d_h = states.shape
    if slots != SLSTM_SLOTS:
        raise ValueError(f"states slots {slots} != {SLSTM_SLOTS}")
    if wx.shape != (batch, time, 4 * d_h):
        raise ValueError(f"wx shape {tuple(wx.shape)} != {(batch, time, 4 * d_h)}")
    if r.shape != (4, d_h):
        raise ValueError(f"r shape {tuple(r.shape)} != {(4, d_h)}")
    h0, bt, has_bt = prepare_h0_block_table(wx, h0, batch, (SLSTM_SLOTS, d_h), block_table)
    validate_cuda_tensors(wx, r, states, h0, name="slstm_residual_amax")
    chunk_pad = resolve_slstm_launch(wx.device, _BLOCK_T).chunk_pad
    n_chunks, n_dtiles = time_tiles(time, d_h, _BLOCK_T, _BLOCK_D, chunk_pad, cap=False)
    res_max = states.new_empty(batch, n_chunks, n_dtiles, dtype=torch.float32)
    _slstm_residual_max_kernel[(batch, n_chunks, n_dtiles)](
        states,
        wx,
        r,
        h0,
        res_max,
        time,
        d_h,
        float(eps),
        *states.stride(),
        *wx.stride(),
        *r.stride(),
        *h0.stride(),
        *res_max.stride(),
        bt,
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        HAS_BT=has_bt,
    )
    return res_max.amax()


def _slstm_fill_residual_max_out(
    residual_max_out: list[float | Tensor] | None,
    states: Tensor,
    wx: Tensor,
    r: Tensor,
    h0: Tensor | None,
    eps: float,
    *,
    block_table: Tensor | None,
) -> Tensor:
    """Append final ``max |F|`` (0-d tensor); ``.item()`` waits in ``_fill_stats``."""
    if residual_max_out is not None:
        residual_max_out.clear()
        residual_max_out.append(
            slstm_residual_amax(
                states,
                wx,
                r,
                h0=h0,
                eps=eps,
                block_table=block_table,
            )
        )
    return states


def slstm_block4_jacobian(
    h_prev: Tensor,
    wx: Tensor,
    r: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Channelwise ``(B,T,4,4,d_h)`` ``J`` for fused eq. 2.6 reverse (CUDA)."""
    if h_prev.device.type != "cuda":
        return slstm_block4_jacobian_eager(h_prev, wx, r, eps=eps)
    batch, time, slots, d_h = h_prev.shape
    if slots != SLSTM_SLOTS:
        raise ValueError(f"h_prev slots {slots} != {SLSTM_SLOTS}")
    if wx.shape != (batch, time, 4 * d_h):
        raise ValueError(f"wx shape {tuple(wx.shape)} != {(batch, time, 4 * d_h)}")
    if r.shape != (4, d_h):
        raise ValueError(f"r shape {tuple(r.shape)} != {(4, d_h)}")
    h_prev = h_prev.contiguous()
    wx = wx.contiguous()
    r = r.contiguous()
    validate_cuda_tensors(h_prev, wx, r, name="slstm_block4_jacobian")
    jac = h_prev.new_empty(batch, time, SLSTM_SLOTS, SLSTM_SLOTS, d_h)
    n_chunks = triton.cdiv(time, _BLOCK_T)
    n_dtiles = triton.cdiv(d_h, _BLOCK_D)
    _slstm_block4_jac_kernel[(batch, n_chunks, n_dtiles)](
        h_prev,
        wx,
        r,
        jac,
        time,
        d_h,
        float(eps),
        *h_prev.stride(),
        *wx.stride(),
        *r.stride(),
        *jac.stride(),
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    return jac


def slstm_block4_jacobian_eager(
    h_prev: Tensor,
    wx: Tensor,
    r: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Eager block4 ``J`` matching ``ParaSLSTM._jac_diag`` / fused ``_slstm_pred_j``."""
    out_dtype = h_prev.dtype
    h_prev_f = h_prev.float()
    wx_f = wx.float()
    r_f = r.float()
    c_prev = h_prev_f[..., SLSTM_CELL, :]
    n_prev = h_prev_f[..., SLSTM_NORMALIZER, :]
    m_prev = h_prev_f[..., SLSTM_STABILIZER, :]
    hh_prev = h_prev_f[..., SLSTM_HIDDEN, :]
    d_h = hh_prev.shape[-1]
    zi, zf, zz, zo = wx_f.split(d_h, dim=-1)
    r_i, r_fg, r_z, r_o = r_f[0], r_f[1], r_f[2], r_f[3]
    z_i = r_i * hh_prev + zi
    z_f = r_fg * hh_prev + zf
    z_z = r_z * hh_prev + zz
    z_o = r_o * hh_prev + zo
    left = z_f + m_prev
    m_new = torch.where(left > z_i, left, z_i)
    alpha = (left > z_i).to(h_prev_f.dtype) + 0.5 * (left == z_i).to(h_prev_f.dtype)
    beta = 1.0 - alpha
    i_t = torch.exp(z_i - m_new)
    f_t = torch.exp(z_f + m_prev - m_new)
    z = torch.tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c_prev + i_t * z
    o = torch.sigmoid(z_o)
    denom = n_new + float(eps)
    dm_dh = alpha * r_fg + beta * r_i
    di_dm = -i_t * alpha
    df_dm = f_t * beta
    di_dh = i_t * (r_i - dm_dh)
    df_dh = f_t * (r_fg - dm_dh)
    dz_dh = (1.0 - z * z) * r_z
    do_dh = o * (1.0 - o) * r_o
    zeros = torch.zeros_like(f_t)
    j_cc = f_t
    j_cn = zeros
    j_cm = df_dm * c_prev + di_dm * z
    j_ch = df_dh * c_prev + di_dh * z + i_t * dz_dh
    j_nc = zeros
    j_nn = f_t
    j_nm = df_dm * n_prev + di_dm
    j_nh = df_dh * n_prev + di_dh
    j_mc = zeros
    j_mn = zeros
    j_mm = alpha
    j_mh = dm_dh
    inv = o / denom
    dn = -o * c_new / denom.square()
    du = c_new / denom
    j_hc = inv * j_cc
    j_hn = dn * j_nn
    j_hm = inv * j_cm + dn * j_nm
    j_hh = inv * j_ch + dn * j_nh + du * do_dh
    return torch.stack(
        (
            torch.stack((j_cc, j_cn, j_cm, j_ch), dim=-2),
            torch.stack((j_nc, j_nn, j_nm, j_nh), dim=-2),
            torch.stack((j_mc, j_mn, j_mm, j_mh), dim=-2),
            torch.stack((j_hc, j_hn, j_hm, j_hh), dim=-2),
        ),
        dim=-3,
    ).to(dtype=out_dtype)


def _compose_chunk_monoids(agg_j: Tensor, agg_r: Tensor, *, chunk_pad: int) -> tuple[Tensor, Tensor]:
    """Compose leaf-chunk monoids → tile ``(P_end, δ_end)``.

    ``agg_j``: ``(B, n_chunks, 16, D)``, ``agg_r``: ``(B, n_chunks, 4, D)``.
    Triton SRAM path when ``n_chunks ≤ pad``; longer tiles reduce in
    ``pad`` windows then recurse (same monoid). Cap pad at 128 for the
    ends/carry kernels (CHUNK_PAD=512 OOM/compile-fails on the carry path).
    """
    pad = min(int(chunk_pad), 128)
    batch, n_chunks, _, d_h = agg_r.shape
    if n_chunks == 0:
        from pararnn.solvers.scan import _fill_ident_block4

        p = agg_j.new_empty(batch, 4, 4, d_h)
        _fill_ident_block4(p)
        return p, agg_r.new_zeros(batch, 4, d_h)
    if n_chunks <= pad:
        p_flat = agg_j.new_empty(batch, 16, d_h)
        d_end = agg_r.new_empty(batch, 4, d_h)
        n_dtiles = (d_h + _CHUNK_D - 1) // _CHUNK_D
        _chunk_monoid_ends_kernel[(batch, n_dtiles)](
            agg_j,
            agg_r,
            p_flat,
            d_end,
            n_chunks,
            d_h,
            *agg_j.stride(),
            *agg_r.stride(),
            *p_flat.stride(),
            *d_end.stride(),
            CHUNK_PAD=pad,
            BLOCK_D=_CHUNK_D,
        )
        return p_flat.view(batch, 4, 4, d_h), d_end
    n_meta = (n_chunks + pad - 1) // pad
    meta_j = agg_j.new_empty(batch, n_meta, 16, d_h)
    meta_r = agg_r.new_empty(batch, n_meta, 4, d_h)
    for i in range(n_meta):
        start = i * pad
        end = min(start + pad, n_chunks)
        p_i, d_i = _compose_chunk_monoids(
            agg_j[:, start:end], agg_r[:, start:end], chunk_pad=pad
        )
        meta_j[:, i] = p_i.reshape(batch, 16, d_h)
        meta_r[:, i] = d_i
    return _compose_chunk_monoids(meta_j, meta_r, chunk_pad=pad)


def _chunk_carries_from_rank(
    agg_j: Tensor, agg_r: Tensor, rank_carry: Tensor, *, chunk_pad: int
) -> Tensor:
    """Exclusive carry per leaf chunk: ``P_excl @ rank_carry + δ_excl``."""
    pad = min(int(chunk_pad), 128)
    batch, n_chunks, _, d_h = agg_r.shape
    carry_buf = agg_r.new_empty(batch, n_chunks, 4, d_h)
    if n_chunks == 0:
        return carry_buf
    if n_chunks <= pad:
        n_dtiles = (d_h + _CHUNK_D - 1) // _CHUNK_D
        _chunk_carries_rank_kernel[(batch, n_dtiles)](
            agg_j,
            agg_r,
            rank_carry,
            carry_buf,
            n_chunks,
            d_h,
            *agg_j.stride(),
            *agg_r.stride(),
            *rank_carry.stride(),
            *carry_buf.stride(),
            CHUNK_PAD=pad,
            BLOCK_D=_CHUNK_D,
        )
        return carry_buf
    n_meta = (n_chunks + pad - 1) // pad
    meta_j = agg_j.new_empty(batch, n_meta, 16, d_h)
    meta_r = agg_r.new_empty(batch, n_meta, 4, d_h)
    for i in range(n_meta):
        start = i * pad
        end = min(start + pad, n_chunks)
        p_i, d_i = _compose_chunk_monoids(
            agg_j[:, start:end], agg_r[:, start:end], chunk_pad=pad
        )
        meta_j[:, i] = p_i.reshape(batch, 16, d_h)
        meta_r[:, i] = d_i
    meta_carry = _chunk_carries_from_rank(meta_j, meta_r, rank_carry, chunk_pad=pad)
    for i in range(n_meta):
        start = i * pad
        end = min(start + pad, n_chunks)
        carry_buf[:, start:end] = _chunk_carries_from_rank(
            agg_j[:, start:end],
            agg_r[:, start:end],
            meta_carry[:, i],
            chunk_pad=pad,
        )
    return carry_buf


class _FusedCpUpsweep:
    """Buffers from ``slstm_fused_cp_upsweep`` for the downsweep apply."""

    __slots__ = (
        "j_loc",
        "r_loc",
        "agg_j",
        "agg_r",
        "p_end",
        "d_end",
        "leaf_t",
        "grid",
        "chunk_pad",
    )

    def __init__(
        self,
        j_loc: Tensor,
        r_loc: Tensor,
        agg_j: Tensor,
        agg_r: Tensor,
        p_end: Tensor,
        d_end: Tensor,
        leaf_t: int,
        grid: tuple[int, int, int],
        chunk_pad: int,
    ) -> None:
        self.j_loc = j_loc
        self.r_loc = r_loc
        self.agg_j = agg_j
        self.agg_r = agg_r
        self.p_end = p_end
        self.d_end = d_end
        self.leaf_t = leaf_t
        self.grid = grid
        self.chunk_pad = chunk_pad


def slstm_fused_cp_upsweep(
    states: Tensor,
    wx: Tensor,
    r: Tensor,
    h0: Tensor,
    *,
    eps: float,
    scan_tile: str = "assoc",
) -> _FusedCpUpsweep:
    """Phase 1: Triton cell+J+local scan; emit tile monoid (no ω update)."""
    validate_cuda_tensors(states, wx, r, h0, name="slstm_fused_cp_upsweep")
    wx = wx.contiguous()
    r = r.contiguous()
    work = states.contiguous()
    h0 = h0.contiguous()
    batch, time, slots, d_h = work.shape
    if slots != SLSTM_SLOTS:
        raise ValueError(f"states slots {slots} != {SLSTM_SLOTS}")
    if r.shape != (4, d_h):
        raise ValueError(f"r shape {tuple(r.shape)} != {(4, d_h)}")
    if wx.shape != (batch, time, 4 * d_h):
        raise ValueError(f"wx shape {tuple(wx.shape)} != {(batch, time, 4 * d_h)}")
    if h0.shape != (batch, SLSTM_SLOTS, d_h):
        raise ValueError(f"h0 shape {tuple(h0.shape)} != {(batch, SLSTM_SLOTS, d_h)}")

    leaf_t = _leaf_block_t(time)
    chunk_pad = resolve_slstm_launch(wx.device, leaf_t).chunk_pad
    n_chunks, n_dtiles = time_tiles(time, d_h, leaf_t, _BLOCK_D, chunk_pad, cap=False)
    grid = (batch, n_chunks, n_dtiles)
    thomas_c, seq = _parse_scan_tile(scan_tile)
    bt = wx.new_zeros(1, dtype=torch.int32)

    j_loc = work.new_empty(batch, time, 16, d_h)
    r_loc = work.new_empty(batch, time, SLSTM_SLOTS, d_h)
    agg_j = work.new_empty(batch, n_chunks, 16, d_h)
    agg_r = work.new_empty(batch, n_chunks, SLSTM_SLOTS, d_h)
    dummy = work

    _slstm_cell_local_scan_kernel[grid](
        work,
        wx,
        r,
        h0,
        j_loc,
        r_loc,
        agg_j,
        agg_r,
        dummy,
        time,
        d_h,
        float(eps),
        1.0,  # omega unused (APPLY_OMEGA=False)
        *work.stride(),
        *h0.stride(),
        *wx.stride(),
        *r.stride(),
        *j_loc.stride(),
        *r_loc.stride(),
        *agg_j.stride(),
        *agg_r.stride(),
        *dummy.stride(),
        bt,
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_T=leaf_t,
        BLOCK_D=_BLOCK_D,
        LOG=False,
        SEQ=seq,
        THOMAS_C=thomas_c,
        HAS_BT=False,
        APPLY_OMEGA=False,
        APPLY_CARRY=False,
        STORE_JR=True,
    )
    # Tile monoid via Triton chunk scan when n_chunks ≤ pad.
    p_end, d_end = _compose_chunk_monoids(agg_j, agg_r, chunk_pad=chunk_pad)
    return _FusedCpUpsweep(
        j_loc, r_loc, agg_j, agg_r, p_end, d_end, leaf_t, grid, chunk_pad
    )


def slstm_fused_cp_downsweep(
    states: Tensor,
    pack: _FusedCpUpsweep,
    rank_carry: Tensor | None,
    *,
    omega: float,
) -> Tensor:
    """Phase 3: apply ``δ = P_within @ carry_chunk + u`` then ω-update.

    ``rank_carry=None`` (or all-zero) uses the Triton inclusive-chunk path.
    """
    work = states.contiguous()
    batch, time, _, d_h = work.shape
    use_fast = rank_carry is None
    if rank_carry is not None and rank_carry.shape != (batch, SLSTM_SLOTS, d_h):
        raise ValueError(
            f"rank_carry shape {tuple(rank_carry.shape)} != {(batch, SLSTM_SLOTS, d_h)}"
        )
    if use_fast:
        incl_r = incl_block_aggregates(
            pack.agg_j,
            pack.agg_r,
            n_state=SLSTM_SLOTS,
            chunk_incl_kernel=_chunk_incl_kernel,
            chunk_pad=pack.chunk_pad,
            block_d=_CHUNK_D,
            logger=log,
        )
        _slstm_apply_update_kernel[pack.grid](
            work,
            pack.j_loc,
            pack.r_loc,
            incl_r,
            time,
            d_h,
            float(omega),
            *work.stride(),
            *pack.j_loc.stride(),
            *pack.r_loc.stride(),
            *incl_r.stride(),
            SLOT_C=SLSTM_CELL,
            SLOT_N=SLSTM_NORMALIZER,
            SLOT_M=SLSTM_STABILIZER,
            SLOT_H=SLSTM_HIDDEN,
            BLOCK_T=pack.leaf_t,
            BLOCK_D=_BLOCK_D,
            HAS_CARRY_BUF=False,
        )
        return work

    carry_buf = _chunk_carries_from_rank(
        pack.agg_j, pack.agg_r, rank_carry.contiguous(), chunk_pad=pack.chunk_pad
    )
    _slstm_apply_update_kernel[pack.grid](
        work,
        pack.j_loc,
        pack.r_loc,
        carry_buf,
        time,
        d_h,
        float(omega),
        *work.stride(),
        *pack.j_loc.stride(),
        *pack.r_loc.stride(),
        *carry_buf.stride(),
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_T=pack.leaf_t,
        BLOCK_D=_BLOCK_D,
        HAS_CARRY_BUF=True,
    )
    return work

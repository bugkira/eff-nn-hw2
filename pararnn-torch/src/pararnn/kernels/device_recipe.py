"""Device-aware launch recipes for fused ParaSLSTM (walk / pack / pad).

Two layers:

1. **Analytic** (``solve_slstm_launch_analytic``) — closed-form knobs from
   ``DeviceProfile`` + window geometry. No wall-clock sweep. Priors are
   walk-path constraints (``block_d ∈ {4,8}``, warps clamp, SRAM pad) and
   coarse consumer/server splits. Three lab cards cannot identify a rich
   model; the formulas stay deliberately low-capacity.

2. **Measured pins** (``_SLSTM_LAUNCH_TABLE``) — optional overrides from
   ``bench_slstm_launch_recipe`` when a ``(cc_major, sm_bucket)`` was timed.
   ``resolve_slstm_launch`` prefers a pin, else analytic.

``@triton.autotune`` stays unused (cold-start cost). ``NewtonConfig``
defaults are unchanged.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import torch

log = logging.getLogger(__name__)

SmBucket = Literal["consumer", "server"]

# Walk-path priors (3060 fused C=128 era + lab sweep).
_DEFAULT_WARPS = 4
_DEFAULT_STAGES_LONG = 2
_DEFAULT_STAGES_SHORT = 1
_DEFAULT_BLOCK_D_LONG = 4
_DEFAULT_BLOCK_D_SHORT = 8
_DEFAULT_CHUNK_PAD = 128
_DEFAULT_PACK_TILES = 128

_SERVER_SM_MIN = 80
# Leaf-style pad budget uses 4×4 Newton lane count (16 J + 4 r).
_MONOID_LANES_BLOCK4 = 20
_CHUNK_D_PAD = 16
_PAD_KAPPA = 1.5
# Opt-in / per-SM shared floors when props look empty.
_S_CTA_FALLBACK = {
    (7,): 64 * 1024,
    (8, 0): 164 * 1024,
    (8, 6): 100 * 1024,
    (8, 9): 100 * 1024,
    (9,): 228 * 1024,
    (10,): 228 * 1024,
}


@dataclass(frozen=True)
class DeviceProfile:
    """CUDA device facts used for recipe lookup (one query per index)."""

    index: int
    name: str
    major: int
    minor: int
    multi_processor_count: int
    total_memory: int
    shared_memory_per_multiprocessor: int

    @classmethod
    def from_device(cls, device: torch.device | str | int | None = None) -> DeviceProfile:
        dev = torch.device(device if device is not None else "cuda")
        if dev.type != "cuda":
            raise ValueError(f"DeviceProfile requires CUDA, got {dev}")
        idx = int(dev.index) if dev.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        major = int(props.major)
        minor = int(props.minor)
        # torch≥2.11 may omit shared_memory_per_multiprocessor; fall back to SM table.
        smem = getattr(props, "shared_memory_per_multiprocessor", None)
        if smem is None:
            smem = _S_CTA_FALLBACK.get((major, minor)) or _S_CTA_FALLBACK.get((major,))
            if smem is None:
                smem = 64 * 1024
        return cls(
            index=idx,
            name=str(props.name),
            major=major,
            minor=minor,
            multi_processor_count=int(props.multi_processor_count),
            total_memory=int(props.total_memory),
            shared_memory_per_multiprocessor=int(smem),
        )

    @property
    def sm_bucket(self) -> SmBucket:
        if self.multi_processor_count >= _SERVER_SM_MIN:
            return "server"
        return "consumer"

    @property
    def is_hbm_class(self) -> bool:
        """Server-ish memory subsystem (A100/H100-class SM count or large SMEM)."""
        if self.sm_bucket == "server":
            return True
        return self.shared_memory_per_multiprocessor >= 160 * 1024


@dataclass(frozen=True)
class SlstmLaunchRecipe:
    """Launch / pack knobs for diag fused ParaSLSTM window walk."""

    num_warps: int
    num_stages: int
    block_d: int
    chunk_pad: int
    pack_tiles: int


def _pow2_floor_clamp(x: float, lo: int, hi: int) -> int:
    x = max(float(lo), min(float(hi), float(x)))
    exp = math.floor(math.log2(max(1.0, x)))
    p = 2 ** exp
    return max(lo, min(hi, p))


def _s_cta_bytes(profile: DeviceProfile) -> int:
    smem = int(profile.shared_memory_per_multiprocessor)
    if smem >= 32 * 1024:
        return smem
    for key, val in _S_CTA_FALLBACK.items():
        if key == (profile.major,) or key == (profile.major, profile.minor):
            return int(val)
    return 64 * 1024


def _base_for_window(window_len: int) -> SlstmLaunchRecipe:
    """Shape-only defaults (no device); long windows use lean D + pipelining."""
    long = int(window_len) >= 128
    return SlstmLaunchRecipe(
        num_warps=_DEFAULT_WARPS,
        num_stages=_DEFAULT_STAGES_LONG if long else _DEFAULT_STAGES_SHORT,
        block_d=_DEFAULT_BLOCK_D_LONG if long else _DEFAULT_BLOCK_D_SHORT,
        chunk_pad=_DEFAULT_CHUNK_PAD,
        pack_tiles=_DEFAULT_PACK_TILES,
    )


def solve_slstm_launch_analytic(
    profile: DeviceProfile,
    window_len: int,
    *,
    batch: int = 2,
    d_h: int = 256,
    monoid_lanes: int = _MONOID_LANES_BLOCK4,
    elem_bytes: int = 4,
) -> SlstmLaunchRecipe:
    """Closed-form walk/pack recipe from device caps + window geometry.

    Targets ``_slstm_window_walk`` (``BLOCK_T = window_len``), not the leaf
    multi-chunk ``BLOCK_T=32`` hierarchy. Candidates stay in the compile-safe
    walk set: ``block_d ∈ {4,8}``, ``num_warps ≤ 4`` when ``block_d=4`` and
    ``C≥128`` (``warps=8`` fails ``TritonGPUOptimizeThreadLocality`` on Ampere).

    Parameters
    ----------
    batch, d_h
        Used only for wave-fill scoring of ``block_d``. Defaults match the
        expert-dump protocol (B=2, d_h=256).
    monoid_lanes, elem_bytes
        SRAM budget for ``chunk_pad`` (4×4 Newton leaf pad proxy).
    """
    win = int(window_len)
    batch = max(1, int(batch))
    d_h = max(1, int(d_h))
    n_sm = max(1, int(profile.multi_processor_count))
    s_cta = _s_cta_bytes(profile)

    # --- block_d ---
    # Short windows: D=4@C∈{32,64} fails Triton locality → D=8.
    # Long windows: score {4,8} by wave efficiency; lean D=4 gets a small prior
    # (lab: D=4 beat D=8 on 3060/2080/A100 @ C=128).
    if win < 128:
        block_d = _DEFAULT_BLOCK_D_SHORT
    else:
        best_d = _DEFAULT_BLOCK_D_LONG
        best_score = -1.0
        for d in (4, 8):
            n_blocks = batch * math.ceil(d_h / d)
            waves = n_blocks / n_sm
            wave_eff = n_blocks / (math.ceil(waves) * n_sm)
            # Penalize severe underfill; do not reward huge oversubscription.
            fill = min(1.0, waves)
            prior = 1.05 if d == 4 else 1.0
            score = wave_eff * (0.25 + 0.75 * fill) * prior
            if score > best_score:
                best_score = score
                best_d = d
        block_d = best_d

    # --- num_warps ---
    # Target ~8 elements/thread: θ = (C·D)/(32·W) ∈ [4,16] ⇒ W ≈ (C·D)/(32·8).
    # Floor at 4 for long windows: lab warps=2 was slower than 4 on all three cards.
    total_elems = max(1, win * block_d)
    target_w = total_elems / (32.0 * 8.0)
    nearest = 2 ** round(math.log2(max(1.0, target_w)))
    lo = 4 if win >= 128 else 2
    num_warps = min(8, max(lo, nearest))
    # warps=8 + block_d=4 + C≥128 fails TritonGPUOptimizeThreadLocality on Ampere.
    if block_d <= 4 and win >= 128:
        num_warps = min(num_warps, 4)

    # --- num_stages ---
    # cp.async exists on CC≥8. Lab A100: stages 1/2/3 tied (~910 ms); winner=1.
    # Consumer Ampere: stages=2 matched the timed pin. Server/HBM → 1.
    if win < 128:
        num_stages = _DEFAULT_STAGES_SHORT
    elif profile.is_hbm_class:
        num_stages = 1
    elif profile.major >= 8:
        num_stages = 2
    else:
        num_stages = 2

    # --- chunk_pad ---
    # Same SRAM sketch as leaf chunk-incl pad; floor 128 (known-good walk default).
    raw_per = monoid_lanes * _CHUNK_D_PAD * elem_bytes * _PAD_KAPPA
    max_pad = s_cta / max(1.0, raw_per)
    chunk_pad = _pow2_floor_clamp(max_pad, 32, 512)
    chunk_pad = max(_DEFAULT_CHUNK_PAD, chunk_pad)
    if not profile.is_hbm_class:
        chunk_pad = min(chunk_pad, 128)

    # --- pack_tiles ---
    # No strong closed form from SM alone; pack trades launches vs peak.
    # Server/HBM → 256 (A100 pin); consumer → 128 (3060/2080 pin).
    pack_tiles = 256 if profile.is_hbm_class else 128

    return SlstmLaunchRecipe(
        num_warps=num_warps,
        num_stages=num_stages,
        block_d=block_d,
        chunk_pad=chunk_pad,
        pack_tiles=pack_tiles,
    )


# Measured pins from ``bench_slstm_launch_recipe`` @ T=131072 B=2 C=128 fbwd.
# Analytic already reproduces these; pins keep resolve stable if the formula moves.
_SLSTM_LAUNCH_TABLE: dict[tuple[int, SmBucket], SlstmLaunchRecipe] = {
    (7, "consumer"): SlstmLaunchRecipe(
        num_warps=4, num_stages=2, block_d=4, chunk_pad=128, pack_tiles=128
    ),
    (8, "consumer"): SlstmLaunchRecipe(
        num_warps=4, num_stages=2, block_d=4, chunk_pad=128, pack_tiles=128
    ),
    (8, "server"): SlstmLaunchRecipe(
        num_warps=4, num_stages=1, block_d=4, chunk_pad=128, pack_tiles=256
    ),
}

_profile_cache: dict[int, DeviceProfile] = {}
_recipe_cache: dict[tuple[int, int], SlstmLaunchRecipe] = {}
_logged: set[tuple[int, int]] = set()
_override: SlstmLaunchRecipe | None = None


def clear_recipe_cache() -> None:
    """Drop profile/recipe caches and the one-shot log set (tests / benches)."""
    _profile_cache.clear()
    _recipe_cache.clear()
    _logged.clear()


def set_slstm_launch_override(recipe: SlstmLaunchRecipe | None) -> None:
    """Force a recipe for benches; ``None`` restores resolve lookup."""
    global _override
    _override = recipe
    clear_recipe_cache()


def get_device_profile(device: torch.device | str | int | None = None) -> DeviceProfile:
    """Cached ``DeviceProfile`` for a CUDA device."""
    if device is None:
        idx = torch.cuda.current_device()
    else:
        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"get_device_profile requires CUDA, got {dev}")
        idx = int(dev.index) if dev.index is not None else torch.cuda.current_device()
    hit = _profile_cache.get(idx)
    if hit is not None:
        return hit
    prof = DeviceProfile.from_device(torch.device("cuda", idx))
    _profile_cache[idx] = prof
    return prof


def _apply_window(base: SlstmLaunchRecipe, window_len: int) -> SlstmLaunchRecipe:
    """Shape clamps on top of a device row (short C keeps safer D / stages=1)."""
    if int(window_len) >= 128:
        return base
    return SlstmLaunchRecipe(
        num_warps=base.num_warps,
        num_stages=_DEFAULT_STAGES_SHORT,
        block_d=_DEFAULT_BLOCK_D_SHORT,
        chunk_pad=base.chunk_pad,
        pack_tiles=base.pack_tiles,
    )


def resolve_slstm_launch(
    device: torch.device | str | int | None,
    window_len: int,
    *,
    batch: int = 2,
    d_h: int = 256,
    prefer_measured: bool = True,
) -> SlstmLaunchRecipe:
    """Resolve walk/pack knobs for ``window_len`` on ``device``.

    Order: override → measured pin (if ``prefer_measured``) → analytic →
    CPU shape fallback. Logs once per ``(device_index, window_len)``.
    """
    win = int(window_len)
    if _override is not None:
        return _apply_window(_override, win)

    if device is not None:
        dev = torch.device(device)
        if dev.type != "cuda":
            return _apply_window(_base_for_window(win), win)
    elif not torch.cuda.is_available():
        return _apply_window(_base_for_window(win), win)

    profile = get_device_profile(device)
    cache_key = (profile.index, win)
    hit = _recipe_cache.get(cache_key)
    if hit is not None:
        return hit

    source = "analytic"
    row = None
    if prefer_measured:
        row = _SLSTM_LAUNCH_TABLE.get((profile.major, profile.sm_bucket))
    if row is not None:
        recipe = _apply_window(row, win)
        source = "measured_pin"
    else:
        recipe = _apply_window(
            solve_slstm_launch_analytic(
                profile, win, batch=batch, d_h=d_h
            ),
            win,
        )

    _recipe_cache[cache_key] = recipe
    if cache_key not in _logged and not torch.compiler.is_compiling():
        _logged.add(cache_key)
        log.info(
            "slstm_launch_recipe",
            extra={
                "source": source,
                "device": profile.name,
                "index": profile.index,
                "cc": f"{profile.major}.{profile.minor}",
                "sm": profile.multi_processor_count,
                "sm_bucket": profile.sm_bucket,
                "smem_per_sm": profile.shared_memory_per_multiprocessor,
                "window_len": win,
                "num_warps": recipe.num_warps,
                "num_stages": recipe.num_stages,
                "block_d": recipe.block_d,
                "chunk_pad": recipe.chunk_pad,
                "pack_tiles": recipe.pack_tiles,
            },
        )
    return recipe


def update_slstm_launch_table(
    major: int,
    sm_bucket: SmBucket,
    recipe: SlstmLaunchRecipe,
) -> None:
    """Install a measured pin (lab sweep). Clears caches."""
    _SLSTM_LAUNCH_TABLE[(int(major), sm_bucket)] = recipe
    clear_recipe_cache()

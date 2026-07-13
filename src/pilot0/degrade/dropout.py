"""Dropouts: non-overlapping zeroed bursts with raised-cosine ramps.

`pct_lost` = fraction of samples fully zeroed. Each burst is `burst_ms` long with
a `ramp_ms` raised-cosine fade at each edge (inside the burst); the fully-zero
core is `burst − 2·ramp`. Burst count is chosen so total core ≈ target. Positions
are deterministic (seeded) and non-overlapping."""

from __future__ import annotations

import numpy as np

from .base import to_float32


def _nonoverlap_starts(n: int, burst: int, k: int, rng) -> list[int]:
    if k * burst >= n:  # pack tightly if we can't fit k spaced bursts
        return list(range(0, max(0, n - burst) + 1, burst))[:k]
    seg = n // k
    starts = []
    for i in range(k):
        lo = i * seg
        hi = min(lo + seg - burst, n - burst)
        s = lo if hi <= lo else int(rng.integers(lo, hi + 1))
        starts.append(min(s, n - burst))
    return starts


def dropouts(
    wav: np.ndarray,
    sr: int,
    pct_lost: float,
    burst_ms: float = 20.0,
    ramp_ms: float = 5.0,
    seed: int = 0,
) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float64)
    n = len(wav)
    burst = max(1, int(round(sr * burst_ms / 1000.0)))
    ramp = min(int(round(sr * ramp_ms / 1000.0)), burst // 2)
    core = max(1, burst - 2 * ramp)
    target_zero = int(round(n * pct_lost / 100.0))
    n_bursts = max(1, int(round(target_zero / core)))

    ramp_win = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp))) if ramp > 0 else None  # 0→1
    gain = np.ones(n)
    rng = np.random.default_rng(seed)
    for s in _nonoverlap_starts(n, burst, n_bursts, rng):
        e = min(s + burst, n)
        seg = np.zeros(e - s)
        length = e - s
        if ramp > 0 and length >= 2 * ramp:
            seg[:ramp] = 1 - ramp_win  # fade out 1→0
            seg[length - ramp :] = ramp_win  # fade in 0→1
        gain[s:e] = np.minimum(gain[s:e], seg)
    return to_float32(wav * gain)

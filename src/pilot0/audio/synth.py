"""Deterministic synthetic clips for the Phase-0 smoke path.

Speech-/tone-like signals that just exercise the encode seam end to end — no
dataset download yet. Real corpora and the degradation grid are Phase 1–2.
"""

from __future__ import annotations

import numpy as np


def synth_clip(seed: int, sr: int = 48000, seconds: float = 4.0) -> np.ndarray:
    """One deterministic ~voiced clip in [-0.5, 0.5], float64, at `sr`."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0 = rng.uniform(90.0, 220.0)
    sig = np.zeros(n)
    for k in range(1, int(rng.integers(4, 9))):
        sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2.0, 5.0) * t)  # syllable-rate
    sig = sig * env + 0.02 * rng.standard_normal(n)  # + breath noise
    peak = float(np.max(np.abs(sig))) or 1.0
    return (0.5 * sig / peak).astype(np.float64)


def synth_batch(n: int = 10, sr: int = 48000, seconds: float = 4.0) -> list[np.ndarray]:
    """`n` deterministic clips (seeds 0..n-1)."""
    return [synth_clip(seed=i, sr=sr, seconds=seconds) for i in range(n)]

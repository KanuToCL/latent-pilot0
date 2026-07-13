"""Clipping parameterised by target %-samples-clipped (L2).

`pct_clipped` sets the fraction of samples at/over the ceiling: the threshold is
the (1 − pct) quantile of |wav|, so the amount of clipping is guaranteed monotone
and always > 0 — unlike 'gain over full-scale', which barely clips a
loudness-normalised master (physics elder)."""

from __future__ import annotations

import numpy as np

from .base import EPS, to_float32


def clip_audio(wav: np.ndarray, sr: int, pct_clipped: float, mode: str = "hard") -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float64)
    target = pct_clipped / 100.0
    thr = float(np.quantile(np.abs(wav), 1.0 - target))
    thr = max(thr, EPS)
    if mode == "hard":
        out = np.clip(wav, -thr, thr)
    elif mode == "soft":
        out = thr * np.tanh(wav / thr)
    else:
        raise ValueError(f"unknown clip mode '{mode}'")
    return to_float32(out)

"""Mains hum: fundamental + harmonics at a target hum-to-signal ratio (dB).

Harmonic amplitudes roll off as 1/k. Base 50 Hz (EU). NOTE: the 1/k roll-off is
a fixed *synthetic* artifact, not physical — real transformer-saturation hum is
harmonically richer and sometimes even-harmonic-dominant; do not read amplitude
structure into it (physics elder)."""

from __future__ import annotations

import numpy as np

from .base import EPS, rms, signal_level, to_float32


def mains_hum(
    wav: np.ndarray, sr: int, hts_db: float, base_hz: float = 50.0, n_harm: int = 4
) -> np.ndarray:
    n = len(wav)
    t = np.arange(n) / sr
    hum = np.zeros(n)
    for k in range(1, n_harm + 1):
        hum += (1.0 / k) * np.sin(2 * np.pi * base_hz * k * t)
    target_rms = signal_level(wav) * (10.0 ** (hts_db / 20.0))
    hum *= target_rms / (rms(hum) + EPS)
    return to_float32(np.asarray(wav, dtype=np.float64) + hum)

"""Broadband additive noise (white / pink) at a target full-band SNR."""

from __future__ import annotations

import numpy as np

from .base import add_noise_at_snr, to_float32


def _pinkify(white: np.ndarray) -> np.ndarray:
    """Shape white noise to a 1/f (pink) magnitude spectrum."""
    spec = np.fft.rfft(white)
    k = np.arange(len(spec), dtype=np.float64)
    k[0] = 1.0
    spec = spec / np.sqrt(k)
    return np.fft.irfft(spec, n=len(white))


def broadband_noise(
    wav: np.ndarray, sr: int, snr_db: float, color: str = "white", seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(wav))
    if color == "pink":
        noise = _pinkify(noise)
    elif color != "white":
        raise ValueError(f"unknown noise color '{color}'")
    return to_float32(add_noise_at_snr(wav, noise, snr_db))

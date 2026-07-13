"""Hiss: high-frequency-shaped noise (a high-pass, not a literal shelf) at a
target SNR measured above a corner.

The corner defaults to 4 kHz. NOTE: 'SNR above 4 kHz' is not comparable across
content arms (speech vs music have very different >4 kHz energy) — fine
within-arm only (physics elder tightening)."""

from __future__ import annotations

import numpy as np

from .base import add_noise_at_snr, to_float32


def hiss(wav: np.ndarray, sr: int, hf_snr_db: float, corner_hz: float = 4000.0, seed: int = 0):
    from scipy.signal import butter, filtfilt

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(wav))
    nyq = sr / 2.0
    wc = min(corner_hz / nyq, 0.99)
    b, a = butter(4, wc, btype="high")
    noise = filtfilt(b, a, noise)  # zero-phase → hiss concentrated above corner
    return to_float32(add_noise_at_snr(wav, noise, hf_snr_db, sr=sr, band=(corner_hz, nyq)))

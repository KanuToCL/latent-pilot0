"""Band-limiting: zero-phase Butterworth low-pass at a target cutoff.

filtfilt gives zero phase (no group delay), so it does not confound time
alignment (physics elder). A cutoff at/above Nyquist for the current rate is
UNDEFINED (it would be a no-op emitting clean audio mislabeled as band-limited) —
so we refuse it loudly. The manifest enumerates valid cells via
`grid.is_applicable`; nothing should ever reach here with cutoff ≥ Nyquist."""

from __future__ import annotations

import numpy as np

from .base import to_float32


def band_limit(wav: np.ndarray, sr: int, cutoff_hz: float, order: int = 8) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float64)
    nyq = sr / 2.0
    if cutoff_hz >= nyq:
        raise ValueError(
            f"band-limit cutoff {cutoff_hz} Hz ≥ Nyquist {nyq} Hz at sr={sr}: undefined "
            "(would be a no-op mislabeled as band-limited). The caller must skip this "
            "(model, level) cell — see grid.is_applicable (L1)."
        )
    from scipy.signal import butter, filtfilt

    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return to_float32(filtfilt(b, a, wav))

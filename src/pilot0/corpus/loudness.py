"""BS.1770 loudness (pyloudnorm). Masters are normalised to a target LUFS BEFORE
degradation and never touched after (L3)."""

from __future__ import annotations

import numpy as np

TARGET_LUFS = -23.0


def measure_lufs(wav: np.ndarray, sr: int) -> float:
    import pyloudnorm as pyln

    return float(pyln.Meter(sr).integrated_loudness(np.asarray(wav, dtype=np.float64)))


def normalize_lufs(wav: np.ndarray, sr: int, target: float = TARGET_LUFS):
    """Return (normalised_wav, measured_lufs, gain), or (None, lufs, nan) if the
    integrated loudness is undefined (e.g. near-silence / too short to gate)."""
    lufs = measure_lufs(wav, sr)
    if not np.isfinite(lufs):
        return None, lufs, float("nan")
    gain = 10.0 ** ((target - lufs) / 20.0)
    return np.asarray(wav, dtype=np.float64) * gain, lufs, gain

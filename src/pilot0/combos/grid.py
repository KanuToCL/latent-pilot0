"""Pairwise degradation combos (§2.2) for the RQ2 additivity test.

Three physically distinct pairs, MID severities only (~10% extra volume): a combo
applies leg A then leg B at the SAME severity on the native-rate master. Its family
label is `A+B` so a combo cell never collides with a single-degradation cell in the
Phase-3 cache. Order is fixed (A then B) and recorded — degradations do not commute
(clip-then-lowpass ≠ lowpass-then-clip), so the pair is an ordered convention, not a
claim of symmetry.
"""

from __future__ import annotations

import numpy as np

from ..degrade.grid import apply_degradation, is_applicable

COMBO_PAIRS = (("noise", "clip"), ("hiss", "bandlimit"), ("hum", "mp3"))
COMBO_SEVERITIES = (2, 3, 4)  # mid only — the extremes are covered by the single grid


def combo_label(a: str, b: str) -> str:
    return f"{a}+{b}"


def combo_severities(a: str, b: str, sr: int) -> tuple[int, ...]:
    """Mid severities where BOTH legs are defined at `sr` (a band-limit cutoff ≥
    Nyquist drops out exactly as it does for the single grid, L1)."""
    return tuple(s for s in COMBO_SEVERITIES if is_applicable(a, s, sr) and is_applicable(b, s, sr))


def apply_combo(native: np.ndarray, sr: int, a: str, b: str, severity: int) -> np.ndarray:
    """Leg A then leg B at `severity` on already-native-rate audio. Returns the
    combined waveform; the per-leg measured metrics are not needed for the additivity
    geometry, so they are discarded here (the single grid audits them in Phase 1)."""
    da = apply_degradation(native, sr, a, severity)
    db = apply_degradation(np.asarray(da.wav, dtype=np.float64), sr, b, severity)
    return db.wav

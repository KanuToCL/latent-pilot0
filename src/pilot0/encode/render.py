"""Render one manifest cell to the waveform a model actually consumes.

L1: resample the −23 LUFS master to the model's native rate (soxr, anti-aliased)
BEFORE degrading, so a label like 'bandlimit 6 kHz' is defined at the rate the
codec sees. The L4 corpus headroom scalar is applied later (at encode time),
uniformly per rate, so an identical render feeds both the headroom pass and the
encode pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..corpus.io import read_audio, to_mono
from ..degrade.base import DegradationResult
from ..degrade.grid import apply_degradation, clean


def resample_to_native(wav: np.ndarray, sr: int, native_sr: int) -> np.ndarray:
    """Anti-aliased master → native-rate (soxr). Same resampler as the real
    backend's front-end, so the degraded audio matches what the codec ingests."""
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if sr == native_sr:
        return wav
    import soxr

    return soxr.resample(wav, sr, native_sr)


def load_master(norm_dir, token: str) -> tuple[np.ndarray, int]:
    wav, sr = read_audio(Path(norm_dir) / f"{token}.wav")
    return to_mono(wav).astype(np.float64), sr


def render_cell(
    master: np.ndarray, master_sr: int, native_sr: int, family: str, severity: int
) -> DegradationResult:
    """Master → native rate (L1) → degrade. `clean` is the resampled master with
    no degradation. Raises for a cell undefined at `native_sr` (e.g. band-limit
    cutoff ≥ Nyquist) — callers must feed only `renderable_rows(manifest, sr)`."""
    native = resample_to_native(master, master_sr, native_sr)
    if family == "clean":
        return clean(native, native_sr)
    return apply_degradation(native, native_sr, family, severity)

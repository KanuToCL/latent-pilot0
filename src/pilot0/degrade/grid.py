"""The degradation grid: 7 families × 5 severities (§2.2), plus `clean`.

Family = the type-classification label. `noise` defaults to white (pink is a
logged sub-variant); `clip` defaults to hard (soft is logged). Severity index
1..5 selects the physical parameter from `levels`.

`is_applicable` / `applicable_severities` gate cells that are undefined at a given
rate (L1) — chiefly a band-limit cutoff ≥ Nyquist. The Phase-2 manifest uses them
to enumerate only valid (family, severity, rate) cells; `apply_degradation`
refuses an invalid cell loudly rather than emitting a mislabeled result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import bandlimit, clip, dropout, hiss, hum, metrics, mp3, noise
from .base import DegradationResult, to_float32


@dataclass(frozen=True)
class Family:
    name: str
    param_name: str
    unit: str
    levels: tuple  # 5 physical params, severity 1..5
    fn: Callable[[np.ndarray, int, float], np.ndarray]


FAMILIES: dict[str, Family] = {
    "noise": Family(
        "noise", "snr_db", "dB", (30, 20, 10, 5, 0),
        lambda w, sr, p: noise.broadband_noise(w, sr, p, color="white"),
    ),
    "hiss": Family(
        "hiss", "hf_snr_db", "dB", (35, 25, 15, 10, 5),
        lambda w, sr, p: hiss.hiss(w, sr, p),
    ),
    "hum": Family(
        "hum", "hts_db", "dB", (-40, -30, -20, -12, -6),
        lambda w, sr, p: hum.mains_hum(w, sr, p),
    ),
    "clip": Family(
        "clip", "pct_clipped", "%", (0.1, 0.5, 1, 3, 8),
        lambda w, sr, p: clip.clip_audio(w, sr, p, mode="hard"),
    ),
    "bandlimit": Family(
        "bandlimit", "cutoff_hz", "Hz", (12000, 8000, 6000, 4000, 3400),
        lambda w, sr, p: bandlimit.band_limit(w, sr, p),
    ),
    "mp3": Family(
        "mp3", "kbps", "kbps", (64, 48, 32, 24, 16),
        lambda w, sr, p: mp3.mp3_transcode(w, sr, int(p)),
    ),
    "dropout": Family(
        "dropout", "pct_lost", "%", (0.5, 1, 2, 5, 10),
        lambda w, sr, p: dropout.dropouts(w, sr, p),
    ),
}

SEVERITIES = (1, 2, 3, 4, 5)


def is_applicable(family: str, severity_index: int, sr: int) -> bool:
    """Whether (family, severity) is a defined degradation at rate `sr`.

    Band-limit cutoffs ≥ Nyquist are undefined (L1); everything else is always
    applicable at the moment (extend here if a future family gains rate limits)."""
    fam = FAMILIES[family]
    param = fam.levels[severity_index - 1]
    if family == "bandlimit":
        return param < sr / 2.0
    return True


def applicable_severities(family: str, sr: int) -> tuple[int, ...]:
    return tuple(s for s in SEVERITIES if is_applicable(family, s, sr))


def apply_degradation(wav: np.ndarray, sr: int, family: str, severity_index: int) -> DegradationResult:
    fam = FAMILIES[family]
    if severity_index not in SEVERITIES:
        raise ValueError(f"severity_index must be 1..5, got {severity_index}")
    param = fam.levels[severity_index - 1]
    if not is_applicable(family, severity_index, sr):
        raise ValueError(
            f"{family} severity {severity_index} ({fam.param_name}={param}{fam.unit}) is "
            f"undefined at sr={sr} (≥ Nyquist); the manifest must skip this cell (L1)."
        )
    wav64 = np.asarray(wav, dtype=np.float64)
    out = fam.fn(wav64, sr, param)
    return DegradationResult(
        wav=to_float32(out),
        sr=sr,
        family=family,
        severity_index=severity_index,
        param=float(param),
        measured=metrics.measure(family, wav64, out, sr, param),
    )


def clean(wav: np.ndarray, sr: int) -> DegradationResult:
    return DegradationResult(to_float32(wav), sr, "clean", 0, float("nan"), {})

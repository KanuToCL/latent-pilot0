"""Level-matched arm primitive: set a degraded clip's gated loudness to its clean
partner's (S1/S7, decision D5/D6).

Estimand, stated precisely because the loose version is wrong: a single gain that
equalises BS.1770-4 INTEGRATED loudness removes the *gated broadband level*
difference from the stimulus. It does NOT remove "the level confound from the
control". Measured (F19): after LUFS-matching a degraded clip to its clean partner,
the energy control's mean-log-total feature is still offset by +4.6 nats at noise/1
and +6.3 at noise/5, because the cue lives in the pause frames BS.1770's -70 LKFS
absolute and -10 LU relative gates exclude — and the pooled STD coordinates are
exactly gain-invariant, so no uniform gain touches them at all. The instrument that
answers "is the control reading loudness?" is the projection split in
`probes/level_split.py`, not this arm.

This module is the primitive plus its guards, deliberately NOT threaded into the
encode pipeline this round: `degrade/` source is hashed into the bank id, so an arm
that changed rendering would silently invalidate the cache
(docs/plans/2026-09-09-level-matched-arm.design.md lists every threading site).

Guards, all of which a corpus run WILL hit:
  - pyloudnorm 0.2.0 raises ValueError below 400 ms of audio (its block size).
  - digital silence measures -inf LUFS; so does anything wholly below the gate.
  - a non-finite measurement on either side means no gain is defined.
Each returns the clip untouched with `applied=False` and a machine-readable
`reason`, never a silent pass-through and never a NaN-scaled waveform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..corpus.loudness import measure_lufs

TOLERANCE_LU = 0.01  # D6: a single-shot gain lands well inside this (measured ~6e-7 LU)


@dataclass(frozen=True)
class LevelMatch:
    gain_db: float  # what was applied (0.0 when applied=False)
    clean_lufs: float  # the target
    degraded_lufs: float  # before matching
    achieved_lufs: float  # after matching — audited, not assumed
    applied: bool
    reason: str  # "matched" | "too_short" | "degraded_not_measurable" | "clean_not_measurable"
    within_tolerance: bool  # |achieved - clean| <= TOLERANCE_LU


def _measure(wav: np.ndarray, sr: int) -> tuple[float, str | None]:
    """LUFS, or (nan, reason) — pyloudnorm raises on short input rather than
    returning -inf, so the two failure shapes are handled in one place."""
    try:
        return measure_lufs(wav, sr), None
    except ValueError:
        return float("nan"), "too_short"


def _result(gain_db, clean_lufs, degraded_lufs, achieved, applied, reason) -> LevelMatch:
    within = bool(np.isfinite(achieved) and np.isfinite(clean_lufs)
                  and abs(achieved - clean_lufs) <= TOLERANCE_LU)
    return LevelMatch(gain_db=gain_db, clean_lufs=clean_lufs, degraded_lufs=degraded_lufs,
                      achieved_lufs=achieved, applied=applied, reason=reason,
                      within_tolerance=within)


def match_loudness(degraded: np.ndarray, clean: np.ndarray, sr: int) -> tuple[np.ndarray, LevelMatch]:
    """Scale `degraded` so its integrated loudness equals `clean`'s.

    Returns `(wav, LevelMatch)`. The achieved loudness is re-measured rather than
    assumed: the gain is exact in theory, but the gates can in principle re-select
    blocks after scaling, and an arm that claims a match must be able to prove it."""
    deg = np.asarray(degraded, dtype=np.float64)
    cln = np.asarray(clean, dtype=np.float64)

    deg_lufs, err = _measure(deg, sr)
    if err is not None:
        return deg, _result(0.0, float("nan"), float("nan"), float("nan"), False, err)
    cln_lufs, err = _measure(cln, sr)
    if err is not None:
        return deg, _result(0.0, float("nan"), deg_lufs, deg_lufs, False, err)

    if not np.isfinite(deg_lufs):  # silence / wholly below the gate -> no gain defined
        return deg, _result(0.0, cln_lufs, deg_lufs, deg_lufs, False, "degraded_not_measurable")
    if not np.isfinite(cln_lufs):
        return deg, _result(0.0, cln_lufs, deg_lufs, deg_lufs, False, "clean_not_measurable")

    gain_db = float(cln_lufs - deg_lufs)
    out = deg * (10.0 ** (gain_db / 20.0))
    achieved, err = _measure(out, sr)
    return out, _result(gain_db, cln_lufs, deg_lufs,
                        float("nan") if err is not None else achieved, True, "matched")

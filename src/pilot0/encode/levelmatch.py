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
  - a gain beyond ±`max_abs_gain_db` (20 dB) is refused. The measured per-cell gains
    are under 1.2 dB (AM7), so 20 dB is not a tuning knob: it is the assertion that
    this clip and this clean partner are the same recording. A 100x scale factor
    arriving from a mispaired manifest row would otherwise be applied silently, and
    the arm would look like it worked.
Each returns the clip untouched with `applied=False` and a machine-readable
`reason`, never a silent pass-through and never a NaN-scaled waveform.

`peak_dbfs_after` is recorded on EVERY path (it describes the waveform actually
returned, matched or not) because a gain that fixes loudness can push peaks past
0 dBFS: the achieved-level audit needs to see clipping, not infer it.

Section map (file order):
  TOLERANCE_LU / MAX_ABS_GAIN_DB  - the two D6 constants
  LevelMatch                      - the per-clip record
  _measure(wav, sr)               - LUFS, or (nan, "too_short")
  _peak_dbfs(wav)                 - peak of the returned waveform; -inf for silence
  _result(...)                    - assemble a LevelMatch, deriving the two audits
  match_loudness(degraded, clean, sr) - the primitive
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..corpus.loudness import measure_lufs

TOLERANCE_LU = 0.01  # D6: a single-shot gain lands well inside this (measured ~6e-7 LU)
MAX_ABS_GAIN_DB = 20.0  # a sanity bound, not a tuning knob — see the module docstring


@dataclass(frozen=True)
class LevelMatch:
    gain_db: float  # what was applied (0.0 when applied=False; the refused magnitude
    # is always recoverable as clean_lufs - degraded_lufs)
    clean_lufs: float  # the target
    degraded_lufs: float  # before matching
    achieved_lufs: float  # after matching — audited, not assumed
    peak_dbfs_after: float  # peak of the RETURNED waveform; -inf for digital silence
    applied: bool
    reason: str  # "matched" | "too_short" | "degraded_not_measurable"
    # | "clean_not_measurable" | "gain_out_of_range"
    within_tolerance: bool  # |achieved - clean| <= TOLERANCE_LU


def _measure(wav: np.ndarray, sr: int) -> tuple[float, str | None]:
    """LUFS, or (nan, reason) — pyloudnorm raises on short input rather than
    returning -inf, so the two failure shapes are handled in one place."""
    try:
        return measure_lufs(wav, sr), None
    except ValueError:
        return float("nan"), "too_short"


def _peak_dbfs(wav: np.ndarray) -> float:
    """Sample peak of `wav` in dBFS. Digital silence is -inf rather than a floored
    sentinel: a clip that was silent and a clip that was merely quiet must not read
    alike in the achieved-level audit."""
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    return float(20.0 * np.log10(peak)) if peak > 0.0 else float("-inf")


def _result(wav, gain_db, clean_lufs, degraded_lufs, achieved, applied, reason) -> LevelMatch:
    within = bool(np.isfinite(achieved) and np.isfinite(clean_lufs)
                  and abs(achieved - clean_lufs) <= TOLERANCE_LU)
    return LevelMatch(gain_db=gain_db, clean_lufs=clean_lufs, degraded_lufs=degraded_lufs,
                      achieved_lufs=achieved, peak_dbfs_after=_peak_dbfs(wav),
                      applied=applied, reason=reason, within_tolerance=within)


def match_loudness(degraded: np.ndarray, clean: np.ndarray, sr: int, *,
                   max_abs_gain_db: float = MAX_ABS_GAIN_DB) -> tuple[np.ndarray, LevelMatch]:
    """Scale `degraded` so its integrated loudness equals `clean`'s.

    Returns `(wav, LevelMatch)`. The achieved loudness is re-measured rather than
    assumed: the gain is exact in theory, but the gates can in principle re-select
    blocks after scaling, and an arm that claims a match must be able to prove it.

    A required gain beyond ±`max_abs_gain_db` is refused (`gain_out_of_range`): at that
    size the two clips are not the same recording, and applying it would hide the
    pairing bug behind a plausible-looking waveform."""
    deg = np.asarray(degraded, dtype=np.float64)
    cln = np.asarray(clean, dtype=np.float64)

    deg_lufs, err = _measure(deg, sr)
    if err is not None:
        return deg, _result(deg, 0.0, float("nan"), float("nan"), float("nan"), False, err)
    cln_lufs, err = _measure(cln, sr)
    if err is not None:
        return deg, _result(deg, 0.0, float("nan"), deg_lufs, deg_lufs, False, err)

    if not np.isfinite(deg_lufs):  # silence / wholly below the gate -> no gain defined
        return deg, _result(deg, 0.0, cln_lufs, deg_lufs, deg_lufs, False, "degraded_not_measurable")
    if not np.isfinite(cln_lufs):
        return deg, _result(deg, 0.0, cln_lufs, deg_lufs, deg_lufs, False, "clean_not_measurable")

    gain_db = float(cln_lufs - deg_lufs)
    if abs(gain_db) > max_abs_gain_db:  # checked BEFORE scaling: nothing is applied
        return deg, _result(deg, 0.0, cln_lufs, deg_lufs, deg_lufs, False, "gain_out_of_range")

    out = deg * (10.0 ** (gain_db / 20.0))
    achieved, err = _measure(out, sr)
    return out, _result(out, gain_db, cln_lufs, deg_lufs,
                        float("nan") if err is not None else achieved, True, "matched")

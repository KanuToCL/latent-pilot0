"""Level-matched arm primitive (D5/D6) and its honest negative control.

Two things are tested here. First that the primitive works: a single BS.1770-4 gain
lands the degraded clip on its clean partner's integrated loudness to well inside
0.01 LU, and every way the measurement can fail — a clip shorter than pyloudnorm's
block, digital silence — returns the audio untouched with a reason instead of a NaN
scale factor.

Second, and more important, that it does NOT do what the audit assumed it would. The
arm removes the gated broadband level difference from the STIMULUS. It does not
remove the level cue from the energy CONTROL: at noise severity 1 the two clips
already have the same integrated loudness, so the gain is ~0 dB, while the control's
mean-log-total feature has moved by more than a nat — the cue lives in the pause
frames that BS.1770's gates exclude (F19). That is why the S7 instrument is the
projection split, not this arm.
"""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.corpus.loudness import measure_lufs
from pilot0.degrade.grid import apply_degradation
from pilot0.encode.levelmatch import TOLERANCE_LU, match_loudness
from pilot0.seam.energy import EnergyEncoder

SR = 16000


def _speech_like(seconds: float = 3.0, duty: float = 0.5, gap_db: float = -70.0) -> np.ndarray:
    """Bursts of tone separated by near-silent gaps — the pause structure BS.1770's
    -70 LKFS absolute and -10 LU relative gates throw away, and the structure the
    energy control pools straight through."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    tone = 0.1 * (np.sin(2 * np.pi * 220.0 * t) + 0.5 * np.sin(2 * np.pi * 900.0 * t))
    burst = ((t * 4.0) % 1.0) < duty  # 4 Hz on/off
    gap = 10 ** (gap_db / 20.0) * np.random.default_rng(0).standard_normal(n)
    return np.where(burst, tone, gap)


def _mean_log_total(wav: np.ndarray) -> float:
    """The energy control's first coordinate, pooled: mean over frames of
    log(mean(x**2))."""
    return float(EnergyEncoder(native_sr=SR).encode(wav, SR)["energy"].frames[:, 0].mean())


# --- the primitive ------------------------------------------------------------


def test_a_single_gain_lands_inside_the_tolerance():
    clean = _speech_like()
    quiet = clean * 10 ** (-6.0 / 20.0)  # 6 dB down
    out, m = match_loudness(quiet, clean, SR)
    assert m.applied and m.reason == "matched"
    assert m.gain_db == pytest.approx(6.0, abs=0.01)
    assert abs(m.achieved_lufs - m.clean_lufs) <= TOLERANCE_LU
    assert m.within_tolerance
    assert measure_lufs(out, SR) == pytest.approx(measure_lufs(clean, SR), abs=TOLERANCE_LU)


def test_matching_a_clip_to_itself_is_a_no_op_gain():
    clean = _speech_like()
    out, m = match_loudness(clean, clean, SR)
    assert m.gain_db == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(out, clean)


def test_short_clip_is_refused_not_crashed():
    """pyloudnorm 0.2.0 raises ValueError below its 400 ms block — the corpus has
    clips near that bound, so the guard is not hypothetical."""
    short = _speech_like(seconds=0.2)
    out, m = match_loudness(short, short, SR)
    assert not m.applied and m.reason == "too_short"
    assert np.allclose(out, short) and not m.within_tolerance


def test_silence_is_refused_not_scaled_by_infinity():
    clean = _speech_like()
    silence = np.zeros(int(SR * 3.0))
    out, m = match_loudness(silence, clean, SR)
    assert not m.applied and m.reason == "degraded_not_measurable"
    assert m.degraded_lufs == -np.inf
    assert np.allclose(out, silence)  # never scaled by a non-finite gain


def test_unmeasurable_clean_partner_is_refused():
    out, m = match_loudness(_speech_like(), np.zeros(int(SR * 3.0)), SR)
    assert not m.applied and m.reason == "clean_not_measurable"


# --- the honest negative control (F19) ----------------------------------------


def test_level_matching_does_not_remove_the_controls_level_cue():
    """noise/1: the degradation is 30 dB down, so the gated loudness barely moves and
    the arm applies essentially no gain — yet the energy control's mean-log-total has
    already shifted by more than a nat, because the noise fills the pause frames the
    loudness gate discards. No uniform gain can remove that."""
    clean = _speech_like()
    degraded = apply_degradation(clean, SR, "noise", 1).wav.astype(np.float64)

    matched, m = match_loudness(degraded, clean, SR)
    assert m.applied and abs(m.gain_db) < 0.1  # the arm has almost nothing to do ...

    shift_before = _mean_log_total(degraded) - _mean_log_total(clean)
    shift_after = _mean_log_total(matched) - _mean_log_total(clean)
    assert shift_before > 1.0  # ... and the control's level coordinate moved anyway
    assert shift_after > 1.0  # ... and is still moved after matching
    assert abs(shift_after - shift_before) < 0.05  # the arm barely touched it

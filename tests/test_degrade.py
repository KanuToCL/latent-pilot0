"""Degradation-library tests: each family's measured effect must match its
target (§3 Phase 1), plus the L3/L4 invariants. All run on the Mac."""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.audio.synth import synth_clip
from pilot0.degrade import FAMILIES, apply_degradation, clean
from pilot0.degrade.mp3 import has_ffmpeg

SR = 48000


@pytest.fixture
def wav():
    return synth_clip(0, sr=SR)


# --- analytic effect matches target per family --------------------------------


def test_noise_hits_target_snr(wav):
    for s, target in enumerate(FAMILIES["noise"].levels, 1):
        r = apply_degradation(wav, SR, "noise", s)
        assert r.measured["snr_db"] == pytest.approx(target, abs=0.5)


def test_hiss_hits_target_hf_snr(wav):
    for s, target in enumerate(FAMILIES["hiss"].levels, 1):
        r = apply_degradation(wav, SR, "hiss", s)
        assert r.measured["hf_snr_db"] == pytest.approx(target, abs=1.5)


def test_hum_hits_target_ratio(wav):
    for s, target in enumerate(FAMILIES["hum"].levels, 1):
        r = apply_degradation(wav, SR, "hum", s)
        assert r.measured["hts_db"] == pytest.approx(target, abs=0.5)


def test_clip_fraction_near_target_and_monotone(wav):
    meas = []
    for s, target in enumerate(FAMILIES["clip"].levels, 1):
        m = apply_degradation(wav, SR, "clip", s).measured["pct_clipped"]
        meas.append(m)
        assert m == pytest.approx(target, abs=0.2 + 0.25 * target)
        assert m > 0
    assert meas == sorted(meas)


def test_bandlimit_attenuates_stopband(wav):
    for s in (1, 2, 3, 4, 5):
        r = apply_degradation(wav, SR, "bandlimit", s)
        assert r.measured["stopband_atten_db"] < -10.0


def test_dropout_fraction_near_target_and_monotone(wav):
    meas = []
    for s, target in enumerate(FAMILIES["dropout"].levels, 1):
        m = apply_degradation(wav, SR, "dropout", s).measured["pct_lost"]
        meas.append(m)
        assert m == pytest.approx(target, rel=0.5, abs=0.3)
    assert meas == sorted(meas)


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg required for MP3 family")
def test_mp3_roundtrip_aligned_and_length_matched(wav):
    for s in (1, 5):
        r = apply_degradation(wav, SR, "mp3", s)
        assert r.measured["len_match"] == 1
        assert np.isfinite(r.wav).all()
        assert not np.array_equal(r.wav, wav.astype(np.float32))


# --- L3 / L4 invariants -------------------------------------------------------


def test_outputs_are_float32_finite_and_same_length(wav):
    for name in FAMILIES:
        if name == "mp3" and not has_ffmpeg():
            continue
        r = apply_degradation(wav, SR, name, 3)
        assert r.wav.dtype == np.float32
        assert np.isfinite(r.wav).all()
        assert len(r.wav) == len(wav)


def test_float32_storage_is_lossless(tmp_path, wav):
    import soundfile as sf

    r = apply_degradation(wav, SR, "noise", 5)  # SNR 0 dB → may exceed 0 dBFS
    path = tmp_path / "x.wav"
    sf.write(path, r.wav, SR, subtype="FLOAT")
    back, _ = sf.read(path, dtype="float32")
    assert np.array_equal(back, r.wav)  # exact → no silent second clipping (L4)


def test_true_peak_is_recorded(wav):
    assert np.isfinite(apply_degradation(wav, SR, "noise", 1).measured["true_peak_dbtp"])


# --- determinism + clean ------------------------------------------------------


def test_degradations_are_deterministic(wav):
    for name in FAMILIES:
        if name == "mp3" and not has_ffmpeg():
            continue
        a = apply_degradation(wav, SR, name, 4).wav
        b = apply_degradation(wav, SR, name, 4).wav
        assert np.array_equal(a, b)


def test_clean_passthrough(wav):
    r = clean(wav, SR)
    assert r.family == "clean" and r.severity_index == 0
    assert np.array_equal(r.wav, wav.astype(np.float32))


def test_severity_out_of_range_raises(wav):
    with pytest.raises(ValueError):
        apply_degradation(wav, SR, "noise", 6)


# --- independent (non-tautological) checks ------------------------------------


def _spectral_slope_db_per_octave(x, sr):
    power = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    band = (freqs > 100) & (freqs < sr / 2 * 0.9)
    slope, _ = np.polyfit(np.log2(freqs[band]), 10 * np.log10(power[band] + 1e-20), 1)
    return slope


def test_white_noise_is_spectrally_flat():
    # Test the generator directly (the SNR scaler is a separate concern).
    white = np.random.default_rng(0).standard_normal(SR * 4)
    assert abs(_spectral_slope_db_per_octave(white, SR)) < 1.0  # ~0 dB/oct


def test_pink_noise_has_pink_slope():
    from pilot0.degrade.noise import _pinkify

    pink = _pinkify(np.random.default_rng(0).standard_normal(SR * 4))
    slope = _spectral_slope_db_per_octave(pink, SR)
    assert -4.5 < slope < -1.5  # 1/f power ≈ −3 dB/octave


def test_clipping_actually_flattens_peaks(wav):
    # Independent of the fraction metric: clipped output must have flat tops.
    r = apply_degradation(wav, SR, "clip", 5)
    ceiling = np.max(np.abs(r.wav))
    assert np.sum(np.abs(r.wav) >= ceiling * (1 - 1e-4)) > 1  # many samples pinned


def test_bandlimit_invalid_above_nyquist_is_refused():
    from pilot0.degrade.bandlimit import band_limit
    from pilot0.degrade.grid import applicable_severities, is_applicable

    w16 = synth_clip(0, sr=16000)  # Nyquist 8000
    assert is_applicable("bandlimit", 3, 16000) is True  # 6000 < 8000
    assert is_applicable("bandlimit", 1, 16000) is False  # 12000 ≥ 8000
    assert is_applicable("bandlimit", 2, 16000) is False  # 8000 ≥ 8000
    assert applicable_severities("bandlimit", 16000) == (3, 4, 5)
    with pytest.raises(ValueError):
        apply_degradation(w16, 16000, "bandlimit", 1)
    with pytest.raises(ValueError):
        band_limit(w16, 16000, 12000)


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg required for MP3 family")
def test_mp3_top_hz_surfaces_lowbitrate_lowpass(wav):
    # The hidden LAME downsample: the lowest bitrate must not exceed the highest.
    top_64 = apply_degradation(wav, SR, "mp3", 1).measured["mp3_top_hz"]  # 64 kbps
    top_16 = apply_degradation(wav, SR, "mp3", 5).measured["mp3_top_hz"]  # 16 kbps
    assert top_16 <= top_64


def test_over_0dbfs_flag_present(wav):
    assert apply_degradation(wav, SR, "noise", 1).measured["over_0dbfs"] in (0, 1)

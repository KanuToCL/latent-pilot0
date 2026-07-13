"""Seam contract tests — all run on the Mac (fake path). Real-backend tests
skip when torch is absent and will execute at GPU-box bring-up."""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.seam.base import (
    Encoder,
    LatentResult,
    measure_frame_rate,
    pool_max,
    pool_mean_std,
)
from pilot0.seam.real import RealBackendUnavailable
from pilot0.seam.registry import REAL_SPECS, available_encoders, make_encoder

SR = 48000


def _wav(seed: int, n: int = SR) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def test_encode_returns_dict_of_declared_variants():
    enc = make_encoder("fake-wavlm")
    out = enc.encode(_wav(1), SR)
    assert set(out) == set(enc.variants) == {"l1", "l6", "l12", "l18", "l24"}
    assert set(make_encoder("fake-encodec24k").encode(_wav(1), SR)) == {"z", "d1", "d2", "d4", "d8"}


def test_fake_encoder_is_deterministic():
    enc = make_encoder("fake-encodec24k")
    wav = _wav(1)
    a, b = enc.encode(wav, SR)["z"], enc.encode(wav, SR)["z"]
    assert np.array_equal(a.frames, b.frames)
    assert a.frame_rate_hz == b.frame_rate_hz


def test_fake_encoder_shape_and_meta_contract():
    enc = make_encoder("fake-wavlm")
    r = enc.encode(_wav(2), SR)["l12"]
    t, d = r.frames.shape
    assert d == enc.latent_dim == r.latent_dim
    assert t == r.n_frames > 0
    assert r.frame_rate_hz > 0
    assert r.meta["backend"] == "fake"
    assert r.meta["variant"] == "l12"
    assert r.meta["native_sr"] == enc.native_sr


def test_variants_are_distinct():
    out = make_encoder("fake-wavlm").encode(_wav(7), SR)
    assert not np.allclose(out["l1"].frames, out["l24"].frames)


def test_pool_mean_std_and_max():
    frames = np.arange(12, dtype=float).reshape(3, 4)
    ms = pool_mean_std(frames)
    assert ms.shape == (8,)
    assert np.allclose(ms[:4], frames.mean(axis=0))
    assert np.allclose(ms[4:], frames.std(axis=0))
    mx = pool_max(frames)
    assert mx.shape == (4,)
    assert np.allclose(mx, frames.max(axis=0))


def test_measure_frame_rate():
    # 100 frames over 2 s → 50 fps
    assert measure_frame_rate(100, 32000, 16000) == pytest.approx(50.0)
    assert measure_frame_rate(0, 0, 16000) == 0.0


def test_different_audio_gives_different_latents():
    enc = make_encoder("fake-encodec24k")
    a, b = enc.encode(_wav(3), SR)["z"], enc.encode(_wav(4), SR)["z"]
    assert not np.allclose(pool_mean_std(a.frames), pool_mean_std(b.frames))


def test_encoder_resamples_to_native_rate():
    wav = _wav(5)
    n24 = make_encoder("fake-encodec24k").encode(wav, SR)["z"].n_frames
    n16 = make_encoder("fake-wavlm").encode(wav, SR)["l12"].n_frames
    assert n24 > n16  # 24k has more samples than 16k after resample


def test_latentresult_validates():
    with pytest.raises(ValueError):
        LatentResult(frames=np.zeros(5), frame_rate_hz=50.0)  # not 2-D
    with pytest.raises(ValueError):
        LatentResult(frames=np.zeros((3, 4)), frame_rate_hz=0.0)  # bad rate


def test_fake_encoder_satisfies_protocol():
    assert isinstance(make_encoder("fake-encodec24k"), Encoder)


def test_registry_lists_fake_and_real():
    avail = available_encoders()
    for base in REAL_SPECS:
        assert avail[f"fake-{base}"] is True
        assert base in avail


def test_availability_reflects_wired_families():
    # All four families are wired as of Phase 5; availability = wired AND the family's
    # backend lib present. So a name reported available MUST have its lib importable —
    # never claim encodable then raise ImportError at encode time (findings S4).
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("no torch — real availability is uniformly False on the Mac")
    import importlib.util

    from pilot0.seam.registry import _FAMILY_REQUIRES, WIRED_FAMILIES

    assert {"encodec", "wavlm", "dac", "mimi"} <= WIRED_FAMILIES
    avail = available_encoders()
    for name, spec in REAL_SPECS.items():
        if avail[name]:
            assert importlib.util.find_spec(_FAMILY_REQUIRES[spec["family"]]) is not None


def test_unknown_encoder_raises():
    with pytest.raises(KeyError):
        make_encoder("fake-nonsense")
    with pytest.raises(KeyError):
        make_encoder("nonsense")


def test_real_backend_guarded_without_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(RealBackendUnavailable):
            make_encoder("encodec24k")
    else:
        pytest.skip("torch present — real backend constructs (validated on the box)")

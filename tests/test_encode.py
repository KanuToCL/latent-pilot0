"""Phase-3 encode & cache tests (Mac, fake seam)."""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.audio.synth import synth_clip
from pilot0.corpus.io import write_audio
from pilot0.corpus.manifest import build_manifest, grid_signature, renderable_rows
from pilot0.corpus.preflight import preflight
from pilot0.degrade.base import true_peak_dbtp
from pilot0.encode.cache import (
    cache_version,
    cell_id,
    code_version,
    corpus_signature,
    is_cached,
    latent_path,
    load_latent,
    save_latent,
)
from pilot0.encode.headroom import CEILING_DBFS, measure_headroom
from pilot0.encode.pipeline import encode_corpus
from pilot0.encode.render import render_cell, resample_to_native
from pilot0.seam.base import LatentResult
from pilot0.seam.registry import make_encoder

SR = 48000


@pytest.fixture
def corpus(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        write_audio(src / f"c{i}.wav", synth_clip(i, sr=SR), SR)
    pf = preflight(src, tmp_path / "out")
    return pf, build_manifest(pf)


# --- render (L1) --------------------------------------------------------------


def test_render_resamples_master_to_native_rate():
    master = synth_clip(0, sr=SR, seconds=2.0)
    d = render_cell(master, SR, 16000, "clean", 0)
    assert d.sr == 16000
    assert d.wav.shape[0] == pytest.approx(len(master) * 16000 / SR, rel=1e-3)


def test_render_degrades_at_native_rate():
    master = synth_clip(1, sr=SR, seconds=2.0)
    d = render_cell(master, SR, 24000, "noise", 5)  # SNR 0 dB
    assert d.family == "noise" and d.sr == 24000
    assert d.wav.shape[0] == render_cell(master, SR, 24000, "clean", 0).wav.shape[0]


def test_render_rejects_bandlimit_above_nyquist():
    master = synth_clip(2, sr=SR, seconds=1.0)
    # 12 kHz cutoff (severity 1) is ≥ Nyquist at 16 kHz → undefined (L1).
    with pytest.raises(ValueError):
        render_cell(master, SR, 16000, "bandlimit", 1)
    assert render_cell(master, SR, 16000, "bandlimit", 5).wav.size > 0  # 3.4 kHz is fine


def test_resample_is_identity_at_same_rate():
    x = synth_clip(3, sr=SR, seconds=0.5)
    assert np.array_equal(resample_to_native(x, SR, SR), x.reshape(-1))


# --- headroom (L4) ------------------------------------------------------------


def test_headroom_pulls_hottest_cell_under_ceiling(corpus):
    pf, man = corpus
    masters = {r.token: (np.asarray(sf_read(pf, r.token)), r.sr) for r in pf.accepted}
    rows = renderable_rows(man, 24000)
    head = measure_headroom(rows, lambda t: masters[t], 24000)

    # Re-render the cell that set the max, attenuate, and confirm it clears −1 dBFS.
    lm = head.loudest
    m, msr = masters[lm["source"]]
    d = render_cell(m, msr, 24000, lm["family"], lm["severity"])
    assert true_peak_dbtp(d.wav * head.scalar, 24000) <= head.ceiling_dbfs + 1e-6
    assert head.scalar <= 1.0  # attenuate only, never amplify


def test_headroom_scalar_is_one_for_quiet_corpus():
    # A −23 LUFS master with no degradation peaks well below −1 dBFS → no attenuation.
    master = synth_clip(0, sr=SR, seconds=2.0)
    rows = [{"source": "m", "family": "clean", "severity": 0}]
    head = measure_headroom(rows, lambda t: (master, SR), 16000)
    assert head.scalar == 1.0
    assert head.max_dbtp < CEILING_DBFS


# --- cache key + roundtrip ----------------------------------------------------


def test_cell_id_depends_on_every_coordinate():
    base = cell_id("src_abc", "noise", 3, 24000)
    assert base == cell_id("src_abc", "noise", 3, 24000)  # stable
    assert base != cell_id("src_xyz", "noise", 3, 24000)  # token
    assert base != cell_id("src_abc", "hum", 3, 24000)  # family
    assert base != cell_id("src_abc", "noise", 4, 24000)  # severity
    assert base != cell_id("src_abc", "noise", 3, 16000)  # rate


def test_code_version_tracks_grid_signature():
    assert grid_signature() in code_version()


def test_cache_version_invalidates_on_corpus_and_ceiling(corpus):
    _, man = corpus
    base = cache_version(man, -1.0)
    assert code_version() in base  # semantics carried through
    assert cache_version(man, -1.0) == base  # stable
    assert cache_version(man, -2.0) != base  # ceiling folds in (finding B)

    drop = man["rows"][0]["source"]
    man2 = dict(man, rows=[r for r in man["rows"] if r["source"] != drop])  # drop a whole source
    assert corpus_signature(man2) != corpus_signature(man)
    assert cache_version(man2, -1.0) != base  # corpus rescales L4 → invalidate (finding C)


def test_save_latent_rejects_non_finite(tmp_path):
    bad = LatentResult(frames=np.array([[np.nan, 0.0]], dtype=np.float32), frame_rate_hz=50.0)
    with pytest.raises(ValueError):
        save_latent(tmp_path / "bad.npz", bad, {"source": "x"})
    assert not (tmp_path / "bad.npz").exists()  # nothing published on failure


def test_save_load_latent_roundtrip(tmp_path):
    frames = np.random.default_rng(0).standard_normal((17, 32)).astype(np.float32)
    src = LatentResult(frames=frames, frame_rate_hz=50.0, meta={"backend": "fake", "variant": "z"})
    path = tmp_path / "cell.npz"
    save_latent(path, src, {"source": "src_abc", "family": "noise", "severity": 3})

    got = load_latent(path)
    assert np.array_equal(got.frames, frames)
    assert got.frame_rate_hz == 50.0
    assert got.meta["variant"] == "z" and got.meta["source"] == "src_abc"


# --- pipeline: cache population + resume ---------------------------------------


def test_encode_corpus_caches_every_renderable_variant(corpus, tmp_path):
    pf, man = corpus
    cache = tmp_path / "cache"
    report = encode_corpus(pf.norm_dir, man, ["fake-encodec24k", "fake-wavlm"], cache)

    expected = 0
    for name in ["fake-encodec24k", "fake-wavlm"]:
        enc = make_encoder(name)
        expected += len(renderable_rows(man, enc.native_sr)) * len(enc.variants)
    assert report.n_latents == expected
    assert sum(s.encoded for s in report.per_encoder) == expected  # all fresh
    assert sum(s.skipped for s in report.per_encoder) == 0


def test_encode_corpus_is_resume_safe(corpus, tmp_path):
    pf, man = corpus
    cache = tmp_path / "cache"
    first = encode_corpus(pf.norm_dir, man, ["fake-encodec24k"], cache)
    again = encode_corpus(pf.norm_dir, man, ["fake-encodec24k"], cache)
    assert sum(s.encoded for s in again.per_encoder) == 0  # nothing re-encoded
    assert again.n_latents == first.n_latents


def test_cached_latent_is_locatable_and_correct(corpus, tmp_path):
    pf, man = corpus
    cache = tmp_path / "cache"
    encode_corpus(pf.norm_dir, man, ["fake-encodec24k"], cache)

    cv, row = cache_version(man, CEILING_DBFS), renderable_rows(man, 24000)[0]
    cid = cell_id(row["source"], row["family"], row["severity"], 24000)
    assert is_cached(cache, "fake-encodec24k", "z", cid, cv)
    lat = load_latent(latent_path(cache, "fake-encodec24k", "z", cid, cv))
    assert lat.meta["source"] == row["source"]
    assert lat.meta["degraded_sha"] and lat.meta["code_version"] == cv


def test_latents_are_deterministic_across_runs(corpus, tmp_path):
    pf, man = corpus
    a, b = tmp_path / "a", tmp_path / "b"
    encode_corpus(pf.norm_dir, man, ["fake-encodec24k"], a)
    encode_corpus(pf.norm_dir, man, ["fake-encodec24k"], b)

    cv, row = cache_version(man, CEILING_DBFS), renderable_rows(man, 24000)[0]
    cid = cell_id(row["source"], row["family"], row["severity"], 24000)
    fa = load_latent(latent_path(a, "fake-encodec24k", "z", cid, cv)).frames
    fb = load_latent(latent_path(b, "fake-encodec24k", "z", cid, cv)).frames
    assert np.array_equal(fa, fb)


def sf_read(pf, token):
    import soundfile as sf

    wav, _ = sf.read(str(pf.norm_dir / f"{token}.wav"), dtype="float64")
    return wav

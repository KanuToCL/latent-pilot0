"""Phase-7 combo tests. Fabricated latents with a KNOWN additivity geometry pin the
cosine (≈1 when the combo sits at the sum of the single displacements, ≈0 when it sits
orthogonal); the grid applicability, the ordered two-leg render, zero-shot transfer,
and an encode→read roundtrip are exercised directly."""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.audio.synth import synth_clip
from pilot0.combos.additivity import additivity
from pilot0.combos.dataset import ComboData, build_combo_data
from pilot0.combos.encode import encode_combos, render_combo
from pilot0.combos.grid import apply_combo, combo_label, combo_severities
from pilot0.combos.transfer import transfer_to_combos
from pilot0.corpus.io import write_audio
from pilot0.corpus.manifest import build_manifest
from pilot0.corpus.preflight import preflight
from pilot0.encode.pipeline import encode_corpus
from pilot0.probes.dataset import ProbeData
from pilot0.seam.registry import make_encoder

SR = 48000
DIM = 8
SEV = 3


def _fab(n_sources: int = 6, additive: bool = True, noise: float = 0.04, seed: int = 0):
    """SINGLE displacements along orthogonal axes; the combo sits either at their sum
    (additive → cos≈1) or on a third axis (non-additive → cos≈0). Each source adds a
    shared latent component to all of its rows (clean/a/b/ab) so EVERY dimension has
    variance — like a real pooled latent — and standardisation stays well-conditioned
    (the source component cancels in every centroid difference)."""
    rng = np.random.default_rng(seed)
    mu_a = np.eye(DIM)[0] * 3.0
    mu_b = np.eye(DIM)[1] * 3.0
    mu_ab = (mu_a + mu_b) if additive else np.eye(DIM)[2] * 3.0

    pX, pfam, psev, psplit, pgroup = [], [], [], [], []
    cX, cpair, cla, clb, csev, csplit, cgroup, csrc = [], [], [], [], [], [], [], []
    for s in range(n_sources):
        g, split = f"s{s}", ("test" if s >= n_sources - 2 else "train")
        src_vec = rng.standard_normal(DIM)  # source identity, shared across this source's rows
        for mu, fam, sev in [(np.zeros(DIM), "clean", 0), (mu_a, "noise", SEV), (mu_b, "clip", SEV)]:
            pX.append(src_vec + mu + noise * rng.standard_normal(DIM))
            pfam.append(fam), psev.append(sev), psplit.append(split), pgroup.append(g)
        cX.append(src_vec + mu_ab + noise * rng.standard_normal(DIM))
        cpair.append("noise+clip"), cla.append("noise"), clb.append("clip")
        csev.append(SEV), csplit.append(split), cgroup.append(g), csrc.append(g)

    probe = ProbeData(
        X=np.asarray(pX), family=np.asarray(pfam), severity=np.asarray(psev, int),
        split=np.asarray(psplit), group=np.asarray(pgroup), source=np.asarray(pgroup),
        encoder="fab", variant="v",
    )
    combo = ComboData(
        X=np.asarray(cX), pair=np.asarray(cpair), leg_a=np.asarray(cla), leg_b=np.asarray(clb),
        severity=np.asarray(csev, int), split=np.asarray(csplit), group=np.asarray(cgroup),
        source=np.asarray(csrc), encoder="fab", variant="v",
    )
    return probe, combo


# --- grid ---------------------------------------------------------------------


def test_combo_severities_gate_on_nyquist():
    assert combo_severities("noise", "clip", 24000) == (2, 3, 4)
    # band-limit cutoff at severity 2 is 8 kHz = Nyquist at 16 kHz → dropped (L1).
    assert combo_severities("hiss", "bandlimit", 16000) == (3, 4)


def test_apply_combo_is_ordered_and_finite():
    native = 0.4 * np.sin(2 * np.pi * 220.0 * np.arange(SR) / SR)
    ab = apply_combo(native, SR, "noise", "clip", SEV)
    ba = apply_combo(native, SR, "clip", "noise", SEV)
    assert ab.shape == native.shape and np.isfinite(ab).all()
    assert not np.allclose(ab, ba)  # degradations do not commute → order matters


def test_render_combo_is_finite_and_length_preserving_at_native_rate():
    """Master already AT the native rate → resample_to_native is the identity, so the
    two-leg render must hand back the same number of samples, all finite. (A master at a
    different rate legitimately changes length; that is L1 resampling, not this test.)"""
    sr = 24000
    master = 0.4 * np.sin(2 * np.pi * 220.0 * np.arange(sr) / sr)
    for a, b in [("noise", "clip"), ("hiss", "bandlimit"), ("hum", "mp3")]:
        wav = render_combo(master, sr, sr, a, b, SEV)
        assert wav.shape == master.shape, f"{a}+{b} changed length"
        assert np.isfinite(wav).all(), f"{a}+{b} produced non-finite audio"


# --- additivity ---------------------------------------------------------------


def test_additivity_cosine_near_one_when_combo_is_the_sum():
    pa = additivity(*_fab(additive=True), n_boot=128).by_cell[("noise+clip", SEV)]
    assert pa.cosine.point > 0.95 and pa.cosine.lo > 0.5  # combo codirectional with Δa+Δb
    assert pa.cosine_std.point > 0.9  # standardised basis agrees on the well-conditioned fab


def test_additivity_cosine_near_zero_when_combo_is_orthogonal():
    pa = additivity(*_fab(additive=False), n_boot=128).by_cell[("noise+clip", SEV)]
    assert abs(pa.cosine.point) < 0.4 and abs(pa.cosine_std.point) < 0.5


def test_additivity_reports_common_source_count():
    res = additivity(*_fab(n_sources=6, additive=True), n_boot=64)
    assert res.by_cell[("noise+clip", SEV)].n_groups == 6  # all sources carry every role


def test_additivity_handles_empty_combo_data():
    probe, _ = _fab()
    empty = ComboData(
        X=np.asarray([]), pair=np.asarray([]), leg_a=np.asarray([]), leg_b=np.asarray([]),
        severity=np.asarray([], int), split=np.asarray([]), group=np.asarray([]),
        source=np.asarray([]), encoder="fab", variant="v",
    )
    assert additivity(probe, empty).by_cell == {}  # no combos encoded → nothing to test, no crash


# --- zero-shot transfer -------------------------------------------------------


def test_transfer_reads_combo_as_a_constituent_leg():
    # Put the combo cluster ON leg-a's single centroid → the single-trained probe must
    # call it "noise" (a constituent), not a third family.
    probe, combo = _fab(additive=True)
    combo = ComboData(**{**combo.__dict__, "X": np.tile(np.eye(DIM)[0] * 3.0, (len(combo.X), 1))})
    res = transfer_to_combos(probe, combo)
    pt = res.by_pair["noise+clip"]
    assert pt.predicted_fraction.get("noise", 0.0) == pytest.approx(1.0)
    assert pt.frac_either_leg == pytest.approx(1.0)


# --- encode → read roundtrip --------------------------------------------------


def test_encode_combos_roundtrip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        write_audio(src / f"c{i}.wav", synth_clip(i, sr=SR, seconds=1.5), SR)
    pf = preflight(src, tmp_path / "out")
    man = build_manifest(pf)
    cache = tmp_path / "cache"
    encode_corpus(pf.norm_dir, man, ["fake-encodec24k"], cache)
    stats = encode_combos(pf.norm_dir, man, ["fake-encodec24k"], cache)
    n_variants = len(make_encoder("fake-encodec24k").variants)
    # 4 sources × (3 pairs × 3 mid severities) cells, each written for every variant.
    assert stats[0].encoded == 4 * 9 * n_variants

    data = build_combo_data(man, "fake-encodec24k", "z", cache)  # one variant read back
    assert len(data.X) == stats[0].encoded // n_variants
    assert set(data.pair) == {"noise+clip", "hiss+bandlimit", "hum+mp3"}
    assert set(data.severity.tolist()) == {2, 3, 4}

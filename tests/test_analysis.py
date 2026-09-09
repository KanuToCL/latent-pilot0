"""Phase-5 analysis tests. The fabricated design matrix injects family identity as
near-orthogonal one-hot directions and severity as a shared linear dimension, so a
working geometry/interpolation/nonlinearity analysis MUST recover that structure.
Frame-level dropout features are tested directly on frames with known zeroing; one
integration test runs the full matrix through the cache."""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.analysis.geometry import geometry
from pilot0.analysis.readability import readability_row
from pilot0.audio.synth import synth_clip
from pilot0.corpus.io import write_audio
from pilot0.corpus.manifest import build_manifest
from pilot0.corpus.preflight import preflight
from pilot0.degrade.grid import FAMILIES
from pilot0.encode.pipeline import encode_corpus
from pilot0.probes.dataset import ProbeData, build_probe_data
from pilot0.probes.frame_dropout import evaluate_frame_dropout, frame_dropout_features
from pilot0.probes.interpolation import evaluate_interpolation
from pilot0.probes.mlp_probe import evaluate_nonlinearity
from pilot0.probes.run import EncoderProbeResult
from pilot0.probes.severity import evaluate_severity_probes
from pilot0.probes.type_probe import evaluate_type_probe
from pilot0.seam.registry import REAL_SPECS, WIRED_FAMILIES, make_encoder

SR = 48000
FAMS = list(FAMILIES)


def _fabricate(n_sources: int = 12, noise: float = 0.3, seed: int = 0) -> ProbeData:
    """Family identity one-hot in dims 0..6, severity in dim 7 — the same fixture the
    Phase-4 probes use, so structure is linearly recoverable and source-disjoint."""
    rng = np.random.default_rng(seed)
    X, fam, sev, split, group = [], [], [], [], []
    for src in range(n_sources):
        sp = "test" if src >= n_sources - 4 else "train"
        g = f"src{src}"
        for fi, f in enumerate(FAMS):
            for s in range(1, 6):
                feat = np.zeros(9)
                feat[fi], feat[7] = 5.0, float(s)
                X.append(feat + noise * rng.standard_normal(9))
                fam.append(f), sev.append(s), split.append(sp), group.append(g)
        X.append(noise * rng.standard_normal(9))
        fam.append("clean"), sev.append(0), split.append(sp), group.append(g)
    return ProbeData(
        X=np.asarray(X), family=np.asarray(fam), severity=np.asarray(sev, int),
        split=np.asarray(split), group=np.asarray(group), encoder="fab", variant="v",
    )


def _map_fields(d: ProbeData):
    return d.X, d.family, d.severity


# --- representation matrix expansion ------------------------------------------


def test_rvq_depth_variants_declared_and_shared_across_families():
    assert REAL_SPECS["encodec24k"]["variants"] == ("z", "d1", "d2", "d4", "d8")
    # DAC's `z` is the QUANTIZER output (F7), so the true pre-quant point is `enc` —
    # a sixth variant, appended so the banked `dac44k/z/*.npz` keep their key (D1).
    assert REAL_SPECS["dac44k"]["variants"] == ("z", "d1", "d2", "d4", "d8", "enc")
    assert WIRED_FAMILIES == {"encodec", "wavlm", "dac", "mimi"}


def test_fake_emits_every_declared_variant():
    for base, spec in REAL_SPECS.items():
        out = make_encoder(f"fake-{base}").encode(synth_clip(0, sr=SR, seconds=1.0), SR)
        assert tuple(out) == spec["variants"]
        for r in out.values():
            assert r.latent_dim == spec["latent_dim"]  # depths share the embedding space


# --- geometry -----------------------------------------------------------------


def test_geometry_directions_are_near_orthogonal_and_diag_unit():
    g = geometry(*_map_fields(_fabricate()))
    assert g.probe_cosine.shape == (7, 7)
    assert np.allclose(np.diag(g.probe_cosine), 1.0, atol=1e-6)
    assert np.allclose(g.probe_cosine, g.probe_cosine.T, atol=1e-6)
    off = g.probe_cosine[~np.eye(7, dtype=bool)]
    assert off.max() < 0.5  # distinct one-hot families → clearly separate directions


def test_geometry_centroid_pca_is_a_valid_spectrum():
    g = geometry(*_map_fields(_fabricate()))
    assert g.centroid_pca_explained.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(g.centroid_pca_explained) <= 1e-9)  # descending
    assert g.centroid_coords.shape[0] == len(g.centroid_labels)


def test_geometry_centroid_count_equals_degraded_conditions():
    d = _fabricate()
    g = geometry(*_map_fields(d))
    deg = d.family != "clean"
    expected = len(set(zip(d.family[deg].tolist(), d.severity[deg].tolist())))
    assert len(g.centroid_labels) == expected  # would catch the "36" doc drift (C1)


def test_geometry_requires_clean_anchor():
    d = _fabricate()
    deg = d.family != "clean"
    with pytest.raises(ValueError):
        geometry(d.X[deg], d.family[deg], d.severity[deg])


# --- interpolation / monotonicity ---------------------------------------------


def test_interpolation_recovers_held_out_level():
    res = evaluate_interpolation(_fabricate())
    assert set(res.by_family) == set(FAMILIES)
    for fam, fi in res.by_family.items():
        assert fi.evaluable, fam  # neighbours 2, 4 and held-out 3 all present
        assert fi.interpolates, fam  # ordering reliable AND positive SRCC CI-lower
        assert fi.interp_frac > 0.9 and fi.srcc.point > 0.9


def test_interpolation_flag_does_not_fire_on_noise():
    # No real severity structure → the co-clearing flag must not pass on luck (B1/P1).
    res = evaluate_interpolation(_fabricate(noise=50.0, seed=2))
    assert sum(fi.interpolates for fi in res.by_family.values()) <= 1


def test_interpolation_beyond_energy_controlled_count():
    res = evaluate_interpolation(_fabricate())
    # Against itself as baseline, no family's CI-lower can exceed its own CI-upper.
    assert res.n_interpolate(res) == 0
    assert res.n_interpolate() == 7  # unconditional count on clean structure


def test_interpolation_curve_is_monotone_on_clean_structure():
    curve = evaluate_interpolation(_fabricate()).by_family["noise"].curve
    vals = [curve[lvl] for lvl in (1, 2, 3, 4, 5)]
    assert vals == sorted(vals)  # predicted severity rises with true severity


# --- nonlinearity gap ---------------------------------------------------------


def test_nonlinearity_gap_small_on_linearly_separable_data():
    res = evaluate_nonlinearity(_fabricate())
    assert res.linear_macro_f1.point > 0.9  # linear already solves it
    assert res.gap.hi < 0.15  # no large nonlinear headroom to gain
    assert np.isfinite([res.gap.lo, res.gap.point, res.gap.hi]).all()


# --- frame-level dropout features ---------------------------------------------


def test_frame_dropout_features_track_zeroed_fraction():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((100, 8)) + 3.0  # non-zero baseline frames

    def with_dropped(k):
        f = base.copy()
        f[:k] = 0.0
        return frame_dropout_features(f)

    frac_low_idx = 3  # first silence fraction (r < 0.5)
    assert with_dropped(40)[frac_low_idx] > with_dropped(5)[frac_low_idx]
    assert with_dropped(0)[frac_low_idx] == pytest.approx(0.0, abs=0.05)


def test_frame_dropout_features_finite_on_silence_and_heavy_dropout():
    assert np.isfinite(frame_dropout_features(np.zeros((50, 8)))).all()  # true silence
    heavy = np.random.default_rng(0).standard_normal((50, 8)) + 3.0
    heavy[:30] = 0.0  # >50% zeroed → median norm is 0, must not blow up to inf/nan
    assert np.isfinite(frame_dropout_features(heavy)).all()


def test_readability_row_flattens_probe_result():
    data = _fabricate()
    res = EncoderProbeResult("fab", "v", evaluate_type_probe(data), evaluate_severity_probes(data), 4)
    row = readability_row(res)
    assert row.name == "fab" and list(row.severity_srcc) == FAMS
    assert len(row.type_macro_f1) == 3


# --- integration: cache → frame-level dropout probe ---------------------------


def test_frame_dropout_probe_runs_through_cache(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):
        write_audio(src / f"c{i}.wav", synth_clip(i, sr=SR, seconds=1.5), SR)
    pf = preflight(src, tmp_path / "out")
    man = build_manifest(pf)
    cache = tmp_path / "cache"
    encode_corpus(pf.norm_dir, man, ["fake-encodec24k"], cache)

    res = evaluate_frame_dropout(man, "fake-encodec24k", "z", cache)
    assert res.n_test > 0 and res.n_train > 0
    for est in (res.frame_srcc, res.pooled_srcc, res.gap):
        assert -1.0 <= est.point <= 1.0 or np.isnan(est.point)

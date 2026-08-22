"""RQ2 shift-vector tests. Every fixture is constructed so the ANSWER IS KNOWN in
closed form — pairing by source, an exactly-computable mean pairwise cosine, a
||Δ|| ladder that is monotone by construction, and a rank-2 planted structure whose
geometry a 2-D projection must reproduce exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.analysis.shift import (
    concentration,
    family_concentration,
    magnitude_curve,
    reduction_persistence,
    shift_vectors,
    standardize_on_degraded,
)

FAMS = ("noise", "hum", "clip")


def _brute_mean_pairwise_cosine(deltas: np.ndarray) -> float:
    """Reference implementation of `concentration`: the literal double loop."""
    u = deltas / np.linalg.norm(deltas, axis=1, keepdims=True)
    n = len(u)
    vals = [float(u[i] @ u[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


# --- 1. shift vectors ---------------------------------------------------------


def _paired_fixture():
    """Sources A and B carry a clean row; source C does not (its degraded row must
    be dropped). Clean anchors differ per source, so mispairing changes the answer."""
    X = np.array(
        [
            [0.0, 0.0],  # A clean
            [1.0, 0.0],  # A noise sev 1  -> Δ = [1, 0]
            [0.0, 2.0],  # A hum   sev 2  -> Δ = [0, 2]
            [10.0, 10.0],  # B clean
            [11.0, 10.0],  # B noise sev 1  -> Δ = [1, 0]
            [5.0, 5.0],  # C noise sev 1  -> DROPPED (no clean row for C)
        ]
    )
    family = np.array(["clean", "noise", "hum", "clean", "noise", "noise"])
    severity = np.array([0, 1, 2, 0, 1, 1])
    source = np.array(["A", "A", "A", "B", "B", "C"])
    return X, family, severity, source


def test_shift_vectors_pairs_each_degraded_row_with_its_own_clean_row():
    sv = shift_vectors(*_paired_fixture())
    expected = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]])  # A/noise, A/hum, B/noise
    assert sv.deltas == pytest.approx(expected, abs=1e-12)
    assert list(sv.family) == ["noise", "hum", "noise"]
    assert list(sv.severity) == [1, 2, 1]
    assert list(sv.source) == ["A", "A", "B"]


def test_shift_vectors_drops_sources_without_a_clean_partner():
    sv = shift_vectors(*_paired_fixture())
    assert sv.n_degraded == 4  # three paired + source C's orphan
    assert sv.n_unpaired_rows == 1
    assert sv.n_unpaired_sources == 1
    assert "C" not in set(sv.source.tolist())


def test_shift_vectors_rejects_a_duplicated_clean_anchor():
    X, family, severity, source = _paired_fixture()
    family = family.copy()
    family[1] = "clean"  # source A now has two clean rows -> ambiguous anchor
    with pytest.raises(ValueError, match="clean row"):
        shift_vectors(X, family, severity, source)


def test_standardize_fits_on_degraded_rows_but_transforms_the_clean_anchor_too():
    X, family, _, _ = _paired_fixture()
    Xs = standardize_on_degraded(X, family)
    assert Xs.shape == X.shape  # clean rows survive; they anchor the shift vectors
    deg = family != "clean"
    assert Xs[deg].mean(axis=0) == pytest.approx(np.zeros(2), abs=1e-12)
    assert Xs[deg].std(axis=0) == pytest.approx(np.ones(2), abs=1e-12)


# --- 2. concentration ---------------------------------------------------------


def test_concentration_of_identical_directions_is_one():
    deltas = np.tile(np.array([1.0, -2.0, 0.5, 3.0]), (25, 1))
    c = concentration(deltas)
    assert c.mean_cosine == pytest.approx(1.0, abs=1e-12)
    assert c.n_rows == 25
    assert c.n_pairs == 25 * 24 // 2  # exact, all pairs — nothing sampled
    assert c.null_scale == pytest.approx(0.5, abs=1e-12)  # 1/sqrt(4)


def test_concentration_of_isotropic_noise_sits_at_the_null_scale():
    # D = 64 -> null 1/8 = 0.125; independent gaussian directions must stay inside 3x.
    deltas = np.random.default_rng(0).standard_normal((200, 64))
    c = concentration(deltas)
    assert c.null_scale == pytest.approx(1.0 / 8.0, abs=1e-12)
    assert abs(c.mean_cosine) < 3.0 / np.sqrt(64)  # 0.375


def test_concentration_matches_the_explicit_pairwise_loop():
    # The closed form (||sum u||^2 - n) / (n(n-1)) must equal the literal mean.
    deltas = np.random.default_rng(1).standard_normal((40, 8)) + np.array([1.0] + [0.0] * 7)
    assert concentration(deltas).mean_cosine == pytest.approx(
        _brute_mean_pairwise_cosine(deltas), abs=1e-12
    )


def test_concentration_drops_zero_norm_rows_instead_of_biasing_the_mean():
    base = np.tile(np.array([0.0, 1.0]), (10, 1))
    withz = np.vstack([base, np.zeros((3, 2))])
    c = concentration(withz)
    assert c.n_zero_norm == 3
    assert c.n_rows == 10
    assert c.mean_cosine == pytest.approx(1.0, abs=1e-12)  # unchanged by the zero rows


def test_concentration_is_undefined_for_fewer_than_two_directions():
    assert np.isnan(concentration(np.ones((1, 5))).mean_cosine)


def test_family_concentration_splits_by_label():
    deltas = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    family = np.array(["a", "a", "b", "b"])
    out = family_concentration(deltas, family)
    assert out["a"].mean_cosine == pytest.approx(1.0, abs=1e-12)  # identical
    assert out["b"].mean_cosine == pytest.approx(-1.0, abs=1e-12)  # antiparallel


# --- 3. magnitude vs severity -------------------------------------------------


def _magnitude_fixture(n_sources: int = 6):
    """||Δ|| = severity exactly, along a family-specific unit direction."""
    dirs = {f: np.eye(3)[i] for i, f in enumerate(FAMS)}
    deltas, family, severity = [], [], []
    for f in FAMS:
        for s in (1, 2, 3, 4, 5):
            for _ in range(n_sources):
                deltas.append(float(s) * dirs[f])
                family.append(f)
                severity.append(s)
    return np.array(deltas), np.array(family), np.array(severity)


def test_magnitude_curve_median_norm_equals_the_planted_severity():
    mc = magnitude_curve(*_magnitude_fixture())
    for f in FAMS:
        for s in (1, 2, 3, 4, 5):
            assert mc.median_norm[f][s] == pytest.approx(float(s), abs=1e-12)
            assert mc.n_by_cell[f][s] == 6


def test_magnitude_curve_spearman_is_exactly_one_on_a_monotone_ladder():
    mc = magnitude_curve(*_magnitude_fixture())
    for f in FAMS:
        assert mc.srcc_by_family[f] == pytest.approx(1.0, abs=1e-12)


def test_magnitude_curve_spearman_is_minus_one_when_the_ladder_is_inverted():
    deltas, family, severity = _magnitude_fixture()
    inverted = deltas / np.linalg.norm(deltas, axis=1, keepdims=True) ** 2  # ||Δ|| = 1/severity
    mc = magnitude_curve(inverted, family, severity)
    for f in FAMS:
        assert mc.srcc_by_family[f] == pytest.approx(-1.0, abs=1e-12)


def test_magnitude_curve_is_undefined_for_a_single_severity_level():
    deltas = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    mc = magnitude_curve(deltas, np.array(["noise"] * 3), np.array([3, 3, 3]))
    assert np.isnan(mc.srcc_by_family["noise"])  # no rank variance in severity


# --- 4. persistence under reduction -------------------------------------------


def _rank2_fixture():
    """All degradation structure lives in the plane span(e0, e1) of a 6-D space, at
    120 degrees between families, with ||Δ|| = severity. The degraded rows are that
    plane plus ONE constant offset, so the top-2 PCs of the rows span exactly the
    plane the Δs live in and a k=2 projection is an isometry on it."""
    offset = np.array([0.0, 0.0, 3.0, -1.0, 0.5, 2.0])
    angles = {f: 2.0 * np.pi * i / 3.0 for i, f in enumerate(FAMS)}
    deltas, family, severity = [], [], []
    for f in FAMS:
        for s in (1, 2, 3):
            d = np.zeros(6)
            d[0], d[1] = np.cos(angles[f]), np.sin(angles[f])
            deltas.append(float(s) * d)
            family.append(f)
            severity.append(s)
    deltas = np.array(deltas)
    X_deg_std = offset + deltas  # every clean anchor sits at `offset`
    return X_deg_std, deltas, np.array(family), np.array(severity)


def test_reduction_persistence_preserves_rank2_structure_at_k2():
    X_deg_std, deltas, family, _ = _rank2_fixture()
    rp = reduction_persistence(X_deg_std, deltas, family, ks=(2,))
    assert rp.labels == sorted(FAMS)
    assert rp.by_k[2].cumulative_explained == pytest.approx(1.0, abs=1e-10)  # rank 2
    # The class-mean cosine matrix must be untouched by the projection.
    assert rp.by_k[2].mean_cosine == pytest.approx(rp.full_mean_cosine, abs=1e-10)


def test_reduction_persistence_reproduces_the_planted_120_degree_cosines():
    X_deg_std, deltas, family, _ = _rank2_fixture()
    rp = reduction_persistence(X_deg_std, deltas, family, ks=(2,))
    off_diag = rp.full_mean_cosine[~np.eye(3, dtype=bool)]
    assert off_diag == pytest.approx(np.full(6, np.cos(2.0 * np.pi / 3.0)), abs=1e-10)  # -0.5
    assert np.diag(rp.full_mean_cosine) == pytest.approx(np.ones(3), abs=1e-10)


def test_reduction_persistence_preserves_per_family_concentration_at_k2():
    X_deg_std, deltas, family, _ = _rank2_fixture()
    rp = reduction_persistence(X_deg_std, deltas, family, ks=(2,))
    for f in FAMS:
        # Within a family every Δ is a positive multiple of one direction -> 1.0.
        assert rp.full_by_family[f].mean_cosine == pytest.approx(1.0, abs=1e-10)
        assert rp.by_k[2].by_family[f].mean_cosine == pytest.approx(1.0, abs=1e-10)
        assert rp.by_k[2].by_family[f].null_scale == pytest.approx(1.0 / np.sqrt(2), abs=1e-12)


def test_reduction_persistence_clips_k_to_the_available_components():
    X_deg_std, deltas, family, _ = _rank2_fixture()
    rp = reduction_persistence(X_deg_std, deltas, family, ks=(2, 8, 32))
    assert rp.n_features == 6
    assert set(rp.by_k) == {2, 6}  # 8 and 32 both clip to the 6 available dims
    assert rp.by_k[6].cumulative_explained == pytest.approx(1.0, abs=1e-10)


def test_reduction_persistence_family_labels_the_deltas_not_the_rows():
    # Deltas and degraded rows need not be the same length (unpaired rows dropped).
    X_deg_std, deltas, family, _ = _rank2_fixture()
    extra = np.vstack([X_deg_std, X_deg_std[:2] + 0.1])  # 11 rows vs 9 deltas
    rp = reduction_persistence(extra, deltas, family, ks=(2,))
    assert rp.by_k[2].mean_cosine.shape == (3, 3)

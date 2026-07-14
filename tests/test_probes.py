"""Phase-4 probe + Gate-1 tests. Unit tests fabricate a design matrix with known
structure (so a working linear probe MUST recover it) and exercise the gate's
CI/energy/underpowered logic directly; one integration test runs the full cache →
probe → gate path."""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.audio.synth import synth_clip
from pilot0.corpus.io import write_audio
from pilot0.corpus.manifest import build_manifest, renderable_conditions
from pilot0.corpus.preflight import preflight
from pilot0.degrade.grid import FAMILIES
from pilot0.encode.pipeline import encode_corpus
from pilot0.probes.dataset import ProbeData, build_probe_data
from pilot0.probes.gate1 import MIN_TEST_GROUPS, Gate1Decision
from pilot0.probes.metrics import (
    Estimate, bootstrap_fraction, bootstrap_over_groups, macro_f1, srcc,
)
from pilot0.probes.run import run_gate1
from pilot0.probes.severity import FamilySeverity, SeverityResult, evaluate_severity_probes
from pilot0.probes.type_probe import TypeResult, evaluate_type_probe
from pilot0.seam.registry import make_encoder

SR = 48000
FAMS = list(FAMILIES)


def _fabricate(n_sources: int = 12, noise: float = 0.3, seed: int = 0) -> ProbeData:
    """Family identity in dims 0..6 (one-hot), severity in dim 7 — recoverable by a
    linear probe. Source-disjoint: last 4 sources are the test split."""
    rng = np.random.default_rng(seed)
    X, fam, sev, split, group = [], [], [], [], []

    def add(feat, f, s, sp, g):
        X.append(feat + noise * rng.standard_normal(9))
        fam.append(f), sev.append(s), split.append(sp), group.append(g)

    for src in range(n_sources):
        sp = "test" if src >= n_sources - 4 else "train"
        g = f"src{src}"
        for fi, f in enumerate(FAMS):
            for s in range(1, 6):
                feat = np.zeros(9)
                feat[fi] = 5.0
                feat[7] = float(s)
                add(feat, f, s, sp, g)
        add(np.zeros(9), "clean", 0, sp, g)

    return ProbeData(
        X=np.asarray(X), family=np.asarray(fam), severity=np.asarray(sev, int),
        split=np.asarray(split), group=np.asarray(group), encoder="fab", variant="v",
    )


# --- metrics ------------------------------------------------------------------


def test_macro_f1_perfect_and_worst():
    y = np.array(["a", "a", "b", "b"])
    assert macro_f1(y, y, ["a", "b"]) == 1.0
    assert macro_f1(y, np.array(["b", "b", "a", "a"]), ["a", "b"]) == 0.0


def test_srcc_monotone_and_degenerate():
    assert srcc(np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40])) == pytest.approx(1.0)
    assert np.isnan(srcc(np.array([1, 2, 3]), np.array([7, 7, 7])))  # no rank variance
    assert np.isnan(srcc(np.array([1, 2]), np.array([1, 2])))  # < 3 points


def test_bootstrap_ci_brackets_point_and_reflects_spread():
    groups = np.repeat(np.arange(6), 5)
    y = np.tile(np.arange(5), 6)
    perfect = bootstrap_over_groups(groups, lambda idx: srcc(y[idx], y[idx]))
    assert perfect.lo <= perfect.point <= perfect.hi
    assert perfect.point == pytest.approx(1.0) and perfect.lo == pytest.approx(1.0)

    rng = np.random.default_rng(0)
    noisy_pred = y + rng.normal(scale=3.0, size=len(y))
    est = bootstrap_over_groups(groups, lambda idx: srcc(y[idx], noisy_pred[idx]))
    assert est.lo < est.point  # a noisy signal has a genuinely lower CI floor


# --- probes recover injected structure ----------------------------------------


def test_type_probe_recovers_family():
    res = evaluate_type_probe(_fabricate())
    assert res.macro_f1.point > 0.9 and res.macro_f1.lo > 0.8
    assert res.confusion.shape == (7, 7)


def test_severity_probe_recovers_ordinal_severity():
    res = evaluate_severity_probes(_fabricate())
    assert set(res.by_family) == set(FAMILIES)
    assert res.n_pass(0.8) == 7  # every family's severity is monotone in dim 7


def test_type_probe_collapses_on_pure_noise():
    assert evaluate_type_probe(_fabricate(noise=50.0, seed=1)).macro_f1.point < 0.6


# --- fail-closed bootstrap primitives (the floors every gate inherits) ---------


def test_bootstrap_over_groups_fails_closed_on_thin_evidence():
    # empty rows → undefined, not a crash
    empty = bootstrap_over_groups(np.array([]), lambda i: 1.0)
    assert np.isnan(empty.lo) and np.isnan(empty.hi)
    # one group under a 3-group floor → CI collapses to nan (winner's-curse guard)
    one_group = bootstrap_over_groups(np.zeros(20, int), lambda i: 0.99, min_groups=MIN_TEST_GROUPS)
    assert one_group.point == 0.99 and np.isnan(one_group.lo) and np.isnan(one_group.hi)
    # >50% of resamples undefined (nan) → CI not trusted even with enough groups
    mostly_nan = bootstrap_over_groups(np.arange(10), lambda i: float("nan"))
    assert np.isnan(mostly_nan.lo) and np.isnan(mostly_nan.hi)


def test_bootstrap_fraction_fails_closed_below_group_floor():
    # a single source ordering "always holds" must NOT read as a reliable fraction
    assert np.isnan(bootstrap_fraction(np.zeros(15, int), lambda i: 1.0, min_groups=MIN_TEST_GROUPS))
    assert np.isnan(bootstrap_fraction(np.array([]), lambda i: 1.0))


# --- gate 1 decision logic ----------------------------------------------------


def _type(lo: float) -> TypeResult:
    return TypeResult(Estimate(lo, lo, lo + 0.05), FAMS, np.zeros((7, 7), int), 10, 10)


def _sev(lo_by_fam: dict[str, float]) -> SeverityResult:
    return SeverityResult({f: FamilySeverity(Estimate(v, v, v), Estimate(v, v, v)) for f, v in lo_by_fam.items()})


def _decision(type_lo, sev, energy, floor_pt, n_groups=MIN_TEST_GROUPS + 2) -> Gate1Decision:
    return Gate1Decision("enc", "v", _type(type_lo), _sev(sev), _sev(energy),
                         Estimate(floor_pt, floor_pt, floor_pt), n_groups)


def test_gate1_passes_only_when_all_clauses_clear():
    dec = _decision(0.90, {f: 0.85 for f in FAMS}, {f: 0.0 for f in FAMS}, floor_pt=0.80)
    assert dec.pass_type and dec.pass_severity and dec.pass_floor and not dec.underpowered
    assert dec.passed and dec.margin_over_floor == pytest.approx(0.10)


def test_gate1_fails_on_thin_floor_margin():
    dec = _decision(0.90, {f: 0.9 for f in FAMS}, {f: 0.0 for f in FAMS}, floor_pt=0.87)
    assert dec.pass_type and dec.pass_severity and not dec.pass_floor and not dec.passed


def test_gate1_fails_when_severity_only_matches_energy_control():
    # Codec severity is high absolutely but the energy control is nearly as high →
    # the codec is reading loudness, not degradation → no family passes (finding B1).
    dec = _decision(0.90, {f: 0.85 for f in FAMS}, {f: 0.83 for f in FAMS}, floor_pt=0.70)
    assert dec.n_severity_pass == 0 and not dec.pass_severity and not dec.passed


def test_gate1_underpowered_cannot_pass():
    dec = _decision(0.95, {f: 0.95 for f in FAMS}, {f: 0.0 for f in FAMS}, floor_pt=0.50,
                    n_groups=MIN_TEST_GROUPS - 1)
    assert dec.underpowered and dec.pass_type and dec.pass_severity and not dec.passed


def test_gate1_exact_thresholds_pass_by_ge():
    # every §8 clause sits EXACTLY on its pre-registered threshold; the gate uses `>=`,
    # so a regression flipping any `>=` to `>` would flip this to fail.
    from pilot0.probes.gate1 import (
        FLOOR_F1_MARGIN, SEVERITY_OVER_ENERGY_MARGIN, SEVERITY_SRCC_MIN, TYPE_MACRO_F1_MIN,
    )
    dec = _decision(
        TYPE_MACRO_F1_MIN,                                              # type lo == min
        {f: SEVERITY_SRCC_MIN for f in FAMS},                          # severity lo == min
        {f: SEVERITY_SRCC_MIN - SEVERITY_OVER_ENERGY_MARGIN for f in FAMS},  # beats energy by exactly the margin
        floor_pt=TYPE_MACRO_F1_MIN - FLOOR_F1_MARGIN,                  # floor margin == FLOOR_F1_MARGIN
    )
    assert dec.pass_type and dec.pass_severity and dec.pass_floor and dec.passed
    assert dec.n_severity_pass == 7


# --- baseline encoders --------------------------------------------------------


def test_logmel_floor_has_delta_features():
    from pilot0.seam.base import Encoder

    enc = make_encoder("logmel")
    assert isinstance(enc, Encoder)
    r = enc.encode(synth_clip(0, sr=SR, seconds=1.0), SR)["mel"]
    assert r.latent_dim == 64 * 3  # static + Δ + ΔΔ (§2.3)
    assert r.meta["deltas"] == 2


def test_energy_control_is_three_band_and_deterministic():
    enc = make_encoder("energy")
    a = enc.encode(synth_clip(1, sr=SR, seconds=1.0), SR)["energy"]
    b = enc.encode(synth_clip(1, sr=SR, seconds=1.0), SR)["energy"]
    assert a.latent_dim == 3 and np.array_equal(a.frames, b.frames)
    assert a.meta["backend"] == "energy"


# --- integration: cache → probe → gate ----------------------------------------


def test_gate1_runs_end_to_end_and_flags_underpowered(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        write_audio(src / f"c{i}.wav", synth_clip(i, sr=SR, seconds=1.5), SR)
    pf = preflight(src, tmp_path / "out")
    man = build_manifest(pf)
    cache = tmp_path / "cache"
    encode_corpus(pf.norm_dir, man, ["logmel", "energy", "fake-encodec24k"], cache)

    data = build_probe_data(man, "fake-encodec24k", "z", cache)
    assert data.X.shape[0] == len(data.family) > 0

    report = run_gate1(man, [("fake-encodec24k", "z")], cache)
    # Floor(16k) and codec(24k) scored on the SAME cells = renderable@16k (finding M1).
    assert report.n_common_conditions == len(renderable_conditions(16000))
    dec = report.decisions[0]
    assert dec.underpowered and not dec.passed  # 4 sources → 1 test group
    assert 0.0 <= dec.type.macro_f1.point <= 1.0

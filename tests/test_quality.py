"""Phase-6 quality-head + Gate-2 tests. Fabricated latents carry a linear quality
signal a working head MUST recover (G2a); G2b's paired head-vs-baseline-on-MOS logic,
the not-evaluable-without-MOS/without-baselines paths, partial-MOS coverage, and the
underpowered short-circuit are exercised directly. One integration test runs
cache → head → gate on synthetic scores; TableScores' §8 rig guard is tested too."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from pilot0.audio.synth import synth_clip
from pilot0.corpus.io import write_audio
from pilot0.corpus.manifest import build_manifest
from pilot0.corpus.preflight import preflight
from pilot0.degrade.grid import FAMILIES
from pilot0.encode.pipeline import encode_corpus
from pilot0.probes.metrics import Estimate
from pilot0.quality.dataset import QualityData, build_quality_data
from pilot0.quality.g2b import FamilyG2b, G2bResult, evaluate_g2b
from pilot0.quality.gate2 import Gate2Decision
from pilot0.quality.head import QualityG2a, evaluate_g2a, fit_head
from pilot0.quality.run import run_gate2
from pilot0.quality.scores import (
    MOS_METRIC,
    NR_BASELINES,
    REF_METRIC,
    FakeScores,
    TableScores,
)

SR = 48000
FAMS = list(FAMILIES)


def _fab_quality(n_sources: int = 12, noise: float = 0.15, seed: int = 0) -> QualityData:
    """Latent dim 0 = quality signal (5 − severity + source offset); ViSQOL tracks it,
    MOS tracks it with a family weight (so MOS ≠ ViSQOL), NR baselines predict MOS."""
    rng = np.random.default_rng(seed)
    w = {f: 0.8 + 0.05 * i for i, f in enumerate(FAMS)}
    X, ref, mos, fam, sev, split, group = [], [], [], [], [], [], []
    nr = {b: [] for b in NR_BASELINES}

    def add(q, f, s, sp, g):
        feat = np.zeros(8)
        feat[0] = q + noise * rng.standard_normal()
        X.append(feat)
        ref.append(q + noise * rng.standard_normal())
        mos.append(q * w.get(f, 1.0) + noise * rng.standard_normal())
        for b in NR_BASELINES:
            nr[b].append(mos[-1] + noise * rng.standard_normal())
        fam.append(f), sev.append(s), split.append(sp), group.append(g)

    for src in range(n_sources):
        sp = "test" if src >= n_sources - 4 else "train"
        g, off = f"src{src}", rng.standard_normal() * 0.3
        for f in FAMS:
            for s in range(1, 6):
                add(5.0 - s + off, f, s, sp, g)
        add(5.0 + off, "clean", 0, sp, g)

    return QualityData(
        X=np.asarray(X), ref=np.asarray(ref), mos=np.asarray(mos),
        nr={b: np.asarray(v) for b, v in nr.items()},
        family=np.asarray(fam), severity=np.asarray(sev, int),
        split=np.asarray(split), group=np.asarray(group), encoder="fab", variant="v",
    )


# --- fake scores --------------------------------------------------------------


def test_fake_scores_deterministic_and_bounded():
    s = FakeScores()
    assert s.get(REF_METRIC, "src0", "noise", 3) == s.get(REF_METRIC, "src0", "noise", 3)
    for m in (REF_METRIC, MOS_METRIC, *NR_BASELINES):
        v = s.get(m, "src1", "clip", 4)
        assert 1.0 <= v <= 5.0


def test_fake_visqol_falls_with_severity():
    s = FakeScores()
    clean = np.mean([s.get(REF_METRIC, f"src{i}", "noise", 0) for i in range(20)])
    harsh = np.mean([s.get(REF_METRIC, f"src{i}", "noise", 5) for i in range(20)])
    assert clean > harsh + 1.0  # ViSQOL clearly higher on clean than severity 5


def test_fake_mos_is_not_a_pure_function_of_visqol():
    s = FakeScores()
    # Family-specific weighting: hiss and dropout at the same severity differ in MOS
    # even where ViSQOL would order them together → MOS carries independent structure.
    assert s.get(MOS_METRIC, "src3", "hiss", 4) != s.get(MOS_METRIC, "src3", "dropout", 4)


# --- table scores: §8 rig guard + subset coverage -----------------------------


def test_tablescores_roundtrip_and_uncovered_cell_is_none(tmp_path):
    p = tmp_path / "scores.json"
    p.write_text(json.dumps({
        "visqol": {"s0|noise|3": 3.1, "s0|clip|4": 2.2},
        "mos": {"s0|noise|3": 2.8},  # speech-only subset: the clip cell has no MOS
    }))
    ts = TableScores.from_json(p)
    assert ts.has("visqol") and ts.has("mos") and not ts.has("nisqa")
    assert ts.get("visqol", "s0", "noise", 3) == 3.1
    assert ts.get("mos", "s0", "clip", 4) is None  # uncovered → dropped, not fabricated


def test_tablescores_rejects_mos_copied_from_visqol(tmp_path):
    p = tmp_path / "rig.json"
    p.write_text(json.dumps({
        "visqol": {"s0|noise|3": 3.1, "s0|clip|4": 2.2},
        "mos": {"s0|noise|3": 3.1, "s0|clip|4": 2.2},  # byte-identical → the §8 rig
    }))
    with pytest.raises(ValueError, match="rig"):
        TableScores.from_json(p)


def test_tablescores_rejects_monotone_copy_of_visqol(tmp_path):
    # The rig defeats a byte check but not a rank check: MOS = 2·ViSQOL+1, plus one
    # cell nudged so it is NOT byte-identical, still ranks ViSQOL-identically.
    ref = {f"s0|noise|{s}": 1.0 + 0.3 * s for s in range(12)}  # ≥ _RIG_MIN_SHARED cells
    mos = {k: 2.0 * v + 1.0 for k, v in ref.items()}
    mos["s0|noise|0"] += 1e-9  # break byte-identity but preserve rank order
    p = tmp_path / "monotone.json"
    p.write_text(json.dumps({"visqol": ref, "mos": mos}))
    with pytest.raises(ValueError, match="rig"):
        TableScores.from_json(p)


def test_build_quality_data_requires_reference_target():
    # No ViSQOL column ⇒ the head has nothing to train on; fail clean, not a deep
    # KeyError inside the cache walk (adversarial W1). Raises before touching the cache.
    class _NoRef:
        def has(self, metric: str) -> bool:
            return metric != REF_METRIC

        def get(self, *a):
            return None

    with pytest.raises(ValueError, match="reference target"):
        build_quality_data({}, "enc", "z", "/nonexistent", _NoRef())


# --- G2a: the head recovers the reference the head was trained on --------------


# n_boot is small in these structural tests: they assert the head recovers a signal
# and that every family is scored, not the exact CI width, so 1000 resamples is waste.
_NB = 128


def test_head_recovers_reference_g2a():
    data = _fab_quality()
    g2a = evaluate_g2a(data, fit_head(data), n_boot=_NB)
    assert g2a.ref_srcc.point > 0.9 and g2a.ref_srcc.lo > 0.7
    assert g2a.ref_lcc.point > 0.9
    assert g2a.n_test_groups == 4  # last 4 of 12 sources
    assert set(g2a.ref_srcc_by_family) == set(FAMS)  # §2.4 per-family view present


# --- G2b: paired head-vs-baseline on MOS, coverage, evaluability ---------------


def test_g2b_evaluates_every_family_when_powered():
    data = _fab_quality()
    res = evaluate_g2b(data, fit_head(data), n_boot=_NB)
    assert res is not None and set(res.by_family) == set(FAMS)
    for fam in FAMS:
        fg = res.by_family[fam]
        assert fg.n_cells > 0 and set(fg.paired_lo) == set(NR_BASELINES)


def test_g2b_not_evaluable_without_mos():
    data = _fab_quality()
    assert evaluate_g2b(replace(data, mos=None), fit_head(data), n_boot=_NB) is None


def test_g2b_not_evaluable_without_baselines():
    data = _fab_quality()
    assert evaluate_g2b(replace(data, nr={}), fit_head(data), n_boot=_NB) is None


def test_g2b_not_evaluable_when_mos_all_nan():
    # MOS collected but zero finite coverage must be not-evaluable at the evaluate_g2b
    # entry point too, not only in build_quality_data (adversarial W3).
    data = _fab_quality()
    all_nan = np.full_like(data.mos, np.nan)
    assert evaluate_g2b(replace(data, mos=all_nan), fit_head(data), n_boot=_NB) is None


def test_g2b_partial_mos_coverage_drops_uncovered_families():
    # A speech-only human study leaves hiss/hum without MOS → those families become
    # NaN and must drop out without crashing; covered families still evaluate (W1).
    data = _fab_quality()
    mos = data.mos.copy()
    mos[np.isin(data.family, ["hiss", "hum"])] = np.nan
    res = evaluate_g2b(replace(data, mos=mos), fit_head(data), n_boot=_NB)
    assert res is not None
    assert res.by_family["hiss"].n_cells == 0 and not res.by_family["hiss"].beats_all
    assert res.by_family["noise"].n_cells > 0


# --- FamilyG2b.beats_all guards ------------------------------------------------


def _family(paired: dict[str, float], n_cells: int = 20, n_groups: int = 4) -> FamilyG2b:
    e = Estimate(0.9, 0.9, 0.9)
    return FamilyG2b(head_mos=e, baselines_mos={b: e for b in paired}, paired_lo=paired,
                     n_cells=n_cells, n_groups=n_groups)


def test_beats_all_requires_all_baselines_positive():
    assert _family({"nisqa": 0.05, "dnsmos": 0.02}).beats_all
    assert not _family({"nisqa": 0.05, "dnsmos": -0.01}).beats_all  # one baseline not beaten


def test_beats_all_requires_min_cells():
    assert not _family({"nisqa": 0.5}, n_cells=2).beats_all  # < MIN_MOS_CELLS


def test_beats_all_requires_min_groups():
    # A family covered by one source (many cells, one group) must NOT count: the
    # cluster bootstrap over a single group collapses to a zero-width CI (physics B1 /
    # adversarial B2). Row count alone is not source-level power.
    assert not _family({"nisqa": 0.5}, n_cells=20, n_groups=1).beats_all
    assert not _family({"nisqa": 0.5}, n_cells=20, n_groups=2).beats_all
    assert _family({"nisqa": 0.5}, n_cells=20, n_groups=3).beats_all


def test_beats_all_false_on_nan_or_empty():
    assert not _family({"nisqa": float("nan")}).beats_all
    assert not _family({}).beats_all  # no baselines at all


# --- gate 2 decision logic ----------------------------------------------------


def _g2a(ref_lo: float, n_groups: int = 5) -> QualityG2a:
    e = Estimate(ref_lo, ref_lo, ref_lo + 0.03)
    return QualityG2a(ref_srcc=e, ref_lcc=e, ref_srcc_by_family={}, n_test_groups=n_groups)


def _g2b(paired_by_fam: dict[str, float]) -> G2bResult:
    return G2bResult(by_family={f: _family({"nisqa": lo}) for f, lo in paired_by_fam.items()})


def test_gate2_passes_when_g2a_and_g2b_clear():
    dec = Gate2Decision("enc", "v", _g2a(0.90), _g2b({f: 0.05 for f in FAMS}))
    assert dec.pass_g2a and dec.n_beats_baseline == 7 and dec.pass_g2b and dec.passed


def test_gate2_fails_g2a_below_threshold():
    dec = Gate2Decision("enc", "v", _g2a(0.80), _g2b({f: 0.05 for f in FAMS}))
    assert not dec.pass_g2a and not dec.passed


def test_gate2_g2b_five_of_seven_is_the_exact_boundary():
    four = {f: (0.05 if i < 4 else -0.01) for i, f in enumerate(FAMS)}  # exactly 4 → fail
    dec4 = Gate2Decision("enc", "v", _g2a(0.90), _g2b(four))
    assert dec4.n_beats_baseline == 4 and not dec4.pass_g2b and not dec4.passed

    five = {f: (0.05 if i < 5 else -0.01) for i, f in enumerate(FAMS)}  # exactly 5 → pass
    dec5 = Gate2Decision("enc", "v", _g2a(0.90), _g2b(five))
    assert dec5.n_beats_baseline == 5 and dec5.pass_g2b and dec5.passed


def test_gate2_g2b_not_evaluable_without_mos_or_baselines():
    dec = Gate2Decision("enc", "v", _g2a(0.90), None)
    assert not dec.g2b_evaluable and dec.n_beats_baseline == 0 and not dec.pass_g2b and not dec.passed


def test_gate2_underpowered_cannot_pass():
    dec = Gate2Decision("enc", "v", _g2a(0.95, n_groups=1), _g2b({f: 0.05 for f in FAMS}))
    assert dec.underpowered and dec.pass_g2a and dec.pass_g2b and not dec.passed


# --- integration: cache → head → gate -----------------------------------------


def test_gate2_runs_end_to_end(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        write_audio(src / f"c{i}.wav", synth_clip(i, sr=SR, seconds=1.5), SR)
    pf = preflight(src, tmp_path / "out")
    man = build_manifest(pf)
    cache = tmp_path / "cache"
    encode_corpus(pf.norm_dir, man, ["fake-encodec24k"], cache)

    report = run_gate2(man, [("fake-encodec24k", "z")], cache, FakeScores())
    assert report.mos_available
    dec = report.decisions[0]
    assert dec.underpowered and not dec.passed  # 4 sources → 1 test group
    assert -1.0 <= dec.g2a.ref_srcc.point <= 1.0
    assert dec.g2b_evaluable  # MOS + baselines present, even if underpowered

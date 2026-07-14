"""Phase-8 release tests. The JSON→figure boundary is exercised on hand-written
fixtures (fast, no pipeline); one integration test runs the real writer on a tiny
corpus and checks every artifact is strict-parseable and flagged FAKE.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pilot0.combos.additivity import AdditivityResult, PairAdditivity
from pilot0.combos.run import ComboReport
from pilot0.combos.transfer import PairTransfer, TransferResult
from pilot0.ood.run import OODReport
from pilot0.ood.teaser import OODTeaser
from pilot0.probes.metrics import Estimate
from pilot0.quality.g2b import FamilyG2b, G2bResult
from pilot0.quality.gate2 import Gate2Decision
from pilot0.quality.head import QualityG2a
from pilot0.quality.run import Gate2Report
from pilot0.quality.scores import FakeScores
from pilot0.release.artifacts import (
    BANNER, CANDIDATES, _additivity_json, _gate2_json, _ood_json,
    build_demo_corpus, write_artifacts,
)
import matplotlib.pyplot as plt  # figures.py already selected the Agg backend on import

from pilot0.release.figures import _RENDERERS, _load, FIGURES, render_figures
from pilot0.serialize import to_jsonable, write_json

FAMS = ["noise", "hiss", "hum", "clip", "bandlimit", "mp3", "dropout"]
KEYS = [f"{n}/{v}" for n, v in CANDIDATES]
PRIMARY = KEYS[0]
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ── serialize ──────────────────────────────────────────────────────────────
def test_to_jsonable_nan_becomes_null():
    assert to_jsonable(float("nan")) is None
    assert to_jsonable([1.0, float("inf"), -float("inf")]) == [1.0, None, None]


def test_to_jsonable_numpy_collapses_to_python():
    assert to_jsonable(np.array([[1.0, 2.0], [3.0, 4.0]])) == [[1.0, 2.0], [3.0, 4.0]]
    assert to_jsonable({"a": np.float64(2.5), "b": np.int64(3)}) == {"a": 2.5, "b": 3}


def test_write_json_is_strict_parseable_with_nan(tmp_path):
    p = write_json(tmp_path / "x.json", {"v": [float("nan"), 1.0]})
    assert json.loads(p.read_text()) == {"v": [None, 1.0]}  # would raise on a bare NaN token


# ── figure fixtures ────────────────────────────────────────────────────────
def _triple(p=0.7):
    return [p, p - 0.1, p + 0.1]


def _heatmap_rows():
    rows = []
    for name, var in [("logmel", "mel"), ("energy", "energy"), *CANDIDATES]:
        rows.append({"name": name, "variant": var, "type_macro_f1": _triple(0.6),
                     "severity_srcc": {f: _triple(0.5) for f in FAMS}, "n_test_groups": 4})
    rows[0]["severity_srcc"]["mp3"] = [None, None, None]  # exercise the masked/"—" cell
    return rows


def _cosine():
    M = (np.eye(len(FAMS)) + 0.1 * (1 - np.eye(len(FAMS)))).tolist()
    return {k: {"labels": FAMS, "mean_cosine": M, "probe_cosine": M,
                "centroid_pca_explained": [0.6, 0.2, 0.1]} for k in KEYS}


def _monotonicity():
    curve = {str(l): float(l) + 0.1 for l in range(1, 6)}
    curve_gap = dict(curve); curve_gap["3"] = None  # a missing level → F2 dashes the point over
    empty = {str(l): None for l in range(1, 6)}  # not evaluable at all → F2 drops it to the "n/a" note

    def fam(f):
        if f == "hum":  # exercise the dropped/"n/a" branch
            return {"srcc": [None, None, None], "curve": empty, "interp_frac": None,
                    "evaluable": False, "interpolates": False}
        gap = f == "bandlimit"  # present-but-non-interpolating → dashed "(n/i)"
        return {"srcc": _triple(0.8), "curve": curve_gap if gap else curve,
                "interp_frac": 0.95, "evaluable": not gap, "interpolates": not gap}
    return {k: {f: fam(f) for f in FAMS} for k in KEYS}


def _additivity():
    cells = {}
    for pair in ("noise+clip", "hiss+bandlimit", "hum+mp3"):
        for sev in (2, 3, 4):
            cells[f"{pair}@{sev}"] = {"pair": pair, "severity": sev, "raw": _triple(0.99),
                                      "std": _triple(0.98), "n_groups": 5}
    return {k: {"mean_cosine": 0.99, "by_cell": cells} for k in KEYS}


def _gate2(g2b: bool):
    def per_key():
        g2b_block = None
        if g2b:
            g2b_block = {"n_beats": 5, "pass": True, "by_family": {
                f: {"head_mos": _triple(0.7), "baselines_mos": {b: _triple(0.5) for b in ("nisqa", "dnsmos", "utmos")},
                    "paired_lo": {b: 0.1 for b in ("nisqa", "dnsmos", "utmos")}, "n_cells": 5,
                    "n_groups": 4, "beats_all": True} for f in FAMS}}
        return {"n_test_groups": 4, "underpowered": False, "passed": g2b,
                "g2a": {"ref_srcc": _triple(0.88), "ref_lcc": _triple(0.85),
                        "by_family": {f: _triple(0.8) for f in FAMS}, "pass": True},
                "g2b": g2b_block,
                "scatter": [[4.2, 4.0], [3.1, 3.3], [2.0, 2.4], [1.5, 1.8]]}
    return {k: per_key() for k in KEYS}


def _ood():
    return {k: {"metrics": ["head", "nisqa", "dnsmos", "utmos"], "grid_disagreement": 0.31,
                "ood_disagreement": 0.97, "calibration_floor": 0.09, "amplifies": True,
                "n_grid": 500, "n_ood": 24} for k in KEYS}


def _write_fixtures(root: Path, *, g2b: bool = True) -> Path:
    write_json(root / "analysis" / "heatmap.json", _heatmap_rows())
    write_json(root / "analysis" / "cosine.json", _cosine())
    write_json(root / "analysis" / "monotonicity.json", _monotonicity())
    write_json(root / "quality" / "gate2.json", _gate2(g2b))
    write_json(root / "combos" / "additivity.json", _additivity())
    write_json(root / "ood" / "teaser.json", _ood())
    return root


# ── figures ────────────────────────────────────────────────────────────────
def test_render_figures_writes_all_six_pngs(tmp_path):
    reports = _write_fixtures(tmp_path / "reports")
    figs = render_figures(reports, tmp_path / "figures")
    assert set(figs) == set(FIGURES)
    for name, path in figs.items():
        blob = path.read_bytes()
        assert blob[:8] == _PNG_MAGIC, f"{name} is not a PNG"
        assert len(blob) > 1000, f"{name} suspiciously small"


def test_f4_renders_when_g2b_not_evaluable(tmp_path):
    reports = _write_fixtures(tmp_path / "reports", g2b=False)
    figs = render_figures(reports, tmp_path / "figures")
    assert figs["F4"].read_bytes()[:8] == _PNG_MAGIC


def test_every_figure_stamps_the_fake_banner(tmp_path):
    # the core seam invariant: a fake figure that escapes must SAY it's fake. A dropped
    # `_finish` banner would still produce a valid PNG, so assert the text is in-frame.
    data = _load(_write_fixtures(tmp_path / "reports"))
    for name, render in _RENDERERS.items():
        fig = render(data)
        assert any(t.get_text() == BANNER for t in fig.texts), f"{name} missing FAKE banner"
        plt.close(fig)


# ── real Phase-8 serializers feed the figures ──────────────────────────────
# The analysis three (heatmap/cosine/monotonicity) are verbatim-moved from the
# Phase-5 audit and covered by `make analyze`; here we prove the NEW serializers
# (gate2/additivity/ood) emit exactly the schema the figures consume — without the
# minutes-long full-corpus run, matching how every other phase tests (fabricate, not
# encode). The whole pipeline is exercised by `make reproduce-figures` and the opt-in
# slow test below.
def _est(p):
    return Estimate(p, p - 0.1, p + 0.1)


def _gate2_report(g2b: bool) -> Gate2Report:
    g2b_block = None
    if g2b:
        g2b_block = G2bResult(by_family={
            f: FamilyG2b(_est(0.7), {b: _est(0.5) for b in ("nisqa", "dnsmos", "utmos")},
                         {b: 0.1 for b in ("nisqa", "dnsmos", "utmos")}, 5, 4) for f in FAMS})
    decisions = tuple(
        Gate2Decision(n, v, QualityG2a(_est(0.88), _est(0.85), {f: _est(0.8) for f in FAMS}, 4), g2b_block)
        for n, v in CANDIDATES)
    scatter = {k: [[4.2, 4.0], [2.0, 2.3], [1.5, 1.8]] for k in KEYS}
    return Gate2Report(decisions=decisions, mos_available=g2b, scatter=scatter)


def _combo_report() -> ComboReport:
    add, trans = {}, {}
    for k in KEYS:
        cells = {(pair, sev): PairAdditivity(pair, sev, _est(0.99), _est(0.98), 5)
                 for pair in ("noise+clip", "hiss+bandlimit", "hum+mp3") for sev in (2, 3, 4)}
        add[k] = AdditivityResult(by_cell=cells)
        trans[k] = TransferResult(by_pair={
            "noise+clip": PairTransfer("noise+clip", "noise", "clip", 8, {"noise": 0.75, "clip": 0.25}, 1.0)})
    return ComboReport(additivity=add, transfer=trans)


def _ood_report() -> OODReport:
    return OODReport(teasers={
        k: OODTeaser(("head", "nisqa", "dnsmos", "utmos"), 0.31, 0.97, 0.09, 500, 24) for k in KEYS}, n_ood=24)


@pytest.mark.parametrize("g2b", [True, False])
def test_gate2_serializer_renders_f4(tmp_path, g2b):
    reports = _write_fixtures(tmp_path / "reports")
    write_json(reports / "quality" / "gate2.json", _gate2_json(_gate2_report(g2b)))
    assert set(json.loads((reports / "quality" / "gate2.json").read_text())) == set(KEYS)
    assert render_figures(reports, tmp_path / "figs")["F4"].read_bytes()[:8] == _PNG_MAGIC


def test_f4_renders_when_g2a_srcc_is_null(tmp_path):
    # a degenerate split makes ref_srcc.point non-finite → serialised as JSON null; F4
    # must render "nan" in the title, never crash the whole suite on f"{None:.2f}".
    reports = _write_fixtures(tmp_path / "reports")
    g2 = json.loads((reports / "quality" / "gate2.json").read_text())
    g2[PRIMARY]["g2a"]["ref_srcc"] = [None, None, None]
    write_json(reports / "quality" / "gate2.json", g2)
    assert render_figures(reports, tmp_path / "figs")["F4"].read_bytes()[:8] == _PNG_MAGIC


def test_additivity_serializer_renders_f3(tmp_path):
    reports = _write_fixtures(tmp_path / "reports")
    write_json(reports / "combos" / "additivity.json", _additivity_json(_combo_report()))
    assert render_figures(reports, tmp_path / "figs")["F3"].read_bytes()[:8] == _PNG_MAGIC


def test_ood_serializer_renders_f6(tmp_path):
    reports = _write_fixtures(tmp_path / "reports")
    write_json(reports / "ood" / "teaser.json", _ood_json(_ood_report()))
    assert render_figures(reports, tmp_path / "figs")["F6"].read_bytes()[:8] == _PNG_MAGIC


def test_banner_marks_outputs_fake():
    assert "FAKE" in BANNER and "NOT results" in BANNER


# The full-corpus mirror of `make reproduce-figures`: real fake-seam encode → all six
# artifacts strict-JSON + flagged fake → all six PNGs. ~4 min, so opt-in (PILOT0_SLOW=1).
@pytest.mark.skipif(not os.environ.get("PILOT0_SLOW"), reason="full-corpus release smoke; set PILOT0_SLOW=1")
def test_write_artifacts_end_to_end(tmp_path):
    out = tmp_path / "reports"
    with tempfile.TemporaryDirectory() as d:
        norm, man, cache = build_demo_corpus(Path(d), n_sources=6, sr=48000)
        paths = write_artifacts(norm, man, CANDIDATES, cache, FakeScores(), out, n_sources=6, sr=48000)
    for path in paths.values():
        json.loads(path.read_text())  # strict: raises on a bare NaN token
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["fake"] is True and prov["banner"] == BANNER
    assert all(p.read_bytes()[:8] == _PNG_MAGIC for p in render_figures(out, out / "figures").values())

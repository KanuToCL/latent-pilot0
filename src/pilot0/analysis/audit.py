"""Phase-5 demo (`make analyze`): synth corpus → fake-encode the matrix → write the
three acceptance artifacts (heatmap / cosine matrix / monotonicity curves) to
reports/analysis/ and print a summary.

BANNER: codec rows use the FAKE encoder — every number is a plumbing check, NOT a
result; only the log-mel floor and energy control are real. The scientific matrix
comes from the GPU box (docs/GPU_BRINGUP.md).
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from ..audio.synth import synth_clip
from ..corpus.io import write_audio
from ..corpus.manifest import build_manifest
from ..corpus.preflight import preflight
from ..encode.pipeline import encode_corpus
from ..probes.gate1 import MIN_TEST_GROUPS
from .run import analyze

CANDIDATES = [("fake-encodec24k", "z"), ("fake-wavlm", "l12")]


def _jsonable(obj):
    """Serialisable + strict-JSON-safe: numpy → list, and every non-finite float
    (NaN from a degenerate SRCC / dropped cell) → null, so `jq`/`JSON.parse` don't
    choke on bare NaN tokens (finding S3)."""
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _heatmap(report) -> list[dict]:
    rows = [report.floor, report.energy, *(c.readability for c in report.candidates.values())]
    return [
        {"name": r.name, "variant": r.variant, "type_macro_f1": r.type_macro_f1,
         "severity_srcc": r.severity_srcc, "n_test_groups": r.n_test_groups}
        for r in rows
    ]


def _cosine(report) -> dict:
    return {
        key: {"labels": c.geometry.labels, "probe_cosine": c.geometry.probe_cosine,
              "mean_cosine": c.geometry.mean_cosine,
              "centroid_pca_explained": c.geometry.centroid_pca_explained}
        for key, c in report.candidates.items()
    }


def _monotonicity(report) -> dict:
    return {
        key: {fam: {"srcc": (fi.srcc.point, fi.srcc.lo, fi.srcc.hi), "curve": fi.curve,
                    "interp_frac": fi.interp_frac, "evaluable": fi.evaluable,
                    "interpolates": fi.interpolates}
              for fam, fi in c.interpolation.by_family.items()}
        for key, c in report.candidates.items()
    }


def main(n_sources: int = 8, sr: int = 48000, out_dir: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        for i in range(n_sources):
            write_audio(src / f"clip{i}.wav", synth_clip(i, sr=sr, seconds=2.0), sr)
        pf = preflight(src, Path(d) / "corpus")
        man = build_manifest(pf)
        cache = Path(d) / "cache"
        encode_corpus(pf.norm_dir, man, ["logmel", "energy", *(n for n, _ in CANDIDATES)], cache)
        report = analyze(man, CANDIDATES, cache)

    reports = Path(out_dir) if out_dir else Path("reports") / "analysis"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "heatmap.json").write_text(json.dumps(_jsonable(_heatmap(report)), indent=2))
    (reports / "cosine.json").write_text(json.dumps(_jsonable(_cosine(report)), indent=2))
    (reports / "monotonicity.json").write_text(json.dumps(_jsonable(_monotonicity(report)), indent=2))

    print("⚠  FAKE codec latents — plumbing check, NOT results (floor + energy are real)\n")
    print(f"common cells scored: {report.n_common_conditions}   artifacts → {reports}/\n")
    for key, c in report.candidates.items():
        mf = c.readability.type_macro_f1
        gap, nl = c.nonlinearity.gap, c.nonlinearity.mlp_macro_f1
        fd = c.frame_dropout
        n_eval = sum(fi.evaluable for fi in c.interpolation.by_family.values())
        n_interp = c.interpolation.n_interpolate(report.energy_interpolation)  # beats loudness (W1)
        power = "UNDERPOWERED" if c.readability.n_test_groups < MIN_TEST_GROUPS else "powered"
        print(f"{key}  (test sources: {c.readability.n_test_groups} — {power})")
        print(f"  type macro-F1 {mf[0]:.3f} [{mf[1]:.3f}, {mf[2]:.3f}]  "
              f"MLP {nl.point:.3f}  nonlinearity gap {gap.point:+.3f} [{gap.lo:+.3f}, {gap.hi:+.3f}]")
        print(f"  dropout severity SRCC: frame {fd.frame_srcc.point:.3f}  pooled {fd.pooled_srcc.point:.3f}  "
              f"gap {fd.gap.point:+.3f} [{fd.gap.lo:+.3f}, {fd.gap.hi:+.3f}]")
        print(f"  monotonicity: {n_interp}/{n_eval} evaluable families interpolate beyond the energy control\n")


if __name__ == "__main__":
    main()

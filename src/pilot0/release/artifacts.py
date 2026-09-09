"""Phase-8 artifact writer. Runs every phase analysis on ONE shared synthetic corpus
and serialises each result to reports/ as the JSON the paper figures (F1–F6) read.

The whole matrix comes from a single corpus + cache so the six figures are mutually
consistent (same sources, same splits, same headroom). Every number is FAKE (fake
codec latents, synthetic ViSQOL/MOS, fabricated NR incumbents) — the provenance file
records `fake: true` and every figure stamps the banner, so a stray artifact can never
be mistaken for a result. At the GPU box the real backends drop into `build_demo_corpus`
and the same writer emits the scientific artifacts unchanged.
"""

from __future__ import annotations

from pathlib import Path

from ..provenance import is_fake, provenance
from ..analysis.run import analyze
from ..analysis.serialize import cosine_matrices, heatmap_rows, monotonicity_curves
from ..audio.synth import synth_clip
from ..combos.encode import encode_combos
from ..combos.run import analyze_combos
from ..corpus.io import write_audio
from ..corpus.manifest import build_manifest
from ..corpus.preflight import preflight
from ..encode.pipeline import encode_corpus
from ..ood.run import run_ood_teaser
from ..quality.run import run_gate2
from ..quality.scores import Scores
from ..seam.registry import candidate_semantics
from ..serialize import write_json

CANDIDATES = [("fake-encodec24k", "z"), ("fake-wavlm", "l12")]


def _est(e) -> list[float]:
    return [e.point, e.lo, e.hi]


def build_demo_corpus(root: Path, n_sources: int, sr: int):
    """Synth corpus → preflight → manifest → encode the singles matrix AND the combos
    into one shared cache. Returns (norm_dir, manifest, cache_dir)."""
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n_sources):
        write_audio(src / f"clip{i}.wav", synth_clip(i, sr=sr, seconds=2.0), sr)
    pf = preflight(src, root / "corpus")
    manifest = build_manifest(pf)
    cache = root / "cache"
    names = ["logmel", "energy", *(n for n, _ in CANDIDATES)]
    encode_corpus(pf.norm_dir, manifest, names, cache)
    encode_combos(pf.norm_dir, manifest, [n for n, _ in CANDIDATES], cache)
    return pf.norm_dir, manifest, cache


def _gate2_json(report) -> dict:
    out = {}
    for dec in report.decisions:
        key = f"{dec.name}/{dec.variant}"
        g2b = None
        if dec.g2b is not None:
            g2b = {"n_beats": dec.n_beats_baseline, "pass": dec.pass_g2b, "by_family": {
                fam: {"head_mos": _est(f.head_mos),
                      "baselines_mos": {b: _est(e) for b, e in f.baselines_mos.items()},
                      "paired_lo": dict(f.paired_lo), "n_cells": f.n_cells,
                      "n_groups": f.n_groups, "beats_all": f.beats_all}
                for fam, f in dec.g2b.by_family.items()}}
        out[key] = {
            "n_test_groups": dec.g2a.n_test_groups, "underpowered": dec.underpowered,
            "passed": dec.passed,
            "g2a": {"ref_srcc": _est(dec.g2a.ref_srcc), "ref_lcc": _est(dec.g2a.ref_lcc),
                    "by_family": {fam: _est(e) for fam, e in dec.g2a.ref_srcc_by_family.items()},
                    "pass": dec.pass_g2a},
            "g2b": g2b,
            "scatter": report.scatter[key],
        }
    return out


def _additivity_json(combo_report) -> dict:
    """`raw`/`std` are this writer's names for `cosine`/`cosine_std` — F3 reads them,
    so the alias stays. Every S6 field below is named exactly as in
    tools/job_c_run.py::additivity_dict (AM2); a test asserts the two key sets match
    modulo that documented alias."""
    return {
        key: {"mean_cosine": add.mean_cosine(), "by_cell": {
            f"{pa.pair}@{pa.severity}": {"pair": pa.pair, "severity": pa.severity,
                                         "raw": _est(pa.cosine), "std": _est(pa.cosine_std),
                                         "cos_to_a": pa.cos_to_a, "cos_to_b": pa.cos_to_b,
                                         "cos_legs": pa.cos_legs, "norm_ratio": pa.norm_ratio,
                                         "r": pa.r, "rel_residual": pa.rel_residual,
                                         "alpha": _est(pa.alpha), "beta": _est(pa.beta),
                                         "n_groups": pa.n_groups, "n_sources": pa.n_sources,
                                         "rows_identical": pa.rows_identical}
            for (_, _), pa in add.by_cell.items()}}
        for key, add in combo_report.additivity.items()
    }


def _transfer_json(combo_report) -> dict:
    return {
        key: {pair: {"leg_a": pt.leg_a, "leg_b": pt.leg_b, "n": pt.n,
                     "frac_either_leg": pt.frac_either_leg,
                     "predicted_fraction": pt.predicted_fraction}
              for pair, pt in trans.by_pair.items()}
        for key, trans in combo_report.transfer.items()
    }


def _ood_json(ood_report) -> dict:
    return {
        key: {"metrics": list(t.metrics), "grid_disagreement": t.grid_disagreement,
              "ood_disagreement": t.ood_disagreement, "calibration_floor": t.calibration_floor,
              "amplifies": t.ood_amplifies_disagreement, "n_grid": t.n_grid, "n_ood": t.n_ood}
        for key, t in ood_report.teasers.items()
    }


def _provenance(candidates, n_sources: int, sr: int) -> dict:
    return provenance(
        fake=is_fake(candidates),
        note="Synthetic ViSQOL/MOS + fabricated NR incumbents on this seam; the "
             "scientific run is the GPU box (docs/GPU_BRINGUP.md).",
        n_sources=n_sources, sample_rate=sr,
        candidates=[f"{n}/{v}" for n, v in candidates],
        # AM8 anchors this at :57, which is inside `_gate2_json`; the candidate metadata
        # this writer actually emits is the provenance block, so the semantics ride here.
        candidate_semantics=candidate_semantics(candidates))


def write_artifacts(
    norm_dir: Path, manifest, candidates, cache_dir: Path, scores: Scores,
    out_dir: Path, *, n_sources: int, sr: int,
) -> dict[str, Path]:
    analysis = analyze(manifest, candidates, cache_dir)
    gate2 = run_gate2(manifest, candidates, cache_dir, scores)
    combos = analyze_combos(manifest, candidates, cache_dir)
    ood = run_ood_teaser(norm_dir, manifest, candidates, cache_dir, scores=scores, sr=sr)

    return {
        "heatmap": write_json(out_dir / "analysis" / "heatmap.json", heatmap_rows(analysis)),
        "cosine": write_json(out_dir / "analysis" / "cosine.json", cosine_matrices(analysis)),
        "monotonicity": write_json(out_dir / "analysis" / "monotonicity.json", monotonicity_curves(analysis)),
        "gate2": write_json(out_dir / "quality" / "gate2.json", _gate2_json(gate2)),
        "additivity": write_json(out_dir / "combos" / "additivity.json", _additivity_json(combos)),
        "transfer": write_json(out_dir / "combos" / "transfer.json", _transfer_json(combos)),
        "ood": write_json(out_dir / "ood" / "teaser.json", _ood_json(ood)),
        "provenance": write_json(out_dir / "provenance.json", _provenance(candidates, n_sources, sr)),
    }

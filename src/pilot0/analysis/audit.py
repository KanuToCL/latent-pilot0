"""Phase-5 demo (`make analyze`): synth corpus → fake-encode the matrix → write the
three acceptance artifacts (heatmap / cosine matrix / monotonicity curves) to
reports/analysis/ and print a summary.

BANNER: codec rows use the FAKE encoder — every number is a plumbing check, NOT a
result; only the log-mel floor and energy control are real. The scientific matrix
comes from the GPU box (docs/GPU_BRINGUP.md).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..audio.synth import synth_clip
from ..corpus.io import write_audio
from ..corpus.manifest import build_manifest
from ..corpus.preflight import preflight
from ..encode.pipeline import encode_corpus
from ..probes.gate1 import MIN_TEST_GROUPS
from ..serialize import write_json
from .run import analyze
from .serialize import cosine_matrices, heatmap_rows, monotonicity_curves

CANDIDATES = [("fake-encodec24k", "z"), ("fake-wavlm", "l12")]


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
    write_json(reports / "heatmap.json", heatmap_rows(report))
    write_json(reports / "cosine.json", cosine_matrices(report))
    write_json(reports / "monotonicity.json", monotonicity_curves(report))

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

"""Phase-7 demo (`make combos-demo`): synth corpus → encode singles + combos →
additivity geometry + zero-shot type-probe transfer, printed per candidate.

BANNER: codec rows use the FAKE encoder — every number is a plumbing check, NOT a
result. It shows the combo pipeline renders/encodes A+B cells into the shared cache
and that the additivity cosine and the zero-shot transfer both compute end-to-end.
The scientific combo analysis comes from the GPU box.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..audio.synth import synth_clip
from ..corpus.io import write_audio
from ..corpus.manifest import build_manifest
from ..corpus.preflight import preflight
from ..encode.pipeline import encode_corpus
from .encode import encode_combos
from .grid import COMBO_SEVERITIES
from .run import analyze_combos

CANDIDATES = [("fake-encodec24k", "z"), ("fake-wavlm", "l12")]


def main(n_sources: int = 16, sr: int = 48000) -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        for i in range(n_sources):
            write_audio(src / f"clip{i}.wav", synth_clip(i, sr=sr, seconds=2.0), sr)
        pf = preflight(src, Path(d) / "corpus")
        man = build_manifest(pf)
        cache = Path(d) / "cache"
        names = [n for n, _ in CANDIDATES]
        encode_corpus(pf.norm_dir, man, names, cache)
        encode_combos(pf.norm_dir, man, names, cache)
        report = analyze_combos(man, CANDIDATES, cache)

    print("⚠  FAKE codec latents — plumbing check, NOT results\n")
    for key in report.additivity:
        add, trans = report.additivity[key], report.transfer[key]
        print(f"{key}")
        print(f"  additivity mean cosine {add.mean_cosine():+.3f}  (1 ⇒ combo codirectional with Δa+Δb)")
        for (pair, sev), pa in add.by_cell.items():
            if sev == COMBO_SEVERITIES[len(COMBO_SEVERITIES) // 2]:  # the middle mid severity
                c, cs = pa.cosine, pa.cosine_std
                print(f"    {pair} @sev{sev}: raw {c.point:+.3f} [{c.lo:+.3f}, {c.hi:+.3f}]  "
                      f"std {cs.point:+.3f}  ({pa.n_groups} sources)")
        for pair, pt in trans.by_pair.items():
            print(f"    {pair}: {pt.frac_either_leg:.0%} of {pt.n} combos read as a constituent leg "
                  f"({pt.leg_a} {pt.predicted_fraction.get(pt.leg_a, 0):.0%} / "
                  f"{pt.leg_b} {pt.predicted_fraction.get(pt.leg_b, 0):.0%})")
        print()


if __name__ == "__main__":
    main()

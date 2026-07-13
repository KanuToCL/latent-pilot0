"""Phase-4 demo: synth corpus → encode (fake codecs + real log-mel floor + energy
control) → Gate-1 table with source-level CIs (`make probe-demo`).

BANNER: the codec rows use the FAKE encoder, so every codec number here is a
plumbing check, NOT a result. Only the log-mel floor and energy control are real.
A small synth corpus is UNDERPOWERED by design (few test sources) — the gate says
so rather than passing on noise. The scientific table comes from the GPU box.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..audio.synth import synth_clip
from ..corpus.io import write_audio
from ..corpus.manifest import build_manifest
from ..corpus.preflight import preflight
from ..encode.pipeline import encode_corpus
from .gate1 import MIN_TEST_GROUPS
from .run import run_gate1

CANDIDATES = [("fake-encodec24k", "z"), ("fake-wavlm", "l12")]


def _sev_line(sev, energy) -> str:
    cells = []
    for fam, fs in sev.by_family.items():
        base = energy.by_family[fam].without_clean.point
        cells.append(f"{fam[:4]}={fs.without_clean.lo:+.2f}/{fs.without_clean.hi:+.2f}(E{base:+.2f})")
    return "  ".join(cells)


def main(n_sources: int = 8, sr: int = 48000) -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        for i in range(n_sources):
            write_audio(src / f"clip{i}.wav", synth_clip(i, sr=sr, seconds=2.0), sr)
        pf = preflight(src, Path(d) / "corpus")
        man = build_manifest(pf)
        cache = Path(d) / "cache"

        encode_corpus(pf.norm_dir, man, ["logmel", "energy", *(n for n, _ in CANDIDATES)], cache)
        report = run_gate1(man, CANDIDATES, cache)

    print("⚠  FAKE codec latents — plumbing check, NOT results (floor + energy are real)\n")
    f = report.floor
    print(f"common cells scored: {report.n_common_conditions}   test sources: {f.n_test_groups} "
          f"(need ≥{MIN_TEST_GROUPS} to be powered)")
    print(f"log-mel floor  type macro-F1 = {f.type.macro_f1.point:.3f} "
          f"[{f.type.macro_f1.lo:.3f}, {f.type.macro_f1.hi:.3f}]\n")
    for dec in report.decisions:
        power = "UNDERPOWERED" if dec.underpowered else "powered"
        print(f"{dec.name}/{dec.variant}  ({power})")
        mf = dec.type.macro_f1
        print(f"  type macro-F1 {mf.point:.3f} [{mf.lo:.3f}, {mf.hi:.3f}]  floor margin {dec.margin_over_floor:+.3f}")
        print(f"  severity SRCC[lo/hi(Energy)]: {_sev_line(dec.severity, dec.energy_severity)}")
        print(f"  severity families passing: {dec.n_severity_pass}/7")
        print(f"  GATE 1: {'PASS' if dec.passed else 'FAIL'} "
              f"(type={dec.pass_type} sev={dec.pass_severity} floor={dec.pass_floor} "
              f"powered={not dec.underpowered})\n")


if __name__ == "__main__":
    main()

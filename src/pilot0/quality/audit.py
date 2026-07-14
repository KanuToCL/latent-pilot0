"""Phase-6 demo (`make quality-demo`): synth corpus → encode → FakeScores → Gate-2
table with source-level CIs.

BANNER: the codec latents are FAKE and the ViSQOL/MOS/baseline scores are SYNTHETIC
— every number is a plumbing check, NOT a result. It shows the machinery runs, that
Gate 2 keeps the ViSQOL-trained head and the MOS-predictor baselines on separate
ground truths (§8), and that G2b is a paired head-vs-baseline test on MOS. The demo
uses enough sources to be POWERED, so the FAIL is a real clause decision (the fake
head can neither recover ViSQOL nor paired-beat the MOS predictors), not the
underpowered short-circuit. The scientific table comes from the GPU box.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ..audio.synth import synth_clip
from ..corpus.io import write_audio
from ..corpus.manifest import build_manifest
from ..corpus.preflight import preflight
from ..degrade.grid import FAMILIES
from ..encode.pipeline import encode_corpus
from .run import run_gate2
from .scores import FakeScores

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
        encode_corpus(pf.norm_dir, man, [n for n, _ in CANDIDATES], cache)
        report = run_gate2(man, CANDIDATES, cache, FakeScores())

    print("⚠  FAKE latents + SYNTHETIC ViSQOL/MOS/baseline scores — plumbing, NOT results\n")
    print(f"MOS ground truth available: {report.mos_available}   (else G2b is not evaluable)\n")
    for dec in report.decisions:
        power = "UNDERPOWERED" if dec.underpowered else "powered"
        r = dec.g2a.ref_srcc
        worst_fam = min((e.point for e in dec.g2a.ref_srcc_by_family.values() if np.isfinite(e.point)),
                        default=float("nan"))
        print(f"{dec.name}/{dec.variant}  (test sources: {dec.g2a.n_test_groups} — {power})")
        print(f"  G2a head-vs-ViSQOL pooled SRCC {r.point:.3f} [{r.lo:.3f}, {r.hi:.3f}]  "
              f"LCC {dec.g2a.ref_lcc.point:.3f}  worst-family {worst_fam:.3f}  → {'PASS' if dec.pass_g2a else 'fail'}")
        if dec.g2b_evaluable:
            worst = {f: min(dec.g2b.by_family[f].paired_lo.values(), default=float("nan"))
                     for f in FAMILIES}
            cells = "  ".join(f"{f[:4]}={worst[f]:+.2f}" for f in FAMILIES)
            print(f"  G2b min paired (head−baseline) SRCC CI-lower per family: {cells}")
            print(f"  G2b families paired-beating all baselines: {dec.n_beats_baseline}/7  "
                  f"→ {'PASS' if dec.pass_g2b else 'fail'}")
        else:
            print("  G2b NOT EVALUABLE (no MOS ground truth)")
        print(f"  GATE 2: {'PASS' if dec.passed else 'FAIL'} "
              f"(g2a={dec.pass_g2a} g2b={dec.pass_g2b} powered={not dec.underpowered})\n")


if __name__ == "__main__":
    main()

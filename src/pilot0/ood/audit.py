"""Phase-7 demo (`make ood-demo`): fit the quality head on the synthetic grid, then
measure no-reference metric DISAGREEMENT on out-of-grid textures vs on the grid.

BANNER: FAKE codec latents + a real grid-fit head + FABRICATED incumbent columns —
plumbing, NOT a result. It shows the OOD path runs end to end (synthesise off-manifold
audio → encode → head predicts → disagreement quantified) and that disagreement is
larger OOD than in-grid, the motivating observation for the F6 figure. Real generative
renders and real NISQA/DNSMOS/UTMOS drop in at the box.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..audio.synth import synth_clip
from ..corpus.io import write_audio
from ..corpus.manifest import build_manifest
from ..corpus.preflight import preflight
from ..encode.pipeline import encode_corpus
from .run import run_ood_teaser

CANDIDATES = [("fake-encodec24k", "z"), ("fake-wavlm", "l12")]


def main(n_sources: int = 16, n_ood: int = 24, sr: int = 48000) -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        for i in range(n_sources):
            write_audio(src / f"clip{i}.wav", synth_clip(i, sr=sr, seconds=2.0), sr)
        pf = preflight(src, Path(d) / "corpus")
        man = build_manifest(pf)
        cache = Path(d) / "cache"
        encode_corpus(pf.norm_dir, man, [n for n, _ in CANDIDATES], cache)
        report = run_ood_teaser(pf.norm_dir, man, CANDIDATES, cache, n_ood=n_ood, sr=sr)

    print("⚠  FAKE latents + real grid-fit head + FABRICATED incumbents — plumbing, NOT results\n")
    for key, t in report.teasers.items():
        print(f"{key}   metrics: {', '.join(t.metrics)}")
        print(f"  in-grid disagreement {t.grid_disagreement:.3f} (n={t.n_grid})   "
              f"OOD disagreement {t.ood_disagreement:.3f} (n={t.n_ood})   "
              f"calibration floor {t.calibration_floor:.3f}")
        print(f"  OOD amplifies metric disagreement (over floor): {t.ood_amplifies_disagreement} "
              f"(by construction — FAKE incumbents)\n")


if __name__ == "__main__":
    main()

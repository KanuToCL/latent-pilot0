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

import numpy as np

from ..audio.synth import synth_clip
from ..corpus.io import write_audio
from ..corpus.manifest import build_manifest, renderable_rows
from ..corpus.preflight import preflight
from ..encode.cache import cache_version
from ..encode.headroom import CEILING_DBFS
from ..encode.pipeline import encode_corpus, memoized_master_loader, resolve_headroom
from ..encode.render import resample_to_native
from ..quality.dataset import build_quality_data
from ..quality.scores import FakeScores
from ..seam.base import pool_mean_std
from ..seam.registry import make_encoder
from .scores import OODNRScores
from .synth import synth_ood_clip
from .teaser import ood_teaser

CANDIDATES = [("fake-encodec24k", "z"), ("fake-wavlm", "l12")]


def _pool_ood(wavs, sr, name, variant, scalar) -> np.ndarray:
    enc = make_encoder(name)
    rows = []
    for w in wavs:
        scaled = (resample_to_native(w, sr, enc.native_sr) * scalar).astype(np.float32)
        rows.append(pool_mean_std(enc.encode(scaled, enc.native_sr)[variant].frames))
    return np.asarray(rows, dtype=np.float64)


def main(n_sources: int = 16, n_ood: int = 24, sr: int = 48000) -> None:
    ood_ids = [f"ood{i}" for i in range(n_ood)]
    ood_wavs = [synth_ood_clip(i, sr=sr, seconds=2.0) for i in range(n_ood)]
    ood_nr = OODNRScores()

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

        cv = cache_version(man, CEILING_DBFS)
        load = memoized_master_loader(pf.norm_dir)
        results = []
        for name, variant in CANDIDATES:
            enc = make_encoder(name)
            scalar = resolve_headroom(cache, cv, enc.native_sr, renderable_rows(man, enc.native_sr), load, CEILING_DBFS).scalar
            grid_data = build_quality_data(man, name, variant, cache, FakeScores())
            ood_X = _pool_ood(ood_wavs, sr, name, variant, scalar)
            results.append((f"{name}/{variant}", ood_teaser(grid_data, ood_X, ood_ids, ood_nr)))

    print("⚠  FAKE latents + real grid-fit head + FABRICATED incumbents — plumbing, NOT results\n")
    for key, t in results:
        print(f"{key}   metrics: {', '.join(t.metrics)}")
        print(f"  in-grid disagreement {t.grid_disagreement:.3f} (n={t.n_grid})   "
              f"OOD disagreement {t.ood_disagreement:.3f} (n={t.n_ood})   "
              f"calibration floor {t.calibration_floor:.3f}")
        print(f"  OOD amplifies metric disagreement (over floor): {t.ood_amplifies_disagreement} "
              f"(by construction — FAKE incumbents)\n")


if __name__ == "__main__":
    main()

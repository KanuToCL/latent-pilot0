"""Phase-7 OOD orchestration: synthesise off-grid textures, encode them through each
candidate at the SAME headroom scalar the grid used, and measure no-reference metric
disagreement (grid-fit head vs fabricated incumbents) on them vs on the grid. Pure
wiring — the disagreement definition lives in `teaser.py`, the fabricated incumbents
in `scores.py`. Shared by `ood.audit` (the demo) and `release.artifacts` (the F6 JSON).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..encode.cache import cache_version
from ..encode.headroom import CEILING_DBFS
from ..encode.pipeline import memoized_master_loader, resolve_headroom
from ..encode.render import resample_to_native
from ..corpus.manifest import renderable_rows
from ..quality.dataset import build_quality_data
from ..quality.scores import FakeScores, Scores
from ..seam.base import pool_mean_std
from ..seam.registry import make_encoder
from .scores import OODNRScores
from .synth import synth_ood_clip
from .teaser import OODTeaser, ood_teaser


@dataclass(frozen=True)
class OODReport:
    teasers: dict[str, OODTeaser]  # "name/variant" -> teaser
    n_ood: int


def _pool_ood(wavs, sr, name, variant, scalar) -> np.ndarray:
    enc = make_encoder(name)
    rows = []
    for w in wavs:
        scaled = (resample_to_native(w, sr, enc.native_sr) * scalar).astype(np.float32)
        rows.append(pool_mean_std(enc.encode(scaled, enc.native_sr)[variant].frames))
    return np.asarray(rows, dtype=np.float64)


def run_ood_teaser(
    norm_dir: Path,
    manifest,
    candidates,
    cache_dir: Path,
    *,
    scores: Scores | None = None,
    ood_nr=None,
    n_ood: int = 24,
    sr: int = 48000,
    seconds: float = 2.0,
    ceiling_dbfs: float = CEILING_DBFS,
) -> OODReport:
    scores = scores or FakeScores()
    ood_nr = ood_nr or OODNRScores()
    ood_ids = [f"ood{i}" for i in range(n_ood)]
    ood_wavs = [synth_ood_clip(i, sr=sr, seconds=seconds) for i in range(n_ood)]

    cv = cache_version(manifest, ceiling_dbfs)
    load = memoized_master_loader(norm_dir)
    teasers: dict[str, OODTeaser] = {}
    for name, variant in candidates:
        enc = make_encoder(name)
        scalar = resolve_headroom(
            cache_dir, cv, enc.native_sr, renderable_rows(manifest, enc.native_sr), load, ceiling_dbfs
        ).scalar
        grid_data = build_quality_data(manifest, name, variant, cache_dir, scores, ceiling_dbfs=ceiling_dbfs)
        ood_X = _pool_ood(ood_wavs, sr, name, variant, scalar)
        teasers[f"{name}/{variant}"] = ood_teaser(grid_data, ood_X, ood_ids, ood_nr)
    return OODReport(teasers=teasers, n_ood=n_ood)

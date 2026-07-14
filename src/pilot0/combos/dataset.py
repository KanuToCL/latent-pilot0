"""Read the combo latents back from the cache and pool them, the combo-side twin of
`probes.dataset`. The manifest supplies each source's split/group (the cache sidecar
does not, exactly as for the singles); the cache is the store. No re-encode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..encode.cache import cache_version, cell_id, latent_path, load_latent
from ..encode.headroom import CEILING_DBFS
from ..seam.base import pool_mean_std
from ..seam.registry import make_encoder
from .grid import COMBO_PAIRS, combo_label, combo_severities


def iter_combo_latents(manifest, name, variant, cache_dir, *, pairs=COMBO_PAIRS, ceiling_dbfs=CEILING_DBFS):
    """Yield `(row, LatentResult)` for every cached combo cell of (name, variant). A
    cell absent from the cache is skipped (encode_combos may not have run for it)."""
    enc = make_encoder(name)
    cv = cache_version(manifest, ceiling_dbfs)
    src_meta = {r["source"]: (r["group"], r["split"]) for r in manifest["rows"]}
    for source, (group, split) in src_meta.items():
        for a, b in pairs:
            for sev in combo_severities(a, b, enc.native_sr):
                path = latent_path(cache_dir, name, variant, cell_id(source, combo_label(a, b), sev, enc.native_sr), cv)
                if path.exists():
                    row = {"source": source, "leg_a": a, "leg_b": b, "pair": combo_label(a, b),
                           "severity": sev, "group": group, "split": split}
                    yield row, load_latent(path)


@dataclass(frozen=True)
class ComboData:
    X: np.ndarray  # [N, 2D] mean+std pooled latents
    pair: np.ndarray  # [N] combo label "a+b"
    leg_a: np.ndarray  # [N] first-leg family
    leg_b: np.ndarray  # [N] second-leg family
    severity: np.ndarray  # [N] shared mid severity of both legs
    split: np.ndarray
    group: np.ndarray
    source: np.ndarray
    encoder: str
    variant: str


def build_combo_data(manifest, name, variant, cache_dir, *, pairs=COMBO_PAIRS, ceiling_dbfs=CEILING_DBFS) -> ComboData:
    X, pair, leg_a, leg_b, sev, split, group, source = ([] for _ in range(8))
    for r, lat in iter_combo_latents(manifest, name, variant, cache_dir, pairs=pairs, ceiling_dbfs=ceiling_dbfs):
        X.append(pool_mean_std(lat.frames))
        pair.append(r["pair"]), leg_a.append(r["leg_a"]), leg_b.append(r["leg_b"])
        sev.append(r["severity"]), split.append(r["split"]), group.append(r["group"]), source.append(r["source"])
    return ComboData(
        X=np.asarray(X, dtype=np.float64),
        pair=np.asarray(pair), leg_a=np.asarray(leg_a), leg_b=np.asarray(leg_b),
        severity=np.asarray(sev, dtype=int), split=np.asarray(split),
        group=np.asarray(group), source=np.asarray(source), encoder=name, variant=variant,
    )

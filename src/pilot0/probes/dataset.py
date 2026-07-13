"""Assemble a probe design matrix from the Phase-3 latent cache.

The manifest is the index (it carries split/group/family/severity per source-disjoint
§2.5); the cache is the store. For one (encoder, variant) we load every renderable
cell's latent, mean+std pool it to a fixed vector (§2.3), and stack — no re-encode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..encode.cache import cache_version, cell_id, latent_path, load_latent
from ..encode.headroom import CEILING_DBFS
from ..corpus.manifest import renderable_rows
from ..seam.base import pool_mean_std
from ..seam.registry import make_encoder

CLEAN = "clean"


def iter_latents(manifest, name, variant, cache_dir, *, ceiling_dbfs=CEILING_DBFS):
    """Yield `(row, LatentResult)` for every renderable cell of (name, variant) from
    the Phase-3 cache — the single cache-walk shared by the pooled design matrix and
    the frame-level dropout probe. No re-encode; the manifest is the index."""
    enc = make_encoder(name)
    cv = cache_version(manifest, ceiling_dbfs)
    for r in renderable_rows(manifest, enc.native_sr):
        cid = cell_id(r["source"], r["family"], r["severity"], enc.native_sr)
        yield r, load_latent(latent_path(cache_dir, name, variant, cid, cv))


@dataclass(frozen=True)
class ProbeData:
    X: np.ndarray  # [N, 2D] mean+std pooled features
    family: np.ndarray  # [N] str label (includes "clean")
    severity: np.ndarray  # [N] int 0..5 (0 = clean)
    split: np.ndarray  # [N] "train"/"test"
    group: np.ndarray  # [N] speaker/track id
    encoder: str
    variant: str

    def mask(self, **eq) -> np.ndarray:
        m = np.ones(len(self.severity), dtype=bool)
        for k, v in eq.items():
            m &= getattr(self, k) == v
        return m

    def select(self, conditions: set[tuple[str, int]]) -> "ProbeData":
        """Restrict to a set of (family, severity) cells — used to score every
        representation on the SAME rows when native rates differ (finding M1)."""
        keep = np.array([(f, int(s)) in conditions for f, s in zip(self.family, self.severity)])
        return ProbeData(
            X=self.X[keep], family=self.family[keep], severity=self.severity[keep],
            split=self.split[keep], group=self.group[keep], encoder=self.encoder, variant=self.variant,
        )

    def n_test_groups(self) -> int:
        return int(len(np.unique(self.group[self.split == "test"])))


def build_probe_data(
    manifest: dict, name: str, variant: str, cache_dir, *, ceiling_dbfs: float = CEILING_DBFS
) -> ProbeData:
    feats, fam, sev, split, group = [], [], [], [], []
    for r, lat in iter_latents(manifest, name, variant, cache_dir, ceiling_dbfs=ceiling_dbfs):
        feats.append(pool_mean_std(lat.frames))
        fam.append(r["family"])
        sev.append(r["severity"])
        split.append(r["split"])
        group.append(r["group"])
    return ProbeData(
        X=np.asarray(feats, dtype=np.float64),
        family=np.asarray(fam),
        severity=np.asarray(sev, dtype=int),
        split=np.asarray(split),
        group=np.asarray(group),
        encoder=name,
        variant=variant,
    )

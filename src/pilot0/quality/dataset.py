"""Assemble the quality design matrix: pooled latents (the head's input) aligned with
the external scores for each cell (ViSQOL target, MOS ground truth, NR-baseline
predictions). Reuses the Phase-3 cache walk; no re-encode. MOS and any baseline may
be absent (box corpus without those annotations) — carried as None so Gate 2 fails
closed rather than fabricating a comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..encode.headroom import CEILING_DBFS
from ..probes.dataset import iter_latents
from ..seam.base import pool_mean_std
from .scores import MOS_METRIC, NR_BASELINES, REF_METRIC, Scores


@dataclass(frozen=True)
class QualityData:
    X: np.ndarray  # [N, 2D] mean+std pooled latents
    ref: np.ndarray  # [N] full-reference training target (ViSQOL)
    mos: np.ndarray | None  # [N] human MOS ground truth, or None if not collected
    nr: dict[str, np.ndarray]  # baseline -> [N] no-reference predictions (absent baselines omitted)
    family: np.ndarray
    severity: np.ndarray
    split: np.ndarray
    group: np.ndarray
    encoder: str
    variant: str

    def n_test_groups(self) -> int:
        return int(len(np.unique(self.group[self.split == "test"])))


def _nan(v: float | None) -> float:
    return float("nan") if v is None else float(v)


def build_quality_data(
    manifest, name, variant, cache_dir, scores: Scores, *, ceiling_dbfs: float = CEILING_DBFS
) -> QualityData:
    if not scores.has(REF_METRIC):  # the head trains on ViSQOL; no target ⇒ Gate 2 unrunnable
        raise ValueError(f"scores has no '{REF_METRIC}' reference target — cannot train the quality head")
    has_mos = scores.has(MOS_METRIC)
    baselines = [b for b in NR_BASELINES if scores.has(b)]

    X, ref, mos, fam, sev, split, group = [], [], [], [], [], [], []
    nr: dict[str, list[float]] = {b: [] for b in baselines}
    for r, lat in iter_latents(manifest, name, variant, cache_dir, ceiling_dbfs=ceiling_dbfs):
        cell = (r["source"], r["family"], r["severity"])
        ref_val = scores.get(REF_METRIC, *cell)
        if ref_val is None:  # no full-reference target for this cell → unusable (G2a trains on it)
            continue
        X.append(pool_mean_std(lat.frames))
        ref.append(float(ref_val))
        if has_mos:  # subset coverage: uncovered cells become nan and drop out of G2b (W1)
            mos.append(_nan(scores.get(MOS_METRIC, *cell)))
        for b in baselines:
            nr[b].append(_nan(scores.get(b, *cell)))
        fam.append(r["family"]), sev.append(r["severity"])
        split.append(r["split"]), group.append(r["group"])

    mos_arr = np.asarray(mos, dtype=np.float64) if has_mos else None
    if mos_arr is not None and not np.isfinite(mos_arr).any():  # collected but zero coverage → not evaluable
        mos_arr = None
    return QualityData(
        X=np.asarray(X, dtype=np.float64),
        ref=np.asarray(ref, dtype=np.float64),
        mos=mos_arr,
        nr={b: np.asarray(v, dtype=np.float64) for b, v in nr.items()},
        family=np.asarray(fam), severity=np.asarray(sev, dtype=int),
        split=np.asarray(split), group=np.asarray(group), encoder=name, variant=variant,
    )

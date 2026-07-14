"""OOD teaser (§9 F6, motivation only, NO claims): reference-free quality metrics
DISAGREE on generative material where they agree on the classical grid.

All four metrics live on the SAME 1–5 quality scale (the head is clipped to [1,5], the
incumbents are MOS-scale), so disagreement is measured DIRECTLY in that shared space:
per clip, the std across the metrics after each is CENTERED by its grid mean (removing
only its calibration offset — a constant anchoring difference is not disagreement). No
per-metric std normalization: dividing each metric by its own grid std would manufacture
spread from a common off-grid drift, so "OOD > grid" would be a normalization artifact
rather than genuine divergence (physics review). The result is in score units.

The head is clipped to [1,5] like any bounded quality metric; its off-manifold
extrapolation (the ridge blowing to ±tens) is itself an OOD-unreliability signal, so
post-clip the head saturates at a boundary OOD and legitimately diverges from the
incumbents there. The head column is the real grid-fit ridge on OOD latents it never
saw; the incumbent columns come from `ood.scores` (FABRICATED with a divergence
signature). The disagreement math is unchanged when the real NISQA/DNSMOS/UTMOS replace
the fabricated columns at bring-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..quality.dataset import QualityData
from ..quality.head import fit_head_model

_SCORE_MIN, _SCORE_MAX = 1.0, 5.0  # a quality score is bounded; the ridge extrapolates off-manifold


@dataclass(frozen=True)
class OODTeaser:
    metrics: tuple[str, ...]
    grid_disagreement: float  # mean per-clip metric spread on in-grid degraded cells (score units)
    ood_disagreement: float  # mean per-clip metric spread on OOD clips (score units)
    calibration_floor: float  # std of the per-metric grid-mean offsets — the OOD-only spread grid-centering leaves even under perfect agreement
    n_grid: int
    n_ood: int

    @property
    def ood_amplifies_disagreement(self) -> bool:
        # genuine only if it clears the calibration floor grid-mean centering imposes OOD
        return self.ood_disagreement > max(self.grid_disagreement, self.calibration_floor)


def _per_clip_spread(columns: dict[str, np.ndarray], centers: dict[str, float]) -> float:
    centered = np.column_stack([columns[m] - centers[m] for m in centers])
    return float(np.nanmean(np.nanstd(centered, axis=1)))


def ood_teaser(grid_data: QualityData, ood_X: np.ndarray, ood_ids: list[str], ood_nr) -> OODTeaser:
    model = fit_head_model(grid_data)
    metric_names = ("head", *grid_data.nr.keys())
    dg = grid_data.severity > 0  # degraded grid cells (clean is trivially top-quality)

    def head(X):
        return np.clip(model.predict(X), _SCORE_MIN, _SCORE_MAX)

    grid_cols = {"head": head(grid_data.X)[dg], **{b: arr[dg] for b, arr in grid_data.nr.items()}}
    ood_cols = {"head": head(ood_X),
                **{b: np.array([ood_nr.get(b, cid) for cid in ood_ids]) for b in grid_data.nr}}

    # center each metric on its GRID mean (remove calibration offset only, keep the
    # shared 1–5 scale); finite guard so an all-nan grid column can't poison the center
    centers = {m: (lambda v: float(v) if np.isfinite(v) else 0.0)(np.nanmean(grid_cols[m])) for m in metric_names}
    return OODTeaser(
        metrics=metric_names,
        grid_disagreement=_per_clip_spread(grid_cols, centers),
        ood_disagreement=_per_clip_spread(ood_cols, centers),
        calibration_floor=float(np.std(list(centers.values()))),
        n_grid=int(dg.sum()),
        n_ood=len(ood_ids),
    )

"""No-reference quality head: a ridge on frozen pooled latents predicting the
full-reference target (ViSQOL) WITHOUT the reference. Source-disjoint train/test,
bootstrap CI over test groups (§2.5).

The head is fit ONCE (`fit_head`) and its predictions are shared by the two Gate-2
clauses, kept in separate modules so they cannot be conflated (§8 rig):
  * G2a here — pooled head-vs-ViSQOL SRCC/LCC on DEGRADED held-out cells. Clean is
    trivially top-quality and would inflate a pooled rank correlation without
    proving the head ranks degradations (the Phase-4 M2 discipline); it still
    anchors TRAINING as a legitimate high-quality point.
  * G2b in g2b.py — the SAME predictions scored against human MOS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..degrade.grid import FAMILIES
from ..probes.gate1 import MIN_TEST_GROUPS
from ..probes.metrics import N_BOOTSTRAP, Estimate, bootstrap_over_groups, lcc, srcc
from .dataset import QualityData


@dataclass(frozen=True)
class QualityG2a:
    ref_srcc: Estimate  # pooled, head vs ViSQOL, degraded test cells (the GATED metric)
    ref_lcc: Estimate  # pooled, secondary
    ref_srcc_by_family: dict[str, Estimate]  # §2.4 per-family view; reported, exposes pooling inflation
    n_test_groups: int


def _head() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def fit_head_model(data: QualityData):
    """The fitted head (StandardScaler→Ridge) trained on the training split, clean
    included as the high-quality anchor. Returned as an estimator so callers outside
    the grid (the Phase-7 OOD teaser) can `.predict` on latents the gate never saw."""
    tr = data.split == "train"
    return _head().fit(data.X[tr], data.ref[tr])


def fit_head(data: QualityData) -> np.ndarray:
    """Predictions for EVERY row from one fit, so G2a and G2b share it."""
    return fit_head_model(data).predict(data.X)


def evaluate_g2a(data: QualityData, pred: np.ndarray, *, n_boot: int = N_BOOTSTRAP) -> QualityG2a:
    te = (data.split == "test") & (data.severity > 0)  # degraded only (M2)
    ref, p, g = data.ref[te], pred[te], data.group[te]
    by_family: dict[str, Estimate] = {}
    for fam in FAMILIES:
        fm = te & (data.family == fam)
        rf, pf, gf = data.ref[fm], pred[fm], data.group[fm]
        by_family[fam] = bootstrap_over_groups(
            gf, lambda i: srcc(rf[i], pf[i]), n=n_boot, min_groups=MIN_TEST_GROUPS
        )
    return QualityG2a(
        ref_srcc=bootstrap_over_groups(g, lambda i: srcc(ref[i], p[i]), n=n_boot, min_groups=MIN_TEST_GROUPS),
        ref_lcc=bootstrap_over_groups(g, lambda i: lcc(ref[i], p[i]), n=n_boot, min_groups=MIN_TEST_GROUPS),
        ref_srcc_by_family=by_family,
        n_test_groups=data.n_test_groups(),
    )

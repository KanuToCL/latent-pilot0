"""Severity probe: one linear (ridge) regressor per family predicting the severity
index, scored by Spearman SRCC on the source-disjoint test split.

Two SRCCs are reported per family (elder finding M2): WITH the clean severity-0
anchor (diagnostic — clean is trivially separable, which inflates the number and
correlates the families) and WITHOUT it (the honest ordinal test: can the probe
order severities 1..5 among themselves). Gate 1 keys on the without-clean CI lower
bound. Rank-invariance does NOT make severity loudness-safe — the energy control
baseline (a level-only representation) is what tests that, at the run level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..degrade.grid import FAMILIES
from .dataset import CLEAN, ProbeData
from .metrics import Estimate, bootstrap_over_groups, srcc


@dataclass(frozen=True)
class FamilySeverity:
    with_clean: Estimate
    without_clean: Estimate


@dataclass(frozen=True)
class SeverityResult:
    by_family: dict[str, FamilySeverity]

    def n_pass(self, threshold: float, baseline: "SeverityResult | None" = None,
              margin: float = 0.0) -> int:
        """Families whose without-clean SRCC CI-lower clears `threshold` AND (if a
        baseline is given) whose CI-lower beats the baseline's CI-UPPER by `margin`
        — the fully conservative comparison, so a baseline that is luckily low on
        this split can't hand the codec a spurious pass (finding N2)."""
        out = 0
        for fam, fs in self.by_family.items():
            if not fs.without_clean.clears(threshold):
                continue
            if baseline is not None:
                base_hi = baseline.by_family[fam].without_clean.hi
                if not (np.isfinite(base_hi) and fs.without_clean.lo >= base_hi + margin):
                    continue
            out += 1
        return out


def _linear_regressor() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def evaluate_severity_probes(data: ProbeData) -> SeverityResult:
    by_family: dict[str, FamilySeverity] = {}
    for fam in FAMILIES:
        anchored = (data.family == fam) | (data.family == CLEAN)  # clean = sev-0 anchor for training
        tr = anchored & (data.split == "train")
        reg = _linear_regressor().fit(data.X[tr], data.severity[tr])

        te_all = anchored & (data.split == "test")
        te_deg = (data.family == fam) & (data.split == "test")  # 1..5, no clean
        by_family[fam] = FamilySeverity(
            with_clean=_srcc_ci(data, reg, te_all),
            without_clean=_srcc_ci(data, reg, te_deg),
        )
    return SeverityResult(by_family=by_family)


def _srcc_ci(data: ProbeData, reg, mask: np.ndarray) -> Estimate:
    y_true, groups = data.severity[mask], data.group[mask]
    y_pred = reg.predict(data.X[mask])
    return bootstrap_over_groups(groups, lambda idx: srcc(y_true[idx], y_pred[idx]))

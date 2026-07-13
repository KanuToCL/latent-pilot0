"""Held-out-severity interpolation — RQ5 monotonicity. Train each family's ridge on
severities {1,2,4,5} only; evaluate on the source-disjoint test split across the
FULL ladder 1..5. Read off:

  * SRCC (with CI) over all held-out rows — a real rank correlation, because the
    test target still varies over 1..5 (scoring on the single held-out level 3
    alone would be a constant target → undefined, the trap this avoids);
  * the monotonicity curve: mean predicted severity per true level;
  * interp_frac — the bootstrap probability, over test-group resamples, that the
    held-out level 3 is INTERPOLATED, i.e. mean-pred(2) < mean-pred(3) < mean-pred(4)
    (strictly increasing only — a backwards-ordinal probe is not interpolation).

A family is `evaluable` only when both trained neighbours (2, 4) and the held-out
level (3) are actually present in the (common) cells; otherwise placing level 3 is
extrapolation, not interpolation (e.g. band-limit at 16 kHz keeps only the widest
cutoffs — findings W2). A single-point ordering check would fire ~1/3 of the time on
noise (findings B1), so `interpolates` requires interp_frac high AND a positive SRCC
CI-lower. Clean (sev 0) is excluded: interpolation is a statement about the 1..5
ladder. Whether the codec interpolates BEYOND loudness is judged against the energy
control at the report level (W1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..degrade.grid import FAMILIES
from .dataset import ProbeData
from .metrics import Estimate, bootstrap_fraction, bootstrap_over_groups, srcc

HELD_OUT_SEVERITY = 3
NEIGHBOURS = (2, 4)
TRAIN_SEVERITIES = (1, 2, 4, 5)
LADDER = (1, 2, 3, 4, 5)
INTERP_FRAC_MIN = 0.90  # ordering must hold in ≥90% of test-group resamples
INTERP_OVER_ENERGY_MARGIN = 0.05  # beat the energy control by the same buffer as G1b (§8)


@dataclass(frozen=True)
class FamilyInterpolation:
    srcc: Estimate  # over held-out sources, full 1..5 ladder
    curve: dict[int, float]  # true level -> mean predicted severity (nan if level absent)
    interp_frac: float  # P(mean-pred(2) < mean-pred(3) < mean-pred(4)) over resamples
    evaluable: bool  # neighbours 2, 4 and the held-out 3 all present in the test cells

    @property
    def interpolates(self) -> bool:
        # Evidence only when the ordering is reliable AND the ridge is genuinely
        # positively ordinal on held-out sources — no inverted branch, no lucky split.
        return (
            self.evaluable
            and np.isfinite(self.srcc.lo)
            and self.srcc.lo > 0
            and np.isfinite(self.interp_frac)
            and self.interp_frac >= INTERP_FRAC_MIN
        )


@dataclass(frozen=True)
class InterpolationResult:
    by_family: dict[str, FamilyInterpolation]

    def n_interpolate(self, baseline: "InterpolationResult | None" = None,
                      margin: float = INTERP_OVER_ENERGY_MARGIN) -> int:
        """Families that interpolate, and (if a baseline is given) whose SRCC CI-lower
        beats the baseline's CI-upper by `margin` — the loudness-controlled count,
        using the same +0.05 buffer as the §8 severity gate G1b (W1)."""
        out = 0
        for fam, fi in self.by_family.items():
            if not fi.interpolates:
                continue
            if baseline is not None:
                base_hi = baseline.by_family[fam].srcc.hi
                if not (np.isfinite(base_hi) and fi.srcc.lo >= base_hi + margin):
                    continue
            out += 1
        return out


def _regressor() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _curve(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    return {
        lvl: float(y_pred[y_true == lvl].mean()) if np.any(y_true == lvl) else float("nan")
        for lvl in LADDER
    }


def _ordering_indicator(y_true: np.ndarray, y_pred: np.ndarray):
    """1.0 if mean-pred is strictly increasing across levels 2, 3, 4 on the resample;
    nan if any of those three levels is absent from it."""
    def indicator(idx: np.ndarray) -> float:
        yt, yp = y_true[idx], y_pred[idx]
        means = []
        for lvl in (NEIGHBOURS[0], HELD_OUT_SEVERITY, NEIGHBOURS[1]):
            sel = yt == lvl
            if not np.any(sel):
                return float("nan")
            means.append(float(yp[sel].mean()))
        return 1.0 if means[0] < means[1] < means[2] else 0.0

    return indicator


def evaluate_interpolation(data: ProbeData) -> InterpolationResult:
    by_family: dict[str, FamilyInterpolation] = {}
    for fam in FAMILIES:
        fam_rows = data.family == fam
        tr = fam_rows & (data.split == "train") & np.isin(data.severity, TRAIN_SEVERITIES)
        te = fam_rows & (data.split == "test") & np.isin(data.severity, LADDER)
        reg = _regressor().fit(data.X[tr], data.severity[tr])

        y_true, y_pred, groups = data.severity[te], reg.predict(data.X[te]), data.group[te]
        curve = _curve(y_true, y_pred)
        evaluable = all(np.isfinite(curve[lvl]) for lvl in (NEIGHBOURS[0], HELD_OUT_SEVERITY, NEIGHBOURS[1]))
        by_family[fam] = FamilyInterpolation(
            srcc=bootstrap_over_groups(groups, lambda i: srcc(y_true[i], y_pred[i])),
            curve=curve,
            interp_frac=bootstrap_fraction(groups, _ordering_indicator(y_true, y_pred)) if evaluable else float("nan"),
            evaluable=evaluable,
        )
    return InterpolationResult(by_family=by_family)

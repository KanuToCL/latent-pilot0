"""Probe metrics with source-level uncertainty.

macro-F1 for type, Spearman SRCC for severity, each with a 95% bootstrap CI
resampled over TEST GROUPS (speakers/tracks), not rows — §2.5. Gate 1 keys on the
CI lower bound, so a thin or lucky split cannot clear a threshold on a point
estimate alone (elder findings B2/M3/W2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score

N_BOOTSTRAP = 1000


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float  # 2.5th percentile over source-resamples
    hi: float  # 97.5th percentile

    def clears(self, threshold: float) -> bool:
        """Pass only if the CI lower bound clears — nan-safe (a degenerate metric
        has lo=nan, which never clears)."""
        return self.lo >= threshold


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> float:
    # Explicit labels so a class absent from a split can't silently drop from the mean.
    return float(f1_score(y_true, y_pred, labels=labels, average="macro"))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def srcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation. Undefined (no rank variance, e.g. a constant
    prediction or a single severity level) → nan, so a degenerate split can't
    masquerade as a passing correlation."""
    if len(y_true) < 3 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def bootstrap_over_groups(
    groups: np.ndarray, stat: Callable[[np.ndarray], float], *, n: int = N_BOOTSTRAP, seed: int = 0
) -> Estimate:
    """`stat(row_index)` computes the metric on a subset of rows. Resample the
    UNIQUE groups with replacement `n` times; each resample gathers all rows of the
    drawn groups (with multiplicity) → a source-level CI. Point estimate uses all
    rows once."""
    if len(groups) == 0:  # a cell dropped entirely by common-rows select → undefined, not a crash
        return Estimate(point=float("nan"), lo=float("nan"), hi=float("nan"))
    order = np.arange(len(groups))
    point = stat(order)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}

    vals = np.empty(n)
    for b in range(n):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_group[g] for g in drawn])
        vals[b] = stat(idx)
    finite = vals[np.isfinite(vals)]
    if finite.size < n * 0.5:  # mostly-undefined resamples → don't trust the CI
        return Estimate(point=point, lo=float("nan"), hi=float("nan"))
    return Estimate(point=point, lo=float(np.percentile(finite, 2.5)), hi=float(np.percentile(finite, 97.5)))

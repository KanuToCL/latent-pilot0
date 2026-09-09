"""Probe metrics with source-level uncertainty.

macro-F1 for type, Spearman SRCC for severity, each with a 95% bootstrap CI
resampled over TEST GROUPS (speakers/tracks), not rows — §2.5. Gate 1 keys on the
CI lower bound, so a thin or lucky split cannot clear a threshold on a point
estimate alone (elder findings B2/M3/W2).

Section map (file order):
  Estimate                    point + 95% CI; `clears()` is the CI-lower gate
  macro_f1 / confusion        type-probe scoring
  srcc / lcc                  rank and linear correlation, nan on a degenerate split
  spearman_ceiling            max SRCC attainable against a TIED severity ladder
  spearman_ceiling_balanced   the equal-count closed form — cross-check only (AM6)
  bootstrap_over_groups       percentile CI resampled over unique groups
  bootstrap_fraction          probability of an ordering over the same resamples
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

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


def lcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson linear correlation (LCC), the §2.4 secondary quality metric. nan on a
    degenerate constant so it can't masquerade as a passing correlation."""
    if len(y_true) < 3 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_ceiling(counts: Sequence[int]) -> float:
    """Largest SRCC any prediction can reach against a severity ladder with `counts[j]`
    rows at level j (levels given in ascending order).

    The ladder is massively tied — 100 rows share each severity — so even a perfect,
    perfectly-ordered prediction is capped. The optimum ranks every level-j row above
    every level-(j−1) row, which makes each level's mean predicted rank its block mean
    rank `r_j`, and Spearman collapses to (F21, general form):

        rho_max = sqrt( 12 / (N(N²−1)) · Σ_j m_j (r_j − (N+1)/2)² ),   N = Σ_j m_j

    valid for UNBALANCED counts; the equal-count closed form OVERSTATES those (+0.07 on
    [3,100,100]), so it is only ever a cross-check (`spearman_ceiling_balanced`, AM6).

    nan below two levels or three rows — the same degenerate cases `srcc` refuses, so a
    ceiling can never masquerade as a number when the correlation itself is undefined."""
    m = np.asarray(counts, dtype=float)
    if m.size and (m <= 0).any():
        raise ValueError(f"level counts must be positive, got {list(counts)}")
    n = float(m.sum())
    if m.size < 2 or n < 3:
        return float("nan")
    starts = np.concatenate([[0.0], np.cumsum(m)[:-1]])
    block_mean_rank = starts + (m + 1.0) / 2.0
    dev2 = (block_mean_rank - (n + 1.0) / 2.0) ** 2
    return float(np.sqrt(12.0 / (n * (n * n - 1.0)) * float((m * dev2).sum())))


def spearman_ceiling_balanced(k: int, m: float) -> float:
    """The equal-count closed form `sqrt(m²(k²−1) / (k²m²−1))` for `k` levels of `m`
    rows. Exact when the ladder IS balanced, and INVALID when it is not — it overstates
    the ceiling, which would gift a candidate margin. Reported as
    `rho_max_balanced_crosscheck` only (AM6). Its m→∞ limit `sqrt(1−1/k²)` is what the
    plan's first draft mistook for the ceiling itself."""
    if k < 2 or m <= 0 or k * m < 3:
        return float("nan")
    return float(np.sqrt(m * m * (k * k - 1.0) / (k * k * m * m - 1.0)))


def bootstrap_over_groups(
    groups: np.ndarray,
    stat: Callable[[np.ndarray], float],
    *,
    n: int = N_BOOTSTRAP,
    seed: int = 0,
    min_groups: int = 1,
) -> Estimate:
    """`stat(row_index)` computes the metric on a subset of rows. Resample the
    UNIQUE groups with replacement `n` times; each resample gathers all rows of the
    drawn groups (with multiplicity) → a source-level CI. Point estimate uses all
    rows once.

    `min_groups` refuses a CI (lo/hi = nan, point kept) when fewer than that many
    distinct groups are present: with one group every resample redraws it, so the
    interval collapses to zero width and would masquerade as a tight, trustworthy
    bound. Callers that need cluster-level protection (per-family G2b under partial
    coverage) pass `MIN_TEST_GROUPS`; the default 1 leaves complete-grid callers
    (Gate 1, Phase 5) untouched."""
    if len(groups) == 0:  # a cell dropped entirely by common-rows select → undefined, not a crash
        return Estimate(point=float("nan"), lo=float("nan"), hi=float("nan"))
    order = np.arange(len(groups))
    point = stat(order)
    uniq = np.unique(groups)
    if len(uniq) < min_groups:  # too few clusters → the resample CI is not trustworthy
        return Estimate(point=point, lo=float("nan"), hi=float("nan"))
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
    # Uncorrected percentile interval (not BCa) — consistent across every gate in the
    # repo; for a bounded difference-of-correlations near ±1 it can be mildly biased,
    # which the CI-lower gating absorbs conservatively.
    return Estimate(point=point, lo=float(np.percentile(finite, 2.5)), hi=float(np.percentile(finite, 97.5)))


def bootstrap_fraction(
    groups: np.ndarray, indicator: Callable[[np.ndarray], float], *, n: int = N_BOOTSTRAP,
    seed: int = 0, min_groups: int = 1,
) -> float:
    """Fraction of TEST-GROUP resamples in which `indicator(row_index)` holds (returns
    1.0 / 0.0, or nan when undefined on that resample → dropped). Returns nan if the
    indicator is undefined on more than half the resamples — the same trust floor as
    the CI. `min_groups` mirrors `bootstrap_over_groups`: fewer than that many distinct
    groups → nan, so a single-source ordering can't masquerade as a reliable fraction.
    Used where the summary wanted is a probability of an ordering, for which a percentile
    CI is the wrong shape."""
    if len(groups) == 0:
        return float("nan")
    uniq = np.unique(groups)
    if len(uniq) < min_groups:  # too few clusters → the fraction is not trustworthy
        return float("nan")
    rng = np.random.default_rng(seed)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = np.empty(n)
    for b in range(n):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        vals[b] = indicator(np.concatenate([rows_by_group[g] for g in drawn]))
    finite = vals[np.isfinite(vals)]
    return float(finite.mean()) if finite.size >= n * 0.5 else float("nan")

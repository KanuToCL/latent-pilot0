"""Degradation-type probe: a linear (multinomial logistic) classifier over the 7
families, trained on degraded rows only (clean has no type). Standardise → fit on
train → macro-F1 (with a source-level bootstrap CI) on the source-disjoint test
split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import CLEAN, ProbeData
from .metrics import Estimate, bootstrap_over_groups, confusion, macro_f1


@dataclass(frozen=True)
class TypeResult:
    macro_f1: Estimate
    labels: list[str]
    confusion: np.ndarray
    n_train: int
    n_test: int


def linear_classifier() -> object:
    """The type probe's standardise→multinomial-logistic pipeline. Public: the MLP
    probe reuses it as the linear baseline so the nonlinearity gap has one source."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=0))


def evaluate_type_probe(data: ProbeData) -> TypeResult:
    degraded = data.family != CLEAN
    tr = degraded & (data.split == "train")
    te = degraded & (data.split == "test")
    labels = sorted(set(data.family[degraded].tolist()))

    clf = linear_classifier().fit(data.X[tr], data.family[tr])
    y_true, y_pred, groups = data.family[te], clf.predict(data.X[te]), data.group[te]

    est = bootstrap_over_groups(groups, lambda idx: macro_f1(y_true[idx], y_pred[idx], labels))
    return TypeResult(
        macro_f1=est,
        labels=labels,
        confusion=confusion(y_true, y_pred, labels),
        n_train=int(tr.sum()),
        n_test=int(te.sum()),
    )

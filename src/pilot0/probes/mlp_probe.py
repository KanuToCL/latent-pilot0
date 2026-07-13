"""Nonlinear (2-layer MLP-256) type probe — the §2.4 secondary. Reported next to
the linear probe so the MLP−linear macro-F1 gap measures how much degradation-type
information is present but NOT linearly decodable (RQ1). The gap is estimated
*paired*: every bootstrap resample scores both probes on the same drawn test
groups and takes the difference, so its CI lower bound tells us whether the
nonlinear lift is real rather than a lucky split.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import CLEAN, ProbeData
from .metrics import Estimate, bootstrap_over_groups, macro_f1
from .type_probe import _linear_classifier


@dataclass(frozen=True)
class NonlinearityResult:
    mlp_macro_f1: Estimate
    linear_macro_f1: Estimate
    gap: Estimate  # mlp − linear, paired over the same resamples


def _mlp_classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256,), max_iter=1000, random_state=0),
    )


def evaluate_nonlinearity(data: ProbeData) -> NonlinearityResult:
    degraded = data.family != CLEAN
    tr = degraded & (data.split == "train")
    te = degraded & (data.split == "test")
    labels = sorted(set(data.family[degraded].tolist()))

    lin = _linear_classifier().fit(data.X[tr], data.family[tr])
    mlp = _mlp_classifier().fit(data.X[tr], data.family[tr])
    y_true, groups = data.family[te], data.group[te]
    y_lin, y_mlp = lin.predict(data.X[te]), mlp.predict(data.X[te])

    return NonlinearityResult(
        mlp_macro_f1=bootstrap_over_groups(groups, lambda i: macro_f1(y_true[i], y_mlp[i], labels)),
        linear_macro_f1=bootstrap_over_groups(groups, lambda i: macro_f1(y_true[i], y_lin[i], labels)),
        gap=bootstrap_over_groups(
            groups, lambda i: macro_f1(y_true[i], y_mlp[i], labels) - macro_f1(y_true[i], y_lin[i], labels)
        ),
    )

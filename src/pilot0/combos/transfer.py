"""Zero-shot probe transfer (§2.4): train the linear type probe on SINGLE degradations
only, then read which single-family label a never-seen combination attracts. For each
pair (a, b) we report the distribution of predicted single labels over the combo clips
— does noise×clip read as noise, as clip, or as something else? — plus the fraction
landing on either constituent leg. Descriptive, not gated; combos from the held-out
TEST sources so the classifier saw neither the combo condition nor the source.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..probes.dataset import CLEAN, ProbeData
from .dataset import ComboData
from .grid import combo_label


@dataclass(frozen=True)
class PairTransfer:
    pair: str
    leg_a: str
    leg_b: str
    n: int  # combo clips scored
    predicted_fraction: dict[str, float]  # single family -> fraction of combo clips predicted as it
    frac_either_leg: float  # fraction predicted as leg a OR leg b


@dataclass(frozen=True)
class TransferResult:
    by_pair: dict[str, PairTransfer]


def transfer_to_combos(probe_data: ProbeData, combo_data: ComboData) -> TransferResult:
    degraded = probe_data.family != CLEAN
    tr = degraded & (probe_data.split == "train")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=0))
    clf.fit(probe_data.X[tr], probe_data.family[tr])

    te = combo_data.split == "test"
    by_pair: dict[str, PairTransfer] = {}
    for a, b in sorted({(x, y) for x, y in zip(combo_data.leg_a[te], combo_data.leg_b[te])}):
        label = combo_label(a, b)
        m = te & (combo_data.pair == label)
        preds = clf.predict(combo_data.X[m])
        n = int(m.sum())
        counts = Counter(preds.tolist())
        fraction = {fam: c / n for fam, c in counts.items()}
        by_pair[label] = PairTransfer(
            pair=label, leg_a=a, leg_b=b, n=n,
            predicted_fraction=fraction,
            frac_either_leg=(counts.get(a, 0) + counts.get(b, 0)) / n,
        )
    return TransferResult(by_pair=by_pair)

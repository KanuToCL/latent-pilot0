"""Run the Gate-2 evaluation: for each candidate build the quality design from the
cache + external scores, fit the head ONCE, and score both clauses off the shared
predictions. Pure orchestration — thresholds live in gate2.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..encode.headroom import CEILING_DBFS
from .dataset import build_quality_data
from .g2b import evaluate_g2b
from .gate2 import Gate2Decision
from .head import evaluate_g2a, fit_head
from .scores import MOS_METRIC, Scores


@dataclass(frozen=True)
class Gate2Report:
    decisions: tuple[Gate2Decision, ...]
    mos_available: bool
    scatter: dict[str, list]  # "name/variant" -> degraded-test (ViSQOL, head-pred) pairs, the exact points G2a scores (for F4)


def run_gate2(
    manifest, candidates, cache_dir, scores: Scores, *, ceiling_dbfs: float = CEILING_DBFS
) -> Gate2Report:
    decisions = []
    scatter: dict[str, list] = {}
    for name, variant in candidates:
        data = build_quality_data(manifest, name, variant, cache_dir, scores, ceiling_dbfs=ceiling_dbfs)
        pred = fit_head(data)  # one fit shared by G2a, G2b, and the F4 scatter below
        te = (data.split == "test") & (data.severity > 0)  # the same mask evaluate_g2a scores on
        scatter[f"{name}/{variant}"] = [[float(r), float(p)] for r, p in zip(data.ref[te], pred[te])]
        decisions.append(
            Gate2Decision(name=name, variant=variant, g2a=evaluate_g2a(data, pred), g2b=evaluate_g2b(data, pred))
        )
    return Gate2Report(decisions=tuple(decisions), mos_available=scores.has(MOS_METRIC), scatter=scatter)

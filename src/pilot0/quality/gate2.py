"""Gate 2 — pre-registered pass/fail (§8). Frozen thresholds; do not tune to data.

  G2a  reference-free quality — head SRCC vs ViSQOL, pooled over DEGRADED cells,
                                CI-lower ≥ 0.85.
  G2b  beats the incumbents  — on ≥ 5/7 families the head PAIRED-beats every NR
                                baseline against HUMAN MOS (per-family paired-diff
                                CI-lower > 0).

The two clauses use different ground truths ON PURPOSE (the proposal's own elder
warning): G2a rewards recovering the reference metric the head was trained on, on
held-out sources; G2b asks whether that reference-free head tracks human judgement
as well as dedicated MOS predictors — so it is scored on MOS, which the head never
saw, not on ViSQOL. No MOS (or no baselines) ⇒ G2b NOT EVALUABLE, never a silent
ViSQOL comparison. CI-lower gating and the MIN_TEST_GROUPS underpowered guard match
Gate 1.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..probes.gate1 import MIN_TEST_GROUPS
from .g2b import G2bResult
from .head import QualityG2a

G2A_REF_SRCC_MIN = 0.85  # pooled head-vs-ViSQOL CI-lower on degraded cells
G2B_MIN_FAMILIES = 5  # of 7


@dataclass(frozen=True)
class Gate2Decision:
    name: str
    variant: str
    g2a: QualityG2a
    g2b: G2bResult | None  # None ⇒ G2b not evaluable

    @property
    def underpowered(self) -> bool:
        return self.g2a.n_test_groups < MIN_TEST_GROUPS

    @property
    def pass_g2a(self) -> bool:
        return self.g2a.ref_srcc.clears(G2A_REF_SRCC_MIN)

    @property
    def g2b_evaluable(self) -> bool:
        return self.g2b is not None

    @property
    def n_beats_baseline(self) -> int:
        return self.g2b.n_beats() if self.g2b is not None else 0

    @property
    def pass_g2b(self) -> bool:
        return self.g2b_evaluable and self.n_beats_baseline >= G2B_MIN_FAMILIES

    @property
    def passed(self) -> bool:
        return not self.underpowered and self.pass_g2a and self.pass_g2b

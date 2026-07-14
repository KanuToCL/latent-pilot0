"""Gate 1 — pre-registered pass/fail (§8). Frozen thresholds; do not tune to the
data. A representation clears Gate 1 only if, on a well-powered source-disjoint
test split, it:

  G1a  reads type          — macro-F1 CI-lower ≥ 0.85
  G1b  reads severity      — without-clean SRCC CI-lower ≥ 0.80 on ≥ 4/7 families
                             AND beats the energy control by ≥ 0.05 on each (so the
                             ordinal signal is not just loudness — finding B1)
  G1c  beats the mel floor — type macro-F1 CI-lower exceeds the floor CI-UPPER by ≥ 0.05
                             (conservative vs the §8 "point" wording, so a lucky-low
                             floor point can't gift the margin — finding N2)

All thresholds use the CI LOWER bound, so a lucky point estimate cannot pass
(B2/M3). A split with fewer than MIN_TEST_GROUPS test sources is UNDERPOWERED and
cannot pass regardless of the numbers (W2).
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import Estimate
from .severity import SeverityResult
from .type_probe import TypeResult

TYPE_MACRO_F1_MIN = 0.85
SEVERITY_SRCC_MIN = 0.80
SEVERITY_MIN_FAMILIES = 4  # of 7
FLOOR_F1_MARGIN = 0.05  # codec must beat the mel floor by ≥ 5 F1 points (0–1 scale)
SEVERITY_OVER_ENERGY_MARGIN = 0.05  # codec severity must beat the level-only control
MIN_TEST_GROUPS = 3  # hard floor; the percentile cluster bootstrap is only truly
#                      trustworthy at ≫3 test groups (real multi-speaker corpus)


@dataclass(frozen=True)
class Gate1Decision:
    name: str
    variant: str
    type: TypeResult
    severity: SeverityResult
    energy_severity: SeverityResult  # level-only control, scored on the same rows
    floor_macro_f1: Estimate
    n_test_groups: int

    @property
    def underpowered(self) -> bool:
        return self.n_test_groups < MIN_TEST_GROUPS

    @property
    def n_severity_pass(self) -> int:
        return self.severity.n_pass(SEVERITY_SRCC_MIN, self.energy_severity, SEVERITY_OVER_ENERGY_MARGIN)

    @property
    def margin_over_floor(self) -> float:
        # Conservative: codec CI-lower vs floor CI-upper, so a lucky-low floor point
        # can't gift the margin (finding N2).
        return self.type.macro_f1.lo - self.floor_macro_f1.hi

    @property
    def pass_type(self) -> bool:
        return self.type.macro_f1.clears(TYPE_MACRO_F1_MIN)

    @property
    def pass_severity(self) -> bool:
        return self.n_severity_pass >= SEVERITY_MIN_FAMILIES

    @property
    def pass_floor(self) -> bool:
        return self.margin_over_floor >= FLOOR_F1_MARGIN

    @property
    def passed(self) -> bool:
        return (
            not self.underpowered
            and self.pass_type
            and self.pass_severity
            and self.pass_floor
        )

"""Ceiling-aware reanalysis of the Gate-1 severity criterion (S1).

Gate 1's G1b asks a representation to reach a without-clean severity SRCC whose CI
LOWER bound clears 0.80 *and* beats the energy control's CI UPPER by 0.05, on at
least 4 of 7 families. Both numbers are compared against a hard cap nobody
computed: with 100 test sources per severity the ladder is 500 tied rows, and the
best any prediction can do is `spearman_ceiling` — 0.9798 at K=5, 0.9428 at K=3.

When the energy control itself sits at that cap, `energy_hi + 0.05` lands ABOVE
it, and the family is unpassable by any representation, perfect ones included.
This module makes that arithmetic explicit per family, and asks the only question
that matters afterwards: how many families could an ORACLE probe — one pinned
exactly at the ceiling, zero-width CI — pass?

Gate 1 is NOT redefined here. Nothing in this module changes a threshold; it
reports what those frozen thresholds imply.

Section map (file order):
  FamilyCeiling / CandidateCeiling   the two result records
  level_counts        test-split rows per severity, on the common grid
  family_ceilings     per family: K, counts, rho_max, energy_hi, required_lo, feasible
  severity_from_json  rebuild a SeverityResult from a serialized gate1 report
  oracle_severity     a probe pinned AT the ceiling, zero-width CI
  oracle_n_pass       that probe's G1b family count, through the frozen n_pass()
  candidate_summary   per candidate: absolute_clears vs margin_passes vs feasible
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..degrade.grid import FAMILIES
from .gate1 import SEVERITY_OVER_ENERGY_MARGIN, SEVERITY_SRCC_MIN
from .metrics import Estimate, spearman_ceiling, spearman_ceiling_balanced
from .severity import FamilySeverity, SeverityResult

CLEAN = "clean"


@dataclass(frozen=True)
class FamilyCeiling:
    family: str
    K: int  # severity levels in the WITHOUT-CLEAN ladder on the common grid
    counts_by_level: dict[int, int]  # severity -> test rows
    N: int
    rho_max: float  # general form (F21) — the number that binds
    rho_max_balanced_crosscheck: float  # cross-check only; invalid when unbalanced (AM6)
    energy_hi: float  # energy control's without-clean CI-upper on this family
    required_lo: float  # CI-lower a candidate must reach to count as a G1b pass
    feasible: bool  # ... and whether the ceiling permits it at all


@dataclass(frozen=True)
class CandidateCeiling:
    key: str
    name: str
    variant: str
    absolute_clears: int  # families clearing the 0.80 absolute bar alone
    margin_passes: int  # families also beating energy_hi + 0.05 (the real G1b count)
    margin_passes_reported: int  # the same number as serialized by the gate run
    feasible_families: int  # families where the criterion is reachable at all


def level_counts(manifest: dict, family: str, grid, *, split: str = "test") -> dict[int, int]:
    """Rows of `split` per severity for `family`, restricted to the cells the gate
    actually scored (`grid` = the common conditions). Clean is excluded: G1b keys on
    the WITHOUT-CLEAN ladder (severity.py, `te_deg`)."""
    allowed = {s for f, s in grid if f == family and f != CLEAN}
    c = Counter(
        int(r["severity"]) for r in manifest["rows"]
        if r["family"] == family and r["split"] == split and int(r["severity"]) in allowed
    )
    return {s: c[s] for s in sorted(c)}


def family_ceilings(report: dict, manifest: dict, grid) -> dict[str, FamilyCeiling]:
    """Per family: the attainable SRCC cap, and the bar G1b sets against it.

    `report` is the inner `report` block of a gate-1 run (floor / energy / decisions).
    `required_lo` is the frozen criterion read forward, never re-tuned: the absolute
    0.80 OR the energy control's CI-upper plus the 0.05 margin, whichever binds."""
    energy = report["energy"]["severity"]["by_family"]
    out: dict[str, FamilyCeiling] = {}
    for fam in FAMILIES:
        counts = level_counts(manifest, fam, grid)
        levels = list(counts.values())
        n = int(sum(levels))
        rho_max = spearman_ceiling(levels)
        hi = _f(energy[fam]["without_clean"]["hi"])
        required = max(SEVERITY_SRCC_MIN, hi + SEVERITY_OVER_ENERGY_MARGIN)
        out[fam] = FamilyCeiling(
            family=fam,
            K=len(levels),
            counts_by_level=counts,
            N=n,
            rho_max=rho_max,
            rho_max_balanced_crosscheck=spearman_ceiling_balanced(len(levels), n / len(levels))
            if levels else float("nan"),
            energy_hi=hi,
            required_lo=required,
            # A non-finite energy bound cannot be beaten conservatively, so the family
            # is not feasible either — the same way n_pass() refuses it.
            feasible=bool(np.isfinite(rho_max) and np.isfinite(required) and required <= rho_max),
        )
    return out


def _f(x) -> float:
    """`to_jsonable` writes nan as JSON null; read it back as nan, not None."""
    return float("nan") if x is None else float(x)


def _est(d: dict) -> Estimate:
    return Estimate(point=_f(d["point"]), lo=_f(d["lo"]), hi=_f(d["hi"]))


def severity_from_json(block: dict) -> SeverityResult:
    """Rebuild a `SeverityResult` from a serialized gate-1 severity block, so the
    frozen `n_pass()` — not a reimplementation of it — does the counting."""
    return SeverityResult(by_family={
        fam: FamilySeverity(with_clean=_est(v["with_clean"]), without_clean=_est(v["without_clean"]))
        for fam, v in block["by_family"].items()
    })


def oracle_severity(ceilings: dict[str, FamilyCeiling]) -> SeverityResult:
    """The best probe physically permitted by the ladder: SRCC exactly at each
    family's ceiling, with a zero-width CI so the CI-lower gate costs it nothing.
    Any real probe's CI-lower is <= this."""
    return SeverityResult(by_family={
        fam: FamilySeverity(with_clean=Estimate(c.rho_max, c.rho_max, c.rho_max),
                            without_clean=Estimate(c.rho_max, c.rho_max, c.rho_max))
        for fam, c in ceilings.items()
    })


def oracle_n_pass(ceilings: dict[str, FamilyCeiling], energy: SeverityResult) -> int:
    """How many families the oracle probe passes under the frozen G1b rule. Below
    `SEVERITY_MIN_FAMILIES` means no representation can pass Gate 1's severity leg on
    this grid — a property of the criterion, not of any codec."""
    return oracle_severity(ceilings).n_pass(SEVERITY_SRCC_MIN, energy, SEVERITY_OVER_ENERGY_MARGIN)


def candidate_summary(report: dict, ceilings: dict[str, FamilyCeiling]) -> list[CandidateCeiling]:
    """Split each candidate's severity verdict into the part the absolute bar decides
    and the part the energy margin decides, next to how many families were reachable."""
    feasible = sum(c.feasible for c in ceilings.values())
    out = []
    for dec in report["decisions"]:
        sev = severity_from_json(dec["severity"])
        energy = severity_from_json(dec["energy_severity"])
        out.append(CandidateCeiling(
            key=f"{dec['name']}:{dec['variant']}",
            name=dec["name"],
            variant=dec["variant"],
            absolute_clears=sev.n_pass(SEVERITY_SRCC_MIN),
            margin_passes=sev.n_pass(SEVERITY_SRCC_MIN, energy, SEVERITY_OVER_ENERGY_MARGIN),
            margin_passes_reported=int(dec["n_severity_pass"]),
            feasible_families=feasible,
        ))
    return out

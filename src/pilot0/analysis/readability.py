"""Readability heatmap data (F1 in the paper): one row per (encoder, variant),
columns = type macro-F1 and per-family severity SRCC. This is the full-matrix
generalisation of the Gate-1 table — floor, energy control, and every codec
variant scored on the SAME common cells so the grid is apples-to-apples (M1).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..degrade.grid import FAMILIES
from ..probes.run import EncoderProbeResult


@dataclass(frozen=True)
class ReadabilityRow:
    name: str
    variant: str
    type_macro_f1: tuple[float, float, float]  # point, lo, hi
    severity_srcc: dict[str, tuple[float, float, float]]  # family -> without-clean point, lo, hi
    n_test_groups: int


def readability_row(row: EncoderProbeResult) -> ReadabilityRow:
    mf = row.type.macro_f1
    sev = {
        fam: (fs.without_clean.point, fs.without_clean.lo, fs.without_clean.hi)
        for fam, fs in row.severity.by_family.items()
    }
    return ReadabilityRow(
        name=row.name,
        variant=row.variant,
        type_macro_f1=(mf.point, mf.lo, mf.hi),
        severity_srcc={fam: sev[fam] for fam in FAMILIES},
        n_test_groups=row.n_test_groups,
    )

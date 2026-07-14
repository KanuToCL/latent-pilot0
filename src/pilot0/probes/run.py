"""Run the Gate-1 evaluation: build the probe design from the cache for the
log-mel floor, the energy control, and each candidate (encoder, variant); score
them all on the SAME renderable cells (the intersection across native rates, so
the floor comparison is apples-to-apples — finding M1); assemble the Gate-1
decisions. Pure orchestration — thresholds live in gate1.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..corpus.manifest import renderable_conditions
from ..encode.headroom import CEILING_DBFS
from ..seam.registry import make_encoder
from .dataset import build_probe_data
from .gate1 import Gate1Decision
from .severity import SeverityResult, evaluate_severity_probes
from .type_probe import TypeResult, evaluate_type_probe

FLOOR = ("logmel", "mel")
ENERGY = ("energy", "energy")


@dataclass(frozen=True)
class EncoderProbeResult:
    name: str
    variant: str
    type: TypeResult
    severity: SeverityResult
    n_test_groups: int


@dataclass(frozen=True)
class Gate1Report:
    floor: EncoderProbeResult
    energy: EncoderProbeResult
    decisions: tuple[Gate1Decision, ...]
    n_common_conditions: int


def common_conditions(names) -> set[tuple[str, int]]:
    """Cells renderable at EVERY candidate's native rate (the apples-to-apples
    intersection, M1). Public: Phase 5 reuses it to score the full matrix on one grid."""
    rates = {make_encoder(n).native_sr for n in names}
    return set.intersection(*(set(renderable_conditions(sr)) for sr in rates))


def evaluate_encoder(
    manifest, name, variant, cache_dir, conditions, *, ceiling_dbfs=CEILING_DBFS
) -> EncoderProbeResult:
    data = build_probe_data(manifest, name, variant, cache_dir, ceiling_dbfs=ceiling_dbfs).select(conditions)
    return EncoderProbeResult(
        name=name,
        variant=variant,
        type=evaluate_type_probe(data),
        severity=evaluate_severity_probes(data),
        n_test_groups=data.n_test_groups(),
    )


def run_gate1(
    manifest, candidates, cache_dir, *, floor=FLOOR, energy=ENERGY, ceiling_dbfs=CEILING_DBFS
) -> Gate1Report:
    names = [floor[0], energy[0], *(n for n, _ in candidates)]
    common = common_conditions(names)

    def ev(name, variant):
        return evaluate_encoder(manifest, name, variant, cache_dir, common, ceiling_dbfs=ceiling_dbfs)

    floor_res, energy_res = ev(*floor), ev(*energy)
    decisions = []
    for name, variant in candidates:
        res = ev(name, variant)
        decisions.append(
            Gate1Decision(
                name=name,
                variant=variant,
                type=res.type,
                severity=res.severity,
                energy_severity=energy_res.severity,
                floor_macro_f1=floor_res.type.macro_f1,
                n_test_groups=res.n_test_groups,
            )
        )
    return Gate1Report(
        floor=floor_res, energy=energy_res, decisions=tuple(decisions), n_common_conditions=len(common)
    )

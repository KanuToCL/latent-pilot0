"""Probes & Gate 1 (Phase 4): linear type/severity probes on the cached latents,
scored (with source-level bootstrap CIs) against the log-mel floor and an energy
control, under the pre-registered Gate-1 thresholds."""

from .dataset import ProbeData, build_probe_data
from .gate1 import (
    FLOOR_F1_MARGIN,
    MIN_TEST_GROUPS,
    SEVERITY_MIN_FAMILIES,
    SEVERITY_OVER_ENERGY_MARGIN,
    SEVERITY_SRCC_MIN,
    TYPE_MACRO_F1_MIN,
    Gate1Decision,
)
from .metrics import Estimate, bootstrap_over_groups, macro_f1, srcc
from .run import EncoderProbeResult, Gate1Report, evaluate_encoder, run_gate1
from .severity import FamilySeverity, SeverityResult, evaluate_severity_probes
from .type_probe import TypeResult, evaluate_type_probe

__all__ = [
    "ProbeData",
    "build_probe_data",
    "Gate1Decision",
    "TYPE_MACRO_F1_MIN",
    "SEVERITY_SRCC_MIN",
    "SEVERITY_MIN_FAMILIES",
    "SEVERITY_OVER_ENERGY_MARGIN",
    "FLOOR_F1_MARGIN",
    "MIN_TEST_GROUPS",
    "Estimate",
    "bootstrap_over_groups",
    "macro_f1",
    "srcc",
    "EncoderProbeResult",
    "Gate1Report",
    "evaluate_encoder",
    "run_gate1",
    "FamilySeverity",
    "SeverityResult",
    "evaluate_severity_probes",
    "TypeResult",
    "evaluate_type_probe",
]

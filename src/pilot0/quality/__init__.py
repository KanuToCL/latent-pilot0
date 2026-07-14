"""No-reference quality head + pre-registered Gate 2 (Phase 6): a ridge on frozen
latents predicting ViSQOL without a reference, scored on ViSQOL (G2a) and — held apart
from the training target — on human MOS against the NR incumbents (G2b)."""

from .dataset import QualityData, build_quality_data
from .g2b import FamilyG2b, G2bResult, evaluate_g2b
from .gate2 import G2A_REF_SRCC_MIN, G2B_MIN_FAMILIES, Gate2Decision
from .head import QualityG2a, evaluate_g2a, fit_head, fit_head_model
from .run import Gate2Report, run_gate2
from .scores import MOS_METRIC, NR_BASELINES, REF_METRIC, FakeScores, Scores, TableScores

__all__ = [
    "build_quality_data",
    "QualityData",
    "fit_head",
    "fit_head_model",
    "evaluate_g2a",
    "QualityG2a",
    "evaluate_g2b",
    "G2bResult",
    "FamilyG2b",
    "Gate2Decision",
    "G2A_REF_SRCC_MIN",
    "G2B_MIN_FAMILIES",
    "run_gate2",
    "Gate2Report",
    "FakeScores",
    "TableScores",
    "Scores",
    "REF_METRIC",
    "MOS_METRIC",
    "NR_BASELINES",
]

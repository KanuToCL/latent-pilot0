"""Full representation-matrix analysis (Phase 5): the readability heatmap, attribute
geometry (cosine matrices), and severity-interpolation curves over the log-mel floor,
the energy control, and every codec variant — all on one common cell grid (M1)."""

from .geometry import GeometryResult, geometry
from .readability import ReadabilityRow, readability_row
from .run import AnalysisReport, CandidateAnalysis, analyze
from .serialize import cosine_matrices, heatmap_rows, monotonicity_curves

__all__ = [
    "analyze",
    "AnalysisReport",
    "CandidateAnalysis",
    "geometry",
    "GeometryResult",
    "readability_row",
    "ReadabilityRow",
    "heatmap_rows",
    "cosine_matrices",
    "monotonicity_curves",
]

"""Phase-5 analysis sweep: over the full representation matrix, assemble the three
acceptance artifacts — the readability heatmap, the attribute cosine matrix, and
the severity-interpolation (monotonicity) curves — plus the linear-vs-MLP
nonlinearity gap and the frame-level dropout probe. Pure orchestration; every
statistic and threshold lives in its own module.

Each candidate's pooled design matrix is built ONCE (one cache walk) and shared
across type/severity/geometry/interpolation/nonlinearity; the frame-level dropout
probe re-walks because it needs per-frame latents, not pooled vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..encode.headroom import CEILING_DBFS
from ..probes.dataset import build_probe_data
from ..probes.frame_dropout import FrameDropoutResult, evaluate_frame_dropout
from ..probes.interpolation import InterpolationResult, evaluate_interpolation
from ..probes.mlp_probe import NonlinearityResult, evaluate_nonlinearity
from ..probes.run import EncoderProbeResult, common_conditions
from ..probes.severity import evaluate_severity_probes
from ..probes.type_probe import evaluate_type_probe
from .geometry import GeometryResult, geometry
from .readability import ReadabilityRow, readability_row

FLOOR = ("logmel", "mel")
ENERGY = ("energy", "energy")


@dataclass(frozen=True)
class CandidateAnalysis:
    readability: ReadabilityRow
    geometry: GeometryResult
    interpolation: InterpolationResult
    nonlinearity: NonlinearityResult
    frame_dropout: FrameDropoutResult


@dataclass(frozen=True)
class AnalysisReport:
    floor: ReadabilityRow
    energy: ReadabilityRow
    energy_interpolation: InterpolationResult  # loudness control for RQ5 monotonicity (W1)
    candidates: dict[str, CandidateAnalysis]  # "name/variant" -> analysis
    n_common_conditions: int


def _selected_data(manifest, name, variant, cache_dir, common, ceiling_dbfs):
    return build_probe_data(manifest, name, variant, cache_dir, ceiling_dbfs=ceiling_dbfs).select(common)


def _probe_result(data, name, variant) -> EncoderProbeResult:
    return EncoderProbeResult(
        name=name, variant=variant, type=evaluate_type_probe(data),
        severity=evaluate_severity_probes(data), n_test_groups=data.n_test_groups(),
    )


def _readability_only(manifest, name, variant, cache_dir, common, ceiling_dbfs) -> ReadabilityRow:
    data = _selected_data(manifest, name, variant, cache_dir, common, ceiling_dbfs)
    return readability_row(_probe_result(data, name, variant))


def _analyze_candidate(manifest, name, variant, cache_dir, common, ceiling_dbfs) -> CandidateAnalysis:
    data = _selected_data(manifest, name, variant, cache_dir, common, ceiling_dbfs)
    return CandidateAnalysis(
        readability=readability_row(_probe_result(data, name, variant)),
        geometry=geometry(data.X, data.family, data.severity),
        interpolation=evaluate_interpolation(data),
        nonlinearity=evaluate_nonlinearity(data),
        frame_dropout=evaluate_frame_dropout(manifest, name, variant, cache_dir, ceiling_dbfs=ceiling_dbfs),
    )


def analyze(
    manifest, candidates, cache_dir, *, floor=FLOOR, energy=ENERGY, ceiling_dbfs=CEILING_DBFS
) -> AnalysisReport:
    names = [floor[0], energy[0], *(n for n, _ in candidates)]
    common = common_conditions(names)
    energy_data = _selected_data(manifest, *energy, cache_dir, common, ceiling_dbfs)
    return AnalysisReport(
        floor=_readability_only(manifest, *floor, cache_dir, common, ceiling_dbfs),
        energy=_readability_only(manifest, *energy, cache_dir, common, ceiling_dbfs),
        energy_interpolation=evaluate_interpolation(energy_data),
        candidates={
            f"{name}/{variant}": _analyze_candidate(manifest, name, variant, cache_dir, common, ceiling_dbfs)
            for name, variant in candidates
        },
        n_common_conditions=len(common),
    )

"""Corpus preparation (Phase 2): preflight (validate + loudness-normalise +
token-rename) and the experiment manifest with speaker/track-disjoint splits."""

from .manifest import (
    assign_splits,
    build_manifest,
    conditions,
    grid_signature,
    renderable_conditions,
    renderable_rows,
    write_manifest,
)
from .preflight import PreflightConfig, PreflightResult, SourceRecord, preflight

__all__ = [
    "PreflightConfig",
    "PreflightResult",
    "SourceRecord",
    "preflight",
    "assign_splits",
    "build_manifest",
    "conditions",
    "grid_signature",
    "renderable_conditions",
    "renderable_rows",
    "write_manifest",
]

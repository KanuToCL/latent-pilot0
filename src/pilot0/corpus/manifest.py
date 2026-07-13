"""Experiment manifest: one row per (source, condition), with source-disjoint
train/test splits. Conditions = clean + the 7×5 degradation grid (§2.2).

Splits are assigned per GROUP (speaker/track), per ARM, by a deterministic
hash-quantile so a group never crosses train/test (§2.5) and the test fraction is
exact and non-empty (not a random hash-bucket that can come out all-train). Rows
are model-agnostic; the Phase-3 renderer must filter each model rate through
`renderable_rows` (a band-limit cutoff ≥ Nyquist is not renderable there)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ..degrade.grid import FAMILIES, SEVERITIES, is_applicable
from .preflight import PreflightResult, SourceRecord
from .provenance import provenance

CLEAN = ("clean", 0)
SPLIT_ALGO = "per-arm-group-quantile-blake2b-v1"


def conditions() -> list[tuple[str, int]]:
    grid = [(family, severity) for family in FAMILIES for severity in SEVERITIES]
    return [CLEAN, *grid]


def renderable_conditions(sr: int) -> list[tuple[str, int]]:
    return [(f, s) for (f, s) in conditions() if f == "clean" or is_applicable(f, s, sr)]


def renderable_rows(manifest: dict, sr: int) -> list[dict]:
    """Manifest rows that are actually renderable at model rate `sr` — the
    Phase-3 render loop's entry point (drops band-limit cells ≥ Nyquist)."""
    allowed = set(renderable_conditions(sr))
    return [r for r in manifest["rows"] if (r["family"], r["severity"]) in allowed]


def grid_signature() -> str:
    """Stable hash of the grid definition, so a reordered family or changed level
    ladder is detectable in the manifest provenance."""
    data = [[f, FAMILIES[f].param_name, FAMILIES[f].unit, list(FAMILIES[f].levels)] for f in FAMILIES]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def _group_key(arm: str, group: str) -> bytes:
    return hashlib.blake2b(f"{arm}/{group}".encode(), digest_size=8).digest()


def assign_splits(records: tuple[SourceRecord, ...], test_frac: float = 0.2) -> dict[str, str]:
    """Map each group_id → 'train'/'test'. Within each arm, groups are ordered by
    hash and the lowest ⌊test_frac·N⌋ (≥1, ≤N−1) go to test — deterministic,
    exact, and non-empty for N ≥ 2."""
    groups_by_arm: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        groups_by_arm[rec.arm].add(rec.group_id)

    split: dict[str, str] = {}
    for arm, groups in groups_by_arm.items():
        ordered = sorted(groups, key=lambda g: _group_key(arm, g))
        n = len(ordered)
        n_test = min(max(1, round(test_frac * n)), n - 1) if n >= 2 else 0
        test_groups = set(ordered[:n_test])
        for g in ordered:
            split[g] = "test" if g in test_groups else "train"
    return split


def build_manifest(pf: PreflightResult, test_frac: float = 0.2, *, assert_grouped: bool = False) -> dict:
    conds = conditions()
    split_map = assign_splits(pf.accepted, test_frac)
    n_groups, n_sources = len(split_map), len(pf.accepted)
    # Fail closed when a multi-clip corpus (VCTK/LibriSpeech) was preflighted without a
    # speaker/track group_fn: every clip its own group → the split leaks speaker identity
    # into both train and test (§2.5, elder finding W1). The operator sets this for such arms.
    if assert_grouped and n_groups == n_sources and n_sources > 1:
        raise ValueError(
            f"assert_grouped: every source is its own group (n_groups == n_sources == {n_sources}). "
            "Pass a speaker/track group_fn to preflight() for a multi-clip corpus."
        )
    if len(set(split_map.values())) < 2 and n_groups > 1:
        raise ValueError(f"splits collapsed to one class {set(split_map.values())} — check test_frac")
    rows = []
    for rec in pf.accepted:
        split = split_map[rec.group_id]
        for family, severity in conds:
            param = None if family == "clean" else FAMILIES[family].levels[severity - 1]
            rows.append(
                {
                    "source": rec.token,
                    "group": rec.group_id,
                    "arm": rec.arm,
                    "split": split,
                    "family": family,
                    "severity": severity,
                    "param": float(param) if param is not None else None,
                    "content_sha": rec.content_sha,
                }
            )
    return {
        "n_sources": len(pf.accepted),
        "n_groups": len(split_map),
        "n_conditions": len(conds),
        "n_rows": len(rows),
        "test_frac": test_frac,
        "split_algo": SPLIT_ALGO,
        "grid_signature": grid_signature(),
        "target_lufs": pf.config.target_lufs,
        "conditions": [{"family": f, "severity": s} for f, s in conds],
        "rows": rows,
        "provenance": provenance(),
    }


def write_manifest(manifest: dict, path) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2))

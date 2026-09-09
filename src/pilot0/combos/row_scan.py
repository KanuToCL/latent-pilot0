"""Design-integrity scan: can `additivity()`'s per-source row-count guard ever fire?

The guard aligns the four roles by sorting each on its source token, which is valid at
exactly one row per source per role. It refuses the cell otherwise. This module answers
whether any cell in the real bank is in that state, WITHOUT loading a single latent.

Why a listing suffices (the scan's validity condition):
  - `probes.dataset.build_probe_data` appends exactly one row per
    `corpus.manifest.renderable_rows(manifest, native_sr)` entry - a pure filter over
    `manifest["rows"]`, no fan-out, no row-level drop. So ProbeData multiplicity per
    (family, severity, source) IS the manifest's.
  - `combos.dataset.build_combo_data` appends at most one row per
    (source, pair, severity): its loop is over a dict keyed by source, and the cache
    path is a deterministic function of that triple. Multiplicity there is <= 1 by
    construction; the listing is scanned anyway rather than argued away.
  - `combos.run.analyze_additivity` passes both straight into `additivity()`; no
    `.select()` sits between, and `select` could only drop rows in any case.

The predicate is duplicates-only. `common` is the four-way intersection of source sets
and each mask is `role & isin(source, common)`, so every count is >= len(common):
`count != len(common)` can only mean multiplicity >= 2. A source MISSING from a role
leaves the intersection instead, and the cell is simply evaluated on fewer sources.

Note the clean role carries no severity filter in `additivity()` (`family == CLEAN` is
computed once, outside the cell loop), so a duplicated clean row for one source would
fire the guard in EVERY cell that source survives into. It is keyed on source alone here
to match.

Section map (file order):
  DuplicateRow                                  - one offending (role, key, severity, source)
  RowScan                                       - counts + duplicates + `ok`
  scan_duplicate_rows(probe_rows, combo_rows)   - the pure predicate
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ..probes.dataset import CLEAN

CLEAN_ROLE = "clean"
SINGLE_ROLE = "single"  # legs a and b are selected the same way, by (family, severity)
COMBO_ROLE = "combo"


@dataclass(frozen=True)
class DuplicateRow:
    role: str  # CLEAN_ROLE | SINGLE_ROLE | COMBO_ROLE
    key: str  # family name, or the combo pair label
    severity: int  # -1 for clean, which is keyed on source alone
    source: str
    n_rows: int  # how many rows carry this key; the guard needs exactly 1

    def __str__(self) -> str:
        sev = "" if self.severity < 0 else f"@{self.severity}"
        return f"{self.role} {self.key}{sev} source={self.source}: {self.n_rows} rows"


@dataclass(frozen=True)
class RowScan:
    n_probe_rows: int
    n_combo_rows: int
    n_keys: int  # distinct (role, key, severity, source) keys counted
    duplicates: tuple[DuplicateRow, ...]

    @property
    def ok(self) -> bool:
        """True when no role/source key is multiply occupied - i.e. the guard in
        `additivity()` cannot fire anywhere in this bank."""
        return not self.duplicates


def scan_duplicate_rows(
    probe_rows: Iterable[tuple[str, int, str]],
    combo_rows: Iterable[tuple[str, int, str]],
) -> RowScan:
    """Count rows per role/source key and report every key holding more than one.

    `probe_rows` yields `(family, severity, source)` - the single-degradation and clean
    cells, exactly as `build_probe_data` stacks them. `combo_rows` yields
    `(pair_label, severity, source)`. Neither needs latents: the first comes from the
    manifest, the second from a cache existence listing.
    """
    counts: Counter[tuple[str, str, int, str]] = Counter()
    n_probe = 0
    for family, severity, source in probe_rows:
        n_probe += 1
        if family == CLEAN:  # no severity filter in additivity(); key on source alone
            counts[(CLEAN_ROLE, CLEAN, -1, source)] += 1
        else:
            counts[(SINGLE_ROLE, family, int(severity), source)] += 1
    n_combo = 0
    for pair, severity, source in combo_rows:
        n_combo += 1
        counts[(COMBO_ROLE, pair, int(severity), source)] += 1

    duplicates = tuple(
        DuplicateRow(role=role, key=key, severity=sev, source=source, n_rows=n)
        for (role, key, sev, source), n in sorted(counts.items()) if n > 1
    )
    return RowScan(n_probe_rows=n_probe, n_combo_rows=n_combo,
                   n_keys=len(counts), duplicates=duplicates)

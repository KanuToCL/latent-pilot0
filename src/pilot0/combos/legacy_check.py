"""D3 regression guard: where source pairing selected exactly the legacy rows, the
raw additivity cosine must reproduce `reports/combos_real.json` bit-for-bit.

Pure: no filesystem, no SystemExit. `tools/additivity_run.py` loads the legacy JSON,
calls `check_legacy`, and turns a failing result into an abort — so the predicate can
be tested without a 140-minute bank pass.

Three buckets, not two. A cell that is unevaluable on BOTH sides (no clip carries all
four roles now, `cosine` null in the legacy report too) proves nothing and disproves
nothing, so it is counted separately instead of being filed as "source pairing changed
the selection". The real bank has six such cells - `hiss+bandlimit@2` on the six 16 kHz
candidates, where band-limit 2 sits at or above Nyquist - and the pre-guard run reported
them as `n_cells_differing: 6`, which read as a discrepancy it never was. The cause is
`_empty_cell` hardcoding `rows_identical=False`, not a NaN comparison: `same_value`
handles NaN correctly.

Losing a number is NOT one of the three buckets. A cell the legacy report measured and
this run cannot evaluate is a regression whichever rows were selected — the reanalysis
went blind where the anchor could see — so it is recorded as a mismatch (`ok` False),
never filed as "source pairing changed the selection". That branch is tested BEFORE
`rows_identical`, which would otherwise swallow it into `n_cells_differing`.

Section map (file order):
  is_unevaluable(value)                 - null / NaN cosine point, either side
  same_value(new, old)                  - bit-identical, JSON null == NaN
  fingerprint_ok(data, expected_sha256) - the anchor file's bytes are the audited bytes
  LegacyCheck                           - counts + mismatches + `ok`
  check_legacy(results, legacy_cells)   - classify every cell into the three buckets
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

MAX_REPORTED_MISMATCHES = 20


def is_unevaluable(value) -> bool:
    """A cosine point of JSON null (or NaN) means the cell produced no number at all.

    Returns a plain `bool`, never `np.bool_`: this feeds an `and`/`or` chain and a
    serialized count, and a numpy scalar there fails an `is True` check silently."""
    return value is None or (isinstance(value, float) and bool(np.isnan(value)))


def same_value(new, old) -> bool:
    """Bit-identical, with JSON null (nan under to_jsonable) treated as nan.

    Both sides go through `is_unevaluable`, so `same_value(None, None)` answers True
    instead of raising: `np.isnan(None)` is a TypeError, and the guard must be able to
    compare two absent numbers without the caller pre-screening them."""
    if is_unevaluable(old):
        return is_unevaluable(new)
    return new == old


def fingerprint_ok(data: bytes, expected_sha256: str) -> bool:
    """The anchor's bytes are the bytes the audit hashed, or they are not the anchor.

    Pure: takes the content, never a path, and returns a verdict rather than aborting —
    what a mismatch means is the caller's decision (`tools/additivity_run.py` refuses to
    use the file as the D3 anchor). Comparison is case-insensitive on the expected digest
    so a hex string copied from a report in either case still matches."""
    return hashlib.sha256(data).hexdigest() == expected_sha256.strip().lower()


@dataclass(frozen=True)
class LegacyCheck:
    n_cells_identical: int  # legacy rows reproduced exactly -> cosine verified
    n_cells_differing: int  # source pairing genuinely changed the selection
    n_cells_both_unevaluable: int  # no number on either side; nothing to compare
    n_cosines_verified: int  # identical cells actually found in the legacy report
    mismatches: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        """Fails on any mismatch, and fails when NO cell is identical - an
        all-different reanalysis is anchored to nothing."""
        return not self.mismatches and self.n_cells_identical > 0

    @property
    def failure(self) -> str | None:
        if self.n_cells_identical == 0:
            return ("no cell has rows_identical - source pairing changed every selection, "
                    "so nothing anchors this reanalysis to the legacy report")
        if self.mismatches:
            return ("legacy cosine not reproduced:\n  "
                    + "\n  ".join(self.mismatches[:MAX_REPORTED_MISMATCHES]))
        return None

    def as_dict(self, legacy_report: str) -> dict:
        return {"n_cells_identical": self.n_cells_identical,
                "n_cells_differing": self.n_cells_differing,
                "n_cells_both_unevaluable": self.n_cells_both_unevaluable,
                "n_cosines_verified": self.n_cosines_verified,
                "legacy_report": legacy_report}


def check_legacy(results: dict, legacy_cells: dict) -> LegacyCheck:
    """Classify every cell of every candidate.

    `results` maps a candidate key to an `AdditivityResult`; `legacy_cells` maps the
    SAME key to `{(pair, severity): legacy_cell_dict}`. Both are supplied by the caller
    so this stays independent of report layout and of the "name/variant" vs
    "name:variant" key spelling.
    """
    identical = differing = both_unevaluable = verified = 0
    mismatches: list[str] = []
    for key, res in results.items():
        if key not in legacy_cells:
            raise KeyError(
                f"no legacy cells for candidate {key!r}: the D3 guard was handed "
                f"{sorted(legacy_cells)} and cannot anchor a candidate absent from them")
        old_cells = legacy_cells[key]
        for cell, pa in sorted(res.by_cell.items()):
            old = old_cells.get(cell)
            new_uneval = is_unevaluable(pa.cosine.point)
            old_uneval = old is None or is_unevaluable(old["cosine"]["point"])
            if new_uneval and old_uneval and old is not None:
                both_unevaluable += 1
                continue
            if new_uneval and not old_uneval:
                # The anchor measured this cell and this run cannot: a lost number is a
                # regression however the rows were selected, so it never reaches the
                # `rows_identical` test that would file it as "differing".
                mismatches.append(f"{key} {cell}: unevaluable now, legacy point "
                                  f"{old['cosine']['point']!r}")
                continue
            if not pa.rows_identical:
                differing += 1
                continue
            identical += 1
            if old is None:
                mismatches.append(f"{key} {cell}: absent from the legacy report")
                continue
            verified += 1
            for name in ("point", "lo", "hi"):
                got, want = getattr(pa.cosine, name), old["cosine"][name]
                if not same_value(got, want):
                    mismatches.append(f"{key} {cell} cosine.{name}: {got!r} != {want!r}")
    return LegacyCheck(n_cells_identical=identical, n_cells_differing=differing,
                       n_cells_both_unevaluable=both_unevaluable,
                       n_cosines_verified=verified, mismatches=tuple(mismatches))

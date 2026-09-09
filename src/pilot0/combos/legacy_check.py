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

Section map (file order):
  is_unevaluable(value)                 - null / NaN cosine point, either side
  same_value(new, old)                  - bit-identical, JSON null == NaN
  LegacyCheck                           - counts + mismatches + `ok`
  check_legacy(results, legacy_cells)   - classify every cell into the three buckets
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_REPORTED_MISMATCHES = 20


def is_unevaluable(value) -> bool:
    """A cosine point of JSON null (or NaN) means the cell produced no number at all.

    Returns a plain `bool`, never `np.bool_`: this feeds an `and`/`or` chain and a
    serialized count, and a numpy scalar there fails an `is True` check silently."""
    return value is None or (isinstance(value, float) and bool(np.isnan(value)))


def same_value(new: float, old) -> bool:
    """Bit-identical, with JSON null (nan under to_jsonable) treated as nan."""
    if is_unevaluable(old):
        return bool(np.isnan(new))
    return new == old


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
            return ("legacy cosine not reproduced on identical rows:\n  "
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
        old_cells = legacy_cells[key]
        for cell, pa in sorted(res.by_cell.items()):
            old = old_cells.get(cell)
            new_uneval = is_unevaluable(pa.cosine.point)
            old_uneval = old is None or is_unevaluable(old["cosine"]["point"])
            if new_uneval and old_uneval and old is not None:
                both_unevaluable += 1
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

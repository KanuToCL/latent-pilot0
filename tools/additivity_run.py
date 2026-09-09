# S6 additivity reanalysis: dominance and recovery on the banked combo latents.
#
# Reads the Phase-3 + combo caches ONLY (no encode, no GPU) and writes
# reports/additivity_real.json + a printed table. The legacy `cosine` answers a
# question that is not identifiable on its own; each cell now also carries which
# leg the combo sits on (cos_to_a / cos_to_b), whether the legs can be separated
# at all (cos_legs), how lopsided they are (norm_ratio), the magnitude the cosine
# discards (r, with the derived rel_residual), and the per-source least-squares
# recovery (alpha, beta) with a speaker bootstrap CI.
#
# Rows are paired by SOURCE across all four roles now, not intersected by speaker
# GROUP. D3 guard: on every cell where the two selections coincide
# (`rows_identical`) the raw cosine must reproduce reports/combos_real.json
# bit-for-bit, and the run FAILS if no cell is identical - that would mean the
# reanalysis had silently changed the ground it stands on.
import json
import os
import time
from pathlib import Path

import numpy as np

from job_b_run import CANDIDATES
from pilot0.combos.grid import COMBO_PAIRS, COMBO_SEVERITIES, combo_label
from pilot0.combos.run import analyze_additivity
from pilot0.probes.run import FLOOR
from pilot0.provenance import is_fake, provenance
from pilot0.seam.registry import candidate_semantics, variant_semantics
from pilot0.serialize import to_jsonable

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "speech"
CACHE = ROOT / "data" / "cache" / "speech"
REPORTS = ROOT / "reports"

ALL_CANDIDATES = [FLOOR, *CANDIDATES]
MID = COMBO_SEVERITIES[len(COMBO_SEVERITIES) // 2]

CAVEATS = [
    "descriptive and IN-SAMPLE: every statistic here is fit on all rows and gates nothing",
    "cosine alone is NOT identifiable: read cos_legs before alpha/beta, and cos_to_b before "
    "believing a high cosine means both legs are present",
    "rel_residual is DERIVED from cosine and r (F23) - it is not an independent measurement",
    "alpha/beta are nan where the legs are collinear: the per-source system is rank-deficient",
    "combo order is A-then-B by convention; degradations do not commute",
]


def key_of(name: str, variant: str) -> str:
    return f"{name}:{variant}"


def write_atomic(path: Path, payload: dict) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def estimate_dict(e) -> dict:
    return {"point": e.point, "lo": e.lo, "hi": e.hi}


def cell_dict(pa) -> dict:
    return {
        "pair": pa.pair, "severity": pa.severity,
        "cosine": estimate_dict(pa.cosine), "cosine_std": estimate_dict(pa.cosine_std),
        "cos_to_a": pa.cos_to_a, "cos_to_b": pa.cos_to_b, "cos_legs": pa.cos_legs,
        "norm_ratio": pa.norm_ratio, "r": pa.r, "rel_residual": pa.rel_residual,
        "alpha": estimate_dict(pa.alpha), "beta": estimate_dict(pa.beta),
        "n_groups": pa.n_groups, "n_sources": pa.n_sources, "rows_identical": pa.rows_identical,
    }


def _same(new: float, old) -> bool:
    """Bit-identical, with JSON null (nan under to_jsonable) treated as nan."""
    if old is None or (isinstance(old, float) and np.isnan(old)):
        return bool(np.isnan(new))
    return new == old


def check_legacy(results: dict, legacy_path: Path) -> dict:
    """D3: where source pairing selected exactly the legacy rows, the raw cosine must
    reproduce the legacy report bit-for-bit. Fails loud on a mismatch, and fails loud
    if NO cell is identical - an all-different reanalysis proves nothing."""
    if not legacy_path.exists():
        raise SystemExit(f"ABORT: {legacy_path} is missing - the D3 regression guard cannot run")
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))["by_candidate"]
    n_identical = n_checked = n_differing = 0
    mismatches = []
    for key, res in results.items():
        name, variant = key.split("/")
        old_cells = {(c["pair"], c["severity"]): c for c in legacy[key_of(name, variant)]["additivity"]["by_cell"]}
        for cell, pa in res.by_cell.items():
            if not pa.rows_identical:
                n_differing += 1
                continue
            n_identical += 1
            old = old_cells.get(cell)
            if old is None:
                mismatches.append(f"{key_of(name, variant)} {cell}: absent from the legacy report")
                continue
            n_checked += 1
            for field, got in (("point", pa.cosine.point), ("lo", pa.cosine.lo), ("hi", pa.cosine.hi)):
                if not _same(got, old["cosine"][field]):
                    mismatches.append(f"{key_of(name, variant)} {cell} cosine.{field}: "
                                      f"{got!r} != {old['cosine'][field]!r}")
    if n_identical == 0:
        raise SystemExit("ABORT (D3): no cell has rows_identical - source pairing changed every "
                         "selection, so nothing anchors this reanalysis to combos_real.json")
    if mismatches:
        raise SystemExit("ABORT (D3): legacy cosine not reproduced on identical rows:\n  "
                         + "\n  ".join(mismatches[:20]))
    return {"n_cells_identical": n_identical, "n_cells_differing": n_differing,
            "n_cosines_verified": n_checked, "legacy_report": str(legacy_path.name)}


def _f(x, width: int = 7) -> str:
    if x is None or not np.isfinite(x):
        return "n/a".rjust(width)
    return f"{x:+.3f}".rjust(width)


HEADER = (f"{'candidate':14s} {'pair':15s} "
          + " ".join(f"{f'cos@{s}':>7s}" for s in COMBO_SEVERITIES)
          + f" | {f'cosA@{MID}':>7s} {f'cosB@{MID}':>7s} {f'cosLegs@{MID}':>9s} "
            f"{f'nRatio@{MID}':>8s} {f'r@{MID}':>7s} {f'relRes@{MID}':>8s} "
            f"{f'alpha@{MID}':>7s} {f'beta@{MID}':>7s} {'nSrc':>5s} {'ident':>5s}")


def print_summary(results: dict) -> None:
    print("\n" + HEADER)
    print("-" * len(HEADER))
    for name, variant in ALL_CANDIDATES:
        res = results[f"{name}/{variant}"]
        head = f"{key_of(name, variant):14s}"
        if not res.by_cell:
            print(f"{head} no combo cells cached")
            continue
        for a, b in COMBO_PAIRS:
            label = combo_label(a, b)
            cells = {s: res.by_cell.get((label, s)) for s in COMBO_SEVERITIES}
            if not any(cells.values()):
                continue
            cos = " ".join(_f(None if cells[s] is None else cells[s].cosine.point) for s in COMBO_SEVERITIES)
            m = cells[MID]
            if m is None:
                print(f"{head} {label:15s} {cos} | (no cell at severity {MID})")
            else:
                print(f"{head} {label:15s} {cos} | {_f(m.cos_to_a)} {_f(m.cos_to_b)} "
                      f"{_f(m.cos_legs, 9)} {_f(m.norm_ratio, 8)} {_f(m.r)} {_f(m.rel_residual, 8)} "
                      f"{_f(m.alpha.point)} {_f(m.beta.point)} {m.n_sources:5d} "
                      f"{('yes' if m.rows_identical else 'NO'):>5s}")
            head = f"{'':14s}"  # candidate printed once, on its first pair


def main() -> None:
    if is_fake(ALL_CANDIDATES):
        raise SystemExit("ABORT: fake encoder in the candidate list - this reanalysis reads real cached latents only")
    REPORTS.mkdir(exist_ok=True)

    man = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    print(f"manifest: {man['n_sources']} sources, {man['n_groups']} groups, {man['n_rows']} rows")
    print(f"candidates: {len(ALL_CANDIDATES)}   pairs: {[combo_label(a, b) for a, b in COMBO_PAIRS]} "
          f"at severities {list(COMBO_SEVERITIES)}", flush=True)

    t0 = time.time()

    def progress(key, res):
        n = sum(1 for c in res.by_cell.values() if c.n_sources)
        print(f"  {key:22s} {n}/{len(res.by_cell)} evaluable cells  "
              f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

    results = analyze_additivity(man, ALL_CANDIDATES, CACHE, on_candidate=progress)
    print(f"\nanalysis done in {(time.time() - t0) / 60:.1f} min", flush=True)

    guard = check_legacy(results, REPORTS / "combos_real.json")
    print(f"D3 guard: {guard['n_cells_identical']} cells with identical rows, "
          f"{guard['n_cosines_verified']} cosines reproduced bit-for-bit, "
          f"{guard['n_cells_differing']} cells where source pairing changed the selection", flush=True)

    out = write_atomic(REPORTS / "additivity_real.json", to_jsonable({
        "by_candidate": {
            key_of(n, v): {
                "key": key_of(n, v), "name": n, "variant": v,
                "semantics": variant_semantics(n, v),
                "mean_cosine": results[f"{n}/{v}"].mean_cosine(),
                "by_cell": [cell_dict(pa) for _cell, pa in sorted(results[f"{n}/{v}"].by_cell.items())],
            } for n, v in ALL_CANDIDATES
        },
        "candidates": [list(c) for c in ALL_CANDIDATES],
        "candidate_semantics": candidate_semantics(ALL_CANDIDATES),
        "combo_pairs": [list(p) for p in COMBO_PAIRS],
        "combo_severities": list(COMBO_SEVERITIES),
        "legacy_regression_guard": guard,
        "caveats": CAVEATS,
        "provenance": provenance(is_fake(ALL_CANDIDATES), corpus="VCTK-0.92 mic1 510x102spk",
                                 stage="s6_additivity_reanalysis"),
    }))
    print(f"\nwrote {out}", flush=True)
    print_summary(results)
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

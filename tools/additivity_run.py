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
# reanalysis had silently changed the ground it stands on. The anchor file is first
# hashed against LEGACY_SHA256: an edited or regenerated combos_real.json is not the
# file the corrections were written against and is refused as an anchor.
#
# The run also EXITS NON-ZERO when any cell was made unevaluable by additivity()'s
# per-source row-count guard (`reason == "duplicate_rows"`). That fires only after the
# report is written and the table printed - the numbers that were computed stay on disk
# and readable, but the exit code refuses to call the pass clean.
#
# `--scan-duplicates` is a separate, seconds-long mode that never loads a latent: it
# asks only whether additivity()'s per-source row-count guard could fire anywhere in
# the bank (pilot0.combos.row_scan states why a listing is sufficient evidence).
#
# Section map (file order):
#   key_of(name, variant)           - "name:variant" report key
#   write_atomic(path, payload)     - tmp + os.replace
#   estimate_dict(e)                - Estimate -> {point, lo, hi}
#   cell_dict(pa)                   - PairAdditivity -> report record
#   read_anchor(path)               - combos_real.json text, refused on a bad fingerprint
#   legacy_cells(path)              - combos_real.json -> check_legacy's input shape
#   guard_or_abort(results, path)   - run the D3 guard, SystemExit on failure
#   duplicate_row_cells(results)    - "key pair@sev" for every duplicate_rows cell
#   probe_rows(manifest, sr)        - (family, severity, source) per probe row
#   combo_rows(manifest, name, variant) - (pair, severity, source) per CACHED combo cell
#   scan_duplicates()               - --scan-duplicates: print the scan, non-zero on a hit
#   _f(x, width)                    - table cell formatter
#   reasons_note(cells)             - "  <- @2: no_common_source" for a whole table row
#   print_summary(results)          - the pair x severity x candidate table
#   main()                          - full reanalysis -> reports/additivity_real.json
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from job_b_run import CANDIDATES
from pilot0.combos.grid import COMBO_PAIRS, COMBO_SEVERITIES, combo_label, combo_severities
from pilot0.combos.legacy_check import check_legacy, fingerprint_ok
from pilot0.combos.row_scan import scan_duplicate_rows
from pilot0.combos.run import analyze_additivity
from pilot0.corpus.manifest import renderable_rows
from pilot0.encode.cache import cache_version, cell_id, is_cached
from pilot0.encode.headroom import CEILING_DBFS
from pilot0.probes.run import FLOOR
from pilot0.provenance import is_fake, provenance
from pilot0.seam.registry import candidate_semantics, make_encoder, variant_semantics
from pilot0.serialize import to_jsonable

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "speech"
CACHE = ROOT / "data" / "cache" / "speech"
REPORTS = ROOT / "reports"

ALL_CANDIDATES = [FLOOR, *CANDIDATES]
MID = COMBO_SEVERITIES[len(COMBO_SEVERITIES) // 2]

# sha256 of the reports/combos_real.json the science spot check audited and every
# correction in docs/DECISIONS.md is anchored to. It lives here, next to the path it
# guards, rather than in combos/legacy_check.py, which knows no report layout.
# It pins the `60f761c`-era file BYTE for byte, so re-running its writer
# (tools/job_c_run.py) will change the digest and `read_anchor` will refuse the result
# - that is the guard working, not a corrupted file. Re-pin this constant deliberately,
# in the same commit that regenerates the report, or the anchor stops meaning anything.
LEGACY_SHA256 = "633a2cbf16df11652648cd8d79aa562b22da33cb2e7145b8305e8573593d2d60"

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
        "reason": pa.reason,  # null when the cell carries numbers
    }


def read_anchor(legacy_path: Path) -> str:
    """The D3 anchor's text, refused unless its bytes hash to LEGACY_SHA256.

    A `combos_real.json` that has been edited or regenerated is not the file the audit
    read and the corrections were written against; anchoring a reanalysis to it would
    verify nothing while looking exactly like a pass. `fingerprint_ok` is the pure
    predicate (tested in tests/test_legacy_check.py); the refusal is this run's policy."""
    if not legacy_path.exists():
        raise SystemExit(f"ABORT: {legacy_path} is missing - the D3 regression guard cannot run")
    data = legacy_path.read_bytes()
    if not fingerprint_ok(data, LEGACY_SHA256):
        raise SystemExit(f"ABORT (D3): {legacy_path} does not hash to the audited fingerprint "
                         f"{LEGACY_SHA256} - it is not the anchor the corrections were "
                         f"written against and will not be used as one")
    return data.decode("utf-8")


def legacy_cells(legacy_path: Path) -> dict:
    """combos_real.json -> `{candidate key: {(pair, severity): cell}}`, the shape
    `pilot0.combos.legacy_check.check_legacy` takes. Keys are spelled "name/variant"
    to match `analyze_additivity`'s result dict, not the report's "name:variant"."""
    legacy = json.loads(read_anchor(legacy_path))["by_candidate"]
    out = {}
    for name, variant in ALL_CANDIDATES:
        cells = legacy[key_of(name, variant)]["additivity"]["by_cell"]
        out[f"{name}/{variant}"] = {(c["pair"], c["severity"]): c for c in cells}
    return out


def guard_or_abort(results: dict, legacy_path: Path) -> dict:
    """Run the D3 guard and turn a failing verdict into an abort. The classification
    itself is pure and tested in tests/test_legacy_check.py."""
    check = check_legacy(results, legacy_cells(legacy_path))
    if not check.ok:
        raise SystemExit(f"ABORT (D3): {check.failure}")
    return check.as_dict(legacy_path.name)


def duplicate_row_cells(results: dict) -> list[str]:
    """`"name:variant pair@severity"` for every cell additivity()'s per-source row-count
    guard made unevaluable, sorted. Pure, so the exit message main() prints is testable
    without a bank pass."""
    return sorted(f"{key.replace('/', ':')} {pair}@{sev}"
                  for key, res in results.items()
                  for (pair, sev), pa in res.by_cell.items()
                  if pa.reason == "duplicate_rows")


# --- --scan-duplicates: is additivity()'s row-count guard reachable at all? ----------


def probe_rows(manifest: dict, native_sr: int):
    """`(family, severity, source)` for every row `build_probe_data` would stack: it
    appends exactly one per renderable manifest row, so no latent need be opened."""
    for r in renderable_rows(manifest, native_sr):
        yield r["family"], int(r["severity"]), r["source"]


def combo_rows(manifest: dict, name: str, variant: str, cache_dir: Path = CACHE):
    """`(pair, severity, source)` for every CACHED combo cell, mirroring
    `iter_combo_latents`' walk with `is_cached` in place of `load_latent`."""
    enc = make_encoder(name)
    cv = cache_version(manifest, CEILING_DBFS)
    for source in {r["source"] for r in manifest["rows"]}:
        for a, b in COMBO_PAIRS:
            label = combo_label(a, b)
            for sev in combo_severities(a, b, enc.native_sr):
                if is_cached(cache_dir, name, variant, cell_id(source, label, sev, enc.native_sr), cv):
                    yield label, sev, source


def scan_duplicates() -> int:
    """Print the per-candidate duplicate scan. Exit code 1 if any cell would trip the
    guard - i.e. if reports/additivity_real.json's pre-guard numbers are in doubt."""
    man = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    print(f"manifest: {man['n_sources']} sources, {man['n_rows']} rows   "
          f"candidates: {len(ALL_CANDIDATES)}   (no latents opened)", flush=True)
    total = 0
    for name, variant in ALL_CANDIDATES:
        sr = make_encoder(name).native_sr
        scan = scan_duplicate_rows(probe_rows(man, sr), combo_rows(man, name, variant))
        total += len(scan.duplicates)
        print(f"  {key_of(name, variant):22s} {sr // 1000:>3d} kHz  "
              f"{scan.n_probe_rows:6d} probe + {scan.n_combo_rows:5d} combo rows  "
              f"{scan.n_keys:6d} role/source keys  duplicates: {len(scan.duplicates)}", flush=True)
        for dup in scan.duplicates[:10]:
            print(f"      {dup}")
    print(f"\ntotal duplicate role/source keys across all candidates: {total}")
    print("=> additivity()'s duplicate-row guard is INERT on this bank" if total == 0
          else "=> the guard WOULD fire: cells above are unevaluable")
    return 1 if total else 0


def _f(x, width: int = 7) -> str:
    if x is None or not np.isfinite(x):
        return "n/a".rjust(width)
    return f"{x:+.3f}".rjust(width)


def reasons_note(cells: dict) -> str:
    """`"  <- @2: no_common_source, @4: duplicate_rows"` for EVERY unevaluable cell in
    one table row, "" when they all carry numbers.

    `cells` maps severity to a PairAdditivity or None (severity not in this candidate's
    grid at all). The cosine columns print "n/a" for a measured nan and for a cell there
    was nothing to measure in, so without this a reader cannot tell them apart - and the
    previous version answered only for the MID severity, leaving the other columns mute
    even when their reason was the very thing that made the row interesting."""
    bad = [f"@{s}: {c.reason}" for s, c in sorted(cells.items())
           if c is not None and not c.evaluable]
    return ("  <- " + ", ".join(bad)) if bad else ""


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
            # Every unevaluable cell in the row names its reason, at every severity, not
            # just the MID one the detail columns come from: a reader must be able to
            # tell "nothing to measure" from "measured nan", and both print "n/a".
            why = reasons_note(cells)
            if m is None:
                print(f"{head} {label:15s} {cos} | (no cell at severity {MID}){why}")
            else:
                print(f"{head} {label:15s} {cos} | {_f(m.cos_to_a)} {_f(m.cos_to_b)} "
                      f"{_f(m.cos_legs, 9)} {_f(m.norm_ratio, 8)} {_f(m.r)} {_f(m.rel_residual, 8)} "
                      f"{_f(m.alpha.point)} {_f(m.beta.point)} {m.n_sources:5d} "
                      f"{('yes' if m.rows_identical else 'NO'):>5s}{why}")
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
        # `PairAdditivity.evaluable` is THE evaluability test, shared with print_summary
        # and mean_cosine. It is not `n_sources`: a duplicate_rows cell reports the
        # sources it paired and still carries no statistics.
        n = sum(1 for c in res.by_cell.values() if c.evaluable)
        skipped = Counter(c.reason for c in res.by_cell.values() if not c.evaluable)
        why = ("  (" + ", ".join(f"{r}: {k}" for r, k in sorted(skipped.items())) + ")") if skipped else ""
        print(f"  {key:22s} {n}/{len(res.by_cell)} evaluable cells{why}  "
              f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

    results = analyze_additivity(man, ALL_CANDIDATES, CACHE, on_candidate=progress)
    print(f"\nanalysis done in {(time.time() - t0) / 60:.1f} min", flush=True)

    guard = guard_or_abort(results, REPORTS / "combos_real.json")
    print(f"D3 guard: {guard['n_cells_identical']} cells with identical rows, "
          f"{guard['n_cosines_verified']} cosines reproduced bit-for-bit, "
          f"{guard['n_cells_differing']} cells where source pairing changed the selection, "
          f"{guard['n_cells_both_unevaluable']} unevaluable on both sides", flush=True)

    out = write_atomic(REPORTS / "additivity_real.json", to_jsonable({
        "by_candidate": {
            key_of(n, v): {
                "key": key_of(n, v), "name": n, "variant": v,
                "semantics": variant_semantics(n, v),
                "mean_cosine": results[f"{n}/{v}"].mean_cosine(),
                # How many cells that mean is over: a mean of 3 evaluable cells and a
                # mean of 9 are different claims and the report must distinguish them.
                "n_cells_averaged": results[f"{n}/{v}"].n_cells_averaged(),
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

    # Last, so the report is on disk and the table has been read: a fired duplicate-row
    # guard means some cells carry no statistics at all, and the exit code has to say so.
    dups = duplicate_row_cells(results)
    if dups:
        raise SystemExit(
            f"FAIL: additivity()'s duplicate-row guard fired on {len(dups)} cell(s). "
            f"{out} is written and its other cells are valid, but these have no "
            f"statistics:\n  " + "\n  ".join(dups))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-duplicates", action="store_true",
                    help="only check whether additivity()'s per-source row-count guard "
                         "could fire on this bank (seconds; opens no latent)")
    args = ap.parse_args()
    raise SystemExit(scan_duplicates() if args.scan_duplicates else main())

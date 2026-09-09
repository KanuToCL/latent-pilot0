# S7 level split: how much of the energy control's severity signal is loudness?
#
# Reads the Phase-3 energy latents ONLY (no encode, no GPU, no level-matched
# corpus) and writes reports/level_split_real.json + a printed table.
#
# The control's pooled vector is six log-energy statistics. A uniform gain moves
# exactly one direction of it - the three log-MEANS together, no log-STD - so the
# vector splits into a 1-d loudness coordinate and a 5-d gain-invariant remainder.
# Scoring the same ridge probe on all three spaces answers the question the
# level-matched arm was supposed to answer, directly and for free. It also bounds
# what that arm could ever have answered: LUFS matching leaves the control's own
# level cue in place (F19), so only this split separates the two.
#
# The projection is done in the RAW pooled space, before any scaler (AM3); the
# ridge standardises after projection, exactly as the gate's probe does.
#
# Section map (file order):
#   write_atomic(path, payload)        - tmp + os.replace
#   assert_gate_grid(n_common)         - pin this run's cell grid to the gate run's
#   low_frame_audit(manifest, grid)    - AM4: frames below -80 dBFS, per family
#   _est(e)                            - Estimate -> {point, lo, hi}
#   _f(x, width)                       - table cell formatter
#   print_summary(res, low, frames)    - 6-d vs level-1d vs invariant-5d table
#   main()                             - -> reports/level_split_real.json
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from job_b_run import CANDIDATES
from pilot0.degrade.grid import FAMILIES
from pilot0.probes.dataset import build_probe_data, iter_latents
from pilot0.probes.level_split import (
    LOW_FRAME_DBFS, PROJECTION_SPACE, SPACES, count_frames_below, eps_bound_dbfs,
    evaluate_level_split,
)
from pilot0.probes.run import ENERGY, FLOOR, common_conditions
from pilot0.provenance import is_fake, provenance
from pilot0.seam.registry import variant_semantics
from pilot0.serialize import to_jsonable

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "speech"
CACHE = ROOT / "data" / "cache" / "speech"
REPORTS = ROOT / "reports"

GRID_NAMES = [FLOOR[0], ENERGY[0], *(n for n, _ in CANDIDATES)]

CAVEATS = [
    "descriptive: this re-scores the CACHED energy latents, it does not re-run Gate 1",
    "the level coordinate is exact for a uniform gain; it is not a loudness model (no BS.1770 gating)",
    f"below {LOW_FRAME_DBFS:.0f} dBFS the encoder's _EPS breaks gain-equivariance - see low_frame_counts",
    "a LUFS-matched arm would NOT remove the level cue measured here (F19); this split is the instrument",
]


def write_atomic(path: Path, payload: dict) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def assert_gate_grid(n_common: int) -> int | None:
    gate = REPORTS / "gate1_real.json"
    if not gate.exists():
        return None
    n_gate = int(json.loads(gate.read_text(encoding="utf-8"))["report"]["n_common_conditions"])
    if n_gate != n_common:
        raise SystemExit(f"ABORT: common cells {n_common} != gate1 {n_gate} - grids differ")
    return n_gate


def low_frame_audit(manifest: dict, grid) -> tuple[dict, dict, dict]:
    """AM4: how exposed is the split to the encoder's _EPS floor? Counts frames below
    LOW_FRAME_DBFS per family, over the same cells the probe scores."""
    low: Counter = Counter()
    frames: Counter = Counter()
    cells: Counter = Counter()
    for row, lat in iter_latents(manifest, ENERGY[0], ENERGY[1], CACHE):
        if (row["family"], row["severity"]) not in grid:
            continue
        fam = row["family"]
        cells[fam] += 1
        frames[fam] += int(lat.frames.shape[0])
        low[fam] += count_frames_below(lat.frames)
    return dict(low), dict(frames), dict(cells)


def _est(e) -> dict:
    return {"point": e.point, "lo": e.lo, "hi": e.hi}


def _f(x, width: int = 6) -> str:
    if x is None or not np.isfinite(x):
        return "n/a".rjust(width)
    return f"{x:+.3f}".rjust(width)


def print_summary(res: dict, low: dict, frames: dict) -> None:
    head = (f"{'family':11s} | {'pooled6d':>7s} {'[lo,hi]':>16s} | {'level1d':>7s} {'[lo,hi]':>16s} "
            f"| {'invar5d':>7s} {'[lo,hi]':>16s} | {'lowFrames':>9s}")
    print("\nwithout-clean severity SRCC (the ladder Gate 1 keys on)")
    print(head)
    print("-" * len(head))
    for fam in FAMILIES:
        cols = []
        for space in SPACES:
            e = res[space].by_family[fam].without_clean
            cols.append(f"{_f(e.point, 7)} [{_f(e.lo)},{_f(e.hi)}]")
        n_low, n_tot = low.get(fam, 0), frames.get(fam, 0)
        pct = f"{100.0 * n_low / n_tot:.2f}%" if n_tot else "n/a"
        print(f"{fam:11s} | " + " | ".join(cols) + f" | {pct:>9s}")


def main() -> None:
    if is_fake([ENERGY]):
        raise SystemExit("ABORT: fake encoder - this reanalysis reads real cached latents only")
    REPORTS.mkdir(exist_ok=True)

    man = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    grid = common_conditions(GRID_NAMES)
    n_gate = assert_gate_grid(len(grid))
    print(f"manifest: {man['n_sources']} sources, {man['n_groups']} groups, {man['n_rows']} rows")
    print(f"common cells: {len(grid)} (gate1: {n_gate})   projection space: {PROJECTION_SPACE}", flush=True)

    t0 = time.time()
    data = build_probe_data(man, ENERGY[0], ENERGY[1], CACHE).select(grid)
    print(f"energy latents loaded: {data.X.shape[0]} rows x {data.X.shape[1]} features "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    res = evaluate_level_split(data)
    print(f"probes fit in {(time.time() - t0) / 60:.1f} min", flush=True)

    low, frames, cells = low_frame_audit(man, grid)
    print(f"frame-floor audit done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    out = write_atomic(REPORTS / "level_split_real.json", to_jsonable({
        "encoder": f"{ENERGY[0]}:{ENERGY[1]}",
        "semantics": variant_semantics(*ENERGY),
        "projection_space": PROJECTION_SPACE,  # AM3
        "spaces": list(SPACES),
        "n_common_conditions": len(grid),
        "n_rows": int(data.X.shape[0]),
        "n_test_groups": data.n_test_groups(),
        "eps_bound_dbfs": eps_bound_dbfs(),  # where _EPS equals the frame's own power
        "low_frame_dbfs": LOW_FRAME_DBFS,
        # Totals span every scored cell INCLUDING clean; `by_family` below covers the
        # seven degradation families only, so the two do not sum to each other.
        "n_frames_total": int(sum(frames.values())),
        "n_frames_below_floor_total": int(sum(low.values())),
        "by_family": {
            fam: {
                "srcc": {space: {"with_clean": _est(res[space].by_family[fam].with_clean),
                                 "without_clean": _est(res[space].by_family[fam].without_clean)}
                         for space in SPACES},
                "n_cells": cells.get(fam, 0),
                "n_frames": frames.get(fam, 0),
                "n_frames_below_floor": low.get(fam, 0),
            } for fam in FAMILIES
        },
        "caveats": CAVEATS,
        "provenance": provenance(is_fake([ENERGY]), corpus="VCTK-0.92 mic1 510x102spk",
                                 stage="s7_energy_level_split"),
    }))

    print_summary(res, low, frames)
    total_low = sum(low.values())
    print(f"\nframes below {LOW_FRAME_DBFS:.0f} dBFS: {total_low} of {sum(frames.values())} "
          f"(_EPS equals the frame's own power at {eps_bound_dbfs():.0f} dBFS)")
    print(f"\nwrote {out}")
    print(f"total {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

# S1 ceiling reanalysis: what Gate 1's severity criterion actually asked for.
#
# Reads reports/gate1_real.json and the corpus manifest ONLY (no cache walk, no
# encode, no GPU) and writes reports/ceiling_real.json + a printed table.
#
# Per family it derives the maximum SRCC attainable against the tied severity
# ladder on the gate's own common grid, then compares it with the bar G1b sets:
# max(0.80, energy control CI-upper + 0.05). Where the bar is above the ceiling the
# family is unpassable by ANY representation, and the run reports how many families
# an oracle probe - one pinned exactly at the ceiling - could pass.
#
# Gate 1 is not redefined: every threshold is imported from probes/gate1.py.
import json
import os
from pathlib import Path

import numpy as np

from job_b_run import CANDIDATES
from pilot0.probes.ceiling import candidate_summary, family_ceilings, oracle_n_pass, severity_from_json
from pilot0.probes.gate1 import SEVERITY_MIN_FAMILIES, SEVERITY_OVER_ENERGY_MARGIN, SEVERITY_SRCC_MIN
from pilot0.probes.run import ENERGY, FLOOR, common_conditions
from pilot0.provenance import is_fake, provenance
from pilot0.seam.registry import candidate_semantics
from pilot0.serialize import to_jsonable

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "speech"
REPORTS = ROOT / "reports"

ALL_CANDIDATES = [FLOOR, *CANDIDATES]
GRID_NAMES = [FLOOR[0], ENERGY[0], *(n for n, _ in CANDIDATES)]

CAVEATS = [
    "the ceiling is a property of the DESIGN (levels x rows), not of any representation",
    "required_lo reads the FROZEN Gate-1 thresholds forward; nothing here re-tunes them",
    "rho_max_balanced_crosscheck is invalid when the level counts are unbalanced - it overstates (AM6)",
    "the oracle probe is a zero-width CI at the ceiling: no real probe can beat it",
]


def write_atomic(path: Path, payload: dict) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def assert_gate_grid(report: dict, n_common: int) -> int:
    """Pin this run's cell grid to the gate run's, exactly as geometry_run does."""
    n_gate = int(report["n_common_conditions"])
    if n_gate != n_common:
        raise SystemExit(f"ABORT: common cells {n_common} != gate1 {n_gate} - grids differ")
    return n_gate


def assert_shared_energy_baseline(report: dict) -> None:
    """`oracle_n_pass` counts against `report.energy.severity`; the gate counted each
    candidate against that candidate's own `energy_severity` block. `run_gate1` sets
    both from the same object, so they must be identical - and if they ever were not,
    the oracle would be measuring a different baseline than the gate did and the
    feasibility claim would not hold. Load-bearing, so it is checked, not assumed."""
    base = report["energy"]["severity"]
    bad = [f"{d['name']}:{d['variant']}" for d in report["decisions"] if d["energy_severity"] != base]
    if bad:
        raise SystemExit("ABORT: energy baseline differs from report.energy.severity for "
                         + ", ".join(bad) + " - the oracle count would not be comparable")


def ceiling_dict(c) -> dict:
    return {
        "family": c.family, "K": c.K, "N": c.N,
        "counts_by_level": {str(s): n for s, n in c.counts_by_level.items()},
        "rho_max": c.rho_max,
        # AM6: cross-check only; invalid when the level counts are unbalanced.
        "rho_max_balanced_crosscheck": c.rho_max_balanced_crosscheck,
        "energy_hi": c.energy_hi, "required_lo": c.required_lo, "feasible": c.feasible,
    }


def print_families(ceilings: dict) -> None:
    head = (f"{'family':11s} {'K':>2s} {'N':>5s} {'counts':>18s} {'rho_max':>8s} "
            f"{'balXcheck':>9s} {'energyHi':>8s} {'requiredLo':>10s}  feasible")
    print("\n" + head)
    print("-" * len(head))
    for f, c in ceilings.items():
        counts = ",".join(str(n) for n in c.counts_by_level.values())
        print(f"{f:11s} {c.K:2d} {c.N:5d} {counts:>18s} {c.rho_max:8.4f} "
              f"{c.rho_max_balanced_crosscheck:9.4f} {c.energy_hi:8.4f} {c.required_lo:10.4f}  "
              f"{'YES' if c.feasible else 'no  <-- unreachable'}")


def print_candidates(summaries: list) -> None:
    head = f"{'candidate':16s} {'absClears':>9s} {'marginPass':>10s} {'reported':>8s} {'feasible':>8s}"
    print("\n" + head)
    print("-" * len(head))
    for s in summaries:
        flag = "" if s.margin_passes == s.margin_passes_reported else "  <-- DIFFERS from the gate run"
        print(f"{s.key:16s} {s.absolute_clears:9d} {s.margin_passes:10d} "
              f"{s.margin_passes_reported:8d} {s.feasible_families:8d}{flag}")


def main() -> None:
    if is_fake(ALL_CANDIDATES):
        raise SystemExit("ABORT: fake encoder in the candidate list - this reanalysis reads a real gate run")
    REPORTS.mkdir(exist_ok=True)

    payload_in = json.loads((REPORTS / "gate1_real.json").read_text(encoding="utf-8"))
    report = payload_in["report"]
    man = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))

    grid = common_conditions(GRID_NAMES)
    n_gate = assert_gate_grid(report, len(grid))
    print(f"manifest: {man['n_sources']} sources, {man['n_groups']} groups, {man['n_rows']} rows")
    print(f"common cells: {len(grid)} (gate1: {n_gate})   "
          f"G1b needs >= {SEVERITY_MIN_FAMILIES}/7 families at SRCC lo >= {SEVERITY_SRCC_MIN} "
          f"and >= energy hi + {SEVERITY_OVER_ENERGY_MARGIN}", flush=True)

    assert_shared_energy_baseline(report)
    ceilings = family_ceilings(report, man, grid)
    energy = severity_from_json(report["energy"]["severity"])
    n_oracle = oracle_n_pass(ceilings, energy)
    summaries = candidate_summary(report, ceilings)
    n_feasible = sum(c.feasible for c in ceilings.values())

    out = write_atomic(REPORTS / "ceiling_real.json", to_jsonable({
        "source_report": "reports/gate1_real.json",
        "n_common_conditions": len(grid),
        "thresholds": {"severity_srcc_min": SEVERITY_SRCC_MIN,
                       "over_energy_margin": SEVERITY_OVER_ENERGY_MARGIN,
                       "min_families": SEVERITY_MIN_FAMILIES},
        "by_family": {f: ceiling_dict(c) for f, c in ceilings.items()},
        "n_feasible_families": n_feasible,
        "oracle_n_pass": n_oracle,
        "gate_severity_leg_reachable": bool(n_oracle >= SEVERITY_MIN_FAMILIES),
        "by_candidate": {s.key: {
            "key": s.key, "name": s.name, "variant": s.variant,
            "absolute_clears": s.absolute_clears, "margin_passes": s.margin_passes,
            "margin_passes_reported": s.margin_passes_reported,
            "feasible_families": s.feasible_families,
        } for s in summaries},
        "candidates": [list(c) for c in ALL_CANDIDATES],
        "candidate_semantics": candidate_semantics(ALL_CANDIDATES),
        "caveats": CAVEATS,
        "provenance": provenance(is_fake(ALL_CANDIDATES), corpus="VCTK-0.92 mic1 510x102spk",
                                 stage="s1_ceiling_reanalysis"),
    }))

    print_families(ceilings)
    print_candidates(summaries)
    top = max(c.rho_max for c in ceilings.values() if np.isfinite(c.rho_max))
    print(f"\nfeasible families: {n_feasible}/7    oracle probe (SRCC pinned at the ceiling, "
          f"max {top:.4f}) passes {n_oracle}/7")
    print(f"G1b needs {SEVERITY_MIN_FAMILIES}: severity leg is "
          f"{'REACHABLE' if n_oracle >= SEVERITY_MIN_FAMILIES else 'UNREACHABLE by any representation'}")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()

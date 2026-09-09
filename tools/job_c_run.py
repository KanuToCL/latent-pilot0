# Job C - Phase 7 (RQ2 combos): the A-then-B pairwise cells.
#
# Two passes, same shape as job_b_run: (1) a warm encode_combos pass over all six
# encoders - the combo workers should already have filled the cache, so this skips
# everything and just re-states the counts; (2) analyze_combos over the SAME 17
# pre-registered candidates Gate 1 used, plus the log-mel floor, writing
# reports/combos_real.json atomically.
#
# Combos REUSE the headroom scalar banked by the singles pass (resolve_headroom
# returns early on the existing headroom/sr_*.json), so z-bar(a+b) lives in the same
# scaling as z-bar(a), z-bar(b) and z-bar(clean) and the additivity cosine is honest.
# Nothing here rescans headroom.
#
# Serialization note (this bit us on Gate 1): dataclasses.asdict silently drops
# @property AND method results, and it would stringify AdditivityResult.by_cell's
# TUPLE keys into "('noise+clip', 2)". Every result field below is written out by
# hand instead - see additivity_dict / transfer_dict.
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

from job_b_run import CANDIDATES, ENCODERS  # single source of truth for both lists
from pilot0.combos.encode import encode_combos
from pilot0.combos.grid import COMBO_PAIRS, COMBO_SEVERITIES, combo_label
from pilot0.combos.run import analyze_combos
from pilot0.probes.run import FLOOR
from pilot0.provenance import is_fake, provenance
from pilot0.seam.registry import variant_semantics
from pilot0.serialize import to_jsonable

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "speech"
CACHE = ROOT / "data" / "cache" / "speech"
REPORTS = ROOT / "reports"

ALL_CANDIDATES = [FLOOR, *CANDIDATES]

CAVEATS = [
    # the two Gate-1 caveats, verbatim from tools/job_b_run.py - they bound this run too
    "winner NOT confirmed on a held-out split (confirmation split unimplemented) - do not declare Gate 1 passed from this run alone",
    "content-controlled corpus: train/test speakers read the same VCTK passages (utt 001-~024); no content-generalization claim",
    # RQ2-combo specific
    "combo order is A-then-B by convention; degradations do not commute",
    "additivity/transfer are descriptive; no pre-registered gate applies",
]


def key_of(name: str, variant: str) -> str:
    """`name:variant`, matching reports/geometry_real.json. analyze_combos keys its own
    dicts `name/variant`; the two are joined once, in main, and never mixed after."""
    return f"{name}:{variant}"


def estimate_dict(e) -> dict:
    """Estimate carries a `clears()` method alongside its three fields; write fields."""
    return {"point": e.point, "lo": e.lo, "hi": e.hi}


def additivity_dict(res) -> dict:
    """mean_cosine is a METHOD (asdict drops it - call it), and by_cell is keyed by the
    tuple (pair, severity), so it is emitted as a list of records rather than a dict
    whose keys to_jsonable would stringify into "('noise+clip', 2)"."""
    return {
        "mean_cosine": res.mean_cosine(),
        "by_cell": [
            {
                "pair": pa.pair,
                "severity": pa.severity,
                "cosine": estimate_dict(pa.cosine),  # RAW pooled-latent basis (primary)
                "cosine_std": estimate_dict(pa.cosine_std),  # standardised basis
                # S6 dominance / recovery. cosine alone is not identifiable: check
                # cos_legs before reading alpha/beta, and cos_to_b before believing
                # a high cosine means both legs are present.
                "cos_to_a": pa.cos_to_a,
                "cos_to_b": pa.cos_to_b,
                "cos_legs": pa.cos_legs,
                "norm_ratio": pa.norm_ratio,
                "r": pa.r,
                "rel_residual": pa.rel_residual,  # DERIVED from cosine and r (F23)
                "alpha": estimate_dict(pa.alpha),
                "beta": estimate_dict(pa.beta),
                "n_groups": pa.n_groups,
                "n_sources": pa.n_sources,
                "rows_identical": pa.rows_identical,
            }
            for _cell, pa in sorted(res.by_cell.items())
        ],
    }


def transfer_dict(res) -> dict:
    """predicted_fraction comes from a Counter, so a family the probe never predicted is
    ABSENT rather than 0.0 - the two leg fractions are pinned explicitly."""
    return {
        "by_pair": [
            {
                "pair": pt.pair,
                "leg_a": pt.leg_a,
                "leg_b": pt.leg_b,
                "n": pt.n,
                "predicted_fraction": dict(pt.predicted_fraction),
                "frac_leg_a": pt.predicted_fraction.get(pt.leg_a, 0.0),
                "frac_leg_b": pt.predicted_fraction.get(pt.leg_b, 0.0),
                "frac_either_leg": pt.frac_either_leg,
            }
            for _label, pt in sorted(res.by_pair.items())
        ]
    }


def write_atomic(path: Path, payload: dict) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _f(x, width: int = 7, signed: bool = True) -> str:
    """nan (a cell with no common sources, e.g. bandlimit@sev2 at 16 kHz) prints n/a."""
    if x is None or not np.isfinite(x):
        return "n/a".rjust(width)
    return f"{x:+.3f}".rjust(width) if signed else f"{x:.3f}".rjust(width)


MID = COMBO_SEVERITIES[len(COMBO_SEVERITIES) // 2]  # the CI / std / nGrp column's severity
HEADER = (f"{'candidate':14s} {'meanCos':>7s} | {'pair':15s} "
          + " ".join(f"{f'cos@{s}':>7s}" for s in COMBO_SEVERITIES)
          + f" {f'[lo,hi]@{MID}':>15s} {f'std@{MID}':>7s} {f'nGrp@{MID}':>9s} "
          + f"| {'either':>6s} {'n':>5s}")


def print_summary(rep) -> None:
    """Raw headline scalars only: the per-cell RAW additivity cosine (+ its CI and the
    standardised-basis cosine at the middle mid severity), the common-source count, and
    the zero-shot transfer's fraction landing on either constituent leg."""
    mid = MID
    print("\n" + HEADER)
    print("-" * len(HEADER))
    for name, variant in ALL_CANDIDATES:
        add, trans = rep.additivity[f"{name}/{variant}"], rep.transfer[f"{name}/{variant}"]
        if not add.by_cell:
            print(f"{key_of(name, variant):14s} {'n/a':>7s} | no combo cells cached")
            continue
        head = f"{key_of(name, variant):14s} {_f(add.mean_cosine()):>7s}"
        for a, b in COMBO_PAIRS:
            label = combo_label(a, b)
            cells = {s: add.by_cell.get((label, s)) for s in COMBO_SEVERITIES}
            if not any(cells.values()):
                continue
            cos = " ".join(_f(None if cells[s] is None else cells[s].cosine.point) for s in COMBO_SEVERITIES)
            c_mid = cells[mid]  # CI / standardised cosine / common-source count, all at MID
            lo = "n/a".rjust(6) if c_mid is None else _f(c_mid.cosine.lo, 6)
            hi = "n/a".rjust(6) if c_mid is None else _f(c_mid.cosine.hi, 6)
            std = None if c_mid is None else c_mid.cosine_std.point
            ng = 0 if c_mid is None else c_mid.n_groups
            pt = trans.by_pair.get(label)
            either = "n/a".rjust(6) if pt is None else f"{pt.frac_either_leg:.3f}".rjust(6)
            n = 0 if pt is None else pt.n
            print(f"{head} | {label:15s} {cos} [{lo},{hi}] {_f(std)} {ng:9d} | {either} {n:5d}")
            head = f"{'':14s} {'':>7s}"  # candidate/meanCos printed once, on its first pair


def main() -> None:
    # ENCODERS carries `energy`, which is not in ALL_CANDIDATES - check both lists, or a
    # fake energy encoder would slip past the candidate-only check.
    if is_fake(ALL_CANDIDATES) or is_fake([(n, "") for n in ENCODERS]):
        raise SystemExit("ABORT: fake encoder in play - RQ2 combos are a real-latent run only")
    REPORTS.mkdir(exist_ok=True)

    man = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    print(f"manifest: {man['n_sources']} sources, {man['n_groups']} groups, {man['n_rows']} rows", flush=True)
    print(f"combo pairs: {[combo_label(a, b) for a, b in COMBO_PAIRS]} at severities {list(COMBO_SEVERITIES)}", flush=True)

    t0 = time.time()
    stats = encode_combos(CORPUS / "norm", man, ENCODERS, CACHE)
    print(f"\ncombo encode pass done in {(time.time() - t0) / 60:.1f} min", flush=True)
    for s in stats:
        print(f"  {s.name:11s} sr={s.native_sr:6d} encoded={s.encoded:7d} skipped={s.skipped:7d}", flush=True)

    t1 = time.time()
    rep = analyze_combos(man, ALL_CANDIDATES, CACHE)
    print(f"\nanalysis done in {(time.time() - t1) / 60:.1f} min", flush=True)

    by_candidate = {}
    for name, variant in ALL_CANDIDATES:
        lib_key = f"{name}/{variant}"  # analyze_combos' key -> this report's key_of()
        by_candidate[key_of(name, variant)] = {
            "key": key_of(name, variant),
            "name": name,
            "variant": variant,
            "semantics": variant_semantics(name, variant),  # what this latent IS (S2/D1/AM8)
            "additivity": additivity_dict(rep.additivity[lib_key]),
            "transfer": transfer_dict(rep.transfer[lib_key]),
        }

    payload = to_jsonable({
        "candidates": [list(c) for c in ALL_CANDIDATES],
        "encoders": list(ENCODERS),
        "combo_pairs": [list(p) for p in COMBO_PAIRS],
        "combo_severities": list(COMBO_SEVERITIES),
        # ComboEncodeStats is plain fields only - asdict is safe here, unlike the results.
        "combo_encode": [dataclasses.asdict(s) for s in stats],
        "by_candidate": by_candidate,
        "caveats": CAVEATS,
        "provenance": provenance(is_fake(ALL_CANDIDATES), corpus="VCTK-0.92 mic1 510x102spk", stage="rq2_combos"),
    })
    out = write_atomic(REPORTS / "combos_real.json", payload)
    print(f"\nwrote {out}", flush=True)

    print_summary(rep)
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

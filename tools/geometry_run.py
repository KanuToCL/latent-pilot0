# RQ2 geometry: descriptive analysis of degradation DIRECTIONS in latent space.
# Reads the Phase-3 latent cache ONLY (build_probe_data walks it; nothing here
# encodes), scores every candidate on the SAME common cells the gate used, and
# writes reports/geometry_real.json + reports/figures/geometry/*.png.
#
# Per candidate: analysis.geometry (probe-weight cosines, class-mean directions,
# centroid PCA) and analysis.shift (per-clip shift vectors -> concentration vs the
# 1/sqrt(D) null, magnitude-vs-severity, persistence under PCA reduction).
#
# Usage:  geometry_run.py                      all 17 candidates + the log-mel floor
#         geometry_run.py logmel:mel encodec24k:z    just those two
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from job_b_run import CANDIDATES  # single source of truth for the 17 pre-registered pairs
from pilot0.analysis import shift
from pilot0.analysis.geometry import geometry
from pilot0.corpus.manifest import renderable_rows
from pilot0.probes.dataset import CLEAN, build_probe_data
from pilot0.probes.run import ENERGY, FLOOR, common_conditions
from pilot0.provenance import is_fake, provenance
from pilot0.seam.registry import candidate_semantics, make_encoder
from pilot0.serialize import to_jsonable

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "speech"
CACHE = ROOT / "data" / "cache" / "speech"
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures" / "geometry"

ALL_CANDIDATES = [FLOOR, *CANDIDATES]
# The cell grid is fixed by the FULL gate name set (M1), never by the CLI filter —
# a two-candidate run must land on the same 34 cells as the 17-candidate run.
GRID_NAMES = [FLOOR[0], ENERGY[0], *(n for n, _ in CANDIDATES)]
FIGURE_KEYS = ("logmel:mel", "encodec24k:z", "wavlm:l1", "dac44k:z", "mimi:semantic")
KS = (2, 8, 32)

CAVEATS = [
    # the two Gate-1 caveats, verbatim from tools/job_b_run.py — they bound this run too
    "winner NOT confirmed on a held-out split (confirmation split unimplemented) - do not declare Gate 1 passed from this run alone",
    "content-controlled corpus: train/test speakers read the same VCTK passages (utt 001-~024); no content-generalization claim",
    # RQ2-specific
    "descriptive and IN-SAMPLE: every statistic here is fit on all rows (train+test) and gates nothing; no CI, no held-out check",
    "all cosines live in the standardised basis (one scaler fit on the degraded rows), so 'near-orthogonal' is a claim about that basis, not the raw latent geometry",
    "concentration is compared against the isotropic null 1/sqrt(D) only - not a permutation test against the corpus's own content structure",
    "||delta|| is in each candidate's OWN standardised units and grows with D - comparable across families/severities within a candidate, never across candidates",
]


def key_of(name: str, variant: str) -> str:
    return f"{name}:{variant}"


def parse_candidates(argv: list[str]):
    if not argv:
        return list(ALL_CANDIDATES)
    index = {key_of(n, v): (n, v) for n, v in ALL_CANDIDATES}
    picked = []
    for arg in argv:
        if arg not in index:
            raise SystemExit(f"unknown candidate '{arg}'\nchoose from: {' '.join(index)}")
        picked.append(index[arg])
    return picked


def assert_gate_grid(n_common: int) -> int | None:
    """Pin this run's cell grid to the Gate-1 run's, if that report exists."""
    gate = REPORTS / "gate1_real.json"
    if not gate.exists():
        return None
    n_gate = int(json.loads(gate.read_text(encoding="utf-8"))["report"]["n_common_conditions"])
    if n_gate != n_common:
        raise SystemExit(f"ABORT: common cells {n_common} != gate1 {n_gate} - grids differ")
    return n_gate


def assemble(manifest: dict, name: str, variant: str, common):
    """Pooled features from the cache + the per-row SOURCE id the shift vectors need.

    ProbeData carries no source column, so it is re-derived from the same
    `renderable_rows` walk `build_probe_data` uses; the family/severity arrays are
    compared element-wise first, so a cache miss or an order change fails loud here
    instead of silently mispairing a clip with another clip's clean anchor."""
    data = build_probe_data(manifest, name, variant, CACHE)
    rows = renderable_rows(manifest, make_encoder(name).native_sr)
    fam = np.asarray([r["family"] for r in rows])
    sev = np.asarray([r["severity"] for r in rows], dtype=int)
    if len(rows) != len(data.family) or not (
        np.array_equal(fam, data.family) and np.array_equal(sev, data.severity)
    ):
        raise RuntimeError(f"{key_of(name, variant)}: cache walk does not match renderable_rows order")
    source = np.asarray([r["source"] for r in rows])
    keep = np.array([(f, int(s)) in common for f, s in zip(data.family, data.severity)])
    return data.select(common), source[keep]


def analyze_candidate(manifest: dict, name: str, variant: str, common) -> dict:
    t0 = time.time()
    data, source = assemble(manifest, name, variant, common)
    t_load = time.time() - t0

    t0 = time.time()
    geo = geometry(data.X, data.family, data.severity)
    t_geo = time.time() - t0

    t0 = time.time()
    Xs = shift.standardize_on_degraded(data.X, data.family)
    sv = shift.shift_vectors(Xs, data.family, data.severity, source)
    conc_overall = shift.concentration(sv.deltas)
    conc_family = shift.family_concentration(sv.deltas, sv.family)
    paired_labels, paired_cosine = shift.mean_direction_cosines(sv.deltas, sv.family)
    mag = shift.magnitude_curve(sv.deltas, sv.family, sv.severity)
    reduction = shift.reduction_persistence(Xs[data.family != CLEAN], sv.deltas, sv.family, ks=KS)
    t_shift = time.time() - t0

    n_features = int(data.X.shape[1])
    return to_jsonable({
        "key": key_of(name, variant),
        "name": name,
        "variant": variant,
        "n_features": n_features,  # pooled mean+std -> 2x the encoder's latent_dim
        "null_scale": 1.0 / np.sqrt(n_features),
        "counts": {
            "rows": int(len(data.family)),
            "clean_rows": int((data.family == CLEAN).sum()),
            "degraded_rows": sv.n_degraded,
            "deltas": int(sv.deltas.shape[0]),
            "unpaired_rows": sv.n_unpaired_rows,
            "unpaired_sources": sv.n_unpaired_sources,
            "sources": int(len(set(source.tolist()))),
            "test_groups": data.n_test_groups(),
        },
        "geometry": {
            "labels": geo.labels,
            "probe_cosine": geo.probe_cosine,
            "mean_cosine": geo.mean_cosine,
            "centroid_pca_explained": geo.centroid_pca_explained,
            "centroid_labels": geo.centroid_labels,
            "centroid_coords": geo.centroid_coords,
        },
        "shift": {
            "concentration": {
                "overall": dataclasses.asdict(conc_overall),
                "by_family": {f: dataclasses.asdict(c) for f, c in conc_family.items()},
                "paired_labels": paired_labels,
                "paired_mean_cosine": paired_cosine,
            },
            "magnitude": dataclasses.asdict(mag),
            "reduction": dataclasses.asdict(reduction),
        },
        "timings_s": {
            "cache_load": t_load, "geometry": t_geo, "shift": t_shift,
            "total": t_load + t_geo + t_shift,
        },
    })


def _conc_line(label: str, cc: dict, extra: str = "") -> str:
    """The concentration C itself, and nothing derived from the null.

    S4: `C / (1/sqrt(D))` was printed as "x N null" and read as a score. It is not one
    - 1/sqrt(D) is the RMS cosine of a single random PAIR in D dimensions, so the ratio
    is a function of the nominal dimension and inflates with duplicated or dead
    channels. The reference is still printed once, above, as a reference.

    nan -> null under to_jsonable, so a degenerate cell prints 'n/a', never crashes."""
    if cc["mean_cosine"] is None:
        return f"    {label:11s}    n/a{extra}"
    return f"    {label:11s} {cc['mean_cosine']:+.4f}   n={cc['n_rows']}{extra}"


def print_candidate(block: dict) -> None:
    c, t = block["counts"], block["timings_s"]
    print(f"\n{block['key']}  D={block['n_features']}  rows={c['rows']} deltas={c['deltas']} "
          f"unpaired={c['unpaired_rows']}  ({t['cache_load']:.0f}s load, {t['geometry']:.0f}s geom, "
          f"{t['shift']:.0f}s shift)", flush=True)
    null = block["null_scale"]
    conc = block["shift"]["concentration"]
    rho = block["shift"]["magnitude"]["srcc_by_family"]
    print(f"  delta concentration C (mean pairwise cosine)   "
          f"single-pair RMS null 1/sqrt(D) = {null:.4f} (reference, not a score)")
    print(_conc_line("pooled", conc["overall"]))
    for fam, cc in conc["by_family"].items():
        r = rho.get(fam)
        print(_conc_line(fam, cc, f"   srcc(sev,||d||)={'n/a' if r is None else f'{r:+.2f}'}"))
    red = block["shift"]["reduction"]
    ks = ", ".join(f"k={k}: {v['cumulative_explained']:.2f}" for k, v in sorted(red["by_k"].items(), key=lambda kv: int(kv[0])))
    print(f"  PCA cumulative explained variance of the degraded rows -> {ks}")


def write_atomic(path: Path, payload: dict) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def render_figures(blocks: dict) -> list[Path]:
    """Only the five headline candidates get figures, and only if this run covered
    them - a filtered run skips the rest instead of failing."""
    keys = [k for k in FIGURE_KEYS if k in blocks]
    if not keys:
        return []
    import geometry_figures  # lazy: a missing matplotlib must not cost us the JSON

    skipped = [k for k in FIGURE_KEYS if k not in blocks]
    if skipped:
        print(f"figures skipped (not in this run): {' '.join(skipped)}", flush=True)
    out = []
    for k in keys:
        out.extend(geometry_figures.render_candidate_figures(blocks[k], FIG_DIR))
    return out


def main(argv: list[str] | None = None) -> None:
    run = parse_candidates(list(sys.argv[1:] if argv is None else argv))
    if is_fake(ALL_CANDIDATES):
        raise SystemExit("ABORT: fake encoder in the candidate list - RQ2 geometry reads real cached latents only")
    REPORTS.mkdir(exist_ok=True)

    man = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    common = common_conditions(GRID_NAMES)
    n_gate = assert_gate_grid(len(common))
    print(f"manifest: {man['n_sources']} sources, {man['n_groups']} groups, {man['n_rows']} rows")
    print(f"common cells: {len(common)} (gate1: {n_gate})   candidates this run: {len(run)}/{len(ALL_CANDIDATES)}", flush=True)

    t_start = time.time()
    blocks: dict[str, dict] = {}
    for name, variant in run:
        blocks[key_of(name, variant)] = analyze_candidate(man, name, variant, common)
        print_candidate(blocks[key_of(name, variant)])

    payload = {
        "candidates": [list(c) for c in run],
        "all_candidates": [list(c) for c in ALL_CANDIDATES],
        "candidate_semantics": candidate_semantics(ALL_CANDIDATES),  # what each variant IS (S2/D1)
        "n_common_conditions": len(common),
        "reduction_ks": list(KS),
        "n_features": {k: b["n_features"] for k, b in blocks.items()},
        "null_scale": {k: b["null_scale"] for k, b in blocks.items()},
        "by_candidate": blocks,
        "caveats": CAVEATS,
        "provenance": provenance(is_fake(run), corpus="VCTK-0.92 mic1 510x102spk", stage="rq2_geometry"),
    }
    out = write_atomic(REPORTS / "geometry_real.json", to_jsonable(payload))
    print(f"\nwrote {out}", flush=True)

    figs = render_figures(blocks)
    for p in figs:
        print(f"  figure {p}")
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

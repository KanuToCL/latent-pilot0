# Job B — first real mini-sweep: encode VCTK corpus (resume-safe) + Gate 1.
# Post-audit: decisions serialized explicitly (asdict drops @property), severity
# printed per decision, provenance.is_fake reused, ROOT from __file__, main guard.
import dataclasses
import json
import time
from pathlib import Path

from pilot0.encode.pipeline import encode_corpus
from pilot0.probes.gate1 import MIN_TEST_GROUPS
from pilot0.probes.run import run_gate1
from pilot0.provenance import is_fake, provenance
from pilot0.serialize import to_jsonable

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "speech"
CACHE = ROOT / "data" / "cache" / "speech"
REPORTS = ROOT / "reports"

ENCODERS = ["logmel", "energy", "encodec24k", "wavlm", "dac44k", "mimi"]
CANDIDATES = (
    [("encodec24k", v) for v in ("z", "d1", "d2", "d4", "d8")]
    + [("wavlm", v) for v in ("l1", "l6", "l12", "l18", "l24")]
    + [("dac44k", v) for v in ("z", "d1", "d2", "d4", "d8")]
    + [("mimi", v) for v in ("semantic", "acoustic")]
)


def decision_dict(dec) -> dict:
    """Explicit: @property verdicts (passed, margins) vanish under asdict."""
    return {
        **dataclasses.asdict(dec),
        "passed": dec.passed,
        "underpowered": dec.underpowered,
        "margin_over_floor": dec.margin_over_floor,
        "n_severity_pass": dec.n_severity_pass,
    }


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    man = json.loads((CORPUS / "manifest.json").read_text())
    print(f"manifest: {man['n_sources']} sources, {man['n_groups']} groups, {man['n_rows']} rows", flush=True)

    t0 = time.time()
    enc_report = encode_corpus(CORPUS / "norm", man, ENCODERS, CACHE)
    print(f"\nencode done in {(time.time() - t0) / 3600:.2f} h", flush=True)
    for s in enc_report.per_encoder:
        print(" ", to_jsonable(dataclasses.asdict(s)), flush=True)

    t1 = time.time()
    g1 = run_gate1(man, CANDIDATES, CACHE)
    print(f"gate1 done in {(time.time() - t1) / 60:.1f} min", flush=True)

    payload = to_jsonable({
        "report": {
            "floor": dataclasses.asdict(g1.floor),
            "energy": dataclasses.asdict(g1.energy),
            "decisions": [decision_dict(d) for d in g1.decisions],
            "n_common_conditions": g1.n_common_conditions,
        },
        "candidates": CANDIDATES,
        "min_test_groups": MIN_TEST_GROUPS,
        "caveats": [
            "winner NOT confirmed on a held-out split (confirmation split unimplemented) - do not declare Gate 1 passed from this run alone",
            "content-controlled corpus: train/test speakers read the same VCTK passages (utt 001-~024); no content-generalization claim",
        ],
        "provenance": provenance(is_fake(CANDIDATES), corpus="VCTK-0.92 mic1 510x102spk", stage="job_b_mini_sweep"),
    })
    out = REPORTS / "gate1_real.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}", flush=True)

    f = g1.floor
    print(f"\ncommon cells: {g1.n_common_conditions}   n_test_groups: {f.n_test_groups} (need >={MIN_TEST_GROUPS})")
    print(f"floor macro-F1 {f.type.macro_f1.point:.3f}")
    for d in g1.decisions:
        tag = "PASS" if d.passed else ("underpowered" if d.underpowered else "fail")
        print(f"  {d.name:11s} {d.variant:9s} {tag:12s} marginFloor={d.margin_over_floor:+.3f} sevPass={d.n_severity_pass}")


if __name__ == "__main__":
    main()

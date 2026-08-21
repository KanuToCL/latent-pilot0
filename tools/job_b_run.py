# Job B — first real mini-sweep: encode VCTK corpus (resume-safe) + Gate 1.
import dataclasses
import json
import time
from pathlib import Path

from pilot0.encode.pipeline import encode_corpus
from pilot0.probes.gate1 import MIN_TEST_GROUPS
from pilot0.probes.run import run_gate1
from pilot0.provenance import provenance
from pilot0.serialize import to_jsonable

ROOT = Path(r"D:\Cosas\04_Coding\MONOREPOS\latent-pilot0")
CORPUS = ROOT / "data" / "corpus" / "speech"
CACHE = ROOT / "data" / "cache" / "speech"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

man = json.loads((CORPUS / "manifest.json").read_text())
print(f"manifest: {man['n_sources']} sources, {man['n_groups']} groups, {man['n_rows']} rows", flush=True)

ENCODERS = ["logmel", "energy", "encodec24k", "wavlm", "dac44k", "mimi"]
CANDIDATES = (
    [("encodec24k", v) for v in ("z", "d1", "d2", "d4", "d8")]
    + [("wavlm", v) for v in ("l1", "l6", "l12", "l18", "l24")]
    + [("dac44k", v) for v in ("z", "d1", "d2", "d4", "d8")]
    + [("mimi", v) for v in ("semantic", "acoustic")]
)

t0 = time.time()
enc_report = encode_corpus(CORPUS / "norm", man, ENCODERS, CACHE)
print(f"\nencode done in {(time.time() - t0) / 3600:.2f} h", flush=True)
for s in enc_report.per_encoder:
    print(" ", to_jsonable(dataclasses.asdict(s)), flush=True)

t1 = time.time()
g1 = run_gate1(man, CANDIDATES, CACHE)
print(f"gate1 done in {(time.time() - t1) / 60:.1f} min", flush=True)

fake = any(n.startswith("fake-") for n, _ in CANDIDATES)  # False -> real provenance
payload = to_jsonable({
    "report": dataclasses.asdict(g1),
    "candidates": CANDIDATES,
    "min_test_groups": MIN_TEST_GROUPS,
    "provenance": provenance(fake, corpus="VCTK-0.92 mic1 510x102spk", stage="job_b_mini_sweep"),
})
out = REPORTS / "gate1_real.json"
out.write_text(json.dumps(payload, indent=2))
print(f"\nwrote {out}", flush=True)

f = g1.floor
print(f"\ncommon cells: {g1.n_common_conditions}   n_test_groups: {f.n_test_groups} (need >={MIN_TEST_GROUPS})")
print(f"floor macro-F1 {f.type.macro_f1.point:.3f}   energy sev pooled {g1.energy.severity.pooled.without_clean.point:+.3f}")
for d in g1.decisions:
    dd = dataclasses.asdict(d)
    status = "PASS" if dd.get("passed") else "fail"
    print(f"  {d.name:11s} {d.variant:9s} {status}  marginFloor={dd.get('margin_over_floor')}")

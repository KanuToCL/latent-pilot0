# GPU_BRINGUP §5 — preflight the staged VCTK wavs with a speaker group_fn,
# build the grouped manifest (fails closed if grouping broke), and print the
# clips-per-group distribution for the mandatory eyeball.
from collections import Counter
from pathlib import Path

from pilot0.corpus.preflight import preflight
from pilot0.corpus.manifest import build_manifest, write_manifest

STAGED = Path(r"D:\Cosas\04_Coding\MONOREPOS\latent-pilot0\data\staged\speech")
OUT = Path(r"D:\Cosas\04_Coding\MONOREPOS\latent-pilot0\data\corpus\speech")

# VCTK: p225_003_mic1.wav -> speaker "p225"
pf = preflight(STAGED, OUT, group_fn=lambda p: p.stem.split("_")[0])

print(f"accepted: {len(pf.accepted)}   rejected: {len(pf.rejected)}")
if pf.rejected:
    reasons = Counter(r for _, r in pf.rejected)
    for reason, n in reasons.most_common():
        print(f"  rejected [{n:3d}] {reason}")

counts = Counter(rec.group_id for rec in pf.accepted)
vals = sorted(counts.values())
mean = sum(vals) / len(vals)
print(f"\nclips-per-group: n_groups={len(counts)}  mean={mean:.2f}  "
      f"min={vals[0]}  max={vals[-1]}")
print(f"group examples: {dict(list(counts.items())[:5])}")
if mean < 1.5:
    raise SystemExit("EYEBALL FAIL: mean clips/group ~1 — group_fn is wrong, Gate 1 would be invalid")

man = build_manifest(pf, assert_grouped=True)
write_manifest(man, OUT / "manifest.json")

splits = Counter(row["split"] for row in man["rows"]) if "rows" in man else None
print(f"\nmanifest written -> {OUT / 'manifest.json'}")
if splits:
    print(f"rows per split: {dict(splits)}")
print("keys:", sorted(man.keys()))

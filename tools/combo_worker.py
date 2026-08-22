# Cache-warmer: encode ONE encoder's full COMBO grid (A-then-B cells) into the SAME
# Phase-3 cache as the singles. Safe to run in parallel with other combo workers —
# cache paths are disjoint per encoder.
#
# Row-sharding is deliberately NOT offered, same as encode_worker.py: headroom must
# stay corpus-global. Combos go further and REUSE the headroom scalar already banked
# by the singles pass (headroom/sr_*.json), so that z-bar(a+b) is comparable to
# z-bar(a), z-bar(b) and z-bar(clean) in the additivity cosine. resolve_headroom
# returns early on the existing JSON — this worker must never trigger a rescan, and a
# row shard would be exactly the way to make it do so.
import json
import sys
import time
from pathlib import Path

from pilot0.combos.encode import encode_combos

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "speech"
CACHE = ROOT / "data" / "cache" / "speech"


def main() -> None:
    name = sys.argv[1]
    man = json.loads((CORPUS / "manifest.json").read_text())
    t0 = time.time()
    print(f"[{name}] combos start", flush=True)
    stats = encode_combos(CORPUS / "norm", man, [name], CACHE)
    s = stats[0]
    print(f"[{name}] combos done in {(time.time() - t0) / 60:.1f} min  "
          f"sr={s.native_sr} encoded={s.encoded} skipped={s.skipped}", flush=True)


if __name__ == "__main__":
    main()

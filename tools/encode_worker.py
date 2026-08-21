# Cache-warmer: encode ONE encoder's full corpus. Safe to run in parallel with
# other workers — cache paths are disjoint per encoder, and the shared headroom
# scan is deterministic + atomically written, so concurrent duplicates converge.
# Row-sharding is deliberately NOT offered: headroom must stay corpus-global.
import json
import sys
import time
from pathlib import Path

from pilot0.encode.pipeline import encode_corpus

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "speech"
CACHE = ROOT / "data" / "cache" / "speech"


def main() -> None:
    name = sys.argv[1]
    man = json.loads((CORPUS / "manifest.json").read_text())
    t0 = time.time()
    print(f"[{name}] start", flush=True)
    rep = encode_corpus(CORPUS / "norm", man, [name], CACHE)
    s = rep.per_encoder[0]
    print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min  encoded={s.encoded} skipped={s.skipped}", flush=True)


if __name__ == "__main__":
    main()

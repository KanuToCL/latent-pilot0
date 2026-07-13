"""Phase-3 demo: synth corpus → manifest → fake-encode → cache summary, then a
second pass to prove resume (`make encode-demo`)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..audio.synth import synth_clip
from ..corpus.io import write_audio
from ..corpus.manifest import build_manifest
from ..corpus.preflight import preflight
from .pipeline import encode_corpus

ENCODERS = ["fake-encodec24k", "fake-wavlm"]


def main(n_sources: int = 6, sr: int = 48000) -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        for i in range(n_sources):
            write_audio(src / f"clip{i}.wav", synth_clip(i, sr=sr), sr)

        pf = preflight(src, Path(d) / "corpus")
        man = build_manifest(pf)
        cache = Path(d) / "cache"

        report = encode_corpus(pf.norm_dir, man, ENCODERS, cache)
        resume = encode_corpus(pf.norm_dir, man, ENCODERS, cache)

    print(f"code_version {report.code_version}")
    for s in report.per_encoder:
        h = s.headroom
        print(
            f"  {s.name:16} sr={s.native_sr:5} rows={s.n_rows:3} × {s.n_variants} var"
            f"  headroom×{h.scalar:.3f} (peak {h.max_dbtp:+.1f}→{h.ceiling_dbfs:+.0f} dBFS)"
            f"  encoded={s.encoded} skipped={s.skipped}"
        )
    print(f"total latents cached: {report.n_latents}")

    resumed = sum(s.encoded for s in resume.per_encoder)
    print(f"resume pass re-encoded {resumed} (want 0 — all served from cache)")


if __name__ == "__main__":
    main()

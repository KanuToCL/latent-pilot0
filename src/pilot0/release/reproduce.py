"""`make reproduce-figures`: from a clean checkout, build one synthetic corpus, write
the F1–F6 JSON artifacts, and render the six figures under reports/. Single corpus →
mutually consistent figures.

BANNER: everything here is FAKE (fake codec latents, synthetic ViSQOL/MOS, fabricated
NR incumbents) — a plumbing check that the paper pipeline runs end to end and every
figure renders. reports/provenance.json records `fake: true`; each PNG stamps the
banner. On the GPU box the real backends replace the fake encoder in `build_demo_corpus`
and the same command emits the scientific figures.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..quality.scores import FakeScores
from .artifacts import BANNER, CANDIDATES, build_demo_corpus, write_artifacts
from .figures import render_figures


def main(out_dir: str = "reports", fig_dir: str | None = None,
         n_sources: int = 16, sr: int = 48000) -> None:
    out = Path(out_dir)
    figs = Path(fig_dir) if fig_dir else out / "figures"
    with tempfile.TemporaryDirectory() as d:
        norm_dir, man, cache = build_demo_corpus(Path(d), n_sources, sr)
        artifacts = write_artifacts(norm_dir, man, CANDIDATES, cache, FakeScores(),
                                    out, n_sources=n_sources, sr=sr)
    figures = render_figures(out, figs)

    print(BANNER + "\n")
    print(f"artifacts → {out}/    figures → {figs}/\n")
    for name, path in {**artifacts, **figures}.items():
        print(f"  {name:12s} {path}")


if __name__ == "__main__":
    main()

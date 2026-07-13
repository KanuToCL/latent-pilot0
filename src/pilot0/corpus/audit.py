"""Phase-2 demo: synth corpus → preflight → manifest → summary (`make manifest-demo`)."""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

from ..audio.synth import synth_clip
from .io import write_audio
from .manifest import build_manifest
from .preflight import preflight


def main(n_sources: int = 6, sr: int = 48000) -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        for i in range(n_sources):
            write_audio(src / f"clip{i}.wav", synth_clip(i, sr=sr), sr)

        pf = preflight(src, Path(d) / "corpus")
        man = build_manifest(pf)

    print(f"preflight: {len(pf.accepted)} accepted, {len(pf.rejected)} rejected")
    print(f"manifest:  {man['n_sources']} sources ({man['n_groups']} groups) × "
          f"{man['n_conditions']} conditions = {man['n_rows']} rows")
    group_split = {r["group"]: r["split"] for r in man["rows"]}
    print("groups per split:", dict(Counter(group_split.values())))
    print(f"grid {man['grid_signature']}  test_frac {man['test_frac']}  {man['split_algo']}")


if __name__ == "__main__":
    main()

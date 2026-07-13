"""Measured-vs-target audit across the degradation grid (`make test-degrade`)."""

from __future__ import annotations

from ..audio.synth import synth_clip
from .grid import FAMILIES, apply_degradation
from .mp3 import has_ffmpeg


def main(sr: int = 48000) -> None:
    wav = synth_clip(0, sr=sr)
    print(f"degradation grid — measured vs target (synthetic clip, {sr} Hz)\n")
    print(f"{'family':10} {'sev':>3} {'target':>12} {'measured':>34} {'peak dBTP':>10}")
    for name, fam in FAMILIES.items():
        if name == "mp3" and not has_ffmpeg():
            print(f"{name:10}  (skipped — ffmpeg not found)")
            continue
        for s in (1, 2, 3, 4, 5):
            r = apply_degradation(wav, sr, name, s)
            target = f"{fam.param_name}={r.param:g}{fam.unit}"
            meas = ", ".join(
                f"{k}={v:.2f}" for k, v in r.measured.items() if k != "true_peak_dbtp"
            )
            print(f"{name:10} {s:>3} {target:>12} {meas:>34} {r.measured['true_peak_dbtp']:>10.2f}")
    print("\n✓ audit complete")


if __name__ == "__main__":
    main()

"""Phase-0 smoke: encode N synthetic clips through the fake stand-ins for the
representations the GPU box will really run, across all their variants, and log
the latent shapes. Acceptance (§3 Phase 0): shapes logged for 10 clips.

On the box, point SMOKE_ENCODERS at the bare names and the same code validates
the real backends.
"""

from __future__ import annotations

import json
from pathlib import Path

from .audio.synth import synth_batch
from .seam.registry import available_encoders, make_encoder

SMOKE_ENCODERS = ["fake-encodec24k", "fake-wavlm"]


def run(n: int = 10, sr: int = 48000, seconds: float = 4.0) -> list[dict]:
    """Encode `n` clips through each smoke encoder; one row per (clip, encoder, variant)."""
    clips = synth_batch(n=n, sr=sr, seconds=seconds)
    encoders = {name: make_encoder(name) for name in SMOKE_ENCODERS}
    rows: list[dict] = []
    for i, wav in enumerate(clips):
        for name, enc in encoders.items():
            for variant, r in enc.encode(wav, sr).items():
                rows.append(
                    {
                        "clip": i,
                        "encoder": name,
                        "variant": variant,
                        "native_sr": enc.native_sr,
                        "frames": list(r.frames.shape),
                        "frame_rate_hz": round(r.frame_rate_hz, 2),
                        "n_frames": r.n_frames,
                    }
                )
    return rows


def main() -> None:
    print("encoder availability:")
    for name, ok in available_encoders().items():
        print(f"  {name:16s} {'yes' if ok else 'no  (needs .[gpu] on the GPU box; dac/mimi Phase 5)'}")

    rows = run()
    print(f"\nencoded {len(rows)} (clip x encoder x variant) latents:")
    hdr = f"{'clip':>4}  {'encoder':16}  {'var':>4}  {'native_sr':>9}  {'frames [T,D]':>14}  {'fps':>6}"
    print(hdr)
    for r in rows:
        print(
            f"{r['clip']:>4}  {r['encoder']:16}  {r['variant']:>4}  {r['native_sr']:>9}  "
            f"{str(r['frames']):>14}  {r['frame_rate_hz']:>6}"
        )

    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "smoke.json").write_text(json.dumps(rows, indent=2))

    n_clips = len({r["clip"] for r in rows})
    assert n_clips == 10, f"expected 10 clips, got {n_clips}"
    print(f"\n✓ smoke ok — {n_clips} clips through {len(SMOKE_ENCODERS)} encoders "
          f"→ reports/smoke.json")


if __name__ == "__main__":
    main()

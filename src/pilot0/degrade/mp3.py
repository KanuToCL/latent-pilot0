"""Low-bitrate MP3 transcode via ffmpeg/libmp3lame.

Encode → decode → cross-correlation-align back to the clean and crop to the
original length. Alignment matters: LAME adds encoder/decoder delay and pads
length, which would otherwise corrupt reference metrics and frame accounting
(physics elder). CBR is pinned for reproducibility; the LAME/ffmpeg version is
recorded in the manifest at Phase 2."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

from .base import to_float32


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _align_to(dec: np.ndarray, ref: np.ndarray) -> np.ndarray:
    from scipy.signal import correlate

    n = len(ref)
    win = min(len(dec), n, 1 << 15)
    a = dec[:win] - dec[:win].mean()
    b = ref[:win] - ref[:win].mean()
    lag = int(np.argmax(correlate(a, b, mode="full"))) - (win - 1)
    if lag > 0:
        dec = dec[lag:]
    elif lag < 0:
        dec = np.concatenate([np.zeros(-lag), dec])
    if len(dec) < n:
        dec = np.concatenate([dec, np.zeros(n - len(dec))])
    return dec[:n]


def mp3_transcode(wav: np.ndarray, sr: int, kbps: int) -> np.ndarray:
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not found on PATH; MP3 family unavailable")
    import soundfile as sf

    wav = np.asarray(wav, dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        inp, comp, out = (os.path.join(d, f) for f in ("in.wav", "a.mp3", "out.wav"))
        sf.write(inp, wav, sr, subtype="FLOAT")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", inp, "-b:a", f"{kbps}k", comp],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", comp, "-ar", str(sr), out],
            check=True,
        )
        dec, _ = sf.read(out, dtype="float32")
    if dec.ndim > 1:
        dec = dec.mean(axis=1)
    return to_float32(_align_to(dec.astype(np.float64), wav.astype(np.float64)))

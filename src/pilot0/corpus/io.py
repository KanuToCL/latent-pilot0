"""Audio IO for the corpus (float32, soundfile)."""

from __future__ import annotations

import numpy as np


def read_audio(path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return wav, int(sr)


def to_mono(wav: np.ndarray) -> np.ndarray:
    return wav.mean(axis=1) if wav.ndim > 1 else wav


def write_audio(path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), np.asarray(wav, dtype=np.float32), sr, subtype="FLOAT")

"""Minimal numpy DSP for the fake backend and smoke path — no scipy, no torch.

These are stand-in transforms so the pipeline runs on the Mac. They make NO
perceptual or fidelity claim. The real backend uses the models' own front-ends,
and Phase-1 degradations use a proper anti-aliased resampler (soxr) — see
docs/DECISIONS.md D4.
"""

from __future__ import annotations

import numpy as np


def resample_linear(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Crude linear-interpolation resample (NOT anti-aliased).

    Adequate only for the fake encoder's shape/plumbing role: it changes the
    frame count with rate so the seam exercises per-rate behaviour.
    """
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if sr == target_sr or wav.size == 0:
        return wav
    n_out = int(round(wav.size * target_sr / sr))
    if n_out <= 1:
        return wav[: max(n_out, 1)]
    x_old = np.linspace(0.0, 1.0, num=wav.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, wav)


def stft_logmag(wav: np.ndarray, n_fft: int = 512, hop: int = 256) -> np.ndarray:
    """Log-magnitude STFT → [T, F] where F = n_fft // 2 + 1. Hann-windowed."""
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if wav.size < n_fft:
        wav = np.pad(wav, (0, n_fft - wav.size))
    win = np.hanning(n_fft)
    n_frames = 1 + (wav.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = wav[idx] * win[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1))
    return np.log1p(mag)

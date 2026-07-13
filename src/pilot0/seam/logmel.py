"""LogMelEncoder — the non-neural baseline representation (the "floor").

Not a fake stand-in: this is a real log-mel spectrogram (numpy DSP, anti-aliased
soxr resample), identical on the Mac and the GPU box. §2.3 pre-registers the floor
as "log-mel (+Δ, mean/std)", so we emit static + Δ + ΔΔ — a stronger floor makes
the ≥5-F1 Gate-1 margin conservative (a too-weak floor would bias the gate toward
GO). Gate 1's floor numbers are meaningful even on the Mac; only the neural-codec
numbers wait for bring-up.
"""

from __future__ import annotations

import numpy as np

from .base import LatentResult, measure_frame_rate


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    # HTK mel scale (O'Shaughnessy 1987; HTK Book, Young et al.).
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """[n_mels, n_fft//2+1] triangular mel filters spanning [0, Nyquist]."""
    n_freq = n_fft // 2 + 1
    fft_hz = np.linspace(0.0, sr / 2.0, n_freq)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(sr / 2.0)), n_mels + 2))
    fb = np.zeros((n_mels, n_freq))
    for m in range(1, n_mels + 1):
        lo, ctr, hi = edges[m - 1], edges[m], edges[m + 1]
        rising = (fft_hz - lo) / max(ctr - lo, 1e-9)
        falling = (hi - fft_hz) / max(hi - ctr, 1e-9)
        fb[m - 1] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


def _delta(x: np.ndarray) -> np.ndarray:
    """First difference along time (edge-padded) — a cheap approximation to the
    canonical windowed-regression MFCC Δ (Furui 1986); adequate for a floor."""
    return np.diff(x, axis=0, prepend=x[:1])


class LogMelEncoder:
    def __init__(self, name: str = "logmel", native_sr: int = 16000, n_mels: int = 64,
                 n_fft: int = 400, hop: int = 160, deltas: int = 2) -> None:
        self.name = name
        self.native_sr = int(native_sr)
        self._n_mels = int(n_mels)
        self._deltas = int(deltas)
        self.latent_dim = int(n_mels) * (1 + self._deltas)  # static (+Δ +ΔΔ), §2.3
        self.variants = ("mel",)
        self._n_fft = int(n_fft)
        self._hop = int(hop)
        self._win = np.hanning(n_fft)
        self._fb = mel_filterbank(n_mels, n_fft, native_sr)

    def encode(self, wav: np.ndarray, sr: int) -> dict[str, LatentResult]:
        wav = np.asarray(wav, dtype=np.float64).reshape(-1)
        if sr != self.native_sr:
            import soxr

            wav = soxr.resample(wav, sr, self.native_sr)
        if wav.size < self._n_fft:
            wav = np.pad(wav, (0, self._n_fft - wav.size))
        n_frames = 1 + (wav.size - self._n_fft) // self._hop
        idx = np.arange(self._n_fft)[None, :] + self._hop * np.arange(n_frames)[:, None]
        power = np.abs(np.fft.rfft(wav[idx] * self._win[None, :], axis=1)) ** 2
        logmel = np.log(power @ self._fb.T + 1e-10)  # [T, n_mels]

        feats = [logmel]
        for _ in range(self._deltas):
            feats.append(_delta(feats[-1]))
        frames = np.concatenate(feats, axis=1)  # [T, n_mels*(1+deltas)]

        fr = measure_frame_rate(n_frames, wav.size, self.native_sr)
        meta = {
            "backend": "logmel",
            "name": self.name,
            "variant": "mel",
            "native_sr": self.native_sr,
            "latent_dim": self.latent_dim,
            "n_frames": int(n_frames),
            "hop": self._hop,
            "deltas": self._deltas,
        }
        return {"mel": LatentResult(frames=frames, frame_rate_hz=fr, meta=meta)}

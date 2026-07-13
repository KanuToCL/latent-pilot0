"""FakeEncoder — deterministic, numpy-only stand-in for a frozen codec/SSL model.

Makes the whole pipeline (encode → cache → probe → analyse) buildable and
testable on the Mac with no torch. Emits ALL declared variants (a distinct
seeded projection each) so downstream code is exercised over the full
representation sweep — see docs/DECISIONS.md D3/D7. Carries real signal but makes
NO perceptual claim and must never appear in a result table.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import LatentResult, measure_frame_rate
from .dsp import resample_linear, stft_logmag


def _variant_seed(base_seed: int, native_sr: int, latent_dim: int, variant: str) -> int:
    """Deterministic across processes (no built-in hash — PYTHONHASHSEED-safe)."""
    h = int.from_bytes(hashlib.blake2b(variant.encode(), digest_size=4).digest(), "big")
    return (base_seed ^ (native_sr * 131) ^ (latent_dim * 977) ^ h) & 0xFFFFFFFF


class FakeEncoder:
    def __init__(
        self,
        name: str,
        native_sr: int,
        latent_dim: int,
        variants: tuple[str, ...],
        n_fft: int = 512,
        hop: int = 256,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.native_sr = int(native_sr)
        self.latent_dim = int(latent_dim)
        self.variants = tuple(variants)
        self._n_fft = int(n_fft)
        self._hop = int(hop)
        n_freq = n_fft // 2 + 1
        self._proj = {
            v: (
                np.random.default_rng(_variant_seed(seed, native_sr, latent_dim, v))
                .standard_normal((n_freq, latent_dim))
                / np.sqrt(n_freq)
            )
            for v in self.variants
        }

    def encode(self, wav: np.ndarray, sr: int) -> dict[str, LatentResult]:
        wav = np.asarray(wav, dtype=np.float64).reshape(-1)
        wav = resample_linear(wav, sr, self.native_sr)
        n_native = wav.size
        logmag = stft_logmag(wav, self._n_fft, self._hop)  # [T, F]
        t = int(logmag.shape[0])
        fr = measure_frame_rate(t, n_native, self.native_sr)
        out: dict[str, LatentResult] = {}
        for v in self.variants:
            frames = logmag @ self._proj[v]  # [T, D]
            meta = {
                "backend": "fake",
                "name": self.name,
                "variant": v,
                "native_sr": self.native_sr,
                "latent_dim": self.latent_dim,
                "n_frames": t,
                "hop": self._hop,
            }
            out[v] = LatentResult(frames=frames, frame_rate_hz=fr, meta=meta)
        return out

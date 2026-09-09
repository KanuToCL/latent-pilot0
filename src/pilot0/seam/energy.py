"""EnergyEncoder — the spectral-band energy control: level, balance, temporal dispersion.

Per-frame [log total energy, log LF energy (<1 kHz), log HF energy (≥4 kHz)] — a
deliberately strong loudness / spectral-tilt representation. Because we never
renormalise (L3) and the L4 headroom scalar preserves relative energies, additive
degradations (noise/hiss/hum) raise RMS with severity and subtractive ones
(clip/bandlimit/mp3/dropout) lower it, so a level meter alone tracks severity. A
codec that cannot BEAT this control on severity is not reading degradation (elder
finding B1). Real DSP, torch-free, runs everywhere.

It is NOT "level-only", which is what this docstring used to claim and what Gate 1's
G1b wording leaned on (S7). Pooled mean+std gives six coordinates and a uniform gain
moves exactly one direction of them — the three log-MEANS together,
`(1,1,1,0,0,0)/√3`. The three log-STDs are gain-invariant, and so is every
spectral-balance contrast among the means, so "beats the energy control" is not the
same claim as "is not reading loudness". `probes/level_split.py` scores the level
coordinate and its 5-d complement separately, which is the honest way to ask.

`_EPS` sits inside the log, so below about −80 dBFS the features stop being
gain-equivariant; the corpus floor is near −60 dBFS, well clear.
"""

from __future__ import annotations

import numpy as np

from .base import LatentResult, measure_frame_rate

_EPS = 1e-10


class EnergyEncoder:
    def __init__(self, name: str = "energy", native_sr: int = 16000,
                 n_fft: int = 400, hop: int = 160, lf_hz: float = 1000.0, hf_hz: float = 4000.0) -> None:
        self.name = name
        self.native_sr = int(native_sr)
        self.latent_dim = 3
        self.variants = ("energy",)
        self._n_fft = int(n_fft)
        self._hop = int(hop)
        self._win = np.hanning(n_fft)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / native_sr)
        # Absolute band levels (not ratios), so — unlike degrade.base.band_power — we
        # deliberately keep the DC and Nyquist bins: no one-sided-spectrum doubling to
        # cancel, preflight already rejects DC offset, and hum sits at 50/60 Hz not DC.
        self._lf = freqs < lf_hz
        self._hf = freqs >= hf_hz

    def encode(self, wav: np.ndarray, sr: int) -> dict[str, LatentResult]:
        wav = np.asarray(wav, dtype=np.float64).reshape(-1)
        if sr != self.native_sr:
            import soxr

            wav = soxr.resample(wav, sr, self.native_sr)
        if wav.size < self._n_fft:
            wav = np.pad(wav, (0, self._n_fft - wav.size))
        n_frames = 1 + (wav.size - self._n_fft) // self._hop
        idx = np.arange(self._n_fft)[None, :] + self._hop * np.arange(n_frames)[:, None]
        frames = wav[idx]
        power = np.abs(np.fft.rfft(frames * self._win[None, :], axis=1)) ** 2
        feat = np.stack(
            [
                np.log(np.mean(frames ** 2, axis=1) + _EPS),
                np.log(power[:, self._lf].sum(axis=1) + _EPS),
                np.log(power[:, self._hf].sum(axis=1) + _EPS),
            ],
            axis=1,
        )  # [T, 3]
        fr = measure_frame_rate(n_frames, wav.size, self.native_sr)
        meta = {"backend": "energy", "name": self.name, "variant": "energy",
                "native_sr": self.native_sr, "latent_dim": 3, "n_frames": int(n_frames)}
        return {"energy": LatentResult(frames=feat, frame_rate_hz=fr, meta=meta)}

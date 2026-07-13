"""The encoder contract every representation backend implements.

Kept tiny and torch-free on purpose: `FakeEncoder` (Mac) and `RealCodecEncoder`
(GPU box) both return `dict[variant -> LatentResult]`, so the encode → cache →
probe → analyse pipeline is written once and never branches on backend.

Design notes (elder review, 2026-07-13):
- One `encode()` emits ALL variants of a model from a single forward pass
  (§2.3 sweeps RVQ depth / WavLM layer / Mimi stream). Variant is part of a
  latent's identity — the cache key is content_hash x name x variant x code.
- `frames` are the primary product; pooling (mean+std, max) lives downstream so
  both the frame-level dropout probe and the pooled probes are supported.
- `frame_rate_hz` is REQUIRED so frame-level probes can align time; the real
  backends measure it at bring-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, eq=False)
class LatentResult:
    """Frame-level latents for one clip through one (encoder, variant).

    frames:        [T, D] float — per-frame latent (the primary product).
    frame_rate_hz: frames per second (measured; required for frame-level probes).
    meta:          backend, name, variant, native_sr, latent_dim, n_frames, ...
    """

    frames: np.ndarray
    frame_rate_hz: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frames.ndim != 2:
            raise ValueError(f"frames must be 2-D [T, D], got shape {self.frames.shape}")
        if not self.frame_rate_hz > 0:
            raise ValueError(f"frame_rate_hz must be > 0, got {self.frame_rate_hz!r}")

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.frames.shape[1])


@runtime_checkable
class Encoder(Protocol):
    """A frozen representation backend. Deterministic: the same waveform at the
    same input rate yields identical latents for every variant."""

    name: str
    native_sr: int
    latent_dim: int
    variants: tuple[str, ...]

    def encode(self, wav: np.ndarray, sr: int) -> dict[str, LatentResult]: ...


# --- pooling (downstream of the encoder; single source of truth) --------------


def pool_mean_std(frames: np.ndarray) -> np.ndarray:
    """[T, D] → [2D] = mean ++ std over frames. Primary pooling (§2.3)."""
    if frames.ndim != 2:
        raise ValueError(f"frames must be 2-D [T, D], got {frames.shape}")
    return np.concatenate([frames.mean(axis=0), frames.std(axis=0)])


def pool_max(frames: np.ndarray) -> np.ndarray:
    """[T, D] → [D] = max over frames. Secondary pooling (§2.3)."""
    if frames.ndim != 2:
        raise ValueError(f"frames must be 2-D [T, D], got {frames.shape}")
    return frames.max(axis=0)


def measure_frame_rate(n_frames: int, n_samples: int, sr: int) -> float:
    """Frames per second = n_frames / (n_samples / sr). Measured, not assumed."""
    duration = n_samples / sr if sr else 0.0
    return (n_frames / duration) if duration > 0 else 0.0

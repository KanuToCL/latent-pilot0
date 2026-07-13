"""Encoder seam — the Mac-dev / GPU-run boundary.

The pipeline is written against `Encoder` and `LatentResult` only. On the Mac it
runs `FakeEncoder` (numpy, no torch); on the GPU box the same code runs the real
frozen backends via `RealCodecEncoder`. Nothing downstream imports torch.
"""

from .base import Encoder, LatentResult, measure_frame_rate, pool_max, pool_mean_std
from .fake import FakeEncoder
from .real import RealBackendUnavailable, RealCodecEncoder
from .registry import REAL_SPECS, WIRED_FAMILIES, available_encoders, make_encoder

__all__ = [
    "Encoder",
    "LatentResult",
    "pool_mean_std",
    "pool_max",
    "measure_frame_rate",
    "FakeEncoder",
    "RealCodecEncoder",
    "RealBackendUnavailable",
    "make_encoder",
    "available_encoders",
    "REAL_SPECS",
    "WIRED_FAMILIES",
]

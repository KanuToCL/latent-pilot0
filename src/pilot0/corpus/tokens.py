"""Opaque, content-derived source IDs. Same audio → same token (dedup + stable
re-runs); the original filename never enters the study. Little-endian float32 is
pinned so tokens are reproducible across platforms."""

from __future__ import annotations

import hashlib

import numpy as np


def content_sha256(wav: np.ndarray, sr: int) -> str:
    h = hashlib.sha256(np.ascontiguousarray(wav, dtype="<f4").tobytes())
    h.update(str(sr).encode())
    return h.hexdigest()


def content_token(wav: np.ndarray, sr: int, prefix: str = "src") -> str:
    return f"{prefix}_{content_sha256(wav, sr)[:16]}"

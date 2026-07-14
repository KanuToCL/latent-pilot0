"""Fabricated no-reference baseline predictions for the OOD clips — the teaser's fake
data source, swapped for real NISQA/DNSMOS/UTMOS at bring-up. Structural signature:
on generative material the incumbents DIVERGE (each carries a different systematic OOD
bias) where on the classical grid they roughly agree. That divergence is the whole
point of the teaser; the absolute values are plumbing, not perceptual claims.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..quality.scores import NR_BASELINES

_SCORE_MIN, _SCORE_MAX = 1.0, 5.0
# Different incumbents mispredict generative material in different directions — the
# reason a reference-free consumer cannot trust any single one out of distribution.
_OOD_BIAS = {"nisqa": 0.9, "dnsmos": -0.7, "utmos": 0.3}


def _rng(*parts) -> np.random.Generator:
    key = "|".join(str(p) for p in parts).encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


@dataclass(frozen=True)
class OODNRScores:
    seed: int = 0

    def _intrinsic(self, clip_id: str) -> float:
        return float(_rng("ood-intrinsic", clip_id, self.seed).normal(3.0, 0.6))

    def get(self, baseline: str, clip_id: str) -> float:
        if baseline not in NR_BASELINES:
            raise KeyError(f"unknown baseline '{baseline}'")
        val = self._intrinsic(clip_id) + _OOD_BIAS[baseline] + _rng(baseline, clip_id, self.seed).normal(0.0, 0.2)
        return float(np.clip(val, _SCORE_MIN, _SCORE_MAX))

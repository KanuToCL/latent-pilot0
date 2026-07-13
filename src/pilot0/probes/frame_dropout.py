"""Frame-level dropout severity probe (§2.4: frame-level probes for dropouts only).

Dropouts zero intermittent ~20 ms bursts. mean+std pooling averages that away — a
clip with 10 % of its frames zeroed looks, in the mean, almost like one with 1 %.
So for the dropout family we build a frame-level feature from the per-frame latent
norm: normalise each frame's L2 norm by the clip's median norm (scale-invariant —
exact for a global gain like the L4 headroom scalar, approximate for a log-domain
rep) and read low order-statistics + the fraction of near-silent frames — a direct
count of the dropout signature.

This probe reports the frame-level SRCC alongside the pooled mean+std SRCC on the
same dropout rows. The gap is precisely "a dropout-tuned frame-norm order-statistic
vs generic mean+std pooling" — NOT the broader "frame-level info helps": mean+std's
std already carries some burst variance, so the pooled comparator is not blind.

The low-norm assumption is representation-dependent (findings W3): it holds where a
zeroed audio burst maps to a low-norm latent frame (EnCodec/DAC, the fake log-mag
path), but a per-frame-normalised SSL rep (WavLM) can flatten frame norm and carry
no dropout signal — an honest false-negative. GPU_BRINGUP step verifies frame norm
actually drops on dropout for each backend before the gap is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..encode.headroom import CEILING_DBFS
from ..seam.base import pool_mean_std
from .dataset import CLEAN, iter_latents
from .metrics import Estimate, bootstrap_over_groups, srcc

FAMILY = "dropout"
_LOW_QUANTILES = (0.05, 0.10, 0.25)
_SILENCE_FRACTIONS = (0.5, 0.25)
_EPS = 1e-10


def frame_dropout_features(frames: np.ndarray) -> np.ndarray:
    """[T, D] → level-invariant frame-norm statistics sensitive to zeroed bursts."""
    norms = np.linalg.norm(frames, axis=1)
    r = norms / (np.median(norms) + _EPS)
    quantiles = np.quantile(r, _LOW_QUANTILES)
    silence = [float(np.mean(r < f)) for f in _SILENCE_FRACTIONS]
    return np.concatenate([quantiles, silence, [float(r.min())]])


@dataclass(frozen=True)
class FrameDropoutResult:
    frame_srcc: Estimate  # ridge on frame-norm statistics
    pooled_srcc: Estimate  # ridge on mean+std pooling, same rows
    gap: Estimate  # frame − pooled, paired over resamples
    n_train: int
    n_test: int


def _regressor() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def evaluate_frame_dropout(manifest, name, variant, cache_dir, *, ceiling_dbfs=CEILING_DBFS) -> FrameDropoutResult:
    frame_X, pooled_X, sev, split, group = [], [], [], [], []
    for r, lat in iter_latents(manifest, name, variant, cache_dir, ceiling_dbfs=ceiling_dbfs):
        if r["family"] not in (FAMILY, CLEAN):  # clean = sev-0 anchor for training
            continue
        frame_X.append(frame_dropout_features(lat.frames))
        pooled_X.append(pool_mean_std(lat.frames))
        sev.append(r["severity"])
        split.append(r["split"])
        group.append(r["group"])

    frame_X, pooled_X = np.asarray(frame_X), np.asarray(pooled_X)
    sev, split, group = np.asarray(sev, dtype=int), np.asarray(split), np.asarray(group)
    tr = split == "train"
    te = (split == "test") & (sev > 0)  # score on graded dropout only (without-clean, finding M2)

    y_true, groups = sev[te], group[te]
    frame_pred = _regressor().fit(frame_X[tr], sev[tr]).predict(frame_X[te])
    pooled_pred = _regressor().fit(pooled_X[tr], sev[tr]).predict(pooled_X[te])

    return FrameDropoutResult(
        frame_srcc=bootstrap_over_groups(groups, lambda i: srcc(y_true[i], frame_pred[i])),
        pooled_srcc=bootstrap_over_groups(groups, lambda i: srcc(y_true[i], pooled_pred[i])),
        gap=bootstrap_over_groups(groups, lambda i: srcc(y_true[i], frame_pred[i]) - srcc(y_true[i], pooled_pred[i])),
        n_train=int(tr.sum()),
        n_test=int(te.sum()),
    )

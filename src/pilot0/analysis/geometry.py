"""Attribute geometry (RQ2): are degradation types near-orthogonal directions in
latent space? Three descriptive, IN-SAMPLE views (fit on all rows including test —
this characterises geometry, it does not gate anything, so `probe_cosine` is NOT
the held-out Gate-1 type probe), all in the STANDARDISED feature space (one scaler
fit on the degraded rows, so no single high-variance dimension dominates a cosine):

  * probe-weight directions — the linear type-probe's per-class weight vectors;
  * class-mean directions   — mean(family) − mean(clean), probe-free;
  * PCA of condition centroids — do the degraded (family, severity) centroids (clean
    excluded; count depends on the common cells — band-limit sheds cutoffs ≥ Nyquist
    per rate) spread along a low-dimensional, severity-ordered structure.

Off-diagonal cosines near 0 ⇒ separable, additive-friendly directions (feeds the
Phase-7 additivity test). Fake latents make this a plumbing check, not a result.
Cosines are taken in the standardised basis, so "near-orthogonal" is a statement
about that basis, not the raw latent geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..probes.dataset import CLEAN

_EPS = 1e-12


@dataclass(frozen=True)
class GeometryResult:
    labels: list[str]  # family order for both cosine matrices
    probe_cosine: np.ndarray  # [F, F] cosine of per-class probe-weight directions
    mean_cosine: np.ndarray  # [F, F] cosine of class-mean − clean directions
    centroid_pca_explained: np.ndarray  # explained-variance ratio of condition centroids
    centroid_labels: list[str]  # "family/severity" per centroid row
    centroid_coords: np.ndarray  # [C, k] centroids projected onto top-k PCs


def _unit_rows(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + _EPS)


def _cosine_matrix(directions: np.ndarray) -> np.ndarray:
    u = _unit_rows(directions)
    return u @ u.T


def geometry(
    X: np.ndarray, family: np.ndarray, severity: np.ndarray, *, n_pcs: int = 3
) -> GeometryResult:
    degraded = family != CLEAN
    if not np.any(family == CLEAN):  # class-mean directions are defined relative to clean
        raise ValueError("geometry() needs clean rows to anchor the class-mean directions")
    labels = sorted(set(family[degraded].tolist()))

    scaler = StandardScaler().fit(X[degraded])
    Xs = scaler.transform(X)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0).fit(Xs[degraded], family[degraded])
    probe_dirs = np.array([clf.coef_[clf.classes_.tolist().index(f)] for f in labels])

    clean_mean = Xs[family == CLEAN].mean(axis=0)
    mean_dirs = np.array([Xs[family == f].mean(axis=0) - clean_mean for f in labels])

    centroids, cent_labels = [], []
    for f in labels:
        for s in sorted(set(severity[family == f].tolist())):
            centroids.append(Xs[(family == f) & (severity == s)].mean(axis=0))
            cent_labels.append(f"{f}/{s}")
    C = np.array(centroids)
    Cc = C - C.mean(axis=0)
    _, sv, vt = np.linalg.svd(Cc, full_matrices=False)
    k = min(n_pcs, vt.shape[0])
    explained = (sv**2) / (np.sum(sv**2) + _EPS)

    return GeometryResult(
        labels=labels,
        probe_cosine=_cosine_matrix(probe_dirs),
        mean_cosine=_cosine_matrix(mean_dirs),
        centroid_pca_explained=explained,
        centroid_labels=cent_labels,
        centroid_coords=Cc @ vt[:k].T,
    )

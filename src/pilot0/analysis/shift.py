"""RQ2 shift vectors: is a degradation a consistent DIRECTION in latent space?

`geometry.py` answers the class-level question (where do the family CENTROIDS sit
relative to clean). This module asks the per-clip question: take Δ = x(degraded) −
x(clean of the SAME source) — a paired displacement that cancels the clip's own
content — and ask

  * concentration — do a family's Δs point the same way (mean pairwise cosine of
    the Δ rows) or scatter like independent directions? The reference scale is the
    null 1/sqrt(D): the RMS cosine between two independent isotropic directions in
    D dims, so "0.31 with a null of 0.06" is readable without a permutation test;
  * magnitude — does ||Δ|| grow with severity (per-cell median + a per-ROW Spearman
    over severity, not over the 3–5 medians, which would be trivially ±1);
  * reduction persistence — refit the geometry after PCA to k dims: does the
    direction structure survive the reduction, or does it live in the tail?

Every statistic here is DESCRIPTIVE and IN-SAMPLE (like `geometry.py`, it gates
nothing) and lives in the SAME standardised space that module uses: one
StandardScaler fit on the DEGRADED rows only, applied to all rows — see
`standardize_on_degraded`, which callers must apply before anything else here.
The scaler's offset cancels in a difference, so the Δs are the raw displacements
rescaled per dimension; "concentration" is therefore a statement about that basis,
not about the raw latent geometry.

Paired class-mean directions (`mean_direction_cosines`) equal `geometry.mean_cosine`
exactly when every source carries a clean row and the family's cells are balanced —
they differ only by which clean rows anchor which degraded rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..probes.dataset import CLEAN
from ..probes.metrics import srcc

_EPS = 1e-12


# --- shared space -------------------------------------------------------------


def standardize_on_degraded(X: np.ndarray, family: np.ndarray) -> np.ndarray:
    """One scaler fit on the DEGRADED rows, applied to ALL rows — clean included,
    because the shift vectors need the clean anchor in the same basis. Identical
    convention to `geometry.geometry`, so cosines from both are comparable."""
    degraded = family != CLEAN
    if not np.any(degraded):
        raise ValueError("standardize_on_degraded() needs degraded rows to fit the scaler")
    return StandardScaler().fit(X[degraded]).transform(X)


def _unit_rows(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + _EPS)


def cosine_matrix(directions: np.ndarray) -> np.ndarray:
    u = _unit_rows(directions)
    return u @ u.T


# --- 1. per-clip shift vectors ------------------------------------------------


@dataclass(frozen=True)
class ShiftVectors:
    deltas: np.ndarray  # [M, D] x(degraded) − x(clean of the same source)
    family: np.ndarray  # [M] family of the degraded row
    severity: np.ndarray  # [M] int severity of the degraded row
    source: np.ndarray  # [M] source id shared by the pair
    n_degraded: int  # degraded rows offered
    n_unpaired_rows: int  # degraded rows dropped (their source has no clean row)
    n_unpaired_sources: int


def shift_vectors(
    X_std: np.ndarray, family: np.ndarray, severity: np.ndarray, source: np.ndarray
) -> ShiftVectors:
    """Δ per degraded row against the clean row of the SAME source. Degraded rows
    whose source carries no clean row are dropped (and counted) rather than paired
    against a population mean — an unpaired Δ would carry the clip's content."""
    clean_rows: dict[str, int] = {}
    for i in np.flatnonzero(family == CLEAN):
        src = source[i]
        if src in clean_rows:  # the manifest gives each source exactly one clean cell
            raise ValueError(f"source {src!r} has more than one clean row — ambiguous anchor")
        clean_rows[src] = int(i)

    degraded = family != CLEAN
    partner = np.array([clean_rows.get(s, -1) for s in source], dtype=int)
    keep = degraded & (partner >= 0)
    lost = degraded & (partner < 0)

    return ShiftVectors(
        deltas=X_std[keep] - X_std[partner[keep]],
        family=family[keep],
        severity=np.asarray(severity, dtype=int)[keep],
        source=source[keep],
        n_degraded=int(degraded.sum()),
        n_unpaired_rows=int(lost.sum()),
        n_unpaired_sources=int(len(set(source[lost].tolist()))),
    )


# --- 2. concentration ---------------------------------------------------------


@dataclass(frozen=True)
class Concentration:
    mean_cosine: float  # mean over all i<j pairs of cos(Δi, Δj)
    null_scale: float  # 1/sqrt(D): RMS cosine of two independent isotropic dirs
    n_rows: int  # rows entering the mean (zero-norm rows already dropped)
    n_pairs: int  # n(n−1)/2 — ALL of them, this is not a sample
    n_zero_norm: int  # dropped: a zero Δ has no direction
    method: str = "exact_all_pairs"


def concentration(deltas: np.ndarray) -> Concentration:
    """Mean pairwise cosine of the Δ rows, EXACTLY — no pair sampling.

    With u_i the unit Δs and S = Σ u_i,  Σ_{i≠j} u_i·u_j = ||S||² − n, so the mean
    over all n(n−1)/2 pairs is (||S||² − n) / (n(n−1)): O(nD) instead of O(n²D),
    and the exact value that sampling would only estimate. (No cancellation risk:
    ||S||² ≈ n only when the mean cosine is ~0, where the residual error is ~1e-16/n,
    orders below the 1/sqrt(D) null.)
    """
    n_features = int(deltas.shape[1]) if deltas.ndim == 2 else 0
    null = 1.0 / np.sqrt(n_features) if n_features else float("nan")

    norms = np.linalg.norm(deltas, axis=1)
    ok = norms > _EPS
    u = deltas[ok] / norms[ok, None]
    n = int(u.shape[0])
    if n < 2:  # a single direction has no pair — undefined, not 1.0
        mean_cos = float("nan")
    else:
        s = u.sum(axis=0)
        mean_cos = float((float(s @ s) - n) / (n * (n - 1)))
    return Concentration(
        mean_cosine=mean_cos,
        null_scale=float(null),
        n_rows=n,
        n_pairs=n * (n - 1) // 2,
        n_zero_norm=int((~ok).sum()),
    )


def family_concentration(deltas: np.ndarray, family: np.ndarray) -> dict[str, Concentration]:
    return {f: concentration(deltas[family == f]) for f in sorted(set(family.tolist()))}


def mean_direction_cosines(
    deltas: np.ndarray, family: np.ndarray, labels: list[str] | None = None
) -> tuple[list[str], np.ndarray]:
    """Cosines between the per-family MEAN Δ directions (the paired analogue of
    `geometry.mean_cosine`). Off-diagonal ≈ 0 ⇒ separable directions."""
    labs = sorted(set(family.tolist())) if labels is None else list(labels)
    dirs = np.array([deltas[family == f].mean(axis=0) for f in labs])
    return labs, cosine_matrix(dirs)


# --- 3. magnitude vs severity -------------------------------------------------


@dataclass(frozen=True)
class MagnitudeCurve:
    median_norm: dict[str, dict[int, float]]  # family -> severity -> median ||Δ||
    n_by_cell: dict[str, dict[int, int]]  # family -> severity -> rows in the cell
    srcc_by_family: dict[str, float]  # per-ROW Spearman(severity, ||Δ||)


def magnitude_curve(
    deltas: np.ndarray, family: np.ndarray, severity: np.ndarray
) -> MagnitudeCurve:
    """Per-(family, severity) median ||Δ|| plus the per-family rank correlation of
    severity with ||Δ||, computed over ROWS (a Spearman over the 3–5 cell medians
    would be ±1 on any monotone curve and says nothing about spread)."""
    norms = np.linalg.norm(deltas, axis=1)
    sev = np.asarray(severity, dtype=int)
    median: dict[str, dict[int, float]] = {}
    counts: dict[str, dict[int, int]] = {}
    rho: dict[str, float] = {}
    for f in sorted(set(family.tolist())):
        m = family == f
        median[f] = {int(s): float(np.median(norms[m & (sev == s)])) for s in sorted(set(sev[m].tolist()))}
        counts[f] = {int(s): int((m & (sev == s)).sum()) for s in sorted(set(sev[m].tolist()))}
        rho[f] = srcc(sev[m], norms[m])  # nan-guarded: a single severity level -> nan
    return MagnitudeCurve(median_norm=median, n_by_cell=counts, srcc_by_family=rho)


# --- 4. persistence under dimensionality reduction ----------------------------


@dataclass(frozen=True)
class ReducedView:
    k: int
    cumulative_explained: float  # of the degraded-row PCA, first k components
    by_family: dict[str, Concentration]
    mean_cosine: np.ndarray  # [F, F] paired class-mean cosines at k dims


@dataclass(frozen=True)
class ReductionPersistence:
    n_features: int  # D of the input space
    labels: list[str]  # family order of every mean_cosine matrix
    explained_variance_ratio: np.ndarray  # per fitted PCA component
    full_by_family: dict[str, Concentration]  # reference: no reduction
    full_mean_cosine: np.ndarray
    by_k: dict[int, ReducedView]


def reduction_persistence(
    X_deg_std: np.ndarray, deltas: np.ndarray, family: np.ndarray, ks=(2, 8, 32)
) -> ReductionPersistence:
    """PCA fit on the standardised DEGRADED rows; the Δs are then projected onto the
    top-k components and the concentration / class-mean cosines recomputed there.

    `family` labels the DELTAS (not the rows of `X_deg_std`); the two have equal
    length only when every source carries a clean partner, so they are not tied.
    Δs are differences, so the PCA mean cancels — project with the components alone.
    """
    labels = sorted(set(family.tolist()))
    n_features = int(deltas.shape[1])
    n_comp = int(min(max(ks), X_deg_std.shape[0], X_deg_std.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0).fit(X_deg_std)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=float)

    by_k: dict[int, ReducedView] = {}
    for k in sorted({int(min(k, n_comp)) for k in ks}):
        z = deltas @ pca.components_[:k].T
        by_k[k] = ReducedView(
            k=k,
            cumulative_explained=float(explained[:k].sum()),
            by_family=family_concentration(z, family),
            mean_cosine=mean_direction_cosines(z, family, labels)[1],
        )
    return ReductionPersistence(
        n_features=n_features,
        labels=labels,
        explained_variance_ratio=explained,
        full_by_family=family_concentration(deltas, family),
        full_mean_cosine=mean_direction_cosines(deltas, family, labels)[1],
        by_k=by_k,
    )

"""RQ2 additivity: does a combined degradation land where the sum of its parts
predicts? For each pair (a, b) at a mid severity, compare the combined displacement
z̄(a+b) − z̄(clean) against the sum of the single displacements Δa + Δb by cosine:

    cos( z̄(a+b) − z̄(clean),  (z̄(a) − z̄(clean)) + (z̄(b) − z̄(clean)) )

The cosine constrains DIRECTION only — cos = 1 means the combo displacement is
codirectional with (not equal to) the sum of parts (dab = k·(Δa+Δb), any k > 0). ≈ 1 ⇒
the codec places the combo along the linear-superposition direction of the two single
attributes (separable, additive-friendly geometry, on the Phase-5 near-orthogonality
finding); ≪ 1 ⇒ the combination interacts nonlinearly.

That question is NOT identifiable from the cosine alone (S6). Two ways it lies:

  dominance   ‖Δa‖ ≫ ‖Δb‖ — a combo that ignores leg b entirely still reads
              additive. Worked case: Δab = Δa = (100,0), Δb = (0,1) → cos = 0.99995.
  collinear   cos(Δa, Δb) ≈ 1 — the two legs span one direction, so no decomposition
              of Δab into them exists and every ratio metric reads "additive".

So each cell also carries `cos_to_a` / `cos_to_b` (which leg the combo sits on),
`cos_legs` (the identifiability condition), `norm_ratio` (leg lopsidedness), `r` and
the derived `rel_residual`, and the recovery test the audit asked for: a per-source
least squares Δab ≈ α·Δa + β·Δb, median over sources with a speaker bootstrap CI.
α ≈ 1 with β ≈ 0 IS dominance; α ≈ β ≈ 1 is genuine superposition.

`r` and `rel_residual` are CENTROID quantities (F23's Δ̄ notation), which is what makes
`rel_residual² = 1 + r² − 2r·cosine` hold exactly against the centroid `cosine`; the
dominance/recovery scalars are per-source medians. §4 groups `r` with the medians;
F23's definition governs.

Basis is reported BOTH ways because the answer is basis-dependent (physics review):
`cosine` in the RAW pooled-latent space (the proposal's z̄ — a superposition test lives
in the codec's own metric, and raw avoids the noise blow-up that standardisation
inflicts on near-constant latent channels), and `cosine_std` in the STANDARDISED basis
(one scaler fit on the degraded singles, equal-weighting every dimension so a few
high-energy channels can't set the verdict). Each cell is restricted to the SOURCES
present in all of clean / a-only / b-only / a+b, so the four role-centroids are means
over the same clips — a speaker group holds several clips, so the old group
intersection did not pair rows at all (D3). `rows_identical` marks the cells where the
two selections coincide and the numbers must therefore reproduce the legacy run
bit-for-bit. Descriptive/in-sample, source-level bootstrap CI; on fake latents this is
a plumbing check, not a result.

Section map (file order):
  PairAdditivity / AdditivityResult   the per-cell record and its container
  _cosine                     nan-guarded cosine of two vectors
  _additivity_cosine          role centroids on a resample, then the cosine
  _stack                      the four role blocks in a fixed order (clean, a, b, ab)
  _displacements              per-source Δa, Δb, Δab, aligned by source
  _recover                    per-source least squares (α, β), nan when unidentifiable
  _per_source_metrics         cos_to_a / cos_to_b / cos_legs / norm_ratio medians + α, β
  _centroid_ratio             r = ‖Δ̄ab‖ / ‖Δ̄a+Δ̄b‖ (F23)
  additivity                  public entry: one PairAdditivity per (pair, severity)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..probes.dataset import CLEAN, ProbeData
from ..probes.gate1 import MIN_TEST_GROUPS
from ..probes.metrics import N_BOOTSTRAP, Estimate, bootstrap_over_groups
from .dataset import ComboData
from .grid import COMBO_SEVERITIES, combo_label

_EPS = 1e-12
_NAN = Estimate(float("nan"), float("nan"), float("nan"))
# |cos(Δa, Δb)| at or above this leaves the 2×2 normal-equation matrix numerically
# singular: infinitely many (α, β) fit equally well, so the recovery refuses to report
# one. This is the identifiability condition, not a quality threshold.
_ILL_CONDITIONED_COS = 0.999


@dataclass(frozen=True)
class PairAdditivity:
    pair: str
    severity: int
    cosine: Estimate  # RAW pooled-latent basis (the proposal's z̄) — primary
    cosine_std: Estimate  # standardised basis — robustness (equal-weights dimensions)
    cos_to_a: float  # median per-source cos(Δab, Δa) — dominance readout
    cos_to_b: float  # median per-source cos(Δab, Δb)
    cos_legs: float  # median per-source cos(Δa, Δb) — the identifiability condition
    norm_ratio: float  # median per-source ‖Δa‖ / ‖Δb‖
    r: float  # ‖Δ̄ab‖ / ‖Δ̄a+Δ̄b‖ on the role centroids (F23)
    rel_residual: float  # DERIVED from cosine and r; carries nothing they do not
    alpha: Estimate  # median per-source least-squares weight on Δa, speaker bootstrap
    beta: Estimate  # ... and on Δb
    n_groups: int  # speaker groups in the legacy intersection (unchanged meaning)
    n_sources: int  # clips carrying all four roles — the population actually used
    rows_identical: bool  # source pairing selected exactly the legacy rows (D3)


@dataclass(frozen=True)
class AdditivityResult:
    by_cell: dict[tuple[str, int], PairAdditivity]

    def mean_cosine(self) -> float:
        """Mean of the per-cell RAW cosines (a summary, not itself a cosine)."""
        vals = [c.cosine.point for c in self.by_cell.values() if np.isfinite(c.cosine.point)]
        return float(np.mean(vals)) if vals else float("nan")


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < _EPS or nv < _EPS:  # a zero displacement (e.g. a role absent) → undefined
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def _additivity_cosine(X: np.ndarray, role: np.ndarray, idx: np.ndarray) -> float:
    """Centroid per role on the resampled rows `idx`, then the additivity cosine. Any
    role absent in this resample → nan (dropped by the bootstrap trust floor)."""
    r = role[idx]
    cents = {}
    for code in (0, 1, 2, 3):  # clean, a, b, ab
        m = r == code
        if not m.any():
            return float("nan")
        cents[code] = X[idx][m].mean(axis=0)
    da, db, dab = cents[1] - cents[0], cents[2] - cents[0], cents[3] - cents[0]
    return _cosine(dab, da + db)


def _stack(Xs: np.ndarray, Xc: np.ndarray, masks) -> np.ndarray:
    clean_m, a_m, b_m, ab_m = masks
    return np.vstack([Xs[clean_m], Xs[a_m], Xs[b_m], Xc[ab_m]])


def _displacements(Xs, Xc, masks, p_source, c_source):
    """Per-source (Δa, Δb, Δab), one row per clip. Each role is re-ordered by source
    token so the three arrays are aligned clip-for-clip — which is exactly what source
    pairing buys and the group intersection could not give."""
    clean_m, a_m, b_m, ab_m = masks
    clean = Xs[clean_m][np.argsort(p_source[clean_m])]
    a = Xs[a_m][np.argsort(p_source[a_m])]
    b = Xs[b_m][np.argsort(p_source[b_m])]
    ab = Xc[ab_m][np.argsort(c_source[ab_m])]
    return a - clean, b - clean, ab - clean


def _recover(da: np.ndarray, db: np.ndarray, dab: np.ndarray) -> tuple[float, float]:
    """Least squares Δab ≈ α·Δa + β·Δb for ONE clip, via the 2×2 normal equations.
    (nan, nan) when the legs are collinear or either is degenerate: the system is then
    rank-deficient and any answer would be an arbitrary pick among infinitely many
    (F23) — the failure mode the cosine hides."""
    naa, nbb = float(da @ da), float(db @ db)
    if naa < _EPS or nbb < _EPS:
        return float("nan"), float("nan")
    nab = float(da @ db)
    if abs(nab) / np.sqrt(naa * nbb) >= _ILL_CONDITIONED_COS:
        return float("nan"), float("nan")
    det = naa * nbb - nab * nab
    ra, rb = float(dab @ da), float(dab @ db)
    return (ra * nbb - rb * nab) / det, (rb * naa - ra * nab) / det


def _nanmedian(vals) -> float:
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmedian(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def _per_source_metrics(da, db, dab, group: np.ndarray, n_boot: int) -> dict:
    """The dominance/recovery block. Every scalar is a per-clip value first and a MEDIAN
    over clips second, so one pathological clip cannot set the readout. α and β
    additionally carry a speaker-group bootstrap CI: the recovery test is the one the
    audit asked for, so it is the one that gets uncertainty."""
    n = da.shape[0]
    ab = np.array([_recover(da[i], db[i], dab[i]) for i in range(n)], dtype=float)
    return {
        "cos_to_a": _nanmedian([_cosine(dab[i], da[i]) for i in range(n)]),
        "cos_to_b": _nanmedian([_cosine(dab[i], db[i]) for i in range(n)]),
        "cos_legs": _nanmedian([_cosine(da[i], db[i]) for i in range(n)]),
        "norm_ratio": _nanmedian([_norm_ratio(da[i], db[i]) for i in range(n)]),
        "alpha": bootstrap_over_groups(group, lambda i: _nanmedian(ab[i, 0]), n=n_boot,
                                       min_groups=MIN_TEST_GROUPS),
        "beta": bootstrap_over_groups(group, lambda i: _nanmedian(ab[i, 1]), n=n_boot,
                                      min_groups=MIN_TEST_GROUPS),
    }


def _norm_ratio(da: np.ndarray, db: np.ndarray) -> float:
    nb = float(np.linalg.norm(db))
    return float(np.linalg.norm(da) / nb) if nb >= _EPS else float("nan")


def _centroid_ratio(da, db, dab) -> float:
    """r = ‖Δ̄ab‖ / ‖Δ̄a+Δ̄b‖ on the role centroids — the magnitude the cosine throws
    away. With it, `rel_residual² = 1 + r² − 2r·cosine` holds exactly (F23), which is
    the point: the residual is derived, never an independent measurement."""
    sum_norm = float(np.linalg.norm(da.mean(axis=0) + db.mean(axis=0)))
    if sum_norm < _EPS:
        return float("nan")
    return float(np.linalg.norm(dab.mean(axis=0)) / sum_norm)


def _empty_cell(label: str, sev: int, n_groups: int, n_sources: int, identical: bool) -> PairAdditivity:
    nan = float("nan")
    return PairAdditivity(label, sev, _NAN, _NAN, nan, nan, nan, nan, nan, nan, _NAN, _NAN,
                          n_groups, n_sources, identical)


def additivity(probe_data: ProbeData, combo_data: ComboData, *, n_boot: int = N_BOOTSTRAP) -> AdditivityResult:
    if combo_data.X.size == 0:  # combos never encoded for this candidate → nothing to test
        return AdditivityResult(by_cell={})
    degraded = probe_data.family != CLEAN
    scaler = StandardScaler().fit(probe_data.X[degraded])
    P_std, C_std = scaler.transform(probe_data.X), scaler.transform(combo_data.X)
    clean_m = probe_data.family == CLEAN

    by_cell: dict[tuple[str, int], PairAdditivity] = {}
    for a, b in sorted({(a, b) for a, b in zip(combo_data.leg_a, combo_data.leg_b)}):
        label = combo_label(a, b)
        for sev in COMBO_SEVERITIES:
            a_m = (probe_data.family == a) & (probe_data.severity == sev)
            b_m = (probe_data.family == b) & (probe_data.severity == sev)
            ab_m = (combo_data.pair == label) & (combo_data.severity == sev)
            # The legacy selection is kept only to answer `rows_identical`: it
            # intersected speaker GROUPS and then took every row of those speakers, so a
            # clip missing one role stayed in through its siblings.
            g_common = set(probe_data.group[a_m]) & set(probe_data.group[b_m]) & set(combo_data.group[ab_m])
            common = (set(probe_data.source[a_m]) & set(probe_data.source[b_m])
                      & set(combo_data.source[ab_m]) & set(probe_data.source[clean_m]))
            if not common:  # no clip carries all four roles → not evaluable
                by_cell[(label, sev)] = _empty_cell(label, sev, len(g_common), 0, False)
                continue

            sel_p = np.isin(probe_data.source, list(common))
            sel_c = np.isin(combo_data.source, list(common))
            masks = (clean_m & sel_p, a_m & sel_p, b_m & sel_p, ab_m & sel_c)
            # The per-source displacements align the four roles by sorting each on its
            # source token, which is only valid at exactly one row per source per role.
            # A duplicated cache cell would otherwise mispair clips silently.
            counts = [int(m.sum()) for m in masks]
            if any(c != len(common) for c in counts):
                raise ValueError(
                    f"{label}@{sev}: {counts} rows across (clean, a, b, ab) for "
                    f"{len(common)} paired sources — a role has duplicate or missing cells"
                )
            g_p, g_c = np.isin(probe_data.group, list(g_common)), np.isin(combo_data.group, list(g_common))
            legacy = (clean_m & g_p, a_m & g_p, b_m & g_p, ab_m & g_c)
            identical = all(bool(np.array_equal(m, l)) for m, l in zip(masks, legacy))

            role = np.concatenate([np.full(int(m.sum()), code) for code, m in enumerate(masks)])
            group = np.concatenate([
                probe_data.group[masks[0]], probe_data.group[masks[1]],
                probe_data.group[masks[2]], combo_data.group[masks[3]],
            ])
            X_raw = _stack(probe_data.X, combo_data.X, masks)
            X_std = _stack(P_std, C_std, masks)
            da, db, dab = _displacements(probe_data.X, combo_data.X, masks,
                                         probe_data.source, combo_data.source)
            src_group = probe_data.group[masks[0]][np.argsort(probe_data.source[masks[0]])]

            cosine = bootstrap_over_groups(group, lambda i: _additivity_cosine(X_raw, role, i),
                                           n=n_boot, min_groups=MIN_TEST_GROUPS)
            r = _centroid_ratio(da, db, dab)
            by_cell[(label, sev)] = PairAdditivity(
                pair=label, severity=sev,
                cosine=cosine,
                cosine_std=bootstrap_over_groups(group, lambda i: _additivity_cosine(X_std, role, i),
                                                 n=n_boot, min_groups=MIN_TEST_GROUPS),
                r=r,
                rel_residual=float(np.sqrt(max(0.0, 1.0 + r * r - 2.0 * r * cosine.point))),
                n_groups=len(g_common), n_sources=len(common), rows_identical=identical,
                **_per_source_metrics(da, db, dab, src_group, n_boot),
            )
    return AdditivityResult(by_cell=by_cell)

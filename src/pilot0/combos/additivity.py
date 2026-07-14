"""RQ2 additivity: does a combined degradation land where the sum of its parts
predicts? For each pair (a, b) at a mid severity, compare the combined displacement
z̄(a+b) − z̄(clean) against the sum of the single displacements Δa + Δb by cosine:

    cos( z̄(a+b) − z̄(clean),  (z̄(a) − z̄(clean)) + (z̄(b) − z̄(clean)) )

The cosine constrains DIRECTION only — cos = 1 means the combo displacement is
codirectional with (not equal to) the sum of parts (dab = k·(Δa+Δb), any k > 0). ≈ 1 ⇒
the codec places the combo along the linear-superposition direction of the two single
attributes (separable, additive-friendly geometry, on the Phase-5 near-orthogonality
finding); ≪ 1 ⇒ the combination interacts nonlinearly.

Basis is reported BOTH ways because the answer is basis-dependent (physics review):
`cosine` in the RAW pooled-latent space (the proposal's z̄ — a superposition test lives
in the codec's own metric, and raw avoids the noise blow-up that standardisation
inflicts on near-constant latent channels), and `cosine_std` in the STANDARDISED basis
(one scaler fit on the degraded singles, equal-weighting every dimension so a few
high-energy channels can't set the verdict). Each cell is restricted to the sources
present in ALL of a-only / b-only / a+b, so the four role-centroids are means over the
SAME population under partial coverage. Descriptive/in-sample, source-level bootstrap
CI; on fake latents this is a plumbing check, not a result.
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


@dataclass(frozen=True)
class PairAdditivity:
    pair: str
    severity: int
    cosine: Estimate  # RAW pooled-latent basis (the proposal's z̄) — primary
    cosine_std: Estimate  # standardised basis — robustness (equal-weights dimensions)
    n_groups: int  # sources present in all four roles (the common population)


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


def additivity(probe_data: ProbeData, combo_data: ComboData, *, n_boot: int = N_BOOTSTRAP) -> AdditivityResult:
    if combo_data.X.size == 0:  # combos never encoded for this candidate → nothing to test
        return AdditivityResult(by_cell={})
    degraded = probe_data.family != CLEAN
    scaler = StandardScaler().fit(probe_data.X[degraded])
    P_std, C_std = scaler.transform(probe_data.X), scaler.transform(combo_data.X)

    by_cell: dict[tuple[str, int], PairAdditivity] = {}
    for a, b in sorted({(a, b) for a, b in zip(combo_data.leg_a, combo_data.leg_b)}):
        label = combo_label(a, b)
        for sev in COMBO_SEVERITIES:
            a_m = (probe_data.family == a) & (probe_data.severity == sev)
            b_m = (probe_data.family == b) & (probe_data.severity == sev)
            ab_m = (combo_data.pair == label) & (combo_data.severity == sev)
            common = set(probe_data.group[a_m]) & set(probe_data.group[b_m]) & set(combo_data.group[ab_m])
            if not common:  # no source carries all three degraded roles → not evaluable
                by_cell[(label, sev)] = PairAdditivity(label, sev, _NAN, _NAN, 0)
                continue

            sel_p, sel_c = np.isin(probe_data.group, list(common)), np.isin(combo_data.group, list(common))
            masks = ((probe_data.family == CLEAN) & sel_p, a_m & sel_p, b_m & sel_p, ab_m & sel_c)
            role = np.concatenate([np.full(int(m.sum()), code) for code, m in enumerate(masks)])
            group = np.concatenate([
                probe_data.group[masks[0]], probe_data.group[masks[1]],
                probe_data.group[masks[2]], combo_data.group[masks[3]],
            ])
            X_raw = _stack(probe_data.X, combo_data.X, masks)
            X_std = _stack(P_std, C_std, masks)
            by_cell[(label, sev)] = PairAdditivity(
                pair=label, severity=sev,
                cosine=bootstrap_over_groups(group, lambda i: _additivity_cosine(X_raw, role, i), n=n_boot, min_groups=MIN_TEST_GROUPS),
                cosine_std=bootstrap_over_groups(group, lambda i: _additivity_cosine(X_std, role, i), n=n_boot, min_groups=MIN_TEST_GROUPS),
                n_groups=len(common),
            )
    return AdditivityResult(by_cell=by_cell)

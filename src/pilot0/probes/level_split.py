"""Split the energy control into its level coordinate and its gain-invariant rest (S7).

Gate 1 asks a codec to beat the "level-only" energy control, and the control's
docstring called itself level-only. It is not. The pooled 6-d vector is
`[mean log P_total, mean log P_LF, mean log P_HF, std of each]`, and a pure gain g
multiplies every band power by g², so it shifts the three MEANS by 2·ln g and moves
the three STDs not at all. The gain direction is therefore exactly

    u = (1, 1, 1, 0, 0, 0) / sqrt(3)

and everything orthogonal to it is gain-invariant by construction. This module
projects onto u and onto an explicit orthonormal 5-d complement, so the question
"how much of the control's severity signal is loudness?" gets a direct answer from
the cached latents — no GPU, no re-encode, no level-matched corpus (F19: LUFS
matching does NOT remove the cue from the control, so the arm could not answer this
even if it ran).

The projection happens in the RAW pooled space, before any scaler (AM3): u is a
statement about log-energies, and standardising first would rotate it into
something else. The ridge pipeline standardises after projection, exactly as
`evaluate_severity_probes` does for the full vector.

Validity bound (F15/AM4): `EnergyEncoder` adds `_EPS = 1e-10` inside the log, so a
frame below about -80 dBFS is no longer gain-equivariant and the split's premise
weakens there. `count_frames_below` counts them so a report can state the exposure
rather than assume it away.

Section map (file order):
  LEVEL_AXIS / INVARIANT_BASIS   the gain direction and its orthonormal complement
  project_level / project_invariant   raw-space projections of a pooled matrix
  split_spaces        the three ProbeData views (6-d, level 1-d, invariant 5-d)
  evaluate_level_split   the same ridge + SRCC CI per family, on each view
  count_frames_below  the -80 dBFS exposure count behind the _EPS bound
"""

from __future__ import annotations

import numpy as np

from ..seam.energy import _EPS
from .dataset import ProbeData
from .severity import SeverityResult, evaluate_severity_probes

POOLED_DIM = 6  # mean+std pooling of the 3-d energy latent
LOW_FRAME_DBFS = -80.0  # where _EPS starts to matter (F15)

# A gain moves all three log-mean bands together and no std at all (F19).
LEVEL_AXIS = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / np.sqrt(3.0)

# Orthonormal basis of u's complement, written out rather than derived from an SVD so
# each coordinate keeps a meaning: two spectral-balance contrasts among the log-mean
# bands, then the three temporal-dispersion coordinates, which are gain-invariant on
# their own (a gain shifts a log, and std ignores a shift).
INVARIANT_BASIS = np.column_stack([
    np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0),  # total vs LF
    np.array([1.0, 1.0, -2.0, 0.0, 0.0, 0.0]) / np.sqrt(6.0),  # (total, LF) vs HF
    np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # std log P_total
    np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),  # std log P_LF
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # std log P_HF
])

SPACES = ("pooled_6d", "level_1d", "invariant_5d")
PROJECTION_SPACE = "raw"  # AM3 — recorded in the report


def _check(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != POOLED_DIM:
        raise ValueError(f"energy pooled features must be [N, {POOLED_DIM}], got {X.shape}")
    return X


def project_level(X: np.ndarray) -> np.ndarray:
    """[N, 1] — the loudness coordinate of the pooled energy vector, in raw log units."""
    return _check(X) @ LEVEL_AXIS[:, None]


def project_invariant(X: np.ndarray) -> np.ndarray:
    """[N, 5] — everything a uniform gain cannot move."""
    return _check(X) @ INVARIANT_BASIS


def split_spaces(data: ProbeData) -> dict[str, ProbeData]:
    """The same rows and labels in three feature spaces, so one probe definition can
    be scored on all of them and the numbers stay comparable."""
    views = {"pooled_6d": _check(data.X), "level_1d": project_level(data.X),
             "invariant_5d": project_invariant(data.X)}
    return {k: ProbeData(X=X, family=data.family, severity=data.severity, split=data.split,
                         group=data.group, source=data.source, encoder=data.encoder,
                         variant=f"{data.variant}:{k}")
            for k, X in views.items()}


def evaluate_level_split(data: ProbeData) -> dict[str, SeverityResult]:
    """Per family, the without-clean severity SRCC (with its bootstrap CI) of the full
    control, of its level coordinate alone, and of its gain-invariant complement.
    Where the invariant 5-d keeps most of the signal, the control is NOT level-only."""
    return {k: evaluate_severity_probes(v) for k, v in split_spaces(data).items()}


def count_frames_below(frames: np.ndarray, floor_dbfs: float = LOW_FRAME_DBFS) -> int:
    """Frames of an energy latent whose mean-square level is below `floor_dbfs`.

    Column 0 is `log(mean(x²) + _EPS)`, so a frame at d dBFS reads `ln(10**(d/10))`
    once _EPS is negligible. Below ~-80 dBFS the epsilon dominates and the coordinate
    stops being gain-equivariant — the count is the report's honest exposure to that
    bound (AM4)."""
    return int((np.asarray(frames, dtype=float)[:, 0] < floor_dbfs / 10.0 * np.log(10.0)).sum())


def eps_bound_dbfs() -> float:
    """The level at which `_EPS` equals the frame's own mean-square power."""
    return float(10.0 * np.log10(_EPS))

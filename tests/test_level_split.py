"""The energy control is not level-only (S7).

`seam/energy.py` called itself a "level-only" representation and Gate 1's G1b is
built on that claim. The pooled vector is six log-energy statistics, and a uniform
gain moves exactly one direction of it: the three log-MEANS together, none of the
three log-STDs. So the control splits cleanly into a 1-d loudness coordinate and a
5-d gain-invariant remainder, and the split says how much of its severity signal is
actually loudness.

The pooled coordinate ORDER is load-bearing here, so it is measured off the real
`EnergyEncoder` rather than assumed. The fixtures keep a ~-60 dBFS noise floor
(AM5) so every frame stays well above the `_EPS` bound at ~-80 dBFS.
"""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.probes.dataset import ProbeData
from pilot0.probes.level_split import (
    INVARIANT_BASIS,
    LEVEL_AXIS,
    count_frames_below,
    evaluate_level_split,
    project_invariant,
    project_level,
    split_spaces,
)
from pilot0.degrade.grid import FAMILIES
from pilot0.seam.base import pool_mean_std
from pilot0.seam.energy import EnergyEncoder

FAMS = list(FAMILIES)
SR = 16000
FLOOR_DB = -60.0
GAINS = (1.0, 0.6, 0.36, 0.22, 0.13)  # severity 1..5, monotone -> a pure gain ladder
TILTS = (0.1, 0.3, 0.5, 0.7, 0.9)  # HF power fraction; f(1-f) is NON-monotone in it


def _clip(rng, *, amp: float = 0.1, hf_frac: float = 0.0, seconds: float = 1.0) -> np.ndarray:
    """A low tone plus (optionally) a high tone at a fixed total power, over a -60 dBFS
    noise floor. `hf_frac` moves power from the 300 Hz band to the 6 kHz band without
    changing the total."""
    t = np.arange(int(SR * seconds)) / SR
    lo = np.sqrt(1.0 - hf_frac) * np.sin(2 * np.pi * 300.0 * t)
    hi = np.sqrt(hf_frac) * np.sin(2 * np.pi * 6000.0 * t)
    return amp * (lo + hi) + 10 ** (FLOOR_DB / 20.0) * rng.standard_normal(t.size)


def _pooled(wav) -> np.ndarray:
    return pool_mean_std(EnergyEncoder(native_sr=SR).encode(wav, SR)["energy"].frames)


def _probe_data(rows) -> ProbeData:
    X, fam, sev, split, group = zip(*rows)
    g = np.asarray(group)
    return ProbeData(X=np.asarray(X), family=np.asarray(fam), severity=np.asarray(sev, int),
                     split=np.asarray(split), group=g, source=g, encoder="energy", variant="energy")


def _ladder(kind: str, n_sources: int = 8) -> ProbeData:
    """Every family gets the same synthetic ladder — `evaluate_severity_probes` scores
    all seven, and an unpopulated family has no test rows for the ridge to predict."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n_sources):
        sp = "test" if s >= n_sources - 4 else "train"
        rows.append((_pooled(_clip(rng)), "clean", 0, sp, f"src{s}"))
        for fam in FAMS:
            for sev in range(1, 6):
                # A gain scales the WHOLE clip, noise floor included — scaling only the
                # tone would move the spectral balance too and stop being a pure gain.
                wav = (GAINS[sev - 1] * _clip(rng) if kind == "gain"
                       else _clip(rng, hf_frac=TILTS[sev - 1]))
                rows.append((_pooled(wav), fam, sev, sp, f"src{s}"))
    return _probe_data(rows)


# --- the geometry the split rests on ------------------------------------------


def test_a_pure_gain_moves_only_the_three_log_means():
    """2*ln(g) on every log-mean band, nothing on the stds — which is why the level
    direction is (1,1,1,0,0,0)/sqrt(3) and its complement is gain-invariant (F19)."""
    rng = np.random.default_rng(1)
    wav = _clip(rng)
    a, b = _pooled(wav), _pooled(0.5 * wav)
    shift = b - a
    assert shift[:3] == pytest.approx(2 * np.log(0.5) * np.ones(3), abs=0.02)
    assert shift[3:] == pytest.approx(np.zeros(3), abs=0.02)


def test_level_axis_and_invariant_basis_are_orthonormal_and_complementary():
    assert LEVEL_AXIS @ LEVEL_AXIS == pytest.approx(1.0)
    assert INVARIANT_BASIS.T @ INVARIANT_BASIS == pytest.approx(np.eye(5))
    assert INVARIANT_BASIS.T @ LEVEL_AXIS == pytest.approx(np.zeros(5))


def test_projection_is_a_gain_equivariant_split():
    rng = np.random.default_rng(2)
    X = np.stack([_pooled(_clip(rng)), _pooled(0.5 * _clip(rng, seconds=1.0))])
    Xg = X + 2 * np.log(0.25) * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # apply a -12 dB gain
    assert project_invariant(Xg) == pytest.approx(project_invariant(X), abs=1e-12)
    assert project_level(Xg) - project_level(X) == pytest.approx(
        np.full((2, 1), 2 * np.log(0.25) * np.sqrt(3.0)), abs=1e-12)


def test_split_spaces_keeps_labels_and_changes_only_the_features():
    d = _ladder("gain", n_sources=4)
    views = split_spaces(d)
    assert views["pooled_6d"].X.shape[1] == 6
    assert views["level_1d"].X.shape[1] == 1
    assert views["invariant_5d"].X.shape[1] == 5
    for v in views.values():
        assert np.array_equal(v.severity, d.severity) and np.array_equal(v.group, d.group)


def test_projection_rejects_a_non_energy_feature_matrix():
    with pytest.raises(ValueError):
        project_level(np.zeros((4, 8)))


# --- the split as an instrument -----------------------------------------------


def test_pure_gain_severity_is_read_by_the_level_coordinate_alone():
    res = evaluate_level_split(_ladder("gain"))
    lvl = res["level_1d"].by_family["noise"].without_clean.point
    inv = res["invariant_5d"].by_family["noise"].without_clean.point
    assert abs(lvl) > 0.95  # a gain ladder IS the level coordinate
    assert abs(inv) < 0.5  # ... and the invariant complement barely sees it


def test_pure_spectral_tilt_severity_is_read_by_the_invariant_complement():
    res = evaluate_level_split(_ladder("tilt"))
    lvl = res["level_1d"].by_family["noise"].without_clean.point
    inv = res["invariant_5d"].by_family["noise"].without_clean.point
    assert abs(inv) > 0.95  # constant total power, moving balance
    assert abs(lvl) < 0.5


def test_every_space_is_scored_on_every_family():
    res = evaluate_level_split(_ladder("gain", n_sources=6))
    assert set(res) == {"pooled_6d", "level_1d", "invariant_5d"}
    for r in res.values():
        assert set(r.by_family) == set(FAMS)
        assert all(np.isfinite(f.without_clean.point) for f in r.by_family.values())


# --- the _EPS validity bound (AM4) --------------------------------------------


def test_count_frames_below_the_eps_bound():
    rng = np.random.default_rng(3)
    loud = EnergyEncoder(native_sr=SR).encode(_clip(rng), SR)["energy"].frames
    assert count_frames_below(loud) == 0  # a -60 dBFS floor is well clear of -80
    silent = EnergyEncoder(native_sr=SR).encode(np.zeros(SR), SR)["energy"].frames
    assert count_frames_below(silent) == silent.shape[0]  # digital silence is all below

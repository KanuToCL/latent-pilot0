"""Gate-1 severity ceiling (S1).

A Spearman correlation between a K-level severity ladder and a continuous
prediction cannot reach 1: the ties inside each level cap it. `spearman_ceiling`
is the cap, and the reanalysis asks whether Gate 1's G1b criterion (SRCC CI-lower
>= 0.80 AND >= energy CI-upper + 0.05, on >= 4/7 families) is even reachable
under it.

Every pinned literal here is regenerated from `scipy.stats.spearmanr` on a
perfectly-ordered untied prediction (AM1) - never typed by hand. The two
UNBALANCED fixtures are the ones that matter: the balanced closed form overstates
them (F21), so it is a cross-check only.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from scipy.stats import spearmanr

from pilot0.probes.ceiling import (
    candidate_summary,
    family_ceilings,
    oracle_n_pass,
    severity_from_json,
)
from pilot0.probes.gate1 import SEVERITY_MIN_FAMILIES, SEVERITY_SRCC_MIN
from pilot0.probes.metrics import spearman_ceiling, spearman_ceiling_balanced

# The reanalysis runner lives in tools/, which is not on the test path by default.
_TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from ceiling_run import ci_excursion_caveat  # noqa: E402

FAMS = ("noise", "hiss", "hum", "clip", "bandlimit", "mp3", "dropout")

# AM1 literals, regenerated from scipy (see `test_ceiling_matches_scipy_exactly`).
PINNED = {
    (100, 100, 100, 100, 100): 0.9797978567,
    (100, 100, 100): 0.9428142795,
    (3, 100, 100): 0.8723164502,
    (100, 100, 100, 100, 98): 0.9797939376,
}

# The energy control's without-clean SRCC CI-upper per family on the real Gate-1
# run (reports/gate1_real.json), rounded to 6 dp. The fabricated report below
# mirrors these so `oracle_n_pass` is exercised against the real bounds.
REAL_ENERGY_HI = {
    "noise": 0.979814, "hiss": 0.979726, "hum": 0.943813, "clip": 0.562177,
    "bandlimit": 0.942846, "mp3": 0.575408, "dropout": 0.742871,
}


def _scipy_ceiling(counts) -> float:
    """SRCC of a perfectly-ordered, untied prediction against the tied ladder."""
    y = np.concatenate([np.full(m, j) for j, m in enumerate(counts)])
    return float(spearmanr(y, np.arange(len(y), dtype=float)).statistic)


# --- the ceiling formula ------------------------------------------------------


@pytest.mark.parametrize("counts", list(PINNED))
def test_ceiling_matches_scipy_exactly(counts):
    assert spearman_ceiling(counts) == pytest.approx(_scipy_ceiling(counts), abs=1e-10)


@pytest.mark.parametrize("counts,expected", list(PINNED.items()))
def test_ceiling_matches_the_pinned_literal(counts, expected):
    assert spearman_ceiling(counts) == pytest.approx(expected, abs=1e-9)


def test_balanced_closed_form_overstates_an_unbalanced_ladder():
    """F21: on [3, 100, 100] the balanced form is +0.07 high. It is reported as a
    cross-check only (AM6) precisely because it would gift a candidate margin."""
    counts = (3, 100, 100)
    balanced = spearman_ceiling_balanced(len(counts), sum(counts) / len(counts))
    assert balanced - spearman_ceiling(counts) == pytest.approx(0.07, abs=0.005)
    # ... and agrees exactly when the ladder IS balanced.
    assert spearman_ceiling_balanced(5, 100) == pytest.approx(spearman_ceiling([100] * 5), abs=1e-12)


def test_ceiling_is_nan_below_two_levels_or_three_rows():
    assert np.isnan(spearman_ceiling([100]))
    assert np.isnan(spearman_ceiling([]))
    assert np.isnan(spearman_ceiling([1, 1]))  # N < 3: srcc() is undefined there too


def test_ceiling_rejects_non_positive_counts():
    with pytest.raises(ValueError):
        spearman_ceiling([100, 0, 100])


# --- fabricated Gate-1 report -------------------------------------------------


def _grid(n_sources: int = 100):
    """The real common grid: clean + 7x5 minus the two band-limit cells that are
    >= Nyquist at 16 kHz, i.e. 34 cells with a K=3 band-limit ladder."""
    cells = {("clean", 0)}
    for f in FAMS:
        for s in range(1, 6):
            if f == "bandlimit" and s in (1, 2):
                continue
            cells.add((f, s))
    return cells


def _manifest(grid, n_test: int = 100, n_train: int = 410):
    rows = []
    for split, n in (("test", n_test), ("train", n_train)):
        for i in range(n):
            for f, s in sorted(grid):
                rows.append({"source": f"{split}{i}", "group": f"g{split}{i // 5}",
                             "family": f, "severity": s, "split": split})
    return {"rows": rows}


def _est(p, lo=None, hi=None):
    return {"point": p, "lo": p if lo is None else lo, "hi": p if hi is None else hi}


def _severity_block(hi_by_family, lo_by_family=None):
    lo_by_family = lo_by_family or {f: h - 0.02 for f, h in hi_by_family.items()}
    return {"by_family": {f: {"with_clean": _est(h), "without_clean": _est(h, lo_by_family[f], h)}
                          for f, h in hi_by_family.items()}}


def _report(candidate_lo=None):
    """Energy bounds mirror the real run; the single candidate is perfect (its
    without-clean CI sits exactly at the K=5 ceiling) unless told otherwise."""
    top = spearman_ceiling([100] * 5)
    lo = candidate_lo if candidate_lo is not None else {f: top for f in FAMS}
    return {
        "energy": {"name": "energy", "variant": "energy",
                   "severity": _severity_block(REAL_ENERGY_HI)},
        "decisions": [{"name": "fab", "variant": "v", "n_test_groups": 20,
                       "n_severity_pass": 0,
                       "severity": _severity_block({f: lo[f] for f in FAMS},
                                                   {f: lo[f] for f in FAMS}),
                       "energy_severity": _severity_block(REAL_ENERGY_HI)}],
        "n_common_conditions": 34,
    }


# --- family ceilings ----------------------------------------------------------


def test_level_counts_and_k_map_come_from_the_common_grid():
    grid = _grid()
    ceil = family_ceilings(_report(), _manifest(grid), grid)
    assert {f: c.K for f, c in ceil.items()} == {f: (3 if f == "bandlimit" else 5) for f in FAMS}
    assert ceil["noise"].counts_by_level == {s: 100 for s in (1, 2, 3, 4, 5)}
    assert ceil["bandlimit"].counts_by_level == {s: 100 for s in (3, 4, 5)}  # train rows excluded
    assert ceil["noise"].N == 500 and ceil["bandlimit"].N == 300


def test_family_ceilings_hit_the_balanced_pins_and_the_crosscheck_agrees_there():
    """This fixture's ladders are BALANCED (100 test rows per level), so it can only
    show that the general form reproduces the balanced pins and that the balanced
    closed form agrees where it is valid. The general form's behaviour on unbalanced
    ladders is what `test_ceiling_matches_scipy_exactly` and
    `test_balanced_closed_form_overstates_an_unbalanced_ladder` cover."""
    grid = _grid()
    ceil = family_ceilings(_report(), _manifest(grid), grid)
    assert ceil["noise"].rho_max == pytest.approx(PINNED[(100,) * 5], abs=1e-9)
    assert ceil["bandlimit"].rho_max == pytest.approx(PINNED[(100,) * 3], abs=1e-9)
    assert ceil["noise"].rho_max_balanced_crosscheck == pytest.approx(ceil["noise"].rho_max, abs=1e-12)


def test_the_energy_control_makes_most_families_infeasible():
    """The S1 result: G1b asks for CI-lower >= energy CI-upper + 0.05, but the energy
    control is itself at or near the ceiling on noise/hiss/hum/bandlimit, so the bar
    sits ABOVE the maximum attainable SRCC. Four of seven families cannot be passed
    by any representation whatsoever."""
    grid = _grid()
    ceil = family_ceilings(_report(), _manifest(grid), grid)
    feasible = {f for f, c in ceil.items() if c.feasible}
    assert feasible == {"clip", "mp3", "dropout"}
    assert ceil["noise"].required_lo > ceil["noise"].rho_max
    assert ceil["clip"].required_lo == pytest.approx(SEVERITY_SRCC_MIN)  # absolute bar binds


def test_oracle_probe_cannot_reach_the_four_family_quorum():
    grid = _grid()
    rep = _report()
    ceil = family_ceilings(rep, _manifest(grid), grid)
    n = oracle_n_pass(ceil, severity_from_json(rep["energy"]["severity"]))
    assert n == 3  # a probe pinned AT the ceiling on every family
    assert n < SEVERITY_MIN_FAMILIES  # ... still fails G1b


# --- candidate summary --------------------------------------------------------


def test_candidate_summary_separates_absolute_clears_from_margin_passes():
    grid = _grid()
    rep = _report()
    ceil = family_ceilings(rep, _manifest(grid), grid)
    (s,) = candidate_summary(rep, ceil)
    assert s.key == "fab:v"
    assert s.absolute_clears == 7  # every family clears the 0.80 absolute bar ...
    assert s.margin_passes == 3  # ... only three survive the energy margin
    assert s.feasible_families == 3
    assert s.margin_passes <= s.feasible_families  # the ceiling bounds the criterion


# --- the CI-excursion caveat is read off the report, not transcribed ----------


def test_the_excursion_caveat_names_exactly_the_families_that_excurse():
    """`energy_hi > rho_max` on noise and band-limit only. The caveat used to hardcode
    "noise +1.6e-5, bandlimit +3.1e-5", which is a statement about one past gate run;
    the families and margins must come from the report being written."""
    grid = _grid()
    ceil = family_ceilings(_report(), _manifest(grid), grid)
    note = ci_excursion_caveat(ceil)
    assert "noise +1.6e-05" in note
    assert "bandlimit +3.2e-05" in note
    for absent in ("hiss", "hum", "clip", "mp3", "dropout"):
        assert absent not in note
    assert "largest excursion is 3.2e-05" in note  # the bigger of the two, not a guess


def test_no_caveat_at_all_when_no_family_excurses():
    """A gate run whose energy control sits below every ceiling must not carry a caveat
    describing an excursion that did not happen."""
    grid = _grid()
    rep = _report()
    rep["energy"]["severity"] = _severity_block({f: 0.5 for f in FAMS})
    assert ci_excursion_caveat(family_ceilings(rep, _manifest(grid), grid)) is None


def test_a_family_with_no_finite_ceiling_is_not_reported_as_an_excursion():
    """A one-level ladder has a nan rho_max; nan comparisons are False, so it drops out
    rather than printing "family +nan"."""
    grid = {("clean", 0), ("noise", 3)}
    rep = _report()
    ceil = family_ceilings(rep, _manifest(grid), grid)
    assert np.isnan(ceil["noise"].rho_max)
    assert ci_excursion_caveat(ceil) is None

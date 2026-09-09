"""RQ2 additivity: dominance and recovery (S6).

The legacy `cosine` answers "is the combo codirectional with Δa + Δb?" and that
question is not identifiable: when one leg dwarfs the other, or when the two legs
are collinear, a combo that ignores leg b entirely still reads as additive. The
worked case is pinned below — Δab = Δa = (100, 0) with Δb = (0, 1) gives
cos = 0.99995.

So each cell now also carries: `cos_to_a` / `cos_to_b` (which leg the combo sits
on), `cos_legs` (the identifiability condition — near ±1 and no per-source fit can
separate the legs), `norm_ratio` (how lopsided the legs are), `r` and the DERIVED
`rel_residual`, and the recovery test the audit actually asked for, a per-source
least squares (α, β) with a speaker bootstrap CI.

The selection changed too: rows are paired by SOURCE across all four roles, not
intersected by speaker GROUP. `rows_identical` says where the two coincide, and on
those cells the numbers must be bit-identical to the legacy run (D3).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from pilot0.combos.additivity import _cosine, additivity
from pilot0.combos.dataset import ComboData
from pilot0.probes.dataset import ProbeData
from pilot0.release.artifacts import _additivity_json

# The second serializer lives in tools/, which is not on the test path by default.
_TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from job_c_run import additivity_dict  # noqa: E402

PAIR, SEV = "noise+clip", 3
LEG_A, LEG_B = "noise", "clip"
N_BOOT = 128


def _fab(mu_a, mu_b, mu_ab, *, n_sources: int = 6, sources_per_group: int = 1,
         drop_combo: tuple[int, ...] = (), seed: int = 0):
    """Exact displacements: every source gets its own random identity vector, added
    to all four of its rows, so it cancels in each role centroid AND in each
    per-source difference. Δa, Δb, Δab are then exactly the mu_* handed in."""
    mu_a, mu_b, mu_ab = (np.asarray(m, dtype=float) for m in (mu_a, mu_b, mu_ab))
    rng = np.random.default_rng(seed)
    dim = mu_a.size

    pX, pfam, psev, psplit, pgroup, psrc = [], [], [], [], [], []
    cX, cpair, cla, clb, csev, csplit, cgroup, csrc = ([] for _ in range(8))
    for s in range(n_sources):
        src, g = f"c{s}", f"g{s // sources_per_group}"
        ident = rng.standard_normal(dim)
        for mu, fam, sev in [(np.zeros(dim), "clean", 0), (mu_a, LEG_A, SEV), (mu_b, LEG_B, SEV)]:
            pX.append(ident + mu)
            pfam.append(fam), psev.append(sev), psplit.append("train"), pgroup.append(g), psrc.append(src)
        if s in drop_combo:
            continue  # this source never got its combo cell encoded
        cX.append(ident + mu_ab)
        cpair.append(PAIR), cla.append(LEG_A), clb.append(LEG_B)
        csev.append(SEV), csplit.append("train"), cgroup.append(g), csrc.append(src)

    probe = ProbeData(
        X=np.asarray(pX), family=np.asarray(pfam), severity=np.asarray(psev, int),
        split=np.asarray(psplit), group=np.asarray(pgroup), source=np.asarray(psrc),
        encoder="fab", variant="v",
    )
    combo = ComboData(
        X=np.asarray(cX), pair=np.asarray(cpair), leg_a=np.asarray(cla), leg_b=np.asarray(clb),
        severity=np.asarray(csev, int), split=np.asarray(csplit), group=np.asarray(cgroup),
        source=np.asarray(csrc), encoder="fab", variant="v",
    )
    return probe, combo


def _cell(mu_a, mu_b, mu_ab, **kw):
    return additivity(*_fab(mu_a, mu_b, mu_ab, **kw), n_boot=N_BOOT).by_cell[(PAIR, SEV)]


# --- the identifiability failure the new metrics exist for ---------------------


def test_pinned_cosine_of_the_dominance_case():
    """Δab = Δa = (100,0) against Δa + Δb = (100,1): the legacy cosine reads 0.99995
    — 'additive' — while leg b is entirely absent from the combo."""
    assert _cosine(np.array([100.0, 0.0]), np.array([100.0, 1.0])) == pytest.approx(
        0.9999500037496875, abs=1e-15)


def test_dominance_is_visible_in_cos_to_b_alpha_and_beta():
    c = _cell([100.0, 0.0], [0.0, 1.0], [100.0, 0.0])
    assert c.cosine.point == pytest.approx(0.9999500037496875, abs=1e-12)  # says "additive"
    assert c.cos_to_a == pytest.approx(1.0, abs=1e-12)
    assert c.cos_to_b == pytest.approx(0.0, abs=1e-12)  # ... but the combo ignores leg b
    assert c.norm_ratio == pytest.approx(100.0, abs=1e-9)  # legs differ by 40 dB
    assert c.alpha.point == pytest.approx(1.0, abs=1e-9)
    assert c.beta.point == pytest.approx(0.0, abs=1e-9)  # leg b recovers zero weight
    assert c.alpha.lo == pytest.approx(1.0, abs=1e-6)  # zero-variance fab → tight CI


def test_collinear_legs_make_the_recovery_unidentifiable():
    """cos_legs ≈ 1 → the (Δa, Δb) design is singular and no (α, β) is recoverable.
    The fit must refuse rather than return one of the infinitely many solutions."""
    c = _cell([10.0, 1.0], [5.0, 0.4], [10.0, 1.0])
    assert c.cos_legs > 0.999
    assert np.isnan(c.alpha.point) and np.isnan(c.beta.point)
    assert c.cosine.point > 0.999  # the legacy cosine still says "additive"


def test_scaled_superposition_reports_r_alpha_beta_and_a_derived_residual():
    c = _cell([3.0, 0.0], [0.0, 3.0], [0.3, 0.3])
    assert c.cosine.point == pytest.approx(1.0, abs=1e-12)  # direction is perfect ...
    assert c.r == pytest.approx(0.1, abs=1e-12)  # ... at a tenth of the magnitude
    assert c.alpha.point == pytest.approx(0.1, abs=1e-12)
    assert c.beta.point == pytest.approx(0.1, abs=1e-12)
    assert c.rel_residual == pytest.approx(0.9, abs=1e-12)
    assert c.cos_legs == pytest.approx(0.0, abs=1e-12)


def test_rel_residual_is_exactly_the_identity_in_cosine_and_r():
    """F23: rel_residual² = 1 + r² − 2r·cosine. It is derived, not measured — it
    carries no information beyond the two numbers it is built from."""
    c = _cell([3.0, 0.0], [0.0, 3.0], [0.3, 0.9])
    expect = np.sqrt(1.0 + c.r ** 2 - 2.0 * c.r * c.cosine.point)
    assert c.rel_residual == pytest.approx(expect, abs=1e-14)


# --- source pairing -----------------------------------------------------------


def test_source_missing_a_role_is_dropped_from_every_role():
    full = _cell([3.0, 0.0], [0.0, 3.0], [3.0, 3.0], n_sources=6)
    dropped = _cell([3.0, 0.0], [0.0, 3.0], [3.0, 3.0], n_sources=6, drop_combo=(0,))
    assert full.n_sources == 6
    assert dropped.n_sources == 5  # the source with no combo cell leaves all four roles


def test_rows_identical_is_true_when_group_and_source_selection_coincide():
    c = _cell([3.0, 0.0], [0.0, 3.0], [3.0, 3.0], n_sources=6)
    assert c.rows_identical is True
    assert c.n_groups == 6 and c.n_sources == 6


def test_rows_identical_is_false_when_a_sibling_clip_lacks_the_combo():
    """Two clips per speaker, one clip missing its combo cell: the speaker is still in
    the GROUP intersection, so the legacy selection keeps the sibling's rows while
    source pairing drops them. This is the case the legacy predicate could not see."""
    c = _cell([3.0, 0.0], [0.0, 3.0], [3.0, 3.0], n_sources=6, sources_per_group=2,
              drop_combo=(0,))
    assert c.rows_identical is False
    assert c.n_groups == 3  # all three speakers survive the group intersection ...
    assert c.n_sources == 5  # ... but only five clips carry all four roles


def test_uncovered_cell_reports_zero_sources_without_crashing():
    """Combo cells cached for clips that carry no singles (a partial encode) — the cell
    is not evaluable, and must come back as nan rather than a crash or a fake number."""
    probe, combo = _fab([3.0, 0.0], [0.0, 3.0], [3.0, 3.0], n_sources=4)
    combo = ComboData(**{**combo.__dict__,
                         "source": np.array([f"other{i}" for i in range(len(combo.X))]),
                         "group": np.array([f"og{i}" for i in range(len(combo.X))])})
    c = additivity(probe, combo, n_boot=N_BOOT).by_cell[(PAIR, SEV)]
    assert c.n_sources == 0 and c.n_groups == 0
    assert np.isnan(c.cosine.point) and np.isnan(c.alpha.point) and np.isnan(c.r)


# --- both serializers (AM2) ----------------------------------------------------

# The two writers predate this work and disagree on two field NAMES for the same
# numbers; `release/figures.py` reads `raw`/`std` and is out of scope (§6), so the
# alias is documented here rather than renamed.
_ALIAS = {"raw": "cosine", "std": "cosine_std"}


def test_both_serializers_emit_the_same_field_set():
    from pilot0.combos.run import ComboReport

    res = additivity(*_fab([3.0, 0.0], [0.0, 3.0], [3.0, 3.0]), n_boot=N_BOOT)
    tool_cell = additivity_dict(res)["by_cell"][0]
    lib_cell = next(iter(_additivity_json(ComboReport(additivity={"k/v": res}, transfer={}))
                         ["k/v"]["by_cell"].values()))
    assert {_ALIAS.get(k, k) for k in lib_cell} == set(tool_cell)


@pytest.mark.parametrize("field", ["cos_to_a", "cos_to_b", "cos_legs", "norm_ratio", "r",
                                   "rel_residual", "alpha", "beta", "n_sources", "rows_identical"])
def test_every_new_field_reaches_both_serializers(field):
    from pilot0.combos.run import ComboReport

    res = additivity(*_fab([3.0, 0.0], [0.0, 3.0], [3.0, 3.0]), n_boot=N_BOOT)
    tool_cell = additivity_dict(res)["by_cell"][0]
    lib_cell = next(iter(_additivity_json(ComboReport(additivity={"k/v": res}, transfer={}))
                         ["k/v"]["by_cell"].values()))
    assert field in tool_cell and field in lib_cell

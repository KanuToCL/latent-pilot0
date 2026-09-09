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


def test_rel_residual_is_the_relative_distance_from_the_sum_of_the_legs():
    """F23: rel_residual = ‖Δ̄ab − (Δ̄a + Δ̄b)‖ / ‖Δ̄a + Δ̄b‖.

    Measured here from the fixture's own role centroids. Restating the implementation's
    identity (√(1 + r² − 2r·cosine)) would only prove numpy can evaluate the same
    expression twice; this reconstructs the geometry the identity is supposed to encode,
    so an error in EITHER r or cosine shows up.

    Δ̄a = (3,0), Δ̄b = (0,3), Δ̄ab = (0.3,0.9)  (the per-source identity vectors cancel
    in every centroid), so Δ̄ab − (Δ̄a+Δ̄b) = (−2.7,−2.1) and the ratio is
    √(2.7² + 2.1²) / √(3² + 3²) = √(11.70/18) = √0.65 ≈ 0.8062257748.
    """
    probe, combo = _fab([3.0, 0.0], [0.0, 3.0], [0.3, 0.9])
    c = additivity(probe, combo, n_boot=N_BOOT).by_cell[(PAIR, SEV)]

    clean = probe.X[probe.family == "clean"].mean(axis=0)
    legs = ((probe.X[probe.family == LEG_A].mean(axis=0) - clean)
            + (probe.X[probe.family == LEG_B].mean(axis=0) - clean))
    d_ab = combo.X.mean(axis=0) - clean
    geometric = float(np.linalg.norm(d_ab - legs) / np.linalg.norm(legs))

    assert geometric == pytest.approx(np.sqrt(0.65), abs=1e-12)  # the arithmetic above
    assert c.rel_residual == pytest.approx(geometric, abs=1e-12)


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
    assert c.reason == "no_common_source"  # nothing to measure, not "measured as nan"


def test_a_duplicated_role_row_makes_the_cell_unevaluable_not_the_batch_fatal():
    """Sorting the four roles on their source tokens only pairs them at one row per
    source per role. A second `noise@3` row for one clip would mispair every clip after
    it, so the cell must be refused — but as ONE unevaluable cell. Raising here would
    abort an 18-candidate, 140-minute batch over a single bad cell while every other
    cell is still answerable."""
    probe, combo = _fab([3.0, 0.0], [0.0, 3.0], [3.0, 3.0], n_sources=4)
    dup = np.flatnonzero((probe.family == LEG_A) & (probe.severity == SEV))[0]
    probe = ProbeData(**{**probe.__dict__,
                         **{f: np.concatenate([getattr(probe, f), getattr(probe, f)[dup:dup + 1]])
                            for f in ("family", "severity", "split", "group", "source")},
                         "X": np.vstack([probe.X, probe.X[dup]])})

    res = additivity(probe, combo, n_boot=N_BOOT)  # must NOT raise
    c = res.by_cell[(PAIR, SEV)]
    assert c.reason == "duplicate_rows"
    assert np.isnan(c.cosine.point) and np.isnan(c.alpha.point)
    assert c.n_sources == 4  # the pairing population is still reported
    assert not c.rows_identical


def test_an_evaluable_cell_carries_no_reason():
    c = _cell([3.0, 0.0], [0.0, 3.0], [3.0, 3.0])
    assert c.reason is None
    assert np.isfinite(c.cosine.point)


def test_n_sources_is_not_the_evaluability_test():
    """A duplicate_rows cell reports the sources it paired and still carries no
    statistics, so `if c.n_sources` counts it as evaluable — which is what the runner's
    progress line used to do. `reason is None` is the test; this pins the difference so
    a future reader does not reintroduce the truthiness check."""
    probe, combo = _fab([3.0, 0.0], [0.0, 3.0], [3.0, 3.0], n_sources=4)
    dup = np.flatnonzero((probe.family == LEG_A) & (probe.severity == SEV))[0]
    probe = ProbeData(**{**probe.__dict__,
                         **{f: np.concatenate([getattr(probe, f), getattr(probe, f)[dup:dup + 1]])
                            for f in ("family", "severity", "split", "group", "source")},
                         "X": np.vstack([probe.X, probe.X[dup]])})

    c = additivity(probe, combo, n_boot=N_BOOT).by_cell[(PAIR, SEV)]
    assert c.n_sources > 0  # truthy ...
    assert c.reason is not None and np.isnan(c.cosine.point)  # ... but nothing to read


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
                                   "rel_residual", "alpha", "beta", "n_sources",
                                   "rows_identical", "reason"])
def test_every_new_field_reaches_both_serializers(field):
    from pilot0.combos.run import ComboReport

    res = additivity(*_fab([3.0, 0.0], [0.0, 3.0], [3.0, 3.0]), n_boot=N_BOOT)
    tool_cell = additivity_dict(res)["by_cell"][0]
    lib_cell = next(iter(_additivity_json(ComboReport(additivity={"k/v": res}, transfer={}))
                         ["k/v"]["by_cell"].values()))
    assert field in tool_cell and field in lib_cell


# --- AM8: a report that names a latent must say what the latent IS ------------------


def test_the_per_candidate_block_says_what_the_latent_is():
    """`dac44k:z` and `encodec24k:z` share a variant letter and mean opposite things.
    Reading the first as "pre-quant" is exactly the S2 error, so combos_real.json's
    per-candidate block carries `semantics` (AM8) - asserted on the block main() writes,
    not on a re-implementation of it."""
    from pilot0.combos.run import ComboReport
    from pilot0.combos.transfer import TransferResult
    from pilot0.seam.registry import variant_semantics
    from job_c_run import candidate_block

    res = additivity(*_fab([3.0, 0.0], [0.0, 3.0], [3.0, 3.0]), n_boot=N_BOOT)
    rep = ComboReport(additivity={"dac44k/z": res}, transfer={"dac44k/z": TransferResult(by_pair={})})

    block = candidate_block("dac44k", "z", rep)
    assert set(block) == {"key", "name", "variant", "semantics", "additivity", "transfer"}
    assert block["key"] == "dac44k:z"
    assert block["semantics"] == variant_semantics("dac44k", "z")
    assert "quantized" in block["semantics"]  # the S2 correction: NOT pre-quantization
    assert block["semantics"] != variant_semantics("encodec24k", "z")  # same letter, other meaning

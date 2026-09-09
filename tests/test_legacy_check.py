"""D3 regression guard (`combos/legacy_check.py`) and the duplicate-row design scan
(`combos/row_scan.py`) — both pure, both previously unreachable without a 140-minute
bank pass.

The guard's three buckets matter: the pre-guard real run reported
`n_cells_differing: 6` for six cells that are unevaluable on BOTH sides
(`hiss+bandlimit@2` on the six 16 kHz candidates), which read as a discrepancy that
did not exist. `test_cells_unevaluable_on_both_sides_are_not_counted_as_differing`
pins the corrected reading.
"""

from __future__ import annotations

import hashlib

import pytest

from pilot0.combos.additivity import AdditivityResult, PairAdditivity
from pilot0.combos.legacy_check import check_legacy, fingerprint_ok, is_unevaluable, same_value
from pilot0.combos.row_scan import scan_duplicate_rows
from pilot0.probes.metrics import Estimate

NAN = float("nan")


def _pa(pair: str, sev: int, cosine, *, identical: bool, n_sources: int = 5) -> PairAdditivity:
    """A PairAdditivity carrying only what the guard reads: cosine and rows_identical."""
    est = Estimate(NAN, NAN, NAN) if cosine is None else Estimate(*cosine)
    return PairAdditivity(
        pair=pair, severity=sev, cosine=est, cosine_std=Estimate(NAN, NAN, NAN),
        cos_to_a=NAN, cos_to_b=NAN, cos_legs=NAN, norm_ratio=NAN, r=NAN, rel_residual=NAN,
        alpha=Estimate(NAN, NAN, NAN), beta=Estimate(NAN, NAN, NAN),
        n_groups=4, n_sources=n_sources, rows_identical=identical,
    )


def _legacy(pair: str, sev: int, cosine) -> dict:
    point, lo, hi = (None, None, None) if cosine is None else cosine
    return {"pair": pair, "severity": sev, "cosine": {"point": point, "lo": lo, "hi": hi}}


def _run(cells, legacy):
    """One candidate, keyed the way analyze_additivity keys its results."""
    results = {"logmel/mel": AdditivityResult(by_cell={(c.pair, c.severity): c for c in cells})}
    return check_legacy(results, {"logmel/mel": {(c["pair"], c["severity"]): c for c in legacy}})


# --- same_value / is_unevaluable ----------------------------------------------------


@pytest.mark.parametrize("value, expected", [(None, True), (NAN, True), (0.0, False), (0.9, False)])
def test_is_unevaluable_covers_json_null_and_nan(value, expected):
    assert is_unevaluable(value) is expected


def test_same_value_matches_nan_to_json_null_but_not_to_a_number():
    assert same_value(NAN, None) is True  # to_jsonable writes nan as null
    assert same_value(NAN, NAN) is True
    assert same_value(NAN, 0.9) is False
    assert same_value(0.9, None) is False


def test_same_value_is_bit_identical_not_approximate():
    assert same_value(0.9797979797979798, 0.9797979797979798) is True
    assert same_value(0.9797979797979798, 0.9797979797979797) is False  # one ulp apart


def test_same_value_compares_two_absent_numbers_without_raising():
    """`np.isnan(None)` is a TypeError. Both legacy JSON and a nan cosine can be absent,
    so the predicate must answer for None/None rather than make the caller pre-screen."""
    assert same_value(None, None) is True
    assert same_value(None, NAN) is True
    assert same_value(None, 0.9) is False


# --- the anchor's fingerprint (C4) ---------------------------------------------------


def test_fingerprint_ok_accepts_exactly_the_hashed_bytes():
    data = b'{"by_candidate": {}}'
    digest = hashlib.sha256(data).hexdigest()
    assert fingerprint_ok(data, digest) is True
    assert fingerprint_ok(data, digest.upper()) is True  # hex case is not evidence


def test_fingerprint_ok_rejects_a_single_changed_byte():
    data = b'{"by_candidate": {}}'
    digest = hashlib.sha256(data).hexdigest()
    assert fingerprint_ok(data + b" ", digest) is False  # whitespace is a different file
    assert fingerprint_ok(b"", digest) is False
    assert fingerprint_ok(data, "0" * 64) is False


# --- the three buckets --------------------------------------------------------------


def test_identical_rows_with_a_reproduced_cosine_pass():
    got = _run([_pa("noise+clip", 3, (0.91, 0.88, 0.94), identical=True)],
               [_legacy("noise+clip", 3, (0.91, 0.88, 0.94))])
    assert got.ok
    assert (got.n_cells_identical, got.n_cosines_verified) == (1, 1)
    assert (got.n_cells_differing, got.n_cells_both_unevaluable) == (0, 0)


@pytest.mark.parametrize("field, bad", [("point", (0.92, 0.88, 0.94)),
                                        ("lo", (0.91, 0.87, 0.94)),
                                        ("hi", (0.91, 0.88, 0.95))])
def test_a_differing_cosine_field_on_identical_rows_fails_the_guard(field, bad):
    got = _run([_pa("noise+clip", 3, (0.91, 0.88, 0.94), identical=True)],
               [_legacy("noise+clip", 3, bad)])
    assert not got.ok
    assert len(got.mismatches) == 1
    assert f"cosine.{field}" in got.mismatches[0]
    assert "legacy cosine not reproduced" in got.failure


def test_a_cell_missing_from_the_legacy_report_fails_the_guard():
    got = _run([_pa("noise+clip", 3, (0.91, 0.88, 0.94), identical=True)],
               [_legacy("hum+mp3", 3, (0.5, 0.4, 0.6))])
    assert not got.ok
    assert got.n_cells_identical == 1 and got.n_cosines_verified == 0  # counted, never verified
    assert "absent from the legacy report" in got.mismatches[0]


def test_rows_identical_false_is_counted_as_differing_and_never_compared():
    """Source pairing legitimately reselects rows; a changed cosine there is expected,
    so the cell is counted and the cosine is not held to the legacy value."""
    got = _run([_pa("noise+clip", 3, (0.91, 0.88, 0.94), identical=True),
                _pa("hum+mp3", 3, (0.10, 0.05, 0.15), identical=False)],
               [_legacy("noise+clip", 3, (0.91, 0.88, 0.94)),
                _legacy("hum+mp3", 3, (0.77, 0.70, 0.80))])  # wildly different: still fine
    assert got.ok
    assert (got.n_cells_identical, got.n_cells_differing) == (1, 1)


def test_cells_unevaluable_on_both_sides_are_not_counted_as_differing():
    """The real bank's six hiss+bandlimit@2 cells at 16 kHz: no clip carries all four
    roles, so `_empty_cell` returns a nan cosine with rows_identical hardcoded False.
    They belong in their own bucket - the pre-guard run filed them as `differing`."""
    got = _run([_pa("noise+clip", 3, (0.91, 0.88, 0.94), identical=True),
                _pa("hiss+bandlimit", 2, None, identical=False, n_sources=0)],
               [_legacy("noise+clip", 3, (0.91, 0.88, 0.94)),
                _legacy("hiss+bandlimit", 2, None)])
    assert got.ok
    assert got.n_cells_both_unevaluable == 1
    assert got.n_cells_differing == 0  # the corrected reading
    assert got.n_cells_identical == 1


@pytest.mark.parametrize("identical", [True, False])
def test_a_cell_that_lost_its_number_fails_the_guard_whatever_the_rows(identical):
    """Unevaluable NOW while the legacy report has a number is a regression: the
    reanalysis went blind where the anchor could see. Reselected rows do not excuse it —
    they explain a CHANGED cosine, never an ABSENT one — so it is a mismatch (`ok`
    False), not the `differing` bucket and not the both-unevaluable one."""
    got = _run([_pa("noise+clip", 3, (0.91, 0.88, 0.94), identical=True),
                _pa("hum+mp3", 3, None, identical=identical, n_sources=0)],
               [_legacy("noise+clip", 3, (0.91, 0.88, 0.94)),
                _legacy("hum+mp3", 3, (0.77, 0.70, 0.80))])
    assert not got.ok
    assert got.n_cells_both_unevaluable == 0
    assert got.n_cells_differing == 0
    assert len(got.mismatches) == 1
    assert "hum+mp3" in got.mismatches[0] and "unevaluable now" in got.mismatches[0]
    assert "0.77" in got.mismatches[0]  # the legacy point it lost


def test_a_cell_the_legacy_report_could_not_evaluate_either_is_still_the_third_bucket():
    """The converse of the regression above: only the NEW side going blind is a failure.
    A number that never existed on either side stays in `n_cells_both_unevaluable`."""
    got = _run([_pa("noise+clip", 3, (0.91, 0.88, 0.94), identical=True),
                _pa("hiss+bandlimit", 2, None, identical=False, n_sources=0)],
               [_legacy("noise+clip", 3, (0.91, 0.88, 0.94)),
                _legacy("hiss+bandlimit", 2, None)])
    assert got.ok and got.n_cells_both_unevaluable == 1


def test_a_candidate_absent_from_the_legacy_cells_names_itself_in_the_error():
    """A bare KeyError here reads as a dict bug. The guard is handed its legacy map by
    the caller, so a missing candidate is a caller error and must say which one."""
    results = {"dac44k/enc": AdditivityResult(
        by_cell={("noise+clip", 3): _pa("noise+clip", 3, (0.9, 0.8, 1.0), identical=True)})}
    with pytest.raises(KeyError) as excinfo:
        check_legacy(results, {"logmel/mel": {}})
    assert "dac44k/enc" in str(excinfo.value)
    assert "logmel/mel" in str(excinfo.value)  # and what it WAS given


def test_no_identical_cell_anywhere_fails_even_with_zero_mismatches():
    got = _run([_pa("noise+clip", 3, (0.91, 0.88, 0.94), identical=False)],
               [_legacy("noise+clip", 3, (0.91, 0.88, 0.94))])
    assert not got.ok
    assert got.mismatches == ()
    assert "nothing anchors this reanalysis" in got.failure


def test_as_dict_carries_the_third_bucket_into_the_report():
    got = _run([_pa("noise+clip", 3, (0.91, 0.88, 0.94), identical=True)],
               [_legacy("noise+clip", 3, (0.91, 0.88, 0.94))])
    assert got.as_dict("combos_real.json") == {
        "n_cells_identical": 1, "n_cells_differing": 0, "n_cells_both_unevaluable": 0,
        "n_cosines_verified": 1, "legacy_report": "combos_real.json",
    }


# --- the duplicate-row design scan --------------------------------------------------


def _probe_rows(sources, families=("clean", "noise", "clip"), severities=(3,)):
    for s in sources:
        yield "clean", 0, s
        for f in families:
            if f == "clean":
                continue
            for sev in severities:
                yield f, sev, s


def _combo_rows(sources, pair="noise+clip", severities=(3,)):
    for s in sources:
        for sev in severities:
            yield pair, sev, s


def test_a_clean_bank_reports_no_duplicates():
    scan = scan_duplicate_rows(_probe_rows(["s1", "s2", "s3"]), _combo_rows(["s1", "s2", "s3"]))
    assert scan.ok
    assert scan.duplicates == ()
    assert (scan.n_probe_rows, scan.n_combo_rows) == (9, 3)  # 3 sources x (clean + 2 singles)
    assert scan.n_keys == 12  # 3 clean + 6 single + 3 combo keys, one row each


def test_a_duplicated_single_row_is_reported_with_its_cell():
    rows = list(_probe_rows(["s1", "s2"])) + [("noise", 3, "s1")]
    scan = scan_duplicate_rows(rows, _combo_rows(["s1", "s2"]))
    assert not scan.ok
    assert len(scan.duplicates) == 1
    dup = scan.duplicates[0]
    assert (dup.role, dup.key, dup.severity, dup.source, dup.n_rows) == ("single", "noise", 3, "s1", 2)


def test_a_duplicated_clean_row_is_keyed_on_source_alone():
    """additivity() computes `family == CLEAN` once, outside the cell loop and with no
    severity filter, so a second clean row for one source poisons every cell it enters."""
    rows = list(_probe_rows(["s1"])) + [("clean", 0, "s1")]
    scan = scan_duplicate_rows(rows, _combo_rows(["s1"]))
    assert [(d.role, d.severity, d.n_rows) for d in scan.duplicates] == [("clean", -1, 2)]


def test_a_duplicated_combo_cell_is_reported():
    scan = scan_duplicate_rows(_probe_rows(["s1"]),
                               list(_combo_rows(["s1"])) + [("noise+clip", 3, "s1")])
    assert [(d.role, d.key, d.n_rows) for d in scan.duplicates] == [("combo", "noise+clip", 2)]


def test_a_source_missing_a_role_is_not_a_duplicate():
    """The guard fires on multiplicity only: a source absent from one role drops out of
    the four-way intersection instead, and the cell is evaluated on fewer sources."""
    rows = [r for r in _probe_rows(["s1", "s2"]) if not (r[0] == "clip" and r[2] == "s2")]
    scan = scan_duplicate_rows(rows, _combo_rows(["s1"]))
    assert scan.ok


def test_the_scan_counts_every_role_source_key_it_saw():
    scan = scan_duplicate_rows(_probe_rows(["s1", "s2"], severities=(2, 3)),
                               _combo_rows(["s1", "s2"], severities=(2, 3)))
    # 2 sources x (1 clean + 2 families x 2 severities) = 10 probe keys, + 4 combo keys
    assert (scan.n_probe_rows, scan.n_combo_rows, scan.n_keys) == (10, 4, 14)
    assert scan.ok


def test_severities_do_not_collide_across_families():
    """Same source, same severity, different family — and same family, different
    severity — are three distinct keys, not one triply-occupied one."""
    scan = scan_duplicate_rows([("noise", 3, "s1"), ("clip", 3, "s1"), ("noise", 2, "s1")], [])
    assert scan.ok and scan.n_keys == 3

"""Pairwise degradation combos (Phase 7): render A+B cells into the shared cache, then
test RQ2 additivity geometry (combo displacement vs sum-of-parts) and zero-shot transfer
of the single-degradation type probe onto never-seen combinations."""

from .additivity import AdditivityResult, PairAdditivity, additivity
from .dataset import ComboData, build_combo_data
from .encode import encode_combos
from .grid import COMBO_PAIRS, COMBO_SEVERITIES, combo_label
from .legacy_check import LegacyCheck, check_legacy
from .row_scan import DuplicateRow, RowScan, scan_duplicate_rows
from .run import ComboReport, analyze_combos
from .transfer import PairTransfer, TransferResult, transfer_to_combos

__all__ = [
    "analyze_combos",
    "ComboReport",
    "additivity",
    "AdditivityResult",
    "PairAdditivity",
    "transfer_to_combos",
    "TransferResult",
    "PairTransfer",
    "build_combo_data",
    "ComboData",
    "encode_combos",
    "combo_label",
    "COMBO_PAIRS",
    "COMBO_SEVERITIES",
    # the two pure checks the S6 reanalysis stands on
    "check_legacy",
    "LegacyCheck",
    "scan_duplicate_rows",
    "RowScan",
    "DuplicateRow",
]

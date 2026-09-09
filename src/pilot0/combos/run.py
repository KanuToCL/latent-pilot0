"""Phase-7 orchestration: for each candidate representation, read the single latents
(the Phase-3 cache) and the combo latents (encode_combos), then run the additivity
geometry and the zero-shot type-probe transfer. Pure wiring — thresholds/definitions
live in the additivity/transfer modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..encode.headroom import CEILING_DBFS
from ..probes.dataset import build_probe_data
from .additivity import AdditivityResult, additivity
from .dataset import build_combo_data
from .transfer import TransferResult, transfer_to_combos


@dataclass(frozen=True)
class ComboReport:
    additivity: dict[str, AdditivityResult]
    transfer: dict[str, TransferResult]


def analyze_combos(manifest, candidates, cache_dir, *, ceiling_dbfs: float = CEILING_DBFS) -> ComboReport:
    add: dict[str, AdditivityResult] = {}
    trans: dict[str, TransferResult] = {}
    for name, variant in candidates:
        pd = build_probe_data(manifest, name, variant, cache_dir, ceiling_dbfs=ceiling_dbfs)
        cd = build_combo_data(manifest, name, variant, cache_dir, ceiling_dbfs=ceiling_dbfs)
        key = f"{name}/{variant}"
        add[key] = additivity(pd, cd)
        trans[key] = transfer_to_combos(pd, cd)
    return ComboReport(additivity=add, transfer=trans)


def analyze_additivity(manifest, candidates, cache_dir, *, ceiling_dbfs: float = CEILING_DBFS,
                       on_candidate=None) -> dict[str, AdditivityResult]:
    """Additivity only — the S6 reanalysis pass. Same cache walk as `analyze_combos`
    minus the zero-shot transfer, so a reanalysis of the banked latents does not pay
    for a probe fit it will not read. `on_candidate(key, result)` is called after each
    candidate so a long run can report progress."""
    out: dict[str, AdditivityResult] = {}
    for name, variant in candidates:
        pd = build_probe_data(manifest, name, variant, cache_dir, ceiling_dbfs=ceiling_dbfs)
        cd = build_combo_data(manifest, name, variant, cache_dir, ceiling_dbfs=ceiling_dbfs)
        key = f"{name}/{variant}"
        out[key] = additivity(pd, cd)
        if on_candidate is not None:
            on_candidate(key, out[key])
    return out

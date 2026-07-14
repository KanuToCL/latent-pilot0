"""G2b: does the reference-free head track human MOS as well as the no-reference
incumbents (NISQA/DNSMOS/UTMOS)? Per family, on the MOS-covered TEST cells, the head
and each baseline are scored by SRCC against MOS — the head trained on ViSQOL (a
proxy), so MOS is neutral ground for all of them (§8 rig avoidance).

A family counts as beaten only if the head PAIRED-beats EVERY baseline: for each
baseline the bootstrap CI-lower of [srcc(head, MOS) − srcc(baseline, MOS)], resampled
over the same test groups, is > 0. The paired difference (shared seed → identical
resamples) is the same rigor Phase 5 uses for the MLP−linear gap, and "beat every
baseline" is "beat the best".

A family is judged only if its MOS-covered TEST cells span ≥ MIN_TEST_GROUPS distinct
groups (not merely ≥ MIN_MOS_CELLS rows): under a partial-MOS subset a family can be
covered by a single source, and a cluster bootstrap over one group collapses to a
zero-width CI that would clear CI-lower > 0 on single-source evidence — exactly the
source-level protection Gate 1 is built on. The bootstrap itself also refuses a CI
below that group floor (`min_groups`), so this is belt-and-suspenders. None ⇒ no MOS
(or all-NaN) or no baselines ⇒ G2b not evaluable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..degrade.grid import FAMILIES
from ..probes.gate1 import MIN_TEST_GROUPS
from ..probes.metrics import N_BOOTSTRAP, Estimate, bootstrap_over_groups, srcc
from .dataset import QualityData

MIN_MOS_CELLS = 3  # a family needs at least this many covered test cells to be judged


@dataclass(frozen=True)
class FamilyG2b:
    head_mos: Estimate  # head vs MOS over all covered cells (reported)
    baselines_mos: dict[str, Estimate]  # each baseline vs MOS (reported)
    paired_lo: dict[str, float]  # baseline -> CI-lower of srcc(head)−srcc(baseline)
    n_cells: int  # covered MOS test cells (rows)
    n_groups: int  # distinct source groups spanning those cells

    @property
    def beats_all(self) -> bool:
        return (
            self.n_groups >= MIN_TEST_GROUPS  # source-level power, not just row count
            and self.n_cells >= MIN_MOS_CELLS
            and bool(self.paired_lo)
            and all(np.isfinite(v) and v > 0 for v in self.paired_lo.values())
        )


@dataclass(frozen=True)
class G2bResult:
    by_family: dict[str, FamilyG2b]

    def n_beats(self) -> int:
        return sum(f.beats_all for f in self.by_family.values())


def evaluate_g2b(data: QualityData, pred: np.ndarray, *, n_boot: int = N_BOOTSTRAP) -> G2bResult | None:
    if data.mos is None or not np.isfinite(data.mos).any() or not data.nr:
        return None
    te = (data.split == "test") & np.isfinite(data.mos)
    by_family: dict[str, FamilyG2b] = {}
    for fam in FAMILIES:
        fm = te & (data.family == fam)  # clean is its own family, so already degraded-only
        mos_f, head_f, g_f = data.mos[fm], pred[fm], data.group[fm]
        head_mos = bootstrap_over_groups(
            g_f, lambda i: srcc(mos_f[i], head_f[i]), n=n_boot, min_groups=MIN_TEST_GROUPS
        )

        baselines_mos: dict[str, Estimate] = {}
        paired_lo: dict[str, float] = {}
        for b, bpred in data.nr.items():
            m = fm & np.isfinite(bpred)
            y_mos, y_head, y_base, g = data.mos[m], pred[m], bpred[m], data.group[m]
            baselines_mos[b] = bootstrap_over_groups(
                g, lambda i: srcc(y_mos[i], y_base[i]), n=n_boot, min_groups=MIN_TEST_GROUPS
            )
            paired_lo[b] = bootstrap_over_groups(
                g, lambda i: srcc(y_mos[i], y_head[i]) - srcc(y_mos[i], y_base[i]),
                n=n_boot, min_groups=MIN_TEST_GROUPS,
            ).lo
        by_family[fam] = FamilyG2b(
            head_mos, baselines_mos, paired_lo, int(fm.sum()), int(len(np.unique(g_f)))
        )
    return G2bResult(by_family=by_family)

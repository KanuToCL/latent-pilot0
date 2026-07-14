"""External quality scores per degradation cell — the Gate-2 seam.

Three roles, deliberately kept distinct so the head cannot be rigged (§8, the
proposal's own elder warning "ViSQOL-trained head vs MOS-predictor incumbents"):

  * a full-reference TRAINING target — ViSQOL (PESQ on the 16 kHz arm). The head
    regresses this from latents WITHOUT the reference.
  * human MOS — the EVALUATION ground truth for G2b. Neither the head (trained on
    ViSQOL, a proxy) nor the no-reference incumbents were fit to it, so scoring all
    of them against MOS is the fair comparison. If MOS is absent, G2b is
    NOT-EVALUABLE — it is never silently scored on ViSQOL.
  * no-reference baselines — NISQA / DNSMOS / UTMOS run directly on the degraded
    audio; each predicts MOS.

On the GPU box these come from precomputed tables (metrics run offline, keyed by
cell). On the Mac a deterministic `FakeScores` synthesises them so the whole Gate-2
machinery is exercised as PLUMBING — the numbers are not perceptual claims.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..probes.metrics import srcc

REF_METRIC = "visqol"  # full-reference training target (PESQ on the 16 kHz arm)
MOS_METRIC = "mos"  # human ground truth for G2b
NR_BASELINES = ("nisqa", "dnsmos", "utmos")  # no-reference incumbents
_SCORE_MIN, _SCORE_MAX = 1.0, 5.0
# §8 rig detection: MOS that ranks ViSQOL-identically is the rig (SRCC is rank-based,
# so a monotone copy — 2·ViSQOL, ViSQOL+ε — defeats the gate while passing a byte
# check). Reject a near-perfect rank match, but only with enough shared cells that a
# genuine (correlated but distinct) MOS could not coincidentally rank-agree this hard.
_RIG_RANK_CORR_MAX = 0.999
_RIG_MIN_SHARED = 8


class Scores(Protocol):
    """Per-cell scalar for a named metric. `has(metric)` lets G2b fail closed when a
    metric was not collected at all; `get` returns None for a cell that metric does
    not cover, so a MOS SUBSET (e.g. speech-only) drops uncovered cells instead of
    crashing — G2a stays on the full corpus, G2b on the covered subset (W1)."""

    def has(self, metric: str) -> bool: ...
    def get(self, metric: str, source: str, family: str, severity: int) -> float | None: ...


def _rng(*parts) -> np.random.Generator:
    key = "|".join(str(p) for p in parts).encode()
    seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return np.random.default_rng(seed)


# Family-specific perceptual sensitivity: MOS falls at different rates per family, so
# MOS is NOT a monotone function of ViSQOL — the reason G2b must be scored on MOS.
_MOS_SENSITIVITY = {
    "noise": 1.15, "hiss": 0.85, "hum": 1.05, "clip": 1.25,
    "bandlimit": 0.75, "mp3": 0.95, "dropout": 1.30, "clean": 1.0,
}
_NR_NOISE = {"nisqa": 0.20, "dnsmos": 0.30, "utmos": 0.25}  # differing skill per baseline


@dataclass(frozen=True)
class FakeScores:
    """Deterministic plumbing scores. ViSQOL and MOS both fall with severity around a
    shared per-source content offset (so they correlate, as they do in reality) but
    MOS is family-weighted and independently noised, so recovering ViSQOL does not
    hand you MOS. NR baselines are noisy MOS predictors with differing skill."""

    seed: int = 0

    def _source_offset(self, source: str) -> float:
        return float(_rng("src", source, self.seed).normal(0.0, 0.3))  # content difficulty

    def _visqol(self, source: str, family: str, severity: int) -> float:
        sev = severity / 5.0
        val = 4.6 - 3.0 * sev + self._source_offset(source) + _rng(REF_METRIC, source, family, severity, self.seed).normal(0.0, 0.15)
        return float(np.clip(val, _SCORE_MIN, _SCORE_MAX))

    def _mos(self, source: str, family: str, severity: int) -> float:
        # Slope 2.6 (not 3.2) so pre-noise MOS stays ≥ 1 even for the most sensitive
        # family (dropout w=1.30 → 4.7−2.6·1.30 = 1.32): a heavier slope clipped
        # dropout/clip to the floor at severities 4–5, replacing the intended
        # family-weight decoupling with a tie-saturation artifact (physics W3).
        sev = severity / 5.0
        w = _MOS_SENSITIVITY[family]
        val = 4.7 - 2.6 * w * sev + self._source_offset(source) + _rng(MOS_METRIC, source, family, severity, self.seed).normal(0.0, 0.25)
        return float(np.clip(val, _SCORE_MIN, _SCORE_MAX))

    def has(self, metric: str) -> bool:
        return metric == REF_METRIC or metric == MOS_METRIC or metric in NR_BASELINES

    def get(self, metric: str, source: str, family: str, severity: int) -> float:
        if metric == REF_METRIC:
            return self._visqol(source, family, severity)
        if metric == MOS_METRIC:
            return self._mos(source, family, severity)
        if metric in NR_BASELINES:  # noisy predictor of MOS with a per-baseline skill
            noise = _rng(metric, source, family, severity, self.seed).normal(0.0, _NR_NOISE[metric])
            return float(np.clip(self._mos(source, family, severity) + noise, _SCORE_MIN, _SCORE_MAX))
        raise KeyError(f"unknown metric '{metric}'")


@dataclass(frozen=True)
class TableScores:
    """Box-side scores from precomputed metric tables (BRINGUP). Keyed by cell; a
    metric absent from the table makes `has` False (G2b not evaluable), and a cell
    absent from a present metric returns None (dropped) — so a MOS subset works."""

    table: dict[str, dict[tuple[str, str, int], float]]

    @classmethod
    def from_json(cls, path) -> "TableScores":
        with open(path) as fh:
            raw = json.loads(fh.read())
        table: dict[str, dict[tuple[str, str, int], float]] = {}
        for m, cells in raw.items():
            table[m] = {}
            for k, v in cells.items():
                try:
                    source, family, sev = k.split("|")
                    table[m][(source, family, int(sev))] = float(v)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"malformed '{m}' score cell key {k!r} = {v!r}: {e}") from e
        cls._reject_rig(table)
        return cls(table=table)

    @staticmethod
    def _reject_rig(table: dict[str, dict[tuple[str, str, int], float]]) -> None:
        """A MOS column that IS ViSQOL (up to a monotone transform) re-introduces the
        §8 rig: the head trained on ViSQOL then "beats" the baselines on a copy of its
        own target. Because the gate is rank-based, reject on RANK identity, not just
        byte identity — a byte check misses `2·ViSQOL` or a one-cell perturbation.

        The check is POOLED over shared cells on purpose. A per-family rank check was
        considered (defense against a copy scrambled across families to keep pooled
        Spearman < 0.999) and rejected: WITHIN a family both ViSQOL and MOS are
        dominated by the severity ladder, so legitimate, genuinely-distinct MOS still
        rank-agrees with ViSQOL per family — a per-family tripwire would false-reject a
        real table and halt the gate at bring-up. The cross-family divergence the
        pooled check keys on is exactly what a real MOS has and a copy does not. The
        residual (a deliberately scrambled per-family copy — fraud, not operator error)
        surfaces as implausibly perfect per-family recovery in G2a's ref_srcc_by_family."""
        if REF_METRIC not in table or MOS_METRIC not in table:
            return
        shared = sorted(set(table[REF_METRIC]) & set(table[MOS_METRIC]))
        if not shared:
            return
        ref_v = np.array([table[REF_METRIC][c] for c in shared], dtype=float)
        mos_v = np.array([table[MOS_METRIC][c] for c in shared], dtype=float)
        if np.array_equal(ref_v, mos_v):
            raise ValueError("MOS table is byte-identical to ViSQOL — that re-introduces the §8 rig")
        if len(shared) >= _RIG_MIN_SHARED:
            rho = srcc(ref_v, mos_v)
            if np.isfinite(rho) and rho >= _RIG_RANK_CORR_MAX:
                raise ValueError(
                    f"MOS ranks ViSQOL-identically over {len(shared)} shared cells "
                    f"(Spearman {rho:.4f} ≥ {_RIG_RANK_CORR_MAX}) — a monotone copy re-introduces the §8 rig"
                )

    def has(self, metric: str) -> bool:
        return metric in self.table

    def get(self, metric: str, source: str, family: str, severity: int) -> float | None:
        return self.table[metric].get((source, family, severity))

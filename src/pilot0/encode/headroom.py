"""L4 corpus headroom: one global attenuation scalar per native rate.

We never renormalise per clip (L3), so a loud additive cell (noise/hum at high
severity) can push the true peak past 0 dBFS. Before encoding we apply a single,
logged scalar per rate = min(1, ceiling / max_true_peak_over_corpus) so nothing
overloads the codec (EnCodec does not self-normalise) while every cell keeps the
SAME scale — relative energies across the severity ladder are preserved, so the
severity↔loudness confound the plan flags is not re-introduced here.

Scalar ≤ 1 (attenuate only, never amplify): amplifying would clip a quieter cell.
Computed once per rate over all renderable cells; the pipeline caches it to JSON
(resume-safe), trading a bounded one-time re-render for a simple, non-buffering,
restartable calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..degrade.base import true_peak_dbtp
from .render import render_cell

CEILING_DBFS = -1.0


@dataclass(frozen=True)
class Headroom:
    native_sr: int
    scalar: float
    max_dbtp: float
    ceiling_dbfs: float
    n_cells: int
    loudest: dict  # the cell that set max_dbtp: {source, family, severity}

    def to_dict(self) -> dict:
        return {
            "native_sr": self.native_sr,
            "scalar": self.scalar,
            "max_dbtp": self.max_dbtp,
            "ceiling_dbfs": self.ceiling_dbfs,
            "n_cells": self.n_cells,
            "loudest": self.loudest,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Headroom":
        return cls(
            native_sr=int(d["native_sr"]),
            scalar=float(d["scalar"]),
            max_dbtp=float(d["max_dbtp"]),
            ceiling_dbfs=float(d["ceiling_dbfs"]),
            n_cells=int(d["n_cells"]),
            loudest=dict(d["loudest"]),
        )


def _scalar_for(max_dbtp: float, ceiling_dbfs: float) -> float:
    if not np.isfinite(max_dbtp):
        return 1.0
    return float(min(1.0, 10.0 ** ((ceiling_dbfs - max_dbtp) / 20.0)))


def measure_headroom(
    rows: list[dict],
    load_master: Callable[[str], tuple[np.ndarray, int]],
    native_sr: int,
    ceiling_dbfs: float = CEILING_DBFS,
) -> Headroom:
    """Render every cell at `native_sr`, track the loudest true peak, and derive
    the attenuation that pulls it to `ceiling_dbfs`. `load_master(token)` returns
    the −23 LUFS master; caching it across rows is the caller's job."""
    max_dbtp = float("-inf")
    loudest: dict = {}
    for row in rows:
        master, master_sr = load_master(row["source"])
        d = render_cell(master, master_sr, native_sr, row["family"], row["severity"])
        if not np.isfinite(d.wav).all():
            raise ValueError(
                f"non-finite render for {row['source']} {row['family']} sev {row['severity']} "
                f"at {native_sr} Hz — would corrupt the headroom scalar and Phase-4 data"
            )
        pk = true_peak_dbtp(d.wav, native_sr)
        if pk > max_dbtp:
            max_dbtp = pk
            loudest = {"source": row["source"], "family": row["family"], "severity": row["severity"]}
    return Headroom(
        native_sr=native_sr,
        scalar=_scalar_for(max_dbtp, ceiling_dbfs),
        max_dbtp=max_dbtp,
        ceiling_dbfs=ceiling_dbfs,
        n_cells=len(rows),
        loudest=loudest,
    )

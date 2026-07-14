"""Phase-7 OOD-teaser tests. The out-of-grid synth is finite and varied; the
fabricated incumbents diverge; and on a grid where the metrics agree, the teaser
reports strictly MORE metric disagreement on OOD clips than in-grid."""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.ood.scores import OODNRScores
from pilot0.ood.synth import OOD_KINDS, synth_ood_clip
from pilot0.ood.teaser import ood_teaser
from pilot0.quality.dataset import QualityData
from pilot0.quality.scores import NR_BASELINES


def _fab_grid(n_sources: int = 10, dim: int = 8, noise: float = 0.05, seed: int = 0) -> QualityData:
    """Latent dim 0 carries quality; ViSQOL and all three baselines track it closely,
    so on the grid the metrics AGREE (low disagreement)."""
    rng = np.random.default_rng(seed)
    X, ref, mos, fam, sev, split, group = [], [], [], [], [], [], []
    nr = {b: [] for b in NR_BASELINES}
    for s in range(n_sources):
        sp, g, off = ("test" if s >= n_sources - 3 else "train"), f"s{s}", rng.standard_normal() * 0.3
        for sv in range(6):
            q = 5.0 - sv + off
            feat = np.zeros(dim)
            feat[0] = q + noise * rng.standard_normal()
            X.append(feat), ref.append(q + noise * rng.standard_normal())
            m = q + noise * rng.standard_normal()
            mos.append(m)
            for b in NR_BASELINES:
                nr[b].append(m + noise * rng.standard_normal())  # baselines agree on grid
            fam.append("noise" if sv > 0 else "clean"), sev.append(sv)
            split.append(sp), group.append(g)
    return QualityData(
        X=np.asarray(X), ref=np.asarray(ref), mos=np.asarray(mos),
        nr={b: np.asarray(v) for b, v in nr.items()},
        family=np.asarray(fam), severity=np.asarray(sev, int),
        split=np.asarray(split), group=np.asarray(group), encoder="fab", variant="v",
    )


def test_ood_synth_kinds_are_finite_and_distinct():
    clips = [synth_ood_clip(i, sr=16000, seconds=1.0) for i in range(len(OOD_KINDS))]
    for w in clips:
        assert np.isfinite(w).all() and np.abs(w).max() <= 0.5 + 1e-9
    # the four kinds are genuinely different waveforms, not just different labels
    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            assert not np.allclose(clips[i], clips[j])


def test_ood_baselines_diverge_and_are_bounded():
    s = OODNRScores()
    vals = [s.get(b, "ood3") for b in NR_BASELINES]
    assert all(1.0 <= v <= 5.0 for v in vals)
    assert max(vals) - min(vals) > 0.5  # incumbents disagree on the same OOD clip
    assert s.get("nisqa", "ood3") == s.get("nisqa", "ood3")  # deterministic


def test_ood_teaser_amplifies_disagreement():
    grid = _fab_grid()
    ood_X = np.random.default_rng(1).standard_normal((12, 8))  # off the training manifold
    t = ood_teaser(grid, ood_X, [f"ood{i}" for i in range(12)], OODNRScores())
    assert t.metrics == ("head", *NR_BASELINES)
    assert t.ood_disagreement > t.grid_disagreement and t.ood_amplifies_disagreement
    assert t.n_grid == grid.severity[grid.severity > 0].size and t.n_ood == 12

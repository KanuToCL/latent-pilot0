"""Smoke-path tests (Mac, fake backend)."""

from __future__ import annotations

from pilot0.seam.registry import make_encoder
from pilot0.smoke import SMOKE_ENCODERS, run


def _expected_rows(n: int) -> int:
    return n * sum(len(make_encoder(name).variants) for name in SMOKE_ENCODERS)


def test_smoke_encodes_all_clip_encoder_variants():
    rows = run(n=10)
    assert len(rows) == _expected_rows(10)
    for r in rows:
        assert r["frames"][0] == r["n_frames"]
        assert r["frame_rate_hz"] > 0


def test_smoke_is_deterministic():
    assert run(n=3) == run(n=3)

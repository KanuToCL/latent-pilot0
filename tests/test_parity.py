"""Fake/real contract parity — runs at GPU-box bring-up, skips on the Mac.

Asserts a real backend returns the SAME contract as its fake stand-in for the
same clip (same variant set; rank-2 frames; positive frame rate; shared meta
keys; latent dim matching the pinned spec) — not the same numbers. This is the
check that makes the seam trustworthy once torch is present.
"""

from __future__ import annotations

import numpy as np
import pytest

from pilot0.audio.synth import synth_clip
from pilot0.seam.registry import REAL_SPECS, make_encoder

pytest.importorskip("torch", reason="real backends require torch (GPU box only)")

_RUNNABLE = ["encodec24k", "wavlm"]  # dac/mimi wired at bring-up (Phase 5)
_META_KEYS = {"backend", "name", "variant", "native_sr", "latent_dim", "n_frames"}


@pytest.mark.parametrize("base", _RUNNABLE)
def test_fake_real_contract_parity(base):
    wav = synth_clip(0, sr=48000)
    real = make_encoder(base).encode(wav, 48000)
    fake = make_encoder(f"fake-{base}").encode(wav, 48000)

    assert set(real) == set(fake) == set(make_encoder(base).variants)
    for v, rr in real.items():
        assert rr.frames.ndim == 2
        assert rr.frame_rate_hz > 0
        assert _META_KEYS <= set(rr.meta)
        assert rr.frames.shape[1] == REAL_SPECS[base]["latent_dim"], (
            f"{base}/{v}: real D={rr.frames.shape[1]} != REAL_SPECS "
            f"{REAL_SPECS[base]['latent_dim']} — update REAL_SPECS + configs (D6)"
        )
        assert np.isfinite(rr.frames).all()

"""Hash of the degradation SEMANTICS — the algorithms, defaults, and DSP
constants that decide the rendered waveform's bytes.

`grid_signature()` covers only the level ladder (family/param/unit/levels); it
would NOT change if noise flipped white→pink, clip flipped hard→soft, or a
constant like the hum base frequency moved. The Phase-3 cache keys on this hash
too, so any such change invalidates stale latents automatically instead of
relying on a human to bump a version string (elder-adversarial finding B).

We hash MODULE SOURCE, so a comment-only edit also invalidates — deliberate:
over-invalidation just re-encodes, whereas under-invalidation silently serves a
mislabeled latent into Phase 4. The grid is pre-registered and frozen after
Phase 1, so churn is not expected in practice.
"""

from __future__ import annotations

import hashlib
import inspect

from . import bandlimit, base, clip, dropout, grid, hiss, hum, mp3, noise

# Modules whose source shapes the rendered WAV. `metrics` is excluded: it only
# fills the (uncached) measured dict, never the audio bytes.
_WAV_MODULES = (base, noise, hiss, hum, clip, bandlimit, mp3, dropout, grid)


def degrade_semantics_version() -> str:
    src = "\n".join(inspect.getsource(m) for m in _WAV_MODULES)
    return hashlib.sha256(src.encode()).hexdigest()[:12]

"""Shared FAKE-marker provenance for any PERSISTED demo/release artifact. A fake JSON
(heatmap, gate2, …) is byte-indistinguishable from a real-box one, so every writer that
persists artifacts drops a co-located `provenance.json`; `fake` is DERIVED from the
encoders in play, never a hand-set flag that could silently lie at bring-up (adversarial
review). Leaf module — stdlib only, no package imports.
"""

from __future__ import annotations

import datetime as _dt
import subprocess

BANNER = "⚠ FAKE latents + synthetic scores — plumbing, NOT results"
_FAKE_PREFIX = "fake-"  # mirrors seam.registry.FAKE_PREFIX; kept literal to stay a leaf


def is_fake(candidates) -> bool:
    """A bundle is FAKE if any candidate is a `fake-*` encoder (the Mac seam). A real
    GPU-box run uses bare names → False, so the marker tracks reality, not a constant."""
    return any(name.startswith(_FAKE_PREFIX) for name, _ in candidates)


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def provenance(fake: bool, **extra) -> dict:
    return {"fake": fake, "banner": BANNER,
            "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "git_sha": git_sha(), **extra}

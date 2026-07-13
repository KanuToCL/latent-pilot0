"""Tool + version provenance. The audio is not redistributed — regeneration
scripts are — so every binary that shapes a sample must be pinned (elder-flagged
reproducibility)."""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version

_PKGS = ("numpy", "scipy", "soxr", "soundfile", "pyloudnorm", "scikit-learn")


def _pkg_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def _ffmpeg_version() -> str:
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        return out.stdout.splitlines()[0] if out.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def provenance() -> dict:
    from .. import __version__

    return {"pilot0": __version__, "ffmpeg": _ffmpeg_version(), **{p: _pkg_version(p) for p in _PKGS}}

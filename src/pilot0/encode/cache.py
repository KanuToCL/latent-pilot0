"""Content-addressed latent cache.

Key = cache_version / name / variant / cell_id, where
`cell_id = blake2b(source_token, family, severity, native_sr)` identifies the
degraded audio (D7's content_hash generalised from the clean clip to a grid
cell). The key is locatable from the manifest WITHOUT rendering, so resume just
tests file existence; the degraded-audio sha lives in the sidecar for an optional
external integrity audit.

`cache_version` folds in everything that changes latent BYTES but not the cell
coordinates: the encode-pipeline version, the grid ladder (`grid_signature`), the
degradation semantics (`degrade_semantics_version`), the headroom ceiling, and a
corpus signature — because the L4 scalar is corpus-global, so adding/removing a
source legitimately rescales every latent and MUST invalidate the cache
(elder-adversarial findings B, C).

Writes are atomic and durable: a unique temp file (mkstemp) is fsync'd, then
`os.replace`d into place — so neither a process kill nor a power loss can leave a
half-written .npz that `is_cached` would accept as complete (findings A, E, F).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from ..corpus.manifest import grid_signature
from ..degrade.version import degrade_semantics_version
from ..seam.base import LatentResult

CODE_VERSION = "enc-v1"  # bump when the encode pipeline / seam semantics change


def code_version() -> str:
    """Byte-affecting semantics that are independent of the corpus and ceiling."""
    return f"{CODE_VERSION}+g{grid_signature()}+d{degrade_semantics_version()}"


def corpus_signature(manifest: dict) -> str:
    """Identity of the source set feeding the L4 headroom scalar."""
    pairs = sorted({(r["source"], r["content_sha"]) for r in manifest["rows"]})
    return hashlib.blake2b(json.dumps(pairs).encode(), digest_size=6).hexdigest()


def cache_version(manifest: dict, ceiling_dbfs: float) -> str:
    return f"{code_version()}+c{ceiling_dbfs:g}+k{corpus_signature(manifest)}"


def cell_id(source_token: str, family: str, severity: int, native_sr: int) -> str:
    key = f"{source_token}|{family}|{severity}|{native_sr}".encode()
    return hashlib.blake2b(key, digest_size=8).hexdigest()


def latent_path(cache_dir, name: str, variant: str, cid: str, cv: str) -> Path:
    return Path(cache_dir) / cv / name / variant / f"{cid}.npz"


def is_cached(cache_dir, name: str, variant: str, cid: str, cv: str) -> bool:
    return latent_path(cache_dir, name, variant, cid, cv).exists()


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"cannot serialise {type(o)} into latent meta")


def atomic_write(path: Path, write: Callable[[object], None]) -> None:
    """Publish `path` atomically and durably: write to a unique temp file, flush +
    fsync, then `os.replace`. `write(fh)` receives the open binary handle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_latent(path: Path, result: LatentResult, extra_meta: dict) -> None:
    if not np.isfinite(result.frames).all():
        raise ValueError(f"refusing to cache non-finite latent for {extra_meta.get('source')}")
    meta = {**result.meta, **extra_meta}
    meta_json = json.dumps(meta, default=_json_default)

    def _write(fh):
        np.savez_compressed(
            fh,
            frames=result.frames.astype(np.float32),
            frame_rate_hz=np.float64(result.frame_rate_hz),
            meta=meta_json,
        )

    atomic_write(path, _write)


def load_latent(path) -> LatentResult:
    with np.load(path, allow_pickle=False) as z:
        frames = z["frames"]
        frame_rate_hz = float(z["frame_rate_hz"])
        meta = json.loads(str(z["meta"]))
    return LatentResult(frames=frames, frame_rate_hz=frame_rate_hz, meta=meta)

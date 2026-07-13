"""Phase-3 orchestration: manifest × encoders → render (L1) → headroom (L4) →
encode → content-addressed cache. Resume-safe at two levels — the per-rate
headroom scalar and every latent are skipped if already on disk — so a run killed
on the GPU box picks up exactly where it stopped.

Backend-agnostic: `encoder_names` are `fake-*` on the Mac and bare names on the
box; nothing here imports torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..corpus.manifest import renderable_rows
from ..corpus.tokens import content_sha256
from ..seam.registry import make_encoder
from .cache import atomic_write, cache_version, cell_id, is_cached, latent_path, save_latent
from .headroom import CEILING_DBFS, Headroom, measure_headroom
from .render import load_master, render_cell


@dataclass(frozen=True)
class EncoderStats:
    name: str
    native_sr: int
    n_rows: int
    n_variants: int
    headroom: Headroom
    encoded: int  # latents written this run
    skipped: int  # latents already cached


@dataclass(frozen=True)
class EncodeReport:
    code_version: str
    per_encoder: tuple[EncoderStats, ...]

    @property
    def n_latents(self) -> int:
        return sum(s.encoded + s.skipped for s in self.per_encoder)


def _memoized_master_loader(norm_dir) -> Callable[[str], tuple[np.ndarray, int]]:
    cache: dict[str, tuple[np.ndarray, int]] = {}

    def load(token: str) -> tuple[np.ndarray, int]:
        if token not in cache:
            cache[token] = load_master(norm_dir, token)
        return cache[token]

    return load


def _headroom_path(cache_dir, cv: str, native_sr: int) -> Path:
    return Path(cache_dir) / cv / "headroom" / f"sr_{native_sr}.json"


def _resolve_headroom(cache_dir, cv, native_sr, rows, load, ceiling_dbfs) -> Headroom:
    path = _headroom_path(cache_dir, cv, native_sr)
    if path.exists():
        return Headroom.from_dict(json.loads(path.read_text()))
    head = measure_headroom(rows, load, native_sr, ceiling_dbfs)
    # Atomic like the latents: existence of this JSON must imply completeness, or a
    # concurrent worker could read a truncated file and abort the run (finding A).
    atomic_write(path, lambda fh: fh.write(json.dumps(head.to_dict(), indent=2).encode()))
    return head


def _encode_rows(enc, rows, load, scalar, cache_dir, cv) -> tuple[int, int]:
    encoded = skipped = 0
    for row in rows:
        cid = cell_id(row["source"], row["family"], row["severity"], enc.native_sr)
        missing = [v for v in enc.variants if not is_cached(cache_dir, enc.name, v, cid, cv)]
        if not missing:
            skipped += len(enc.variants)
            continue
        skipped += len(enc.variants) - len(missing)

        master, master_sr = load(row["source"])
        d = render_cell(master, master_sr, enc.native_sr, row["family"], row["severity"])
        wav = (d.wav * scalar).astype(np.float32)
        degraded_sha = content_sha256(wav, enc.native_sr)
        latents = enc.encode(wav, enc.native_sr)
        for v in missing:
            extra = {
                "source": row["source"],
                "family": row["family"],
                "severity": row["severity"],
                "param": row["param"],
                "measured": d.measured,  # achieved SNR / over_0dbfs / mp3_top_hz for Phase-4
                "headroom_scalar": scalar,
                "degraded_sha": degraded_sha,
                "code_version": cv,
            }
            save_latent(latent_path(cache_dir, enc.name, v, cid, cv), latents[v], extra)
            encoded += 1
    return encoded, skipped


def encode_corpus(
    norm_dir,
    manifest: dict,
    encoder_names: list[str],
    cache_dir,
    *,
    ceiling_dbfs: float = CEILING_DBFS,
) -> EncodeReport:
    cv = cache_version(manifest, ceiling_dbfs)
    load = _memoized_master_loader(norm_dir)
    per_encoder: list[EncoderStats] = []
    for name in encoder_names:
        enc = make_encoder(name)
        rows = renderable_rows(manifest, enc.native_sr)
        head = _resolve_headroom(cache_dir, cv, enc.native_sr, rows, load, ceiling_dbfs)
        encoded, skipped = _encode_rows(enc, rows, load, head.scalar, cache_dir, cv)
        per_encoder.append(
            EncoderStats(
                name=name,
                native_sr=enc.native_sr,
                n_rows=len(rows),
                n_variants=len(enc.variants),
                headroom=head,
                encoded=encoded,
                skipped=skipped,
            )
        )
    return EncodeReport(code_version=cv, per_encoder=tuple(per_encoder))

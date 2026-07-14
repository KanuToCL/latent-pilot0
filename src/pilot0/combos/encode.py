"""Render + encode the pairwise-combo cells into the SAME Phase-3 cache as the
singles, reusing the corpus L4 headroom scalar (per native rate) so that z̄(a+b) is
directly comparable to z̄(a), z̄(b) and z̄(clean) in the additivity test — a different
scalar would rotate the displacement vectors and corrupt the cosine.

Resume-safe (skips already-cached cells) like `encode_corpus`, and backend-agnostic
(`fake-*` on the Mac, bare names on the box). Intended to run after `encode_corpus`
on the same (manifest, cache): the headroom JSON and the single latents already
exist, so only the A+B cells are rendered here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..corpus.manifest import renderable_rows
from ..corpus.tokens import content_sha256
from ..degrade.grid import apply_degradation
from ..encode.cache import cache_version, cell_id, is_cached, latent_path, save_latent
from ..encode.headroom import CEILING_DBFS
from ..encode.pipeline import memoized_master_loader, resolve_headroom
from ..encode.render import resample_to_native
from ..seam.registry import make_encoder
from .grid import COMBO_PAIRS, apply_combo, combo_label, combo_severities


@dataclass(frozen=True)
class ComboEncodeStats:
    name: str
    native_sr: int
    encoded: int
    skipped: int


def render_combo(master, master_sr, native_sr, a, b, severity) -> np.ndarray:
    """Master → native rate (L1) → leg A → leg B, matching the single-cell render so
    the combo audio is what the codec would ingest."""
    native = resample_to_native(master, master_sr, native_sr)
    return apply_combo(native, native_sr, a, b, severity)


def _sources(manifest: dict) -> list[tuple[str, str, str]]:
    """One (source, group, split) per source — a source's group/split is constant
    across its condition rows."""
    seen: dict[str, tuple[str, str]] = {}
    for r in manifest["rows"]:
        seen.setdefault(r["source"], (r["group"], r["split"]))
    return [(s, g, sp) for s, (g, sp) in seen.items()]


def encode_combos(
    norm_dir,
    manifest: dict,
    encoder_names: list[str],
    cache_dir,
    *,
    ceiling_dbfs: float = CEILING_DBFS,
    pairs=COMBO_PAIRS,
) -> tuple[ComboEncodeStats, ...]:
    cv = cache_version(manifest, ceiling_dbfs)
    load = memoized_master_loader(norm_dir)
    stats: list[ComboEncodeStats] = []
    for name in encoder_names:
        enc = make_encoder(name)
        head = resolve_headroom(
            cache_dir, cv, enc.native_sr, renderable_rows(manifest, enc.native_sr), load, ceiling_dbfs
        )
        encoded = skipped = 0
        for source, _group, _split in _sources(manifest):
            for a, b in pairs:
                label = combo_label(a, b)
                for sev in combo_severities(a, b, enc.native_sr):
                    cid = cell_id(source, label, sev, enc.native_sr)
                    # enc.name (not the loop `name`) matches how encode_corpus keys the
                    # singles, so combos and singles never split across directories.
                    missing = [v for v in enc.variants if not is_cached(cache_dir, enc.name, v, cid, cv)]
                    skipped += len(enc.variants) - len(missing)
                    if not missing:
                        continue
                    master, master_sr = load(source)
                    wav = (render_combo(master, master_sr, enc.native_sr, a, b, sev) * head.scalar).astype(np.float32)
                    latents = enc.encode(wav, enc.native_sr)
                    degraded_sha = content_sha256(wav, enc.native_sr)
                    for v in missing:
                        save_latent(
                            latent_path(cache_dir, enc.name, v, cid, cv),
                            latents[v],
                            {"source": source, "family": label, "severity": sev,
                             "combo": [a, b], "headroom_scalar": head.scalar,
                             "degraded_sha": degraded_sha, "code_version": cv},
                        )
                        encoded += 1
        stats.append(ComboEncodeStats(name=name, native_sr=enc.native_sr, encoded=encoded, skipped=skipped))
    return tuple(stats)

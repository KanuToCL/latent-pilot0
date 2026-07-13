"""Encode & cache (Phase 3): render each manifest cell at the model's native rate
(L1), apply the corpus headroom scalar (L4), encode through the `Encoder` seam,
and store latents in a resume-safe content-addressed cache."""

from .cache import (
    cache_version,
    cell_id,
    code_version,
    corpus_signature,
    latent_path,
    load_latent,
    save_latent,
)
from .headroom import CEILING_DBFS, Headroom, measure_headroom
from .pipeline import EncodeReport, EncoderStats, encode_corpus
from .render import load_master, render_cell, resample_to_native

__all__ = [
    "cache_version",
    "cell_id",
    "code_version",
    "corpus_signature",
    "latent_path",
    "load_latent",
    "save_latent",
    "CEILING_DBFS",
    "Headroom",
    "measure_headroom",
    "EncodeReport",
    "EncoderStats",
    "encode_corpus",
    "load_master",
    "render_cell",
    "resample_to_native",
]

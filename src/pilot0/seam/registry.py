"""Encoder registry: `make_encoder(name)` and `available_encoders()`.

`fake-<base>` builds a numpy `FakeEncoder` mirroring the real `<base>` model's
native rate, latent dim, and variant set, so Mac shapes match the GPU box. A bare
`<base>` builds the torch-gated `RealCodecEncoder`.

REAL_SPECS is the runtime source of truth; configs/models.yaml mirrors it and a
test asserts they never drift (tests/test_config.py).

Every variant also carries a `semantics` string saying what its latent IS, read
through `variant_semantics()`. S2 found the label "pre-quant z" travelling into
three docs and two HTML pages while DAC's `z` was the QUANTIZER output; the string
is data with a drift guard so a report can never carry the numbers without the
meaning.
"""

from __future__ import annotations

from .energy import EnergyEncoder
from .fake import FakeEncoder
from .logmel import LogMelEncoder
from .real import RealBackendUnavailable, RealCodecEncoder

# Real model specs. `variants` are the representation points swept in §2.3.
# `latent_dim` values are nominal stand-ins for shape parity, reconciled against
# the true checkpoints at bring-up (docs/DECISIONS.md D6). RVQ-depth variants
# `d{k}` are the sum of the first k codebook embeddings — the same embedding space
# as the family's continuous encoder output, so latent_dim is shared across a
# model's variants. The name `z` does NOT mean the same thing in both codec
# families: EnCodec's is the continuous encoder output, DAC's is the quantizer
# output (S2/F7), which is why every variant carries `semantics`.
_DEPTH_SEMANTICS = {
    f"d{k}": f"dequantized sum of the first {k} RVQ codebook embeddings" for k in (1, 2, 4, 8)
}

REAL_SPECS: dict[str, dict] = {
    "encodec24k": dict(
        family="encodec", native_sr=24000, latent_dim=128,
        checkpoint="facebook/encodec_24khz", variants=("z", "d1", "d2", "d4", "d8"),
        semantics={"z": "continuous pre-quantization encoder output", **_DEPTH_SEMANTICS},
    ),
    "wavlm": dict(
        family="wavlm", native_sr=16000, latent_dim=1024,
        checkpoint="microsoft/wavlm-large", variants=("l1", "l6", "l12", "l18", "l24"),
        semantics={f"l{i}": f"transformer hidden state, layer {i}" for i in (1, 6, 12, 18, 24)},
    ),
    "dac44k": dict(
        family="dac", native_sr=44100, latent_dim=1024,
        checkpoint="descript/dac_44khz", variants=("z", "d1", "d2", "d4", "d8", "enc"),
        semantics={
            # DAC.encode() overwrites its z with the quantizer output (dac/model/dac.py
            # :243–247). The codebook count is not asserted anywhere, so this is NOT
            # labelled "= d9" (F7).
            "z": "quantized: DAC.encode() output, sum over all of the model's codebooks",
            "enc": "continuous pre-quantization encoder output (model.encoder)",
            **_DEPTH_SEMANTICS,
        },
    ),
    "mimi": dict(
        family="mimi", native_sr=24000, latent_dim=512,
        checkpoint="kyutai/mimi", variants=("semantic", "acoustic"),
        semantics={
            "semantic": "dequantized codebook 0, the WavLM-distilled semantic stream",
            "acoustic": "dequantized residual codebooks 1.., the acoustic stream",
        },
    ),
}

# Kwargs `RealCodecEncoder.__init__` takes. `semantics` is registry metadata, not an
# encoder argument, so it is filtered here rather than threaded through the backends.
_ENCODER_FIELDS = ("family", "native_sr", "latent_dim", "checkpoint", "variants")

# Families whose real encode() is implemented (torch-gated, BRINGUP-validated on
# the box). Availability = torch installed AND family wired, so the map never
# claims an encodable backend that would raise NotImplementedError. All four are
# wired as of Phase 5; GPU_BRINGUP step 1 shape-checks each before any encoding.
WIRED_FAMILIES = {"encodec", "wavlm", "dac", "mimi"}

FAKE_PREFIX = "fake-"
FLOOR_NAME = "logmel"
ENERGY_NAME = "energy"
BASELINES = {FLOOR_NAME: LogMelEncoder, ENERGY_NAME: EnergyEncoder}
# The baselines ride in the same candidate lists as the codecs (ALL_CANDIDATES =
# [FLOOR, *CANDIDATES]), so the semantics lookup must answer for them too.
# ASCII only: these strings are printed by the run tools onto a cp1252 console.
BASELINE_SEMANTICS: dict[str, dict[str, str]] = {
    FLOOR_NAME: {"mel": "log-mel filterbank magnitudes, the spectral floor baseline"},
    ENERGY_NAME: {"energy": "per-frame log band energies (total, <1 kHz, >=4 kHz), the "
                            "spectral-band energy control"},
}

# Non-torch import each family's real encode() needs. Checked (import-free, via
# find_spec) so the box doesn't report a backend available when a sub-dep is
# missing — it would otherwise raise ImportError only at encode time (S4).
_FAMILY_REQUIRES = {"encodec": "transformers", "wavlm": "transformers",
                    "mimi": "transformers", "dac": "dac"}


def make_encoder(name: str):
    """Construct an encoder by name. `fake-*` on the Mac; bare names on the box;
    `logmel` (spectral floor) and `energy` (level control) are real, torch-free
    baselines available everywhere."""
    if name in BASELINES:
        return BASELINES[name]()
    if name.startswith(FAKE_PREFIX):
        base = name[len(FAKE_PREFIX) :]
        spec = REAL_SPECS.get(base)
        if spec is None:
            raise KeyError(f"unknown fake encoder '{name}': base '{base}' not in REAL_SPECS")
        return FakeEncoder(
            name=name,
            native_sr=spec["native_sr"],
            latent_dim=spec["latent_dim"],
            variants=spec["variants"],
        )
    spec = REAL_SPECS.get(name)
    if spec is None:
        raise KeyError(f"unknown encoder '{name}'")
    return RealCodecEncoder(name=name, **_encoder_fields(spec))


def _encoder_fields(spec: dict) -> dict:
    return {k: spec[k] for k in _ENCODER_FIELDS}


def variant_semantics(name: str, variant: str) -> str:
    """What the (name, variant) latent IS, in one phrase. Required by every report and
    figure writer's candidate metadata (D1) so the numbers can never travel without the
    meaning — the S2 failure mode. `fake-<base>` answers as `<base>`: the fake mirrors
    the real variant set, so it must mirror the claim about it too."""
    base = name[len(FAKE_PREFIX):] if name.startswith(FAKE_PREFIX) else name
    by_variant = BASELINE_SEMANTICS.get(base) or (REAL_SPECS.get(base) or {}).get("semantics")
    if by_variant is None:
        raise KeyError(f"unknown encoder '{name}'")
    if variant not in by_variant:
        raise KeyError(f"encoder '{base}' has no variant '{variant}'")
    return by_variant[variant]


def candidate_semantics(candidates) -> dict[str, str]:
    """`{'name:variant': semantics}` for a candidate list — the block every report
    writer embeds beside its candidate list (D1/AM8)."""
    return {f"{n}:{v}": variant_semantics(n, v) for n, v in candidates}


def available_encoders() -> dict[str, bool]:
    """Map every encoder name → is it *encodable* here. `fake-*` always True; real
    names True only where torch AND the family's backend library are installed AND
    the family is wired."""
    import importlib.util

    out: dict[str, bool] = {b: True for b in BASELINES}  # torch-free baselines, always available
    for base in REAL_SPECS:
        out[f"{FAKE_PREFIX}{base}"] = True
    for name, spec in REAL_SPECS.items():
        try:
            RealCodecEncoder(name=name, **_encoder_fields(spec))  # raises without torch → Mac reports all False
        except RealBackendUnavailable:
            out[name] = False
            continue
        # soxr resamples the input to native rate for every real family (_to_native).
        libs = (_FAMILY_REQUIRES[spec["family"]], "soxr")
        has_libs = all(importlib.util.find_spec(m) is not None for m in libs)
        out[name] = spec["family"] in WIRED_FAMILIES and has_libs
    return out

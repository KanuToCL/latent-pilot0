"""Encoder registry: `make_encoder(name)` and `available_encoders()`.

`fake-<base>` builds a numpy `FakeEncoder` mirroring the real `<base>` model's
native rate, latent dim, and variant set, so Mac shapes match the GPU box. A bare
`<base>` builds the torch-gated `RealCodecEncoder`.

REAL_SPECS is the runtime source of truth; configs/models.yaml mirrors it and a
test asserts they never drift (tests/test_config.py).
"""

from __future__ import annotations

from .energy import EnergyEncoder
from .fake import FakeEncoder
from .logmel import LogMelEncoder
from .real import RealBackendUnavailable, RealCodecEncoder

# Real model specs. `variants` are the representation points swept in §2.3.
# `latent_dim` values are nominal stand-ins for shape parity, reconciled against
# the true checkpoints at bring-up (docs/DECISIONS.md D6).
REAL_SPECS: dict[str, dict] = {
    "encodec24k": dict(
        family="encodec", native_sr=24000, latent_dim=128,
        checkpoint="facebook/encodec_24khz", variants=("z",),
    ),
    "wavlm": dict(
        family="wavlm", native_sr=16000, latent_dim=1024,
        checkpoint="microsoft/wavlm-large", variants=("l1", "l6", "l12", "l18", "l24"),
    ),
    "dac44k": dict(
        family="dac", native_sr=44100, latent_dim=1024,
        checkpoint="descript/dac_44khz", variants=("z",),
    ),
    "mimi": dict(
        family="mimi", native_sr=24000, latent_dim=512,
        checkpoint="kyutai/mimi", variants=("semantic", "acoustic"),
    ),
}

# Families whose real encode() is implemented today; others are wired at bring-up
# (Phase 5). Availability reflects this so the map never claims an encodable
# backend that would raise NotImplementedError.
WIRED_FAMILIES = {"encodec", "wavlm"}

FAKE_PREFIX = "fake-"
FLOOR_NAME = "logmel"
ENERGY_NAME = "energy"
BASELINES = {FLOOR_NAME: LogMelEncoder, ENERGY_NAME: EnergyEncoder}


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
    return RealCodecEncoder(name=name, **spec)


def available_encoders() -> dict[str, bool]:
    """Map every encoder name → is it *encodable* here. `fake-*` always True; real
    names True only where torch is installed AND the family is wired."""
    out: dict[str, bool] = {b: True for b in BASELINES}  # torch-free baselines, always available
    for base in REAL_SPECS:
        out[f"{FAKE_PREFIX}{base}"] = True
    for name, spec in REAL_SPECS.items():
        try:
            RealCodecEncoder(name=name, **spec)
            out[name] = spec["family"] in WIRED_FAMILIES
        except RealBackendUnavailable:
            out[name] = False
    return out

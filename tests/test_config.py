"""Drift guard: configs/models.yaml must mirror REAL_SPECS exactly.

Bring-up (D6) reconciles latent dims against the real checkpoints; this test
fails loudly if the yaml and the code source-of-truth ever disagree. `semantics`
is compared too: S2 found "pre-quant z" travelling into three docs and two HTML
pages while the DAC latent was quantized, so the label is now data with a drift
guard, not prose."""

from __future__ import annotations

import pathlib

import yaml

from pilot0.seam.registry import REAL_SPECS


def test_real_specs_match_models_yaml():
    path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    doc = yaml.safe_load(path.read_text())["encoders"]
    assert set(doc) == set(REAL_SPECS)
    for name, spec in REAL_SPECS.items():
        y = doc[name]
        assert y["family"] == spec["family"]
        assert y["native_sr"] == spec["native_sr"]
        assert y["latent_dim"] == spec["latent_dim"]
        assert y["checkpoint"] == spec["checkpoint"]
        assert tuple(y["variants"]) == tuple(spec["variants"])
        assert y["semantics"] == spec["semantics"]


def test_every_variant_has_a_semantics_string():
    for name, spec in REAL_SPECS.items():
        assert set(spec["semantics"]) == set(spec["variants"]), name

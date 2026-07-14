"""Off-manifold (OOD) teaser (Phase 7): synthesise generative-style textures, encode them
at the grid's headroom scalar, and measure no-reference metric disagreement OOD vs on the
classical grid — the §9 F6 motivation (fabricated incumbents on this seam)."""

from .run import OODReport, run_ood_teaser
from .scores import OODNRScores
from .synth import synth_ood_clip
from .teaser import OODTeaser, ood_teaser

__all__ = [
    "run_ood_teaser",
    "OODReport",
    "ood_teaser",
    "OODTeaser",
    "OODNRScores",
    "synth_ood_clip",
]

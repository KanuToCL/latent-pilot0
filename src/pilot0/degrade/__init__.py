"""Degradation library (Phase 1): 7 parametric families with free labels.

Public API is the grid: `apply_degradation`, `clean`, `FAMILIES`. Individual
family functions live in their own modules; `metrics` measures the achieved
effect for the measured-vs-target acceptance table.
"""

from .base import DegradationResult
from .grid import FAMILIES, SEVERITIES, apply_degradation, clean

__all__ = ["DegradationResult", "FAMILIES", "SEVERITIES", "apply_degradation", "clean"]

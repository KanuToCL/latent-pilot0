"""Write & release (Phase 8): one synthetic corpus → strict-JSON artifacts (F1–F6
source) → rendered figures.

The barrel exports only the torch/matplotlib-FREE artifact API. `figures` and `reproduce`
are imported explicitly (`from pilot0.release.figures import ...`) because they pull the
optional `figures` extra (matplotlib) — keeping it off any code path that only writes JSON.
"""

from .artifacts import CANDIDATES, build_demo_corpus, write_artifacts

__all__ = ["build_demo_corpus", "write_artifacts", "CANDIDATES"]

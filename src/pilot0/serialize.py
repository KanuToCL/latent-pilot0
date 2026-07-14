"""Strict-JSON serialisation shared by the analysis demo and the Phase-8 release
writer. `to_jsonable` collapses numpy arrays to lists and every non-finite float
(a NaN from a degenerate SRCC or a dropped cell) to null, so the artifacts parse
under strict `json.loads`/`jq`/`JSON.parse` — no bare `NaN` tokens.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2))
    return path

"""Compatibility package for the historic ``phi_utils`` import path."""

from __future__ import annotations

import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from philosophers_stone import (  # noqa: E402,F401
    Config,
    DefaultConfig,
    checkpoint_available,
    infer_brain_health,
    infer_brain_health_from_specs,
    load_model,
)

__all__ = [
    "Config",
    "DefaultConfig",
    "checkpoint_available",
    "infer_brain_health",
    "infer_brain_health_from_specs",
    "load_model",
]

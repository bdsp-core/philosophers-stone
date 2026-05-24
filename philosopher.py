#!/usr/bin/env python
"""Backward-compatible CLI wrapper for source checkouts."""

from __future__ import annotations

import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from philosophers_stone.cli import main


if __name__ == "__main__":
    main()

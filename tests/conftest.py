"""Pytest configuration for mess-methods tests.

Adds the package ``src/`` directory to ``sys.path`` so that the
validation submodule can be imported as ``validation`` (matching the
public import path used by external consumers).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

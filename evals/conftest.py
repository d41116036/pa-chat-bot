"""Pytest hooks so `app` and `evals` import cleanly without manual PYTHONPATH."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_root = str(_PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""Run the maintained read-only November 2024 reproduction and retention audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.validate_legacy_experiments import main


if __name__ == "__main__":
    raise SystemExit(main())

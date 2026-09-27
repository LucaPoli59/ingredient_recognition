"""Generate the blind-gated deterministic report for a selector campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ingredient_selection.analysis import analyze_campaign
from src.ingredient_selection.data import SelectorDataBundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/input/yummly")
    args = parser.parse_args()
    bundle = SelectorDataBundle.load(args.data_root)
    print(json.dumps(analyze_campaign(args.output_dir, bundle), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

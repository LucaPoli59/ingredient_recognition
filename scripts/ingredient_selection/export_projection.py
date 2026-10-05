"""Export the approved P6 vocabulary definition; never rewrites split metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ingredient_selection.projection import export_projection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, help="Optional write-once destination for verification")
    args = parser.parse_args()
    result = export_projection(ROOT, args.campaign, args.output)
    print(json.dumps({key: result[key] for key in (
        "projection_id", "label_count", "class_order_hash", "artifact_hash",
        "outcome_counts", "test_split_accessed", "metadata_exported",
    )}, indent=2))


if __name__ == "__main__":
    main()

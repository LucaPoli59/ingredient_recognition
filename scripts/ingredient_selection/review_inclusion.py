"""Apply adopted D6 to saved v3 scores, preserving the original D4 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ingredient_selection.inclusion import review_inclusion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    report = review_inclusion(ROOT, args.campaign,
                              progress=lambda message: print(message, file=sys.stderr, flush=True))
    print(json.dumps({key: report[key] for key in (
        "policy_id", "outcome_counts", "sensitivity_counts", "inclusion_rule_hash", "artifact_hash",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

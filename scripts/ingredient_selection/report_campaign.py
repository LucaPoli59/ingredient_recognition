"""Generate the reproducible P4 full-profile summary and scientific figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ingredient_selection.reporting import generate_campaign_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = generate_campaign_report(args.output_dir)
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "analysis_scope": report["analysis_scope"],
        "visible_label_count": report["visible_label_count"],
        "outcome_counts": report["outcome_counts"],
        "profile_rule_hash": report["profile_rule_hash"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect a fresh campaign plan; --write prepares empty studies, never trains."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.training.benchmark_preparation import campaign_plan, prepare_campaign, inspect_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write", action="store_true", help="prepare new empty studies; refuse overwrite")
    modes.add_argument("--verify", action="store_true", help="verify an existing prepared handoff read-only")
    args = parser.parse_args()
    command = prepare_campaign if args.write else inspect_campaign if args.verify else campaign_plan
    print(json.dumps(command(args.run_name), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

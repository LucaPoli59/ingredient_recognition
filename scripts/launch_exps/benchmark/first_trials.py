"""Print the first-release plan, or explicitly run its five serial trials."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.training.benchmark_protocol import policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepared", action="store_true", help="explicitly start the verified empty R3 campaign")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"run_name": args.run_name, "models": policy()["models"],
                          "trials_per_model": 1, "epochs_per_trial": 40,
                          "next_state": "awaiting_cost_review"}, indent=2))
        return
    from src.training.benchmark_queue import first_trials
    print(json.dumps(first_trials(args.run_name, prepared=args.prepared), indent=2))


if __name__ == "__main__":
    main()

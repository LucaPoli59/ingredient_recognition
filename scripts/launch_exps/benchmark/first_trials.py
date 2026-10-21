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
    parser.add_argument("--full-budget", action="store_true", help="reference0 then49 further attempts/model, without another cost pause")
    parser.add_argument("--cost-reviewed", action="store_true", help="accept the historical-cost estimate for the entire release")
    args = parser.parse_args()
    if args.full_budget and not (args.prepared and args.cost_reviewed):
        parser.error("--full-budget requires --prepared --cost-reviewed")
    if not args.execute:
        print(json.dumps({"run_name": args.run_name, "models": policy()["models"],
                          "trials_per_model": 50 if args.full_budget else 1, "epochs_per_trial": 40,
                          "next_state": "complete" if args.full_budget else "awaiting_cost_review"}, indent=2))
        return
    from src.training.benchmark_queue import first_trials
    options = {"prepared": args.prepared}
    if args.full_budget:
        options.update(full_budget=True, cost_reviewed=True)
    print(json.dumps(first_trials(args.run_name, **options), indent=2))


if __name__ == "__main__":
    main()

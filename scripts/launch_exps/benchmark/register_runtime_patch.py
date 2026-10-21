"""Inspect/register the final-checkpoint repair without starting any trial."""

import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.training.benchmark_provenance import policy, inspect_patch, register_checkpoint_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--accept-missing-final", action="store_true")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("a simple existing campaign name is required")
    if args.apply != args.accept_missing_final:
        parser.error("registration requires both --apply and --accept-missing-final")
    base = ROOT / "experiments/benchmark_v1"
    campaign = base / args.run_name
    queue = json.loads((campaign / "campaign_queue.json").read_text())
    if queue["status"] != "awaiting_cost_review":
        parser.error("the first-release queue must be finished and awaiting review")
    with (base / ".controller.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        studies = [campaign / name for name in policy()["models"]]
        # Check every study before registering any of them.
        inspected = {path.name: inspect_patch(path) for path in studies}
        result = ({path.name: register_checkpoint_patch(path, accept_missing_final=True) for path in studies}
                  if args.apply else inspected)
        if not args.apply:
            result = {name: {**{key: value for key, value in item.items() if key != "preserved_files"},
                             "preserved_file_count": len(item.get("preserved_files", {}))}
                      for name, item in result.items()}
    print(json.dumps({"applied": args.apply, "training_started": False, "studies": result}, indent=2))


if __name__ == "__main__":
    main()

"""Inspect the Phase 6 recipe by default; training requires --execute explicitly."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.training.benchmark_protocol import policy, make_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=policy()["models"])
    parser.add_argument("--run-name")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--additional-trials", type=int, default=1)
    parser.add_argument("--cost-reviewed", action="store_true")
    parser.add_argument("--export-config", type=Path, help="new config file only; does not construct a model")
    args = parser.parse_args()
    if not args.execute:
        if args.resume or args.cost_reviewed or args.additional_trials != 1:
            parser.error("execution options require --execute")
        if args.export_config:
            if not args.model or args.export_config.exists():
                parser.error("config export requires --model and a new output file")
            make_config(args.model, ROOT / "experiments/benchmark_v2/CONFIG_ONLY").save_to_file(args.export_config)
        print(json.dumps(policy(), indent=2))
        return
    if args.export_config or not args.model or not args.run_name or Path(args.run_name).name != args.run_name:
        parser.error("execution requires --model and a simple --run-name, without config export")
    from src.training.benchmark_campaign import execute, CAMPAIGN_ROOT
    print(json.dumps(execute(CAMPAIGN_ROOT / args.run_name / args.model,
                             args.model, resume=args.resume, additional_trials=args.additional_trials,
                             cost_reviewed=args.cost_reviewed), indent=2))


if __name__ == "__main__":
    main()

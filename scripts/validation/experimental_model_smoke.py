"""Rerunnable Phase 5.5 probes and real train/validation/consumer qualification."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.experimental_qualification import MODEL_TYPES, execute_job, run_portfolio, serve_dashboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", help="new isolated directory under experiments/runtime_smoke")
    parser.add_argument("--models", nargs="+", choices=MODEL_TYPES, help="default: all three models")
    parser.add_argument("--serve-dashboard", choices=MODEL_TYPES, help="serve a completed smoke for browser checks")
    parser.add_argument("--port", type=int, default=8068)
    parser.add_argument("--worker", choices=("probe", "smoke"), help=argparse.SUPPRESS)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=MODEL_TYPES, help=argparse.SUPPRESS)
    parser.add_argument("--adaptation", choices=("full", "frozen_encoder"), help=argparse.SUPPRESS)
    parser.add_argument("--physical-batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--selected", action="store_true", help="selected-task browser check; also an internal worker field")
    args = parser.parse_args()
    if args.serve_dashboard:
        if not args.run_name or args.worker or args.models or any((args.job, args.model, args.adaptation, args.physical_batch)):
            parser.error("dashboard serving requires --run-name and a completed model only")
        serve_dashboard(args.run_name, args.serve_dashboard, args.selected, args.port)
        return 0
    if args.worker:
        if not all((args.job, args.model, args.adaptation, args.physical_batch)) or args.run_name or args.models:
            parser.error("worker requires its exact job/model/adaptation/batch fields only")
        return execute_job(args.job, args.model, args.adaptation, args.physical_batch, args.selected, args.worker)
    if not args.run_name or any((args.job, args.model, args.adaptation, args.physical_batch, args.selected)):
        parser.error("provide --run-name; worker fields are internal")
    print(run_portfolio(args.run_name, args.models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

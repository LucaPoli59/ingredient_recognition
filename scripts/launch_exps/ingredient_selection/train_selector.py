"""Rerun Phase 3-D1-v3 for 40 epochs with effective batch 128 and a CUDA gate.

Examples (from the repository, using the WSL ML interpreter):
    python scripts/launch_exps/ingredient_selection/train_selector.py
    python scripts/launch_exps/ingredient_selection/train_selector.py --run-name repeat_01
    python scripts/launch_exps/ingredient_selection/train_selector.py --probe-batches

Capacity trials are disposable. A fresh model is constructed for the campaign.
"""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.ingredient_selection.batching import batch_candidates
from src.ingredient_selection.protocol import PROTOCOL_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=PROTOCOL_ID)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--probe-batches", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be one directory name")
    output = ROOT / "analysis_outputs/ingredient_selection" / args.run_name
    experiment = ROOT / "experiments/ingredient_selection" / args.run_name
    gate = ROOT / "analysis_outputs/ingredient_selection" / f"{args.run_name}_resource_gate.json"
    command = [sys.executable, str(ROOT / "scripts/ingredient_selection/run_campaign.py"),
               "--num-workers", str(args.num_workers), "--output-dir", str(output),
               "--experiment-dir", str(experiment), "--resource-gate-output", str(gate)]
    if args.probe_batches:
        for physical in batch_candidates(128):
            version = PROTOCOL_ID.rsplit("-", 1)[-1]
            trial_path = ROOT / f"analysis_outputs/ingredient_selection/capacity_{version}" / f"batch_{physical:03d}.json"
            trial_command = command[:-2] + ["--resource-gate-output", str(trial_path),
                                           "--capacity-probe", "--physical-batch-size", str(physical)]
            result = subprocess.run(trial_command, cwd=ROOT)
            if result.returncode not in (0, 2):
                raise RuntimeError(f"capacity trial {physical} failed for a reason other than CUDA OOM")
            if result.returncode == 0:
                print(f"Largest feasible divisor in quick probes: {physical}. "
                      "Set MAX_ALLOWED_BATCH_SIZE, then run the full-epoch gate.", flush=True)
                return 0
        raise RuntimeError("no physical batch size fits the GPU")
    if output.exists() and any(output.iterdir()) and not args.gate_only:
        parser.error("campaign output exists; use --run-name for a fresh rerun")
    subprocess.run(command + ["--resource-gate-only"], cwd=ROOT, check=True)
    if not args.gate_only:
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Serial first-trial release, followed by the declared measured-cost review."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

from src.training.benchmark_campaign import execute
from src.training.benchmark_baselines import run_baselines
from src.training.benchmark_protocol import policy, policy_hash


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def first_trials(run_name):
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("a simple new run name is required")
    repo = Path(__file__).resolve().parents[2]
    root = repo / "experiments" / "benchmark_v1" / run_name
    root.mkdir(parents=True, exist_ok=False)
    state_path = root / "campaign_queue.json"
    state = {"run_name": run_name, "policy_hash": policy_hash(), "pid": os.getpid(),
             "started_at": timestamp(), "status": "preparing_baselines",
             "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
             "release": "one_unpruned_40_epoch_trial_per_model",
             "models": [{"model_id": name, "status": "pending"} for name in policy()["models"]]}
    save_state(state_path, state)
    current = None
    try:
        run_baselines(root / "baselines")
        state["baselines"] = "baselines/baseline_summary.json"
        for current in state["models"]:
            current.update(status="running", started_at=timestamp())
            state.update(status="running", updated_at=timestamp())
            save_state(state_path, state)
            print(f"Starting {current['model_id']} trial 0 at {current['started_at']}", flush=True)
            current["result"] = execute(root / current["model_id"], current["model_id"])
            current.update(status="complete", completed_at=timestamp())
            save_state(state_path, state)
        state.update(status="awaiting_cost_review", completed_at=timestamp())
        save_state(state_path, state)
        return state
    except BaseException as error:
        if current is not None:
            current.update(status="failed", failed_at=timestamp(), error=f"{type(error).__name__}: {error}")
        state.update(status="failed", failed_at=timestamp(), error=f"{type(error).__name__}: {error}")
        save_state(state_path, state)
        traceback.print_exc()
        raise

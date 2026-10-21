"""Serial benchmark releases with explicit historical-cost budget approval."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

from src.training.benchmark_campaign import execute, CAMPAIGN_ROOT
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


def first_trials(run_name, *, prepared=False, full_budget=False, cost_reviewed=False):
    if full_budget and not (prepared and cost_reviewed):
        raise ValueError("full-budget release requires prepared studies and explicit cost approval")
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("a simple new run name is required")
    repo = Path(__file__).resolve().parents[2]
    root = CAMPAIGN_ROOT / run_name
    if prepared:
        from src.training.benchmark_preparation import inspect_campaign
        preparation = inspect_campaign(run_name)
    else:
        root.mkdir(parents=True, exist_ok=False)
    state_path = root / "campaign_queue.json"
    state = {"run_name": run_name, "policy_hash": policy_hash(), "pid": os.getpid(),
             "started_at": timestamp(), "status": "preparing_baselines",
             "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
             "release": "full_fifty_attempt_budget" if full_budget else "one_unpruned_40_epoch_trial_per_model",
             "models": [{"model_id": name, "status": "pending"} for name in policy()["models"]]}
    if full_budget:
        state["cost_approval"] = {"basis": "historical_first_release_estimate",
            "decision": "2026-10-21 user waived repeat cost review; no cost-only trials",
            "projection": preparation["cost_projection"], "attempts_per_model": 50,
            "post_reference_pause": False}
    save_state(state_path, state)
    current = None
    try:
        run_baselines(root / "baselines")
        state["baselines"] = "baselines/baseline_summary.json"
        phases = ("reference", "remaining") if full_budget else ("reference",)
        for phase in phases:
            for current in state["models"]:
                current.update(status="running", started_at=timestamp())
                state.update(status="running", phase=phase, updated_at=timestamp())
                save_state(state_path, state)
                print(f"Starting {current['model_id']} {phase} release at {current['started_at']}", flush=True)
                options = ({"prepared": True} if prepared else {}) if phase == "reference" else {
                    "resume": True, "additional_trials": policy()["hpo"]["started_trials_per_model"] - 1,
                    "cost_reviewed": True}
                result = execute(root / current["model_id"], current["model_id"], **options)
                current["reference_result" if full_budget and phase == "reference" else "result"] = result
                if full_budget and phase == "remaining" and (result["trials_started"] != 50 or result["remaining"] != 0):
                    raise ValueError("full-budget release returned inconsistent attempt accounting")
                current.update(status="reference_complete" if full_budget and phase == "reference" else "complete",
                               completed_at=timestamp())
                save_state(state_path, state)
        state.update(status="complete" if full_budget else "awaiting_cost_review", completed_at=timestamp())
        save_state(state_path, state)
        return state
    except BaseException as error:
        if current is not None:
            current.update(status="failed", failed_at=timestamp(), error=f"{type(error).__name__}: {error}")
        state.update(status="failed", failed_at=timestamp(), error=f"{type(error).__name__}: {error}")
        save_state(state_path, state)
        traceback.print_exc()
        raise

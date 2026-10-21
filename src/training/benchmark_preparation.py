"""Non-training preparation and read-only inspection of the Phase 6 v2 campaign."""

from datetime import datetime, timezone
import json
from pathlib import Path

from src.training import benchmark_campaign as campaign
from src.training.benchmark_protocol import policy, policy_hash
from src.training.benchmark_provenance import sha256

REPO = Path(__file__).resolve().parents[2]
COST_RECORD = REPO / "src_scratches/phase6_first_release_audit/summary.json"
COST_RECORD_HASH = "2a8eaa78edbe26b48437bba35f1538685fa5d943242aaa8e2b6b44b9d955cd0e"


def campaign_root(run_name):
    if not isinstance(run_name, str) or Path(run_name).name != run_name or run_name in ("", ".", "..") or "\\" in run_name:
        raise ValueError("a simple new campaign name is required")
    return campaign.CAMPAIGN_ROOT / run_name


def campaign_plan(run_name):
    root, p = campaign_root(run_name), policy()
    return {"run_name": run_name, "protocol_id": p["protocol_id"], "policy_hash": policy_hash(),
            "training_authorized": False, "status": "plan_only", "campaign_root": str(root),
            "models": [{"model_id": name, "study_name": f"{p['protocol_id']}-{name}",
                        "study_path": str(root / name), "started_attempt_cap": p["hpo"]["started_trials_per_model"],
                        "epochs": p["epochs"], "optimizer_updates_per_epoch": p["optimizer_updates_per_epoch"],
                        "complete_updates": p["epochs"] * p["optimizer_updates_per_epoch"],
                        "physical_batch": p["physical_batches"][name], "effective_batch": p["effective_batch"],
                        "accumulation": p["effective_batch"] // p["physical_batches"][name]}
                       for name in p["models"]],
            "first_release_attempts_per_model": 1, "next_state_after_first_release": "awaiting_cost_review",
            "full_hpo_attempt_allowance": len(p["models"]) * p["hpo"]["started_trials_per_model"],
            "plateau": p["hpo"]["plateau"], "old_trial_import": False,
            "q3_completed_selector_restarted": False}


def cost_projection():
    """Only recorded costs are consumed; old AP/weights never enter the new studies."""
    if sha256(COST_RECORD) != COST_RECORD_HASH:
        raise ValueError("the reviewed historical cost record changed")
    saved = json.loads(COST_RECORD.read_text())
    cap = policy()["hpo"]["started_trials_per_model"]
    seconds = {row["model_id"]: row["release_wall_seconds"] for row in saved["models"]}
    if set(seconds) != set(policy()["models"]):
        raise ValueError("historical cost protocols differ")
    return {"source": str(COST_RECORD.relative_to(REPO)), "source_sha256": COST_RECORD_HASH,
            "spent_v1_wall_seconds": sum(seconds.values()), "spent_v1_attempts": 5,
            "reference_release_wall_seconds": seconds,
            "new_hpo_full_run_equivalent_days": sum(seconds.values()) * cap / 86400,
            "interpretation": "conditional unpruned arithmetic, not an ETA, bound or measured GPU-active time",
            "excluded": "later Q2/Q3/Q4 work, new inference/storage overhead and interruptions",
            "pruning_savings": "unknown; old five attempts are spent work outside the new allowance"}


def prepare_campaign(run_name):
    """Create five empty journals/configurations/snapshots, without any trial allocation."""
    plan = campaign_plan(run_name)
    root = campaign_root(run_name)
    if root.exists():
        raise FileExistsError(root)
    costs = cost_projection()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for model_id in policy()["models"]:
            study, _ = campaign.prepare_study(root / model_id, model_id)
            if study.trials:
                raise ValueError("preparation must leave every study empty")
            records.append({"model_id": model_id, "trials_started": 0,
                            "prepared_study_sha256": sha256(root / model_id / "prepared_study.json")})
        manifest = {"schema_version": 1, "status": "prepared_not_launched",
                    "prepared_at": datetime.now(timezone.utc).isoformat(), "plan": plan,
                    "cost_projection": costs, "studies": records}
        campaign.write_new(root / "preparation_manifest.json", manifest)
        return inspect_campaign(run_name)
    except BaseException as error:
        campaign.write_new(root / "preparation_failure.json", {"error": f"{type(error).__name__}: {error}",
            "training_started": False, "completed_studies": records})
        raise


def inspect_campaign(run_name):
    """Validate the whole prepared handoff read-only; never opens Optuna for writing."""
    root = campaign_root(run_name)
    manifest = json.loads((root / "preparation_manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "prepared_not_launched"
            or manifest.get("plan") != campaign_plan(run_name)
            or manifest.get("cost_projection") != cost_projection()
            or [row["model_id"] for row in manifest.get("studies", [])] != policy()["models"]
            or (root / "campaign_queue.json").exists() or (root / "preparation_failure.json").exists()):
        raise ValueError("prepared campaign identity changed or launch/preparation already began")
    for row in manifest["studies"]:
        directory = root / row["model_id"]
        if row["trials_started"] != 0 or sha256(directory / "prepared_study.json") != row["prepared_study_sha256"]:
            raise ValueError("prepared study manifest differs")
        campaign.inspect_prepared_study(directory, row["model_id"])
    return manifest

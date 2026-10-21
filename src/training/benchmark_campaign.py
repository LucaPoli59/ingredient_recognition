"""Serial, isolated Phase 6 HPO over the canonical model_training function.

    No historical journal replay/deletion, silent retry, automatic budget
    extension, or implicit run when producing a plan. Resume is supported only
    between finished trials, with the exact trusted local sampler state.
"""

import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time
import zipfile
from importlib.metadata import version
from unittest.mock import patch

import lightning as lgn
import optuna
from optuna.trial import TrialState
import torch

from src.training.benchmark_protocol import policy, policy_hash, make_config
from src.training.benchmark_stopping import eligible_complete_trials

CAMPAIGN_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "benchmark_v2"


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def _storage(root):
    return optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(root / "journal.log")))


def _save_sampler(root, study):
    # Only read files created by this local controller; pickle is not an import format.
    payload = pickle.dumps({"sampler": study.sampler, "trials": len(study.trials), "optuna": optuna.__version__})
    envelope = hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload
    temporary = root / "sampler.tmp"
    with temporary.open("xb") as stream:
        stream.write(envelope)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / "sampler.bin")


def open_study(root, model_id, *, resume=False):
    root = Path(root)
    p = policy()
    if model_id not in p["models"]:
        raise ValueError("unknown benchmark model")
    identity = {"policy_hash": policy_hash(), "model_id": model_id,
                "study_name": f"{p['protocol_id']}-{model_id}", "optuna": optuna.__version__}
    if resume:
        if any(not (root / name).is_file() for name in ("study_identity.json", "journal.log", "sampler.bin")):
            raise ValueError("study resume inputs are missing; no study may be recreated")
        if json.loads((root / "study_identity.json").read_text()) != identity:
            raise ValueError("study protocol/model/package identity differs")
        digest, payload = (root / "sampler.bin").read_bytes().split(b"\n", 1)
        if hashlib.sha256(payload).hexdigest().encode() != digest:
            raise ValueError("sampler snapshot checksum differs")
        snapshot = pickle.loads(payload)
        sampler = snapshot["sampler"]
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_new(root / "study_identity.json", identity)
        sampler = optuna.samplers.TPESampler(seed=p["hpo"]["sampler_seed"], n_startup_trials=p["hpo"]["startup_trials"])
    pruner = optuna.pruners.HyperbandPruner(min_resource=p["hpo"]["min_resource"],
        max_resource=p["hpo"]["max_resource"], reduction_factor=p["hpo"]["reduction_factor"])
    if resume:
        study = optuna.load_study(storage=_storage(root), study_name=identity["study_name"],
                                 sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(storage=_storage(root), study_name=identity["study_name"],
                                   sampler=sampler, pruner=pruner, direction="maximize")
        if p.get("study_initialization") == "fresh_no_trial_import":
            study.set_user_attr("benchmark_policy_hash", policy_hash())
            study.set_user_attr("benchmark_model_id", model_id)
            study.set_user_attr("started_trial_budget", p["hpo"]["started_trials_per_model"])
    if p.get("study_initialization") == "fresh_no_trial_import":
        if (study.user_attrs.get("benchmark_policy_hash") != policy_hash()
                or study.user_attrs.get("benchmark_model_id") != model_id
                or study.user_attrs.get("started_trial_budget") != p["hpo"]["started_trials_per_model"]):
            raise ValueError("fresh study policy/model/budget identity differs; trial import is unsupported")
    if resume:
        if (snapshot["optuna"] != optuna.__version__ or snapshot["trials"] != len(study.trials)
                or any(t.state in (TrialState.RUNNING, TrialState.WAITING) for t in study.trials)):
            raise ValueError("interrupted/out-of-sync study needs explicit recovery; no trial was rewritten")
        if study.direction != optuna.study.StudyDirection.MAXIMIZE:
            raise ValueError("benchmark study must maximize AP")
    else:
        _save_sampler(root, study)
    return study


def remaining_trials(study):
    # Failed and pruned attempts consume the budget too. No automatic replacements.
    remaining = policy()["hpo"]["started_trials_per_model"] - len(study.trials)
    if remaining < 0:
        raise ValueError("study exceeds its declared started-trial budget")
    return remaining


def select_trial(study):
    completed = eligible_complete_trials(study)
    if not completed:
        raise ValueError("no eligible complete 40-epoch/15000-update trial")
    return min(completed, key=lambda t: (-t.value, t.number))


def _source_snapshot(root):
    from src.training.benchmark_provenance import write_source_snapshot
    write_source_snapshot(root)


def environment_identity(model_id):
    """Read-only preflight; never download, construct a model or claim a GPU cap."""
    import sklearn
    import torchvision
    from src.models.benchmark_anchors import ARTIFACTS, sha256, verified_dino_source
    artifacts = ARTIFACTS | {
        "efficientnet_v2_s": ("efficientnet_v2_s-dd5fe13b.pth", "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"),
        "p2_s": ("efficientnet_v2_s-dd5fe13b.pth", "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"),
        "maxvit_t": ("maxvit_t-bc5ab103.pth", "bc5ab103d47a7c6c02dc35bf65796b3a6cccf3d51bce330326dbdc634a3ac0e1"),
    }
    name, expected = artifacts[model_id]
    path = Path(torch.hub.get_dir()) / "checkpoints" / name
    if sha256(path) != expected:
        raise ValueError("the approved original weight artifact differs")
    if model_id == "dinov2_b14_anchor":
        verified_dino_source()
    return {"torch": torch.__version__, "torchvision": torchvision.__version__, "lightning": lgn.__version__,
            "xformers": version("xformers") if model_id == "dinov2_b14_anchor" else None,
            "optuna": optuna.__version__, "sklearn": sklearn.__version__,
            "original_artifact": name, "sha256": expected, "bytes": path.stat().st_size}


def _study_inputs(root, model_id):
    from src.ingredient_selection.runtime import resolve_projection
    root = Path(root).resolve()
    allowed = CAMPAIGN_ROOT.resolve()
    if not root.is_relative_to(allowed) or root == allowed:
        raise ValueError("fresh benchmark study must be isolated below experiments/benchmark_v2")
    projection = resolve_projection(policy()["selected_projection"])
    config = make_config(model_id, root / "template")
    projection.verify_metadata(config.datamodule["data_dir"])
    environment = environment_identity(model_id)
    data_identity = {"metadata_hashes": dict(projection.metadata_hashes),
                     "base_class_order": list(projection.base_class_order)}
    return root, config, environment, data_identity


def _validate_saved_inputs(root, environment, data_identity):
    if (json.loads((root / "policy.json").read_text()) != policy()
            or json.loads((root / "data_identity.json").read_text()) != data_identity):
        raise ValueError("saved benchmark policy/data identity changed")
    if json.loads((root / "environment.json").read_text()) != environment:
        raise ValueError("benchmark environment/original-artifact identity changed")


PREPARED_FILES = ("study_identity.json", "journal.log", "sampler.bin", "policy.json", "environment.json",
                  "data_identity.json", "configuration_template.json", "source_inventory.json", "source_snapshot.zip")


def prepare_study(root, model_id):
    """Create a fresh empty study and immutable launch inputs; never ask/enqueue/fit."""
    from src.training.benchmark_provenance import sha256, validate_study_sources
    root, config, environment, data_identity = _study_inputs(root, model_id)
    study = open_study(root, model_id)
    _source_snapshot(root)
    write_new(root / "policy.json", policy())
    write_new(root / "environment.json", environment)
    files = PREPARED_FILES
    if model_id == "dinov2_b14_anchor":
        from src.models.benchmark_anchors import verified_dino_source
        source = verified_dino_source()
        with zipfile.ZipFile(root / "dinov2_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((source / "dinov2").rglob("*.py")) + [source / "hubconf.py", source / "LICENSE"]:
                archive.write(path, path.relative_to(source))
        files += ("dinov2_source.zip",)
    write_new(root / "data_identity.json", data_identity)
    config.save_to_file(root / "configuration_template.json")
    source_identity = validate_study_sources(root)
    write_new(root / "prepared_study.json", {"schema_version": 1, "status": "prepared_empty",
        "model_id": model_id, "policy_hash": policy_hash(), "trials_started": 0,
        "files": {name: sha256(root / name) for name in files}})
    return study, source_identity


def inspect_prepared_study(root, model_id):
    """Read-only input/source validation, including the untouched empty journal/sampler."""
    from src.training.benchmark_provenance import sha256, validate_study_sources
    root, _, environment, data_identity = _study_inputs(root, model_id)
    _validate_saved_inputs(root, environment, data_identity)
    record = json.loads((root / "prepared_study.json").read_text())
    files = set(PREPARED_FILES) | ({"dinov2_source.zip"} if model_id == "dinov2_b14_anchor" else set())
    if (record.get("schema_version") != 1 or record.get("status") != "prepared_empty"
            or record.get("policy_hash") != policy_hash() or record.get("model_id") != model_id
            or record.get("trials_started") != 0 or set(record.get("files", {})) != files
            or any(sha256(root / name) != digest for name, digest in record["files"].items())
            or any(root.glob("trial_*"))):
        raise ValueError("prepared study inputs changed or the study has already started")
    validate_study_sources(root)
    return record


def execute(root, model_id, *, resume=False, prepared=False, additional_trials=1, cost_reviewed=False):
    """Serialize campaign controllers on the single-device WSL host."""
    import fcntl
    # Keep the existing host-wide mutex shared with retained v1 controllers.
    lock_root = Path(__file__).resolve().parents[2] / "experiments" / "benchmark_v1"
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another benchmark controller owns the single-device lock") from error
        return _execute(root, model_id, resume=resume, prepared=prepared,
                        additional_trials=additional_trials, cost_reviewed=cost_reviewed)


def _execute(root, model_id, *, resume=False, prepared=False, additional_trials=1, cost_reviewed=False):
    """Explicitly invoked by Phase 6.2, not by configuration inspection/tests."""
    from src.training.commons import model_training, load_datamodule, set_torch_constants
    from src.training.experimental_runtime import prepare_experimental_config
    from src.data_processing.images_recipes import ImagesRecipesBaseDataModule
    from src.training.benchmark_provenance import validate_study_sources

    root = Path(root).resolve()
    if type(additional_trials) is not int or additional_trials < 1:
        raise ValueError("additional_trials must be positive")
    if additional_trials > policy()["hpo"]["started_trials_per_model"]:
        raise ValueError("requested release exceeds the fixed started-trial budget")
    if prepared and resume:
        raise ValueError("prepared first launch and finished-trial resume are distinct")
    if (resume or additional_trials > 1) and not cost_reviewed:
        raise ValueError("acknowledge the measured-cost review before releasing multiple/further trials")
    if resume or prepared:
        root, _, environment, data_identity = _study_inputs(root, model_id)
        _validate_saved_inputs(root, environment, data_identity)
        if prepared:
            inspect_prepared_study(root, model_id)
        source_identity = validate_study_sources(root)
        study = open_study(root, model_id, resume=True)
        if (prepared and study.trials) or (resume and not study.trials):
            raise ValueError("empty studies require prepared launch; started studies require resume")
    else:
        study, source_identity = prepare_study(root, model_id)
    if additional_trials > remaining_trials(study):
        raise ValueError("requested release exceeds the remaining fixed budget")

    def objective(trial):
        p = policy()
        frozen = model_id == "dinov2_b14_anchor"
        lr_range = p["search"]["frozen_lr" if frozen else "full_lr"]
        lr = trial.suggest_float("lr", *lr_range, log=True)
        wd = trial.suggest_float("weight_decay", *p["search"]["weight_decay"], log=True)
        folder = root / f"trial_{trial.number}"
        folder.mkdir(exist_ok=False)
        write_new(folder / "source_identity.json", source_identity)
        trial.set_user_attr("source_identity", source_identity)
        stopping = {"protocol_id": p["protocol_id"], "policy_hash": policy_hash(),
                    "hpo": p["hpo"], "eligible_trials": p["selection"]["eligible_trials"],
                    "required_completed_epochs": 40, "required_optimizer_updates": 15000}
        write_new(folder / "stopping_policy.json", stopping)
        trial.set_user_attr("stopping_policy", stopping)
        set_torch_constants()  # canonical spawn; numeric benchmark settings override its legacy defaults below
        lgn.seed_everything(42, workers=True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        trial_config = make_config(model_id, folder, lr=lr, weight_decay=wd)
        data = load_datamodule(trial_config)  # all-split metadata only; no test images
        prepare_experimental_config(trial_config, data)
        trial_config.save_to_file(folder / "trial_config.json")
        start = time.monotonic()
        try:
            with patch.object(ImagesRecipesBaseDataModule, "test_dataloader", side_effect=RuntimeError("test image evaluation forbidden")), \
                 patch.object(ImagesRecipesBaseDataModule, "predict_dataloader", side_effect=RuntimeError("predict image evaluation forbidden")):
                trainer, model = model_training(trial_config, data, trainer_kwargs={"trial": trial})
            if trainer.current_epoch != 40 or trainer.global_step != 15000:
                raise ValueError("completed benchmark trial has an unexpected epoch/update horizon")
            trial.set_user_attr("completed_epochs", 40)
            trial.set_user_attr("optimizer_updates", trainer.global_step)
            trial.set_user_attr("best_checkpoint", trainer.checkpoint_callback.best_model_path)
            trial.set_user_attr("final_checkpoint", trainer.last_checkpoint_path)
            return float(trainer.checkpoint_callback.best_model_score)
        finally:
            trial.set_user_attr("session_wall_seconds", time.monotonic() - start)

    if not resume:
        # Enqueued fixed parameters are recorded by Optuna, never substituted after sampling.
        study.enqueue_trial({"lr": 1e-3 if model_id == "dinov2_b14_anchor" else 1e-4, "weight_decay": 1e-4})
    try:
        study.optimize(objective, n_trials=additional_trials, n_jobs=1,
                       callbacks=[lambda s, t: _save_sampler(root, s)], gc_after_trial=True)
    finally:
        # Exceptions are not swallowed; FAIL consumes its attempt and halts this release.
        if all(t.state.is_finished() for t in study.trials):
            _save_sampler(root, study)
    winner = select_trial(study)
    from src.training.benchmark_predictions import export_best_validation
    predictions = export_best_validation(root, winner.number, winner.user_attrs["best_checkpoint"], winner.value)
    return {"trials_started": len(study.trials), "remaining": remaining_trials(study),
            "best_trial": winner.number, "best_value": winner.value,
            "best_checkpoint": winner.user_attrs["best_checkpoint"], "validation_predictions": predictions}

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
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=8)
    pruner = optuna.pruners.HyperbandPruner(min_resource=5, max_resource=40, reduction_factor=3)
    if resume:
        study = optuna.load_study(storage=_storage(root), study_name=identity["study_name"],
                                 sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(storage=_storage(root), study_name=identity["study_name"],
                                   sampler=sampler, pruner=pruner, direction="maximize")
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
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE
                 and t.user_attrs.get("completed_epochs") == 40 and t.value is not None]
    if not completed:
        raise ValueError("no eligible complete 40-epoch trial")
    return min(completed, key=lambda t: (-t.value, t.number))


def _source_snapshot(root):
    repo = Path(__file__).resolve().parents[2]
    paths = sorted(set(p for directory in ("src", "settings", "scripts/launch_exps/benchmark")
                       for p in (repo / directory).rglob("*.py")) | set((repo / "src/training/resources").glob("*.json")))
    inventory = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with zipfile.ZipFile(root / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(repo))
    write_new(root / "source_inventory.json", {
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "files": inventory, "zip_sha256": hashlib.sha256((root / "source_snapshot.zip").read_bytes()).hexdigest()})


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


def execute(root, model_id, *, resume=False, additional_trials=1, cost_reviewed=False):
    """Serialize campaign controllers on the single-device WSL host."""
    import fcntl
    allowed = Path(__file__).resolve().parents[2] / "experiments" / "benchmark_v1"
    allowed.mkdir(parents=True, exist_ok=True)
    with (allowed / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another benchmark controller owns the single-device lock") from error
        return _execute(root, model_id, resume=resume, additional_trials=additional_trials, cost_reviewed=cost_reviewed)


def _execute(root, model_id, *, resume=False, additional_trials=1, cost_reviewed=False):
    """Explicitly invoked by Phase 6.2, not by configuration inspection/tests."""
    from src.training.commons import model_training, load_datamodule, set_torch_constants
    from src.training.experimental_runtime import prepare_experimental_config
    from src.data_processing.images_recipes import ImagesRecipesBaseDataModule
    from src.ingredient_selection.runtime import resolve_projection

    root = Path(root).resolve()
    if type(additional_trials) is not int or additional_trials < 1:
        raise ValueError("additional_trials must be positive")
    if (resume or additional_trials > 1) and not cost_reviewed:
        raise ValueError("review the first trial's measured time before releasing additional trials")
    # Prevent accidentally opening an existing study in another legacy location.
    allowed = Path(__file__).resolve().parents[2] / "experiments" / "benchmark_v1"
    if not root.is_relative_to(allowed) or root == allowed:
        raise ValueError("benchmark study must be isolated below experiments/benchmark_v1")
    projection = resolve_projection(policy()["selected_projection"])
    config = make_config(model_id, root / "template")
    projection.verify_metadata(config.datamodule["data_dir"])
    environment = environment_identity(model_id)
    study = open_study(root, model_id, resume=resume)
    if resume:
        if json.loads((root / "environment.json").read_text()) != environment:
            raise ValueError("benchmark environment/original-artifact identity changed")
        saved = json.loads((root / "source_inventory.json").read_text())
        repo = Path(__file__).resolve().parents[2]
        if any(hashlib.sha256((repo / name).read_bytes()).hexdigest() != digest for name, digest in saved["files"].items()):
            raise ValueError("benchmark sources changed; explicit protocol review is required")
    else:
        _source_snapshot(root)
        write_new(root / "policy.json", policy())
        write_new(root / "environment.json", environment)
        if model_id == "dinov2_b14_anchor":
            from src.models.benchmark_anchors import verified_dino_source
            source = verified_dino_source()
            with zipfile.ZipFile(root / "dinov2_source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted((source / "dinov2").rglob("*.py")) + [source / "hubconf.py", source / "LICENSE"]:
                    archive.write(path, path.relative_to(source))
        write_new(root / "data_identity.json", {"metadata_hashes": dict(projection.metadata_hashes),
                                                "base_class_order": list(projection.base_class_order)})
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

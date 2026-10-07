"""Historical full-task configurations transferred to the frozen selected task."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import lightning as lgn
import torch

from settings.config import EXPERIMENTS_PATH, PROJECT_PATH, YUMMLY_PATH
from src.commons.config_enc_dec import encode_config
from src.commons.exp_config import ExpConfig
from src.data_processing.labels_encoders import MultiLabelBinarizer
from src.ingredient_selection.runtime import PROJECTION_ID, resolve_projection
from src.lightning.lgn_trainers import BaseTrainer
from src.training.commons import load_datamodule, model_training, set_torch_constants


PRESETS = {
    "resnet": {
        "source": "experiments/basic_v5/resnets_htuning/trial_77/trial_config.json",
        "sha256": "9328d3138b4f47dffc10b29d2f830dd19b1de074130aee327e6d393baea26ffc",
        "model": "src.models.resnet.Resnet18",
        "run_name": "resnet18_from_full_trial77_d6_v1",
        "physical_batch": 128,
    },
    "dinov2": {
        "source": "experiments/basic_v5/dinov2_htuning_v1/trial_61/trial_config.json",
        "sha256": "ecce5bf7e156f83fe750cde6cfb25ab59a299238fd6a84a1af3afd0b84f15561",
        "model": "src.models.dinov2.DinoV2B14",
        "run_name": "dinov2_b14_lp_from_full_trial61_d6_v1",
        "physical_batch": 32,
    },
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def build_config(family, *, run_name=None, workers=8, physical_batch=None):
    preset = PRESETS[family]
    source = Path(PROJECT_PATH) / preset["source"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != preset["sha256"]:
        raise ValueError("historical source configuration differs from the reviewed artifact")
    original = ExpConfig.load_from_file(source)
    projection = resolve_projection(PROJECTION_ID)
    model_type = original.torch_model["type"]
    if (f"{model_type.__module__}.{model_type.__name__}" != preset["model"]
            or original.torch_model["num_classes"] != len(projection.base_class_order)
            or original.datamodule.get("ingredient_projection") is not None):
        raise ValueError("expected the reviewed full-task model configuration")
    projection.validate_data_config(original.datamodule["metadata_filename"],
                                    original.datamodule["feature_label"], original.datamodule["category"])
    physical = preset["physical_batch"] if physical_batch is None else physical_batch
    logical = original.hp["batch_size"]
    if type(physical) is not int or not 0 < physical <= logical or logical % physical:
        raise ValueError("physical batch must be a positive exact divisor of the logical batch")
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a non-negative integer")
    name = preset["run_name"] if run_name is None else run_name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("run name must contain only letters, digits, underscores and hyphens")

    hp = copy.deepcopy(original.hp)
    hp.pop("torch_model_type", None)  # Historical unused DummyModel alias, not the actual architecture.
    hp["torch_model"]["num_classes"] = len(projection.class_order)
    hp["log_per_ingredient_metrics"] = True
    for metric in hp["metrics"].values():
        if "num_labels" in metric["init_params"]:
            metric["init_params"]["num_labels"] = len(projection.class_order)
    dm = copy.deepcopy(original.datamodule)
    encoder = MultiLabelBinarizer(classes=list(projection.class_order))
    encoder.fit()
    dm.update(data_dir=YUMMLY_PATH, num_workers=workers, ingredient_projection=PROJECTION_ID,
              label_encoder=encoder.to_config())
    trial = Path(EXPERIMENTS_PATH) / "selected_v5" / name / "trial_0"
    trainer = copy.deepcopy(original.trainer)
    trainer.update(type=BaseTrainer, save_dir=str(trial), debug=False)
    config = ExpConfig(hp_=hp, dm_=dm, tr_=trainer,
                       lgg_log_exp_config=True,
                       lgg_wandb_notes="Historical full-task hyperparameters transferred to D6; not final benchmark")
    contract = {
        "schema_version": 1, "family": family, "source_config": preset["source"],
        "source_config_sha256": preset["sha256"], "seed": 42,
        "precision": "16-mixed", "physical_batch": physical,
        "logical_batch": logical, "accumulate_grad_batches": logical // physical,
        "check_val_every_n_epoch": 2,
        "initialization": "fresh pretrained backbone and fresh selected-task head; no source checkpoint",
        "config": encode_config(config.config),
    }
    return config, json.loads(json.dumps(contract))


class LaunchContract(lgn.Callback):
    def __init__(self, contract):
        self.contract = contract

    def on_save_checkpoint(self, trainer, module, checkpoint):
        checkpoint["selected_vocab_launch"] = self.contract

    def on_fit_start(self, trainer, module):
        if (module.batch_size != self.contract["physical_batch"]
                or trainer.accumulate_grad_batches != self.contract["accumulate_grad_batches"]):
            raise ValueError("runtime batching differs from the saved launch contract")

    def on_load_checkpoint(self, trainer, module, checkpoint):
        if checkpoint.get("selected_vocab_launch") != self.contract:
            raise ValueError("checkpoint belongs to a different selected-task launch contract")


def validate_resume(config, contract, manifest, checkpoint):
    if manifest.get("contract") != contract or manifest.get("contract_hash") != _digest(contract):
        raise ValueError("resume settings differ from the saved launch; use the original arguments")
    if manifest.get("status") == "completed":
        raise ValueError("completed runs cannot be resumed; choose a new run name")
    if checkpoint.get("selected_vocab_launch") != contract:
        raise ValueError("checkpoint belongs to a different selected-task launch contract")
    restored = ExpConfig.load_from_ckpt_data(checkpoint)
    if (restored.datamodule.get("ingredient_projection") != config.datamodule["ingredient_projection"]
            or restored.label_encoder != config.label_encoder
            or restored.torch_model["num_classes"] != config.torch_model["num_classes"]):
        raise ValueError("checkpoint vocabulary/encoder differs from the requested task")


def run_selected(config, contract, *, resume=False):
    trial = Path(config.trainer["save_dir"])
    run = trial.parent
    manifest_path = run / "launch_manifest.json"
    checkpoint_path = None
    if not resume and run.exists():
        raise FileExistsError("run already exists; use --resume for an interrupted run or a new --run-name")
    files = ["src/training/selected_vocab.py", "src/training/commons.py", "src/commons/exp_config.py",
             "src/commons/config_enc_dec.py", "src/models/commons.py", "src/lightning/lgn_models.py",
             "src/lightning/lgn_trainers.py", "src/lightning/custom_callbacks.py",
             "src/data_processing/images_recipes.py", "src/data_processing/labels_encoders.py",
             "src/data_processing/transformations.py", "src/ingredient_selection/runtime.py",
             "settings/config.py", "src/models/resnet.py" if contract["family"] == "resnet" else "src/models/dinov2.py"]
    sources = {name: hashlib.sha256((Path(PROJECT_PATH) / name).read_bytes()).hexdigest() for name in files}
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("runtime_source_sha256") != sources:
            raise ValueError("runtime sources changed since launch; review before resuming")
        checkpoint_path = trial / "checkpoints" / "last.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_resume(config, contract, manifest, checkpoint)
        del checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("these launchers require CUDA in the WSL ML environment")
    # The projection verifies metadata hashes; preparation reads metadata, never test images.
    set_torch_constants()
    lgn.seed_everything(contract["seed"], workers=True)
    dm = load_datamodule(config)
    if dm.label_encoder.to_config() != config.label_encoder:
        raise ValueError("prepared encoder differs from the saved selected order")
    if not resume:
        run.mkdir(parents=True, exist_ok=False)
        trial.mkdir()
        config.save_to_file(trial / "launch_config.json")
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_PATH,
                                  check=True, capture_output=True, text=True).stdout.strip()
        manifest = {"contract": contract, "contract_hash": _digest(contract), "attempts": [],
                    "git_base_revision": revision, "runtime_source_sha256": sources,
                    "torch_version": torch.__version__, "lightning_version": lgn.__version__,
                    "cuda_device": torch.cuda.get_device_name(0)}
    scratch = run / "trainer_scratch"
    scratch.mkdir(exist_ok=True)
    attempt = {"started_at": datetime.now(timezone.utc).isoformat(), "resumed": resume}
    manifest["attempts"].append(attempt)

    def record(status):
        manifest["status"] = status
        attempt["status"] = status
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    record("running")
    try:
        with ExitStack() as stack:
            # Scope the legacy resource cap and scratch cleanup to this launch only.
            stack.enter_context(patch.object(config.torch_model["type"], "max_allowed_batch_size",
                                             property(lambda model: contract["physical_batch"])))
            stack.enter_context(patch("src.lightning.lgn_trainers.EXPERIMENTS_TRASH_PATH", str(scratch)))
            stack.enter_context(patch.object(dm, "test_dataloader", side_effect=RuntimeError("test evaluation forbidden")))
            stack.enter_context(patch.object(dm, "predict_dataloader", side_effect=RuntimeError("prediction split forbidden")))
            trainer, model = model_training(config, dm, ckpt_path=checkpoint_path, trainer_kwargs={
                "precision": contract["precision"],
                "check_val_every_n_epoch": contract["check_val_every_n_epoch"],
            } | _callback_kwargs(trial, contract))
        attempt["optimizer_updates"] = trainer.global_step
    except KeyboardInterrupt:
        record("interrupted")
        raise
    except Exception:
        record("failed")
        raise
    else:
        record("completed")
        return trainer, model
    finally:
        attempt["ended_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _callback_kwargs(trial, contract):
    from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar
    from src.lightning.custom_callbacks import FullModelCheckpoint
    checkpoint = FullModelCheckpoint(dirpath=trial / "checkpoints", monitor="val_loss", mode="min",
                                     save_top_k=2, save_last=True, every_n_epochs=2,
                                     filename="epoch={epoch}-vloss={val_loss:.3f}")
    return {"model_checkpoint_callback": checkpoint,
            "callbacks": [checkpoint, LaunchContract(contract), RichProgressBar(leave=True),
                          LearningRateMonitor(logging_interval="epoch", log_momentum=True)]}


def main(family, argv=None):
    parser = argparse.ArgumentParser(description=f"Retrain {family} on the frozen 59-label D6 task; no HPO/test evaluation")
    parser.add_argument("--dry-run", action="store_true", help="show config without model construction or training")
    parser.add_argument("--resume", action="store_true", help="resume only this run's validated last checkpoint")
    parser.add_argument("--run-name")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--physical-batch", type=int, help="exact divisor of 128; logical batch stays unchanged")
    args = parser.parse_args(argv)
    config, contract = build_config(family, run_name=args.run_name, workers=args.workers,
                                    physical_batch=args.physical_batch)
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        return
    run_selected(config, contract, resume=args.resume)

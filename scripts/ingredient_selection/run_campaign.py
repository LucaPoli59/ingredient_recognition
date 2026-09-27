"""Run or resource-check the frozen Phase 3-D1 selector campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightning as lgn
import numpy as np
import torch
import torchvision
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.ingredient_selection.artifacts import SelectorArtifactStore, read_json, write_json
from src.ingredient_selection.data import SelectorDataBundle, SelectorDataModule
from src.ingredient_selection.protocol import (
    PROTOCOL_ID,
    SelectorProtocol,
    build_pilot_cohort,
    compute_pos_weight,
    configure_deterministic_runtime,
    git_revision_and_cleanliness,
    ordered_values_hash,
    require_clean_tracked_worktree,
    sha256_file,
    sha256_json,
    tensor_hash,
)
from src.models.efficientnet import EfficientNetV2SSelector
from src.training.ingredient_selection import (
    SelectorAuditCallback,
    SelectorLightningModule,
    run_resource_gate,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _weight_file() -> Path:
    filename = EfficientNetV2SSelector.WEIGHTS_URL.rsplit("/", 1)[-1]
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def _verify_weight_file() -> dict[str, object]:
    path = _weight_file()
    if not path.is_file():
        raise FileNotFoundError(
            f"pinned EfficientNetV2-S weights are not cached at {path}; load them before the campaign"
        )
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != EfficientNetV2SSelector.WEIGHTS_SIZE or digest != EfficientNetV2SSelector.WEIGHTS_SHA256:
        raise ValueError("cached EfficientNetV2-S weights do not match the frozen size/SHA-256")
    return {
        "enum": EfficientNetV2SSelector.WEIGHTS_ENUM,
        "url": EfficientNetV2SSelector.WEIGHTS_URL,
        "size": size,
        "sha256": digest,
        "offline_load_verified": True,
    }


def _campaign_identity(
        bundle: SelectorDataBundle,
        protocol: SelectorProtocol,
        revision: str,
        positive_counts: list[int],
        pos_weight: torch.Tensor,
        head_hash: str,
) -> dict[str, object]:
    return {
        "protocol_id": protocol.protocol_id,
        "git_revision": revision,
        "seed": protocol.seed,
        "class_order": list(bundle.class_names),
        "class_order_hash": ordered_values_hash(bundle.class_names),
        "metadata_sha256": {
            "train": bundle.train.metadata_sha256,
            "val": bundle.val.metadata_sha256,
        },
        "model": {
            "type": "EfficientNetV2SSelector",
            "weights": EfficientNetV2SSelector.WEIGHTS_ENUM,
            "input_size": [384, 384],
            "full_backbone_trainable": True,
            "classifier": "Dropout(p=0.2,inplace=True)+Linear(1280,165,bias=True)",
            "initial_head_hash": head_hash,
        },
        "transform": {
            "name": "4B-D1-full-frame-fit-pad-v1",
            "resize": "long-side-384-short-side-round-half-up",
            "interpolation": "bilinear",
            "antialias": True,
            "padding": "center-ImageNet-mean-right-bottom-odd-residual",
            "train_augmentation": "RandomHorizontalFlip(p=0.5)",
            "validation_augmentation": None,
        },
        "loss": {
            "type": "BCEWithLogitsLoss",
            "reduction": "mean",
            "pos_weight_formula": "(N_train-P_c)/P_c",
            "positive_counts": positive_counts,
            "positive_counts_hash": ordered_values_hash(positive_counts),
            "pos_weight": pos_weight.tolist(),
            "pos_weight_hash": tensor_hash(pos_weight),
        },
        "optimizer": {
            "type": "AdamW",
            "lr": protocol.learning_rate,
            "betas": list(protocol.adam_betas),
            "eps": protocol.adam_eps,
            "weight_decay": protocol.weight_decay,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "parameter_groups": 1,
            "gradient_clipping": None,
        },
        "scheduler": {
            "type": "SequentialLR",
            "warmup": {"type": "LinearLR", "start_factor": 0.1, "end_factor": 1.0, "total_iters": 2},
            "milestones": [2],
            "main": {"type": "CosineAnnealingLR", "T_max": 18, "eta_min": 1e-6},
            "interval": "epoch",
        },
        "execution": {
            "max_epochs": 20,
            "batch_size": 8,
            "drop_last": False,
            "gradient_accumulation": 1,
            "precision": "32-true",
            "early_stopping": False,
            "swa": False,
            "audit_epochs": list(protocol.audit_epochs),
            "planned_steps": 119_920,
        },
    }


def _validate_resource_gate(
        path: Path,
        bundle: SelectorDataBundle,
        protocol: SelectorProtocol,
        weights: dict[str, object],
        revision: str,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"resource gate not found: {path}; run this command with --resource-gate-only first"
        )
    gate = read_json(path)
    expected_data = {
        "class_order_hash": ordered_values_hash(bundle.class_names),
        "metadata_sha256": {
            "train": bundle.train.metadata_sha256,
            "val": bundle.val.metadata_sha256,
        },
    }
    if gate.get("protocol_id") != protocol.protocol_id:
        raise ValueError("resource gate protocol does not match the campaign")
    if gate.get("data_identity") != expected_data:
        raise ValueError("resource gate data/class identity does not match the campaign")
    if gate.get("weights") != weights:
        raise ValueError("resource gate weight identity does not match the campaign")
    if gate.get("git_revision") != revision or not gate.get("tracked_worktree_clean"):
        raise ValueError("resource gate must come from the same clean tracked revision as the campaign")
    if not gate.get("measurement", {}).get("passed"):
        raise ValueError("resource gate did not pass")
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "sha256": sha256_file(path),
        "measurement": gate["measurement"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/input/yummly")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis_outputs/ingredient_selection" / PROTOCOL_ID)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "experiments/ingredient_selection" / PROTOCOL_ID)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resource-gate-only", action="store_true")
    parser.add_argument("--resource-gate-output", type=Path, default=None)
    args = parser.parse_args(argv)

    protocol = SelectorProtocol()
    runtime = configure_deterministic_runtime(protocol.seed)
    lgn.seed_everything(protocol.seed, workers=True)
    bundle = SelectorDataBundle.load(args.data_root, expected_classes=protocol.num_classes)
    datamodule = SelectorDataModule(
        bundle,
        batch_size=protocol.batch_size,
        num_workers=args.num_workers,
        seed=protocol.seed,
        pin_memory=False,
    )
    positive_counts = bundle.train.supports.astype(int).tolist()
    pilot = build_pilot_cohort(bundle.class_names, positive_counts)
    resource_gate_path = args.resource_gate_output or (
        ROOT / "analysis_outputs/ingredient_selection" / f"{PROTOCOL_ID}_resource_gate.json"
    )

    if args.resource_gate_only:
        weights = _verify_weight_file()
        gate_revision, gate_clean, gate_status = git_revision_and_cleanliness(ROOT)
        result = {
            "protocol_id": protocol.protocol_id,
            "created_at_utc": utc_now(),
            "deterministic_runtime": runtime,
            "pilot_artifact_hash_generated_before_model": pilot["artifact_hash"],
            "data_identity": {
                "class_order_hash": ordered_values_hash(bundle.class_names),
                "metadata_sha256": {
                    "train": bundle.train.metadata_sha256,
                    "val": bundle.val.metadata_sha256,
                },
            },
            "weights": weights,
            "git_revision": gate_revision,
            "tracked_worktree_clean": gate_clean,
            "tracked_changes_at_gate": gate_status,
            "measurement": run_resource_gate(datamodule, protocol=protocol),
        }
        write_json(resource_gate_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    revision = require_clean_tracked_worktree(ROOT)
    store = SelectorArtifactStore(args.output_dir)
    store.initialize()
    store.write_pilot(pilot)
    weights = _verify_weight_file()
    resource_gate = _validate_resource_gate(resource_gate_path, bundle, protocol, weights, revision)

    # The cohort exists on disk before the seeded model/head is constructed.
    lgn.seed_everything(protocol.seed, workers=True)
    model = EfficientNetV2SSelector(num_classes=protocol.num_classes)
    head_hash = sha256_json({
        "weight": tensor_hash(model.classifier_target_layer.weight),
        "bias": tensor_hash(model.classifier_target_layer.bias),
    })
    pos_weight = compute_pos_weight(bundle.train.targets)
    identity = _campaign_identity(
        bundle, protocol, revision, positive_counts, pos_weight, head_hash
    )
    manifest = {
        "schema_version": 1,
        "campaign_identity": identity,
        "campaign_identity_hash": sha256_json(identity),
        "status": "running",
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "command": [sys.executable, *sys.argv],
        "repository_root": ".",
        "tracked_worktree_clean_at_start": True,
        "pilot_artifact_hash": pilot["artifact_hash"],
        "weights": weights,
        "resource_gate": resource_gate,
        "deterministic_runtime": runtime,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "lightning": lgn.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "artifacts": {
            "output_dir": str(Path(args.output_dir).resolve().relative_to(ROOT)),
            "experiment_dir": str(Path(args.experiment_dir).resolve().relative_to(ROOT)),
        },
    }
    store.write_manifest(manifest)

    module = SelectorLightningModule(model, pos_weight, protocol)
    checkpoint_dir = Path(args.experiment_dir) / "checkpoints"
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        every_n_epochs=2,
        save_top_k=-1,
        save_last=True,
        save_weights_only=False,
        filename="epoch={epoch:02d}",
    )
    audit = SelectorAuditCallback(store, run_id=protocol.protocol_id, protocol=protocol)
    trainer = lgn.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=protocol.max_epochs,
        min_epochs=protocol.max_epochs,
        precision="32-true",
        deterministic=True,
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        callbacks=[checkpoint, audit],
        logger=CSVLogger(save_dir=args.experiment_dir, name="logs"),
        enable_checkpointing=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        check_val_every_n_epoch=None,
        log_every_n_steps=50,
        default_root_dir=args.experiment_dir,
    )
    try:
        trainer.fit(module, datamodule=datamodule)
    except BaseException:
        manifest = read_json(store.manifest_path)
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = utc_now()
        store.write_manifest(manifest)
        raise
    manifest = read_json(store.manifest_path)
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = utc_now()
    final_checkpoint = Path(checkpoint.last_model_path)
    manifest["final_checkpoint"] = {
        "path": str(final_checkpoint.resolve().relative_to(ROOT)),
        "sha256": sha256_file(final_checkpoint),
    }
    store.write_manifest(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

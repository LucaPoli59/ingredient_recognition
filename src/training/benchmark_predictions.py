"""Ordered validation inference from a retained, already-selected checkpoint."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from src.training.benchmark_baselines import load_metadata_pair
from src.training.benchmark_campaign import write_new
from src.training.benchmark_protocol import epoch_metrics, policy_hash


def collect_validation(model, loader, *, expected_targets, class_order, device):
    if list(model.output_class_order) != list(class_order):
        raise ValueError("validation export class order differs")
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("validation export requires the complete sequential record order")
    original_device, was_training = model.device, model.training
    logits, targets = [], []
    try:
        model.to(device).eval()
        with torch.inference_mode():
            for inputs, truth in loader:
                if inputs.ndim != 4:
                    raise ValueError("validation export requires image batches")
                outputs = model(inputs.to(device))
                if outputs.shape != truth.shape or outputs.shape[1] != len(class_order):
                    raise ValueError("validation output shape differs from its targets")
                logits.append(outputs.float().cpu().numpy())
                targets.append(truth.cpu().numpy())
        logits, targets = np.concatenate(logits), np.concatenate(targets)
        if not np.array_equal(targets, expected_targets):
            raise ValueError("validation export record/target order differs from metadata")
        return logits, targets, epoch_metrics(logits, targets)
    finally:
        model.to(original_device).train(was_training)


def export_best_validation(study_root, trial_number, checkpoint_path, expected_ap):
    from src.commons.exp_config import ExpConfig
    from src.training.commons import load_datamodule
    from src.training.experimental_runtime import load_model_for_experiment
    from src.ingredient_selection.data import _encode_targets

    root = Path(study_root)
    destination = root / "validation" / f"trial_{trial_number}"
    checkpoint_hash = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    if destination.exists():
        saved = json.loads((destination / "prediction_manifest.json").read_text())
        if (saved["checkpoint_sha256"] != checkpoint_hash or saved["policy_hash"] != policy_hash()
                or saved["archive_sha256"] != hashlib.sha256((destination / "validation_predictions.npz").read_bytes()).hexdigest()):
            raise ValueError("existing validation prediction identity differs")
        return saved
    config = ExpConfig.load_from_file(root / f"trial_{trial_number}" / "trial_config.json")
    projection, rows = load_metadata_pair(config.datamodule["data_dir"])
    classes = list(projection.base_class_order)
    expected = _encode_targets(rows["val"], classes, projection.target_field)
    module = load_model_for_experiment(config, checkpoint_path=checkpoint_path)
    data = load_datamodule(config)  # all-split metadata only, no test image loader
    data.transform_plain, data.transform_aug = module.transform_plain, module.transform_aug
    data.batch_size = module.batch_size
    data.setup("fit")
    names = [row["image"] for row in rows["val"]]
    if [Path(path).name for path in data.val_dataset.images_paths] != names:
        raise ValueError("validation image order differs from frozen metadata")
    logits, targets, metrics = collect_validation(module, data.val_dataloader(), expected_targets=expected,
                                                  class_order=classes, device="cuda")
    if abs(metrics["val_macro_ap"] - expected_ap) > 1e-6:
        raise ValueError("retained checkpoint AP does not reproduce within absolute tolerance 1e-6")
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "validation_predictions.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, logits=logits, targets=targets, class_order=np.array(classes),
                            record_ids=np.array([str(row["id"]) for row in rows["val"]]), image_names=np.array(names))
    result = {"split": "val", "trial_number": trial_number, "policy_hash": policy_hash(),
              "checkpoint": str(Path(checkpoint_path).relative_to(root)), "checkpoint_sha256": checkpoint_hash,
              "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "metadata_sha256": dict(projection.metadata_hashes)["val"], "class_order": classes,
              "records": len(targets), "metrics": metrics, "reproduction_ap_absolute_tolerance": 1e-6,
              "test_image_inference": False}
    write_new(destination / "prediction_manifest.json", result)
    return result

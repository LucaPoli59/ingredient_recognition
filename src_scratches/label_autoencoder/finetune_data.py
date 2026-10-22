"""Read-only preparation and image integrity for the backbone fine-tuning pilot."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data_processing.experimental_transforms import ExperimentalFitPad224
from src_scratches.label_autoencoder import compare, image_models


ARMS = ("direct", "ae32_frozen_decoder", "ae32_trainable_decoder")
PATH_KEYS = ("data_dir", "source_run", "feature_cache", "autoencoder", "weights")
INT_KEYS = ("expected_labels", "seed", "epochs", "patience", "batch_size", "workers", "threads")
FLOAT_KEYS = ("min_delta", "backbone_lr", "head_lr", "decoder_lr", "latent_weight")


def read_config(path):
    config = json.loads(Path(path).read_text())
    expected = {"schema_version", "metadata", "arms", *PATH_KEYS, *INT_KEYS, *FLOAT_KEYS}
    if set(config) != expected or config["schema_version"] != 1:
        raise ValueError("Unsupported or incomplete fine-tuning configuration")
    for key in INT_KEYS:
        minimum = 0 if key in ("seed", "workers") else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"Invalid {key}")
    if config["seed"] >= 2 ** 32:
        raise ValueError("Seed must be below 2**32")
    for key in FLOAT_KEYS:
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
        if key.endswith("_lr") and value == 0:
            raise ValueError(f"{key} must be positive")
    if (not isinstance(config["arms"], list) or not config["arms"] or
            any(arm not in ARMS for arm in config["arms"]) or
            len(set(config["arms"])) != len(config["arms"])):
        raise ValueError("Select distinct known arms")
    for key in (*PATH_KEYS, "metadata"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"Invalid {key}")
    return config


def resolved(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else compare.ROOT / path).resolve()


def image_records(cache, manifest, provenance):
    """Reuse the original image-byte inventory; cached feature arrays are unnecessary."""
    if manifest["status"] != "complete" or manifest.get("cross_split_identical_image_hashes") != 0:
        raise ValueError("Feature manifest must be complete and split-isolated")
    rows = {split: [] for split in ("train", "val")}
    for chunk in manifest["chunks"]:
        split = chunk["split"]
        if split not in rows or chunk["start"] != len(rows[split]):
            raise ValueError("Non-contiguous image inventory")
        path = (cache / chunk["images"]).resolve()
        if path.parent != cache or compare.sha256(path) != chunk["images_sha256"]:
            raise ValueError("Image inventory path/hash mismatch")
        batch = json.loads(path.read_text())
        if len(batch) != chunk["stop"] - chunk["start"] or not batch:
            raise ValueError("Invalid image chunk length")
        for record in batch:
            name, digest = record["image"], record["sha256"]
            if (not isinstance(name, str) or not name or Path(name).name != name or
                    not isinstance(digest, str) or len(digest) != 64 or
                    any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Malformed image inventory")
        rows[split].extend(batch)
    for split, records in rows.items():
        source = provenance[split]
        cached = manifest["identity"]["metadata"][split]
        if any(source[key] != cached[key] for key in ("rows", "sha256")):
            raise ValueError("Cached metadata differs from current targets")
        expected = [r["image"] for r in json.loads(Path(source["path"]).read_text())]
        if [r["image"] for r in records] != expected:
            raise ValueError("Image row order/population differs from target metadata")
    if {r["sha256"] for r in rows["train"]} & {r["sha256"] for r in rows["val"]}:
        raise ValueError("Byte-identical train/validation image overlap")
    return rows


def prepare(config_path):
    """Validate inputs on CPU; no image reads, backbone passes, training or CUDA calls."""
    config_path = Path(config_path).resolve()
    config = read_config(config_path)
    paths = {key: resolved(config[key]) for key in PATH_KEYS}
    train_y, val_y, labels, provenance = compare.load_labels(
        paths["data_dir"], config["metadata"], "ingredients_target")
    if len(labels) != config["expected_labels"]:
        raise ValueError("Unexpected vocabulary size")
    run = paths["source_run"]
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Warm-start source run is incomplete")
    contract = manifest["contract"]
    if contract["labels"] != labels or any(
        contract[split + "_sha256"] != provenance[split]["sha256"] for split in ("train", "val")
    ):
        raise ValueError("Warm-start vocabulary or metadata mismatch")
    artifacts = {"config": config_path, "source_manifest": run / "manifest.json",
                 "source_summary": run / "summary.json", "autoencoder": paths["autoencoder"],
                 "weights": paths["weights"],
                 "feature_manifest": paths["feature_cache"] / "feature_manifest.json"}
    hashes = {key: compare.sha256(path) for key, path in artifacts.items()}
    for key, field in (("autoencoder", "autoencoder_sha256"), ("weights", "backbone_weights_sha256"),
                       ("feature_manifest", "feature_manifest_sha256")):
        if hashes[key] != contract[field]:
            raise ValueError(f"Changed source artifact: {key}")
    cache_manifest = json.loads(artifacts["feature_manifest"].read_text())
    transform = ExperimentalFitPad224.from_config(contract["transform"])
    identity = cache_manifest["identity"]
    if (identity["weights_sha256"] != hashes["weights"] or identity["transform"] != transform.to_config()
            or transform.horizontal_flip_probability != 0):
        raise ValueError("Feature/image preprocessing contract mismatch")
    if identity["transform_source_sha256"] != compare.sha256(compare.ROOT / "src/data_processing/experimental_transforms.py"):
        raise ValueError("Transform implementation changed since source run")
    rows = image_records(paths["feature_cache"], cache_manifest, provenance)
    teacher, ae_saved = compare.load_autoencoder(paths["autoencoder"])
    teacher.requires_grad_(False).eval()
    if (ae_saved["latent_dim"] != 32 or ae_saved["labels"] != labels or
            ae_saved["train_sha256"] != provenance["train"]["sha256"]):
        raise ValueError("AE32 teacher does not match this task")
    source_arms = {arm["method"]: arm for arm in summary["arms"]}
    heads, specifications = {}, {}
    for name in ("direct", "ae32_aligned"):
        artifacts[name] = run / f"{name}.pt"
        hashes[name] = compare.sha256(artifacts[name])
        if hashes[name] != source_arms[name]["checkpoint_sha256"]:
            raise ValueError(f"Changed warm-start checkpoint: {name}")
        head, saved = image_models.load_head(artifacts[name])
        if (saved["kind"] != name or saved["labels"] != labels or saved["contract"] != contract
                or saved["feature_dim"] != 512 or saved["latent_dim"] != 32):
            raise ValueError("Head checkpoint contract mismatch")
        if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
            raise ValueError("Nonfinite warm-start state")
        if (head.feature_std <= 0).any():
            raise ValueError("Nonpositive feature scale")
        heads[name], specifications[name] = head, {k: v for k, v in saved.items() if k != "state_dict"}
    ae_head = heads["ae32_aligned"]
    if (ae_head.latent_std <= 0).any() or any(
        not torch.equal(value, ae_head.decoder.state_dict()[key])
        for key, value in teacher.decoder.state_dict().items()
    ):
        raise ValueError("AE warm-start decoder/scales differ from the teacher")
    for attr in ("feature_mean", "feature_std"):
        if not torch.equal(getattr(heads["direct"], attr), getattr(ae_head, attr)):
            raise ValueError("Warm-start heads use different feature normalization")
    scripts = [Path(__file__), Path(image_models.__file__), Path(compare.__file__),
               Path(__file__).with_name("finetune.py"),
               compare.ROOT / "src/data_processing/experimental_transforms.py"]
    plan = {"status": "prepared_not_started", "training_started": False,
            "config": config, "contract": contract, "data": provenance,
            "artifacts": {key: {"path": str(artifacts[key]), "sha256": digest} for key, digest in hashes.items()},
            "scripts": {str(p.relative_to(compare.ROOT)): compare.sha256(p) for p in scripts},
            "initialization": "Existing direct head and shared ae32_aligned head; ImageNet ResNet-18 weights.",
            "checkpoint_selection": "Minimum validation ingredient BCE, including epoch-zero warm start.",
            "threshold_policy": "All distinct validation scores (primary F1), original 0.05-0.95 grid (secondary).",
            "gpu_gate": {"physical_index": 0, "samples": 3, "min_free_mib": 6144, "max_utilization_pct": 10},
            "test_split_accessed": False, "image_bytes_checked_now": False,
            "image_integrity": "Inventory checked now; actual bytes checked on every future dataset read."}
    return {"plan": plan, "config": config, "paths": paths, "heads": heads, "specifications": specifications,
            "teacher": teacher, "labels": labels, "train_y": train_y, "val_y": val_y, "rows": rows}


class RecipeImages(Dataset):
    def __init__(self, store, rows, labels, codes, transform):
        if not len(rows) == len(labels) == len(codes):
            raise ValueError("Image/target/code population mismatch")
        self.store, self.rows = Path(store).resolve(), rows
        self.labels, self.codes, self.transform = labels, codes, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        record = self.rows[index]
        path = (self.store / record["image"]).resolve()
        if path.parent != self.store:
            raise ValueError("Image resolves outside the shared store")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Image bytes changed: {record['image']}")
        with Image.open(io.BytesIO(raw)) as image:
            pixels = self.transform(image)
        return pixels, torch.from_numpy(self.labels[index]), torch.from_numpy(self.codes[index])

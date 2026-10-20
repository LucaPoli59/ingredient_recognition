"""Train-fitted prior controls on frozen train/validation metadata only."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from src.ingredient_selection.data import _encode_targets, _normalise_cuisine
from src.ingredient_selection.runtime import resolve_projection
from src.training.benchmark_campaign import write_new
from src.training.benchmark_protocol import epoch_metrics, policy, policy_hash


def prior_scores(train_targets, train_cuisines, validation_cuisines):
    y = np.asarray(train_targets)
    cuisines = np.asarray(train_cuisines)
    if (y.ndim != 2 or not len(y) or not np.isin(y, (0, 1)).all()
            or cuisines.shape != (len(y),)):
        raise ValueError("binary train targets and one cuisine per train record are required")
    prevalence = y.mean(axis=0)
    groups = {}
    for name in sorted(set(cuisines)):
        subset = y[cuisines == name]
        # Same Laplace-smoothed cuisine prior used by the retained selector controls.
        groups[name] = (subset.sum(axis=0) + 1.) / (len(subset) + 2.)
    rows = len(validation_cuisines)
    return {
        "all_negative": np.zeros((rows, y.shape[1]), dtype=np.float64),
        "train_prevalence": np.broadcast_to(prevalence, (rows, y.shape[1])).copy(),
        "train_cuisine": np.stack([groups.get(name, prevalence) for name in validation_cuisines]),
    }


def probability_logits(values):
    probability = np.clip(np.asarray(values, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(probability) - np.log1p(-probability)


def load_metadata_pair(data_root):
    projection = resolve_projection(policy()["selected_projection"])
    hashes = dict(projection.metadata_hashes)
    records = {}
    for split in ("train", "val"):
        path = Path(data_root) / split / projection.metadata_filename
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != hashes[split]:
            raise ValueError(f"{split} metadata differs from the frozen generation")
        rows = json.loads(content)
        if not rows or len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError("empty metadata or duplicate split record IDs")
        records[split] = rows
    return projection, records


def run_baselines(output, data_root=None):
    from settings.config import YUMMLY_PATH
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    projection, records = load_metadata_pair(data_root or YUMMLY_PATH)
    order = list(projection.base_class_order)
    targets = {split: _encode_targets(rows, order, projection.target_field) for split, rows in records.items()}
    if len(records["train"]) != policy()["train_records"] or len(records["val"]) != policy()["validation_records"]:
        raise ValueError("the benchmark row population differs")
    scores = prior_scores(targets["train"], [_normalise_cuisine(row.get("cuisine")) for row in records["train"]],
                          [_normalise_cuisine(row.get("cuisine")) for row in records["val"]])
    indices = list(projection.base_class_indices)
    result = {"protocol_id": policy()["protocol_id"], "policy_hash": policy_hash(),
              "status": "validation_only_uncalibrated_controls", "train_records": len(targets["train"]),
              "validation_records": len(targets["val"]), "class_order": order,
              "selected_projection": projection.to_config(), "test_metadata_read": False,
              "cuisine_formula": "(train_cuisine_positives+1)/(train_cuisine_records+2)",
              "unknown_cuisine": "unsmoothed_train_prevalence", "metrics": {}}
    for task, columns in (("full165", list(range(len(order)))), ("selected59", indices)):
        result["metrics"][task] = {
            name: epoch_metrics(probability_logits(probabilities[:, columns]), targets["val"][:, columns])
            for name, probabilities in scores.items()}
    root.mkdir(parents=True, exist_ok=False)
    archive = root / "validation_controls.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, targets=targets["val"], class_order=np.array(order),
                            record_ids=np.array([str(row["id"]) for row in records["val"]]),
                            image_names=np.array([row["image"] for row in records["val"]]),
                            selected_base_indices=np.array(indices), **scores)
    repo = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(__file__).with_name("benchmark_protocol.py"),
             repo / "src/ingredient_selection/data.py", repo / "src/ingredient_selection/runtime.py"]
    result["provenance"] = {
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "metadata_hashes": {name: dict(projection.metadata_hashes)[name] for name in ("train", "val")},
        "source_hashes": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    write_new(root / "baseline_summary.json", result)
    return result

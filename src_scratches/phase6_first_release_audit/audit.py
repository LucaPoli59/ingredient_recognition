"""Read-only artifact/cost audit of the five Phase 6.2 reference trials.

No study replay, model construction, image inference, or source repair. Only
trusted locally produced full checkpoints are deserialized, on CPU.
"""

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.ingredient_selection.data import _encode_targets
from src.training.benchmark_baselines import load_metadata_pair
from src.training.benchmark_protocol import epoch_metrics, policy, policy_hash


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def decoded(values):
    """Inspect primitive serialized identity without importing saved classes."""
    result = {}
    for key, (kind, value) in values.items():
        result[key] = (decoded(value) if kind == "config" else
                       json.loads(value) if kind in ("list", "ndarray") else
                       None if kind == "None" else value)
    return result


def checkpoint_identity(path, classes, model_id):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["hyper_parameters"]["benchmark_policy_hash"] == policy_hash()
    assert checkpoint["experimental_output_class_order"] == classes
    assert checkpoint["experimental_model_contract"]["model_id"] == model_id
    assert checkpoint["ingredient_projection"] is None
    assert checkpoint["optimizer_states"] and checkpoint["lr_schedulers"]
    state = hashlib.sha256()
    for name, value in sorted(checkpoint["state_dict"].items()):
        assert value.device.type == "cpu"
        assert not value.is_floating_point() or torch.isfinite(value).all()
        state.update(name.encode())
        state.update(str((value.dtype, tuple(value.shape))).encode())
        state.update(value.contiguous().numpy().tobytes())
    result = {"sha256": digest(path), "epoch_one_based": checkpoint["epoch"] + 1,
              "optimizer_updates": checkpoint["global_step"], "state_sha256": state.hexdigest(),
              "complete_optimizer_scheduler_state": True}
    del checkpoint
    gc.collect()
    return result


def audit(root):
    root = root.resolve()
    inputs = {}

    def remember(path):
        inputs[str(Path(path).relative_to(REPO))] = digest(path)

    def read(path):
        remember(path)
        return json.loads(Path(path).read_text())

    queue = read(root / "campaign_queue.json")
    assert queue["status"] == "awaiting_cost_review"
    assert queue["policy_hash"] == policy_hash()
    assert [model["model_id"] for model in queue["models"]] == policy()["models"]
    projection, metadata = load_metadata_pair(REPO / "data/input/yummly")
    classes = list(projection.base_class_order)
    expected_targets = _encode_targets(metadata["val"], classes, projection.target_field)
    expected_ids = np.array([str(row["id"]) for row in metadata["val"]])
    expected_images = np.array([row["image"] for row in metadata["val"]])
    remember(REPO / "src/training/resources/phase6_protocol_v1.json")
    remember(REPO / "src/ingredient_selection/resources/ingredients_selected_v5_d6_v1.json")
    for split in ("train", "val"):
        remember(REPO / "data/input/yummly" / split / projection.metadata_filename)
    baseline = read(root / "baselines/baseline_summary.json")
    remember(root / "baselines/validation_controls.npz")
    assert digest(root / "baselines/validation_controls.npz") == baseline["provenance"]["archive_sha256"]
    records = []
    for item in queue["models"]:
        model_id = item["model_id"]
        print(f"Auditing {model_id}", file=sys.stderr, flush=True)
        assert item["status"] == "complete"
        folder = root / model_id
        source = read(folder / "source_inventory.json")
        assert source["base_revision"] == queue["base_revision"]
        remember(folder / "source_snapshot.zip")
        assert digest(folder / "source_snapshot.zip") == source["zip_sha256"]
        with zipfile.ZipFile(folder / "source_snapshot.zip") as archive:
            assert set(archive.namelist()) == set(source["files"])
            for name, sha in source["files"].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == sha
                assert digest(REPO / name) == sha
        assert read(folder / "policy.json") == policy()
        identity = read(folder / "study_identity.json")
        assert identity["policy_hash"] == policy_hash() and identity["model_id"] == model_id
        data = read(folder / "data_identity.json")
        assert data["base_class_order"] == classes
        assert data["metadata_hashes"] == dict(projection.metadata_hashes)
        environment = read(folder / "environment.json")
        artifact = Path(torch.hub.get_dir()) / "checkpoints" / environment["original_artifact"]
        assert artifact.stat().st_size == environment["bytes"] and digest(artifact) == environment["sha256"]
        if model_id == "dinov2_b14_anchor":
            remember(folder / "dinov2_source.zip")
            with zipfile.ZipFile(folder / "dinov2_source.zip") as archive:
                assert len(archive.namelist()) == 141
                cached = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
                assert all(archive.read(name) == (cached / name).read_bytes() for name in archive.namelist())
        journal_path = folder / "journal.log"
        remember(journal_path)
        events = [json.loads(line) for line in journal_path.read_text().splitlines()]
        assert sum(event["op_code"] == 4 for event in events) == 1
        attrs = {}
        for event in events:
            if event["op_code"] == 8:
                attrs.update(event["user_attr"])
        states = [event for event in events if event["op_code"] == 6]
        assert states[-1]["state"] == 1
        assert attrs["completed_epochs"] == 40 and attrs["optimizer_updates"] == 15000
        assert states[-1]["values"] == [item["result"]["best_value"]]
        remember(folder / "sampler.bin")
        checksum, payload = (folder / "sampler.bin").read_bytes().split(b"\n", 1)
        assert hashlib.sha256(payload).hexdigest().encode() == checksum
        costs_path = folder / "trial_0/epoch_costs.jsonl"
        remember(costs_path)
        costs = [json.loads(line) for line in costs_path.read_text().splitlines()]
        assert [row["epoch"] for row in costs] == list(range(1, 41))
        assert [row["optimizer_updates"] for row in costs] == [375 * epoch for epoch in range(1, 41)]
        assert all(np.isfinite(row["seconds"]) and row["seconds"] > 0 for row in costs)
        best = min(costs, key=lambda row: (-row["val_macro_ap"], row["epoch"]))
        assert best["val_macro_ap"] == item["result"]["best_value"]
        config = decoded(read(folder / "trial_0/trial_config.json"))
        hp = config["hyper_parameters"]
        assert hp["benchmark_policy_hash"] == policy_hash() and hp["output_class_order"] == classes
        assert hp["batch_size"] == 128 and hp["weighted_loss"] is False
        assert hp["lr"] == (1e-3 if model_id == "dinov2_b14_anchor" else 1e-4)
        assert hp["weight_decay"] == 1e-4
        prediction = read(folder / "validation/trial_0/prediction_manifest.json")
        assert prediction == item["result"]["validation_predictions"]
        assert prediction["policy_hash"] == policy_hash() and prediction["test_image_inference"] is False
        assert prediction["metadata_sha256"] == dict(projection.metadata_hashes)["val"]
        archive_path = folder / "validation/trial_0/validation_predictions.npz"
        remember(archive_path)
        assert digest(archive_path) == prediction["archive_sha256"]
        with np.load(archive_path, allow_pickle=False) as arrays:
            assert arrays["logits"].shape == (5996, 165) and np.isfinite(arrays["logits"]).all()
            for key, expected in (("class_order", classes), ("targets", expected_targets),
                                  ("record_ids", expected_ids), ("image_names", expected_images)):
                assert np.array_equal(arrays[key], expected)
            metrics = epoch_metrics(arrays["logits"], arrays["targets"])
        assert abs(metrics["val_macro_ap"] - item["result"]["best_value"]) <= 1e-6
        best_path = folder / prediction["checkpoint"]
        last_path = folder / "trial_0/checkpoints/last.ckpt"
        assert best_path != last_path
        best_checkpoint = checkpoint_identity(best_path, classes, model_id)
        last_checkpoint = checkpoint_identity(last_path, classes, model_id)
        remember(best_path)
        remember(last_path)
        assert best_checkpoint["sha256"] == prediction["checkpoint_sha256"]
        assert best_checkpoint["epoch_one_based"] == best["epoch"]
        assert best_checkpoint["optimizer_updates"] == 375 * best["epoch"]
        seconds = [row["seconds"] for row in costs]
        elapsed = (datetime.fromisoformat(item["completed_at"]) - datetime.fromisoformat(item["started_at"])).total_seconds()
        records.append({"model_id": model_id, "epochs": 40, "optimizer_updates": 15000,
                        "source_entries": len(source["files"]), "source_zip_sha256": source["zip_sha256"],
                        "epoch_median_seconds": statistics.median(seconds), "epoch_min_seconds": min(seconds),
                        "epoch_max_seconds": max(seconds), "epoch_sum_seconds": sum(seconds),
                        "trial_session_seconds": attrs["session_wall_seconds"], "release_wall_seconds": elapsed,
                        "best_epoch": best["epoch"], "metrics": metrics,
                        "best_checkpoint": best_checkpoint, "last_checkpoint": last_checkpoint,
                        "final_checkpoint_valid": last_checkpoint["epoch_one_based"] == 40 and last_checkpoint["optimizer_updates"] == 15000,
                        "remaining_99_full_run_equivalent_days": 99 * elapsed / 86400})
    # Verify that the inspected campaign artifacts were not changed during the audit.
    assert all(digest(REPO / name) == sha for name, sha in inputs.items())
    total = sum(row["release_wall_seconds"] for row in records)
    return {"campaign": str(root.relative_to(REPO)), "policy_hash": policy_hash(),
            "base_revision": queue["base_revision"], "completed_at": queue["completed_at"],
            "queue_status": queue["status"], "all_training_exports_verified": True,
            "all_final_checkpoints_valid": all(row["final_checkpoint_valid"] for row in records),
            "test_metadata_or_images_read_by_audit": False, "models": records,
            "release_wall_seconds_sum": total, "remaining_99_full_run_equivalent_days": 99 * total / 86400,
            "projection_is_conditional_not_eta_or_measured_gpu_active_time": True,
            "input_sha256": inputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(REPO / "experiments/benchmark_v1/phase62-20261018-v1")
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "input_sha256"}, indent=2))

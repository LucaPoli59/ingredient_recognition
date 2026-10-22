"""Resumable CPU-only, frozen ResNet-18 features for the ingredient head pilot."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

# CPU isolation must precede importing torch, including when launched without env.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
import torch
import torchvision
from src_scratches.label_autoencoder.image_models import load_backbone

from src.data_processing.experimental_transforms import ExperimentalFitPad224
from src_scratches.label_autoencoder.compare import ROOT, DEFAULT_DATA, sha256


FEATURE_ALGORITHM = "resnet18-gap512-fp32-channels-last-fit-pad224-v1"
LEGACY_EXTRACTOR_SHA256 = "d56056645d4e45483f885724e66c123389156d5b429f5656aad382d68a8eb91d"


def migrate_identity_v1(identity):
    """Recognize the exact first pilot implementation; preserve it in manifest history."""
    if identity.get("schema_version") != 1 or identity.get("extractor_sha256") != LEGACY_EXTRACTOR_SHA256:
        raise ValueError("Unknown legacy feature implementation; cannot reuse cache")
    upgraded = dict(identity)
    upgraded.pop("threads")
    upgraded.pop("extractor_sha256")
    upgraded.update(schema_version=2, feature_algorithm=FEATURE_ALGORITHM)
    return upgraded


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_image_rows(data_dir, metadata):
    if Path(metadata).name != metadata:
        raise ValueError("metadata must be a filename")
    rows, sources = {}, {}
    for split in ("train", "val"):
        path = data_dir / split / metadata
        raw = path.read_bytes()
        records = json.loads(raw)
        if not isinstance(records, list) or not records:
            raise ValueError(f"{split}: expected a nonempty list")
        names = [record["image"] for record in records]
        if any(not isinstance(name, str) or not name or Path(name).name != name for name in names):
            raise ValueError("Images must be basenames in the shared image store")
        rows[split] = names
        sources[split] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(names)}
    return rows, sources


def validate_chunks(output, manifest, image_rows):
    """Check committed chunks before resuming; uncommitted final files can be overwritten."""
    positions = {"train": 0, "val": 0}
    for chunk in manifest["chunks"]:
        split, start, stop = chunk["split"], chunk["start"], chunk["stop"]
        if start != positions[split] or not start < stop <= len(image_rows[split]):
            raise ValueError("Non-contiguous feature chunks")
        for key in ("features", "images"):
            path = output / chunk[key]
            if sha256(path) != chunk[key + "_sha256"]:
                raise ValueError(f"Corrupt cache artifact: {path.name}")
        features = np.load(output / chunk["features"], allow_pickle=False)
        if features.shape != (stop - start, 512) or not np.isfinite(features).all():
            raise ValueError("Invalid feature shape or values")
        images = json.loads((output / chunk["images"]).read_text())
        if [row["image"] for row in images] != image_rows[split][start:stop]:
            raise ValueError("Feature/image row-order mismatch")
        positions[split] = stop
    return positions


def extract(args, backbone=None):
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    rows, sources = read_image_rows(args.data_dir, args.metadata)
    transform = ExperimentalFitPad224(horizontal_flip_probability=0)
    identity = {
        "schema_version": 2, "feature_algorithm": FEATURE_ALGORITHM,
        "backbone": "torchvision-resnet18-imagenet1k-v1-frozen-gap512",
        "weights_sha256": sha256(args.weights), "metadata": sources,
        "image_store": str((args.data_dir / "imgs/standard").resolve()),
        "transform": transform.to_config(),
        "transform_source_sha256": sha256(ROOT / "src/data_processing/experimental_transforms.py"),
        "batch_size": args.batch_size, "chunk_size": args.chunk_size,
        "versions": {"torch": str(torch.__version__), "torchvision": torchvision.__version__, "numpy": np.__version__},
    }
    manifest_path = args.output / "feature_manifest.json"
    if args.output.exists():
        if not args.resume or not manifest_path.exists():
            raise FileExistsError("Existing cache requires --resume and its feature manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"].get("schema_version") == 1:
            upgraded = migrate_identity_v1(manifest["identity"])
            if upgraded != identity:
                raise ValueError("Legacy feature cache identity changed; use a new output directory")
            manifest["initial_identity_v1"] = manifest["identity"]
            for session in manifest["sessions"]:
                session["threads"] = manifest["identity"]["threads"]
                session["extractor_sha256"] = manifest["identity"]["extractor_sha256"]
            manifest["identity"] = upgraded
            manifest["identity_upgrade"] = {"utc": datetime.now(timezone.utc).isoformat(),
                "reason": "Separate execution thread count/source history from unchanged feature algorithm"}
        if manifest["identity"] != identity:
            raise ValueError("Feature cache identity changed; use a new output directory")
        positions = validate_chunks(args.output, manifest, rows)
        if manifest["status"] == "complete":
            return manifest
    else:
        args.output.mkdir(parents=True)
        manifest = {"identity": identity, "status": "running", "chunks": [],
                    "started_utc": datetime.now(timezone.utc).isoformat(), "sessions": []}
        positions = {"train": 0, "val": 0}
    manifest["status"] = "running"
    manifest.pop("error", None)
    manifest["sessions"].append({"started_utc": datetime.now(timezone.utc).isoformat(),
                                 "pid": os.getpid(), "scheduler": os.sched_getscheduler(0) if hasattr(os, "sched_getscheduler") else None,
                                 "threads": args.threads, "extractor_sha256": sha256(Path(__file__)),
                                 "model_source_sha256": sha256(Path(__file__).with_name("image_models.py")),
                                 "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                                 "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "positions": dict(positions)})
    atomic_json(manifest_path, manifest)
    model = backbone if backbone is not None else load_backbone(args.weights)
    model.requires_grad_(False).eval()
    image_store = args.data_dir / "imgs/standard"
    try:
        with threadpool_limits(limits=args.threads), torch.inference_mode():
            for split in ("train", "val"):
                for start in range(positions[split], len(rows[split]), args.chunk_size):
                    stop = min(start + args.chunk_size, len(rows[split]))
                    feature_batches, image_records = [], []
                    for offset in range(start, stop, args.batch_size):
                        batch = []
                        for name in rows[split][offset:min(offset + args.batch_size, stop)]:
                            image_path = image_store / name
                            if image_path.resolve().parent != image_store.resolve():
                                raise ValueError("Image resolves outside shared store")
                            raw = image_path.read_bytes()
                            image_records.append({"image": name, "sha256": hashlib.sha256(raw).hexdigest()})
                            with Image.open(io.BytesIO(raw)) as image:
                                batch.append(transform(image))
                        values = torch.stack(batch).to(memory_format=torch.channels_last)
                        features = model(values).cpu().numpy()
                        if features.shape != (len(batch), 512) or not np.isfinite(features).all():
                            raise ValueError("Backbone produced invalid features")
                        feature_batches.append(features)
                    stem = f"{split}_{start:06d}_{stop:06d}"
                    feature_path = args.output / (stem + ".npy")
                    image_path = args.output / (stem + "_images.json")
                    temporary = feature_path.with_suffix(".npy.tmp")
                    with temporary.open("wb") as handle:
                        np.save(handle, np.concatenate(feature_batches), allow_pickle=False)
                    temporary.replace(feature_path)
                    atomic_json(image_path, image_records)
                    manifest["chunks"].append({"split": split, "start": start, "stop": stop,
                        "features": feature_path.name, "features_sha256": sha256(feature_path),
                        "images": image_path.name, "images_sha256": sha256(image_path)})
                    positions[split] = stop
                    manifest["progress"] = dict(positions)
                    manifest["sessions"][-1]["elapsed_seconds"] = time.perf_counter() - started
                    atomic_json(manifest_path, manifest)
                    print(f"features {split}: {stop}/{len(rows[split])}; "
                          f"session {time.perf_counter() - started:.1f}s", flush=True)
        hashes = {split: set() for split in rows}
        for chunk in manifest["chunks"]:
            images = json.loads((args.output / chunk["images"]).read_text())
            hashes[chunk["split"]].update(row["sha256"] for row in images)
        overlap = len(hashes["train"] & hashes["val"])
        manifest["cross_split_identical_image_hashes"] = overlap
        if overlap:
            raise ValueError(f"Found {overlap} byte-identical train/validation images; do not compare heads")
        if torch.cuda.is_initialized():
            raise RuntimeError("Unexpected CUDA context in CPU feature extraction")
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["sessions"][-1]["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(manifest_path, manifest)
    return manifest


def load_features(cache, provenance):
    """Load only a complete cache with exact source hashes and validated artifact order."""
    manifest = json.loads((cache / "feature_manifest.json").read_text())
    if manifest["status"] != "complete" or manifest.get("cross_split_identical_image_hashes") != 0:
        raise ValueError("Feature cache is incomplete or fails split isolation")
    for split in ("train", "val"):
        cached = manifest["identity"]["metadata"][split]
        if any(cached[key] != provenance[split][key] for key in ("sha256", "rows")):
            raise ValueError("Feature cache metadata differs from current label rows")
    # Re-read the tiny metadata sources to establish row order, not the image bytes.
    names = {split: [row["image"] for row in json.loads(Path(provenance[split]["path"]).read_text())]
             for split in ("train", "val")}
    positions = validate_chunks(cache, manifest, names)
    if any(positions[split] != provenance[split]["rows"] for split in positions):
        raise ValueError("Feature cache has missing rows")
    matrices = {split: np.concatenate([np.load(cache / chunk["features"], allow_pickle=False)
                                      for chunk in manifest["chunks"] if chunk["split"] == split])
                for split in ("train", "val")}
    return matrices["train"], matrices["val"], manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metadata", default="ingredients_target_v5_metadata.json")
    parser.add_argument("--weights", type=Path, default=Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.chunk_size, args.threads) < 1:
        parser.error("batch-size, chunk-size and threads must be positive")
    return args


if __name__ == "__main__":
    extract(parse_args())

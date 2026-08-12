"""Audit and reproduce the retained November 2024 ingredient-selection evidence.

The command is deliberately read-only unless ``--write-manifest`` is supplied.
It verifies the selected retention manifest, reconstructs the historical
four-run top-quartile train-F1 rule, checks the projected metadata, and loads
the three executable checkpoint anchors without instantiating the current
training stack.  It does not migrate, rewrite, or delete legacy artifacts.

Examples
--------
    python scripts/validate_legacy_experiments.py --write-manifest
    python scripts/validate_legacy_experiments.py
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import io
import json
import math
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MANIFEST_RELATIVE_PATH = Path(
    "src_scratches/ingredient_selection_reconstruction/retention_manifest.json"
)
EXPECTED_METADATA_HASHES = {
    "train": "69189FD97ADB0AAAB6748874F44CD50DAA6FC7D1A739DFE9040EC28018E5623B",
    "val": "382D24E2C99DE5C93EEC6EF7E29F00935BB5166C42C98EC37458DA9E3474B102",
    "test": "0E37DFE11219CD4DCF21185E4FA2AA8D6F218BCF10DC1B8E7EE72D1F4F163FF3",
}
EXPECTED_METADATA_COUNTS = {"train": 50866, "val": 4802, "test": 4854}
EXPECTED_SELECTED_LABEL_INDICES = [
    9,
    15,
    18,
    19,
    25,
    29,
    30,
    33,
    40,
    44,
    46,
    48,
    49,
    51,
    56,
    57,
    60,
    61,
    64,
    69,
    70,
    72,
    88,
    91,
    96,
    104,
    106,
    111,
    113,
    119,
    136,
    142,
    149,
    151,
    154,
    156,
    164,
    168,
    170,
    180,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def add_entry(
    entries: list[dict[str, Any]],
    root: Path,
    relative_path: str,
    group: str,
    role: str,
    anchor: bool = False,
) -> None:
    path = root / Path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"retained artifact is missing: {relative_path}")
    entries.append(
        {
            "path": relative_path.replace("\\", "/"),
            "group": group,
            "role": role,
            "anchor": anchor,
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
    )


def add_trial_files(
    entries: list[dict[str, Any]],
    root: Path,
    experiment: str,
    trial_ids: Iterable[int],
    group: str,
    include_trial_config: bool,
    anchor_trial: int,
) -> None:
    for trial_id in trial_ids:
        trial = f"{experiment}/trial_{trial_id}"
        if include_trial_config:
            add_entry(entries, root, f"{trial}/trial_config.json", group, "trial configuration")
        add_entry(entries, root, f"{trial}/hparams.yaml", group, "saved hyperparameters")
        add_entry(entries, root, f"{trial}/metrics.csv", group, "per-trial metrics")
        if trial_id == anchor_trial:
            add_entry(
                entries,
                root,
                f"{trial}/best_model.ckpt",
                group,
                "executable checkpoint anchor",
                anchor=True,
            )


def build_manifest(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    h1 = "experiments/basic/resnets_htuning{25k}"
    for filename in ("best_trials.csv", "full_metrics.csv", "hparam_config.json", "hparam_gen_config.json", "trials_info.csv"):
        add_entry(entries, root, f"{h1}/{filename}", "H1", "aggregate tuning evidence")
    add_trial_files(entries, root, h1, (21, 2, 18), "H1", True, 21)

    h2 = "experiments/basic/resnets_training_BM_F1_INGS"
    for filename in ("full_f1_train.csv", "full_labels_stats.csv", "full_metrics.csv"):
        add_entry(entries, root, f"{h2}/{filename}", "H2", "aggregate selection evidence")
    add_trial_files(entries, root, h2, (0, 1, 2, 3), "H2", False, 0)

    h3 = "experiments/basic/resnets_htuning_sel{10k}"
    for filename in ("best_trials.csv", "full_metrics.csv", "hparam_config.json", "hparam_gen_config.json", "trials_info.csv"):
        add_entry(entries, root, f"{h3}/{filename}", "H3", "aggregate selected-vocabulary tuning evidence")
    add_trial_files(entries, root, h3, (64, 66, 68), "H3", True, 64)

    h4 = root / "experiments/basic/resnets_htuning_sel{10k}_test"
    for path in sorted(h4.rglob("*")):
        if path.is_file():
            add_entry(entries, root, rel(root, path), "H4", "selected-vocabulary validation evidence")

    for split in ("train", "val", "test"):
        add_entry(
            entries,
            root,
            f"data/input/yummly/{split}/sel_ing_2410_metadata.json",
            "metadata",
            f"immutable projected legacy metadata ({split})",
        )

    for filename in (
        "analize_f1_ings.ipynb",
        "analize_test_f1.ipynb",
        "fake_study.ipynb",
        "refine_study_data.ipynb",
        "visualize_study_data.ipynb",
    ):
        add_entry(entries, root, f"scripts/analize_exps/{filename}", "provenance", "analysis notebook")
    for relative_path in (
        "scripts/launch_exps/resnet/htuning_resnets.py",
        "scripts/launch_exps/resnet/train_resnets_bs_f1_ings.py",
        "scripts/launch_exps/resnet/htuning_resnets_sel_ings.py",
        "scripts/launch_exps/test_best_for_f1.py",
        "experiments/journal.log",
        "experiments/journal_trash.log",
    ):
        add_entry(entries, root, relative_path, "provenance", "historical launch or journal provenance")

    entries.sort(key=lambda item: item["path"])
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_relative_paths": True,
        "read_only_verification": True,
        "purpose": "Minimum retained evidence for Data work package 2.1c historical compatibility.",
        "groups": {
            "H1": "initial ResNet tuning context",
            "H2": "four-run full-vocabulary train-F1 selection evidence",
            "H3": "selected-vocabulary tuning context",
            "H4": "selected-vocabulary validation evidence",
            "metadata": "immutable projected legacy metadata",
            "provenance": "analysis and launch provenance",
        },
        "artifacts": entries,
    }


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def quantile(values: list[float], probability: float) -> float:
    """Match pandas/numpy's default linear quantile for a one-dimensional list."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def reproduce_selection(root: Path) -> dict[str, Any]:
    aggregate = root / "experiments/basic/resnets_training_BM_F1_INGS/full_f1_train.csv"
    with aggregate.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    label_columns = [str(index) for index in range(183)]
    if set(label_columns) - set(rows[0] if rows else ()):
        raise AssertionError("full_f1_train.csv does not contain all 183 label columns")

    selected_by_trial: dict[str, list[int]] = {}
    thresholds: dict[str, float] = {}
    for trial_id in range(4):
        trial_rows = [row for row in rows if int(row["trial"]) == trial_id]
        if len(trial_rows) != 40 or {int(row["epoch"]) for row in trial_rows} != set(range(40)):
            raise AssertionError(f"H2 trial {trial_id} does not contain one row for each epoch 0..39")
        maxima = {
            label: max(float(row[label]) for row in trial_rows)
            for label in label_columns
        }
        threshold = quantile(list(maxima.values()), 0.75)
        selected = [int(label) for label, value in maxima.items() if value >= threshold]
        if len(selected) != 46:
            raise AssertionError(f"H2 trial {trial_id} top quartile has {len(selected)} labels, expected 46")
        selected_by_trial[str(trial_id)] = sorted(selected)
        thresholds[str(trial_id)] = threshold

    intersection = sorted(set(selected_by_trial["0"]) & set(selected_by_trial["1"]) & set(selected_by_trial["2"]) & set(selected_by_trial["3"]))
    if intersection != EXPECTED_SELECTED_LABEL_INDICES:
        raise AssertionError(f"historical 40-label intersection changed: {intersection}")

    metadata = load_json(root / "data/input/yummly/train/metadata.json")
    classes = sorted({label for record in metadata for label in record["ingredients_ok"]}) + ["<UNK>"]
    if len(classes) != 183:
        raise AssertionError(f"legacy class order has {len(classes)} classes, expected 183")
    selected_names = [classes[index] for index in intersection]
    return {
        "source": "experiments/basic/resnets_training_BM_F1_INGS/full_f1_train.csv",
        "metric": "maximum train F1 over epochs 0..39",
        "threshold": "per-trial 0.75 quantile (q3), inclusive",
        "selected_by_trial": selected_by_trial,
        "thresholds": thresholds,
        "intersection_indices": intersection,
        "intersection_labels": selected_names,
        "class_order": classes,
        "class_order_size": len(classes),
    }


def verify_metadata(root: Path, selected_labels: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"splits": {}, "image_root": "data/input/yummly/imgs/standard"}
    image_root = root / "data/input/yummly/imgs/standard"
    for split, expected_count in EXPECTED_METADATA_COUNTS.items():
        selected_path = root / f"data/input/yummly/{split}/sel_ing_2410_metadata.json"
        original_path = root / f"data/input/yummly/{split}/metadata.json"
        selected = load_json(selected_path)
        original = load_json(original_path)
        selected_hash = sha256(selected_path)
        if selected_hash != EXPECTED_METADATA_HASHES[split]:
            raise AssertionError(
                f"{split} selected metadata hash changed: {selected_hash}"
            )
        if len(selected) != expected_count:
            raise AssertionError(f"{split} selected metadata has {len(selected)} records, expected {expected_count}")
        if len({record["id"] for record in selected}) != len(selected):
            raise AssertionError(f"{split} selected metadata contains duplicate ids")
        for record in selected:
            labels = set(record["ingredients_ok"])
            if not labels <= selected_labels or len(labels) < 3:
                raise AssertionError(f"{split} selected metadata violates the 40-label/minimum-3 rule")
            image = image_root / record["image"]
            if not image.is_file():
                raise AssertionError(f"{split} image is not available in shared store: {record['image']}")
        # Open one image from both metadata generations to test the shared layout.
        try:
            from PIL import Image

            for record in (original[0], selected[0]):
                with Image.open(image_root / record["image"]) as image:
                    image.verify()
        except ImportError as exc:
            raise AssertionError("Pillow is required for the shared-image smoke check") from exc
        result["splits"][split] = {
            "original_records": len(original),
            "selected_records": len(selected),
            "selected_min_targets": min(len(record["ingredients_ok"]) for record in selected),
            "selected_max_targets": max(len(record["ingredients_ok"]) for record in selected),
            "sample_image_and_target_batch": True,
        }
    return result


def verify_checkpoints(root: Path) -> list[dict[str, Any]]:
    anchors = [
        ("H1", "experiments/basic/resnets_htuning{25k}/trial_21/best_model.ckpt", 183),
        ("H2", "experiments/basic/resnets_training_BM_F1_INGS/trial_0/best_model.ckpt", 183),
        ("H3", "experiments/basic/resnets_htuning_sel{10k}/trial_64/best_model.ckpt", 41),
    ]
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            import torch
    except ImportError as exc:
        raise AssertionError("Torch is required to load the retained checkpoint anchors") from exc

    loaded: list[dict[str, Any]] = []
    for group, relative_path, expected_classes in anchors:
        checkpoint = torch.load(root / relative_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", {})
        candidates = [
            value
            for key, value in state_dict.items()
            if key.endswith("classifier.2.weight") or key.endswith("fc.weight")
        ]
        if not candidates:
            raise AssertionError(f"{group} checkpoint has no recognizable classifier weight")
        output_classes = int(candidates[-1].shape[0])
        if output_classes != expected_classes:
            raise AssertionError(f"{group} checkpoint has {output_classes} outputs, expected {expected_classes}")
        loaded.append(
            {
                "group": group,
                "path": relative_path,
                "epoch": int(checkpoint["epoch"]),
                "global_step": int(checkpoint["global_step"]),
                "output_classes": output_classes,
            }
        )
    return loaded


def verify_saved_config_contract(root: Path, expected_classes: list[str]) -> dict[str, Any]:
    """Check the saved metadata/target contract and the serialized H2 classes."""

    def data_config(path: Path) -> dict[str, Any]:
        config = load_json(path)["datamodule_hyper_parameters"][1]
        return {
            "metadata_filename": config["metadata_filename"][1],
            "feature_label": config["feature_label"][1],
        }

    result = {
        "H1": data_config(root / "experiments/basic/resnets_htuning{25k}/trial_21/trial_config.json"),
        "H3": data_config(root / "experiments/basic/resnets_htuning_sel{10k}/trial_64/trial_config.json"),
    }
    if result["H1"] != {"metadata_filename": "metadata.json", "feature_label": "ingredients_ok"}:
        raise AssertionError(f"H1 saved target contract changed: {result['H1']}")
    if result["H3"] != {"metadata_filename": "sel_ing_2410_metadata.json", "feature_label": "ingredients_ok"}:
        raise AssertionError(f"H3 saved target contract changed: {result['H3']}")

    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            import torch
    except ImportError as exc:
        raise AssertionError("Torch is required to inspect the serialized H2 class order") from exc
    checkpoint = torch.load(
        root / "experiments/basic/resnets_training_BM_F1_INGS/trial_0/best_model.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    datamodule = checkpoint["datamodule_hyper_parameters"]
    serialized_classes = json.loads(datamodule["label_encoder"][1]["classes"][1])
    if serialized_classes != expected_classes:
        raise AssertionError("H2 serialized label class order differs from legacy metadata order")
    if datamodule["metadata_filename"][1] != "metadata.json" or datamodule["feature_label"][1] != "ingredients_ok":
        raise AssertionError("H2 saved target contract changed")
    result["H2"] = {
        "metadata_filename": datamodule["metadata_filename"][1],
        "feature_label": datamodule["feature_label"][1],
        "serialized_class_order": "matches legacy metadata",
    }
    return result


def verify_configuration_evidence(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for trial_id in range(4):
        path = root / f"experiments/basic/resnets_training_BM_F1_INGS/trial_{trial_id}/hparams.yaml"
        text = path.read_text(encoding="utf-8")
        weighted_match = re.search(r"^weighted_loss:\s*(true|false)\s*$", text, re.MULTILINE | re.IGNORECASE)
        augmentation_match = re.search(r"^\s+trns_aug:\s*(.*?)\s*$", text, re.MULTILINE)
        if weighted_match is None or weighted_match.group(1).lower() != "false":
            raise AssertionError(f"H2 trial {trial_id} is not recorded as unweighted")
        result[str(trial_id)] = {
            "weighted_loss": False,
            "trns_aug": augmentation_match.group(1).strip() if augmentation_match else None,
            "source": f"experiments/basic/resnets_training_BM_F1_INGS/trial_{trial_id}/hparams.yaml",
        }
    return result


def verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        raise AssertionError("retention manifest has no artifacts")
    seen: set[str] = set()
    for artifact in artifacts:
        path = artifact["path"]
        if path in seen:
            raise AssertionError(f"duplicate manifest path: {path}")
        seen.add(path)
        file_path = root / Path(path)
        if not file_path.is_file():
            raise AssertionError(f"manifest artifact is missing: {path}")
        actual_size = file_path.stat().st_size
        actual_hash = sha256(file_path)
        if actual_size != artifact["size"] or actual_hash != artifact["sha256"]:
            raise AssertionError(f"manifest hash/size mismatch: {path}")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--write-manifest", action="store_true", help="write the current relative-path retention manifest")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest_path = (args.manifest or root / MANIFEST_RELATIVE_PATH).resolve()

    try:
        if args.write_manifest:
            manifest = build_manifest(root)
            write_report(manifest_path, manifest)
        elif not manifest_path.is_file():
            raise AssertionError(f"retention manifest not found: {manifest_path}")
        else:
            manifest = load_json(manifest_path)

        verify_manifest(root, manifest)
        selection = reproduce_selection(root)
        selected_labels = set(selection["intersection_labels"])
        metadata = verify_metadata(root, selected_labels)
        checkpoints = verify_checkpoints(root)
        saved_config_contract = verify_saved_config_contract(root, selection["class_order"])
        configurations = verify_configuration_evidence(root)
        print(json.dumps({
            "status": "PASS",
            "manifest": rel(root, manifest_path) if manifest_path.is_relative_to(root) else str(manifest_path),
            "manifest_artifacts": len(manifest["artifacts"]),
            "historical_selection": selection,
            "metadata": metadata,
            "checkpoint_anchors": checkpoints,
            "saved_config_contract": saved_config_contract,
            "configuration_evidence": configurations,
            "writes_performed": [rel(root, manifest_path)] if args.write_manifest else [],
        }, indent=2, ensure_ascii=False))
        return 0
    except (AssertionError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

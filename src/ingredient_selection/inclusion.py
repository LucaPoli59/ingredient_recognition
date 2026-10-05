"""Versioned D6 reanalysis of saved validation scores; never rewrites D4."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.ingredient_selection.inclusion_reporting import render_inclusion_plot
from src.ingredient_selection.inclusion_statistics import (
    classify_inclusion,
    paired_cluster_bootstrap,
)


POLICY_ID = "phase3-d6-held-out-quality-v1"
LATE_EPOCHS = (32, 34, 36, 38, 40)
SOURCE_FILES = (
    "src/ingredient_selection/__init__.py",
    "src/ingredient_selection/inclusion.py",
    "src/ingredient_selection/inclusion_statistics.py",
    "src/ingredient_selection/inclusion_reporting.py",
    "scripts/ingredient_selection/review_inclusion.py",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace changed D6 artifact: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f"refusing to replace changed D6 artifact: {path}")
    finally:
        Path(temporary).unlink(missing_ok=True)


def _verify_hash(payload: dict) -> None:
    body = dict(payload)
    expected = body.pop("artifact_hash", None)
    if expected != _hash_json(body):
        raise ValueError("invalid canonical artifact hash")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    # Preserve the original metrics.py float64 rounding and saturated ties,
    # without importing the training stack or altering its source-bound file.
    values = np.asarray(logits, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def load_inputs(repo_root: Path, campaign: Path) -> dict[str, Any]:
    """Validate the retained campaign and recover validation-only image groups."""
    repo_root, campaign = repo_root.resolve(), campaign.resolve()
    manifest = _load(campaign / "campaign_manifest.json")
    identity = manifest["campaign_identity"]
    if (manifest.get("status") != "completed"
            or identity.get("protocol_id") != "phase3-d1-v3"
            or manifest.get("campaign_identity_hash") != _hash_json(identity)
            or identity.get("seed") != 42
            or identity.get("execution", {}).get("max_epochs") != 40):
        raise ValueError("requires the completed, hash-valid Phase 3 v3 campaign")
    names = identity["class_order"]
    if (len(names) != 165 or len(set(names)) != 165
            or identity["class_order_hash"] != _hash_json(names)):
        raise ValueError("invalid frozen 165-class order")
    rule = _load(campaign / "profile_rule.json")
    _verify_hash(rule)
    if (rule["campaign_identity_hash"] != manifest["campaign_identity_hash"]
            or rule["classifier_source_sha256"] != _hash_file(
                repo_root / "src/ingredient_selection/metrics.py")
            or rule["pilot_evidence_sha256"] != _hash_file(
                campaign / "pilot_profile_evidence.csv")):
        raise ValueError("original source-bound D4 rule or pilot has changed")
    report = _load(campaign / "p4_profile_report.json")
    validation = _load(campaign / "validation_summary.json")
    evidence = _csv(campaign / "profile_evidence.csv")
    evidence_hash = _hash_file(campaign / "profile_evidence.csv")
    if (report["profile_rule_hash"] != rule["artifact_hash"]
            or report["campaign_identity_hash"] != manifest["campaign_identity_hash"]
            or validation["campaign_identity_hash"] != manifest["campaign_identity_hash"]
            or report["profile_evidence_sha256"] != evidence_hash
            or validation["profile_evidence_sha256"] != evidence_hash
            or validation.get("analysis_scope") != "full"
            or validation.get("test_split_accessed") is not False
            or [r["class_name"] for r in evidence] != names
            or [int(r["class_index"]) for r in evidence] != list(range(165))):
        raise ValueError("D4 full evidence, class order or test-isolation record changed")
    inputs = {name: _hash_file(campaign / name) for name in (
        "campaign_manifest.json", "profile_rule.json", "profile_evidence.csv",
        "pilot_profile_evidence.csv", "p4_profile_report.json", "validation_summary.json",
        "metrics_per_label_epoch.csv",
    )}
    source_identity = manifest["source_identity"]
    snapshot_path = campaign / "source_snapshot.zip"
    if (identity.get("source_identity_hash") != source_identity["sha256"]
            or source_identity["sha256"] != _hash_json(source_identity["files"])
            or _hash_file(snapshot_path) != manifest["source_snapshot"]["sha256"]):
        raise ValueError("training source snapshot/inventory hash differs from manifest")
    with zipfile.ZipFile(snapshot_path) as archive:
        if (len(archive.namelist()) != len(source_identity["files"])
                or set(archive.namelist()) != set(source_identity["files"])
                or any(hashlib.sha256(archive.read(name)).hexdigest() != digest
                       for name, digest in source_identity["files"].items())):
            raise ValueError("training snapshot members differ from source inventory")
    inputs["source_snapshot.zip"] = _hash_file(snapshot_path)
    data_root = repo_root / "data/input/yummly"
    metadata = {}
    for split in ("train", "val"):
        path = data_root / split / "ingredients_target_v5_metadata.json"
        digest = _hash_file(path)
        if digest != identity["metadata_sha256"][split]:
            raise ValueError(f"{split} metadata differs from frozen campaign")
        inputs[f"metadata/{split}"] = digest
        records = _load(path)
        if not records or len({str(r["id"]) for r in records}) != len(records):
            raise ValueError(f"invalid or duplicate {split} record identities")
        metadata[split] = records
    if sorted({name for r in metadata["train"] for name in r["ingredients_target"]}) != names:
        raise ValueError("train metadata does not define the saved class order")
    name_to_index = {name: index for index, name in enumerate(names)}
    encoded = {}
    for split, records in metadata.items():
        target = np.zeros((len(records), len(names)), dtype=np.uint8)
        for row, record in enumerate(records):
            labels = record["ingredients_target"]
            if not isinstance(labels, list) or set(labels) - name_to_index.keys():
                raise ValueError(f"invalid {split} targets")
            for name in labels:
                target[row, name_to_index[name]] = 1
        encoded[split] = target
        supports = target.sum(axis=0)
        if any(int(evidence[i][f"{split}_support"]) != supports[i] for i in range(165)):
            raise ValueError(f"{split} supports differ from retained evidence")
    val_records = metadata["val"]
    ids = [str(r["id"]) for r in val_records]
    targets = encoded["val"]
    metrics = _csv(campaign / "metrics_per_label_epoch.csv")
    keyed = {(r["split"], int(r["audit_epoch"]), int(r["class_index"])): r for r in metrics}
    expected = {(s, e, c) for s in ("train", "val") for e in range(0, 41, 2) for c in range(165)}
    if len(keyed) != len(metrics) or set(keyed) != expected:
        raise ValueError("metric table has missing/duplicate/unexpected split/epoch/class keys")
    for (split, _, index), row in keyed.items():
        if (row["run_id"] != "phase3-d1-v3" or row["class_name"] != names[index]
                or int(row["records"]) != len(metadata[split])
                or int(row["support"]) != int(encoded[split][:, index].sum())
                or not np.isfinite(float(row["average_precision"]))):
            raise ValueError("invalid metric identity, support or finite AP")
    if any(r.get("trajectory_complete") != "True" for r in evidence):
        raise ValueError("incomplete original evidence trajectory")
    scores = []
    for epoch in LATE_EPOCHS:
        relative = f"audit_scores/validation_epoch_{epoch:02d}.npz"
        path = campaign / relative
        inputs[relative] = _hash_file(path)
        with np.load(path, allow_pickle=False) as archive:
            logits = archive["logits"]
            if (set(archive.files) != {"record_ids", "targets", "logits"}
                    or archive["record_ids"].astype(str).tolist() != ids
                    or not np.array_equal(archive["targets"], targets)
                    or logits.shape != targets.shape or not np.isfinite(logits).all()):
                raise ValueError(f"invalid/alignment-mismatched validation scores at {epoch}")
            scores.append(_sigmoid(logits))
    image_root = (data_root / "imgs/standard").resolve()
    image_rows = []
    for record in val_records:
        image_path = (image_root / str(record["image"])).resolve()
        if not image_path.is_relative_to(image_root) or not image_path.is_file():
            raise ValueError("validation image escaped the image root or is absent")
        image_rows.append([str(record["id"]), str(record["image"]), _hash_file(image_path)])
    group_sizes = Counter(r[2] for r in image_rows)
    groups = {
        "schema_version": 1, "split": "val", "recovered_on": "2026-10-05",
        "record_count": len(ids), "cluster_count": len(group_sizes),
        "size_distribution": dict(sorted(Counter(group_sizes.values()).items())),
        "ordered_rows_sha256": _hash_json(image_rows),
        "columns": ["record_id", "image", "sha256"], "rows": image_rows,
        "provenance_limit": "Current image bytes; no campaign-time image hashes were saved.",
    }
    groups["artifact_hash"] = _hash_json(groups)
    return {
        "manifest": manifest, "d4_rule": rule, "evidence": evidence,
        "inputs_sha256": inputs, "names": names, "targets": targets,
        "scores": np.stack(scores), "metrics": keyed, "groups": groups,
        "cluster_ids": [r[2] for r in image_rows],
    }


def _source_snapshot(repo_root: Path) -> tuple[dict, bytes]:
    inventory = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for relative in SOURCE_FILES:
            content = (repo_root / relative).read_bytes()
            inventory[relative] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    return inventory, buffer.getvalue()


def review_inclusion(
        repo_root: str | Path, campaign_dir: str | Path, *,
        progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Freeze the approved D6 rule and report; do not export P6 metadata."""
    root, campaign = Path(repo_root).resolve(), Path(campaign_dir).resolve()
    output = campaign / "inclusion_d6_v1"
    basis = load_inputs(root, campaign)
    inventory, snapshot = _source_snapshot(root)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              text=True, capture_output=True, check=True).stdout.strip()
    if (output / "inclusion_rule.json").exists():
        frozen_rule = _load(output / "inclusion_rule.json")
        _verify_hash(frozen_rule)
        revision = frozen_rule["git_base_revision"]
    rule = {
        "schema_version": 1, "policy_id": POLICY_ID, "adopted_on": "2026-10-05",
        "interpretation": "outcome-informed exploratory post-P4 amendment",
        "campaign_identity_hash": basis["manifest"]["campaign_identity_hash"],
        "original_d4_rule_hash": basis["d4_rule"]["artifact_hash"],
        "class_order_hash": basis["manifest"]["campaign_identity"]["class_order_hash"],
        "quality_floor": 0.20, "max_validation_iqr": 0.03,
        "positive_excess_lower_bound_required": True,
        "late_epochs": list(LATE_EPOCHS), "statistic": "median_of_separate_checkpoint_AP",
        "bootstrap": {"unit": "exact_image_sha256_cluster", "cluster_order": "first_occurrence",
                      "draws": 1000, "max_attempts": 10000, "seed": "42000 + class_index",
                      "generator": "NumPy PCG64", "percentiles": [2.5, 97.5],
                      "quantile_method": "linear", "paired_constant_baseline": "resampled_prevalence"},
        "sensitivity_floors": [0.15, 0.20, 0.25],
        "diagnostic_only": ["train_support", "train_gain", "train_ap", "train_validation_gap", "cuisine_prior"],
        "git_base_revision": revision, "source_inventory": inventory,
        "source_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "inputs_sha256": basis["inputs_sha256"],
        "validation_groups_hash": basis["groups"]["artifact_hash"],
        "test_split_accessed": False, "exports_projection": False,
    }
    rule["artifact_hash"] = _hash_json(rule)
    output.mkdir(exist_ok=True)
    _write_once(output / "inclusion_rule.json", _json_bytes(rule))
    _write_once(output / "source_snapshot.zip", snapshot)
    _write_once(output / "validation_image_groups.json", _json_bytes(basis["groups"]))
    if progress:
        progress(f"Validated {len(basis['targets'])} records / {basis['groups']['cluster_count']} image groups; rule frozen.")
    rows = []
    numeric_diagnostics = (
        "train_support", "train_prevalence", "val_support", "train_initial_ap",
        "train_initial_to_late_gain", "train_early_to_late_gain", "train_late_median_ap",
        "train_late_iqr", "train_near_to_late_shift", "train_minus_val_late_gap",
        "cuisine_prior_ap", "image_vs_cuisine_ap_advantage",
    )
    for index, name in enumerate(basis["names"]):
        stat = paired_cluster_bootstrap(basis["targets"][:, index], basis["scores"][:, :, index],
                                        basis["cluster_ids"], seed=42000 + index)
        expected = [float(basis["metrics"][("val", e, index)]["average_precision"]) for e in LATE_EPOCHS]
        old = basis["evidence"][index]
        if (not np.allclose(stat["checkpoint_ap"], expected, atol=1e-12, rtol=0)
                or not np.isclose(stat["q"], float(old["val_late_median_ap"]), atol=1e-12, rtol=0)
                or not np.isclose(stat["iqr"], float(old["val_late_iqr"]), atol=1e-12, rtol=0)):
            raise ValueError(f"saved-score AP/trajectory parity failed for class {index}")
        diagnostics = {k: float(old[k]) for k in numeric_diagnostics}
        diagnostics["val_near_to_late_shift"] = stat["q"] - float(np.median([
            float(basis["metrics"][("val", e, index)]["average_precision"])
            for e in (30, 32, 34, 36, 38)]))
        diagnostics["flags"] = {
            "final_train_support_below_500": diagnostics["train_support"] < 500,
            "train_gain_below_0_10": diagnostics["train_initial_to_late_gain"] < .10,
            "train_ap_below_0_35": diagnostics["train_late_median_ap"] < .35,
            "train_validation_gap_above_0_50": diagnostics["train_minus_val_late_gap"] > .50,
            "cuisine_advantage_below_0_10": diagnostics["image_vs_cuisine_ap_advantage"] < .10,
        }
        rows.append({
            "class_index": index, "class_name": name, "statistics": stat,
            "decision": classify_inclusion(stat), "diagnostics": diagnostics,
            "d4_outcome": old["provisional_outcome"], "d4_reasons": old["profile_reasons"],
            "sensitivity": {f"{floor:.2f}": classify_inclusion(stat, quality_floor=floor)
                            for floor in rule["sensitivity_floors"]},
        })
        if progress and (index + 1) % 20 == 0:
            progress(f"Computed paired intervals: {index + 1}/165 labels.")
    original = {r["class_name"] for r in rows if r["d4_outcome"] == "generalizable_candidate"}
    included = {r["class_name"] for r in rows if r["decision"]["included"]}
    ordered = lambda members: [n for n in basis["names"] if n in members]
    report = {
        "schema_version": 1, "policy_id": POLICY_ID,
        "interpretation": rule["interpretation"], "inclusion_rule_hash": rule["artifact_hash"],
        "campaign_identity_hash": rule["campaign_identity_hash"],
        "class_order_hash": rule["class_order_hash"], "label_count": len(rows),
        "test_split_accessed": False, "exports_projection": False,
        "point_AP_parity_checked": True,
        "groups_summary": {k: v for k, v in basis["groups"].items() if k != "rows"},
        "outcome_counts": dict(sorted(Counter(r["decision"]["outcome"] for r in rows).items())),
        "eligible_names": ordered(included),
        "versus_d4": {"retained": ordered(included & original), "added": ordered(included - original),
                      "removed": ordered(original - included)},
        "sensitivity_counts": {f"{floor:.2f}": dict(sorted(Counter(
            r["sensitivity"][f"{floor:.2f}"]["outcome"] for r in rows).items()))
            for floor in rule["sensitivity_floors"]},
        "limitations": ["Single seed/configuration; conditional on the reference selector.",
                        "Nominal per-label intervals; not simultaneous or post-selection guarantees.",
                        "Current image/score bytes hashed now, not at campaign launch.",
                        "No direct-visibility, intrinsic-unlearnability or final-test claim."],
        "labels": rows,
    }
    report["artifact_hash"] = _hash_json(report)
    if _source_snapshot(root)[0] != inventory:
        raise RuntimeError("analysis source changed during execution")
    for relative, expected_hash in basis["inputs_sha256"].items():
        path = (root / "data/input/yummly" / relative.split("/")[1] /
                "ingredients_target_v5_metadata.json") if relative.startswith("metadata/") else campaign / relative
        if _hash_file(path) != expected_hash:
            raise RuntimeError(f"analysis input changed during execution: {relative}")
    image_root = (root / "data/input/yummly/imgs/standard").resolve()
    for _, image_name, expected_hash in basis["groups"]["rows"]:
        image_path = (image_root / image_name).resolve()
        if not image_path.is_relative_to(image_root) or _hash_file(image_path) != expected_hash:
            raise RuntimeError("validation image bytes changed during execution")
    table = io.StringIO(newline="")
    fields = ["class_index", "class_name", "outcome", "reasons", "q", "q_lower", "q_upper",
              "prevalence", "difference_lower", "difference_upper", "iqr", "d4_outcome"]
    writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({"class_index": row["class_index"], "class_name": row["class_name"],
                         "outcome": row["decision"]["outcome"],
                         "reasons": ";".join(row["decision"]["reasons"]),
                         "d4_outcome": row["d4_outcome"],
                         **{k: row["statistics"][k] for k in fields[4:-1]}})
    _write_once(output / "inclusion_evidence.csv", table.getvalue().encode("utf-8"))
    _write_once(output / "inclusion_decision_map.svg", render_inclusion_plot(report))
    _write_once(output / "inclusion_report.json", _json_bytes(report))
    return report

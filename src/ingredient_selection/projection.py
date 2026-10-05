"""Publish the approved D6 vocabulary without recomputing selection or opening data."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


PROJECTION_ID = "ingredients_selected_v5_d6_v1"
POLICY_ID = "phase3-d6-held-out-quality-v1"
APPROVED_RULE_HASH = "851c52cc485279895cf369368fe62076ad3be7318a80e9d3109a20b186dbbcdc"
APPROVED_REPORT_HASH = "4f065e1bcdb2158d1c52e529970b4d47f2dd5165873e2bf03fbe599e5ca4861f"
SOURCE_FILES = (
    "src/ingredient_selection/projection.py",
    "scripts/ingredient_selection/export_projection.py",
)
RESOURCE_PATH = f"src/ingredient_selection/resources/{PROJECTION_ID}.json"


def _hash(value: Any) -> str:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _verify(payload: dict) -> None:
    body = dict(payload)
    expected = body.pop("artifact_hash", None)
    if expected != _hash(body):
        raise ValueError("invalid canonical artifact hash")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_projection(manifest: dict, rule: dict, report: dict,
                     source_inventory: dict[str, str]) -> dict:
    """Package existing decisions, retaining the base column identities and reasons."""
    _verify(rule)
    _verify(report)
    identity = manifest["campaign_identity"]
    names = identity["class_order"]
    if (manifest.get("status") != "completed"
            or identity.get("protocol_id") != "phase3-d1-v3"
            or manifest["campaign_identity_hash"] != _hash(identity)
            or len(names) != 165 or len(set(names)) != 165
            or not all(isinstance(name, str) and name for name in names)
            or identity["class_order_hash"] != _hash(names)):
        raise ValueError("invalid completed campaign or base class order")
    for payload in (rule, report):
        if (payload.get("schema_version") != 1
                or payload.get("policy_id") != POLICY_ID
                or payload.get("campaign_identity_hash") != manifest["campaign_identity_hash"]
                or payload.get("class_order_hash") != identity["class_order_hash"]
                or payload.get("test_split_accessed") is not False
                or payload.get("exports_projection") is not False):
            raise ValueError("D6 identity, schema or isolation mismatch")
    if (report["inclusion_rule_hash"] != rule["artifact_hash"]
            or report.get("label_count") != 165
            or report.get("point_AP_parity_checked") is not True):
        raise ValueError("D6 report does not reference its verified rule")
    rows = report["labels"]
    if (len(rows) != 165
            or [r["class_index"] for r in rows] != list(range(165))
            or [r["class_name"] for r in rows] != names
            or any(type(r["class_index"]) is not int for r in rows)):
        raise ValueError("report rows differ from the base class order")
    groups = {key: [] for key in ("included", "uncertain", "below_quality_floor")}
    excluded = []
    for row in rows:
        decision = row["decision"]
        outcome = decision["outcome"]
        if (outcome not in groups
                or decision["included"] is not (outcome == "included")
                or not isinstance(decision["reasons"], list)
                or not all(isinstance(reason, str) for reason in decision["reasons"])
                or not isinstance(decision["axis_status"], dict)):
            raise ValueError("inconsistent report decision")
        groups[outcome].append(row["class_index"])
        if not decision["included"]:
            excluded.append({"base_class_index": row["class_index"],
                             "outcome": outcome, "reasons": decision["reasons"],
                             "axis_status": decision["axis_status"]})
    selected = groups["included"]
    selected_names = [names[index] for index in selected]
    counts = dict(Counter(r["decision"]["outcome"] for r in rows))
    if (not selected or len(selected) == len(names)
            or counts != report["outcome_counts"]
            or selected_names != report["eligible_names"]):
        raise ValueError("report membership/counts disagree or projection is not a proper subset")
    for split in ("train", "val"):
        if identity["metadata_sha256"][split] != rule["inputs_sha256"][f"metadata/{split}"]:
            raise ValueError("rule and campaign metadata hashes disagree")
    projection = {
        "schema_version": 1, "projection_id": PROJECTION_ID, "frozen_on": "2026-10-05",
        "policy_id": POLICY_ID, "interpretation": report["interpretation"],
        "base_vocabulary": {
            "metadata_filename": "ingredients_target_v5_metadata.json",
            "target_field": "ingredients_target", "class_order": names,
            "class_order_hash": identity["class_order_hash"],
            "metadata_sha256": {split: identity["metadata_sha256"][split]
                                for split in ("train", "val")},
        },
        "class_order": selected_names, "class_order_hash": _hash(selected_names),
        "base_class_indices": selected, "label_count": len(selected),
        "outcome_counts": counts, "groups_base_indices": groups,
        "excluded_decisions": excluded,
        "evidence": {
            "campaign_protocol_id": identity["protocol_id"],
            "campaign_identity_hash": manifest["campaign_identity_hash"],
            "inclusion_rule_hash": rule["artifact_hash"],
            "inclusion_report_hash": report["artifact_hash"],
            "d6_source_snapshot_sha256": rule["source_snapshot_sha256"],
            "campaign_relative_rule": "inclusion_d6_v1/inclusion_rule.json",
            "campaign_relative_report": "inclusion_d6_v1/inclusion_report.json",
            "row_lookup": "labels[base_class_index]",
        },
        "export_source_inventory": source_inventory,
        "usage_contract": {
            "replaces_default": False, "shared_across_model_categories": True,
            "record_policy": "preserve original split, record order and all records, including empty projections",
            "target_policy": "intersect existing ingredients_target with class_order; never refit vocabulary per split",
            "full_model_comparison": "slice full-model output columns by base_class_indices in this order",
            "uncertain_policy": "excluded from primary projection; not evidence of unlearnability",
            "below_floor_policy": "excluded from primary projection; below the operational floor, not unlearnable",
            "training_integration": "P7; this artifact does not change runtime configuration",
            "random_controls": "Phase 6; match size/support using the linked evidence, not arbitrary label dropout",
        },
        "test_split_accessed": False, "metadata_exported": False,
        "limitations": report["limitations"],
    }
    projection["artifact_hash"] = _hash(projection)
    return projection


def _write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace changed projection: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"refusing to replace changed projection: {path}")
    finally:
        Path(temporary).unlink(missing_ok=True)


def export_projection(repo_root: str | Path, campaign_dir: str | Path,
                      output: str | Path | None = None) -> dict:
    """Verify the approved artifact identities and publish an immutable definition."""
    root, campaign = Path(repo_root).resolve(), Path(campaign_dir).resolve()
    output = Path(output).resolve() if output is not None else root / RESOURCE_PATH
    paths = {"manifest": campaign / "campaign_manifest.json",
             "rule": campaign / "inclusion_d6_v1/inclusion_rule.json",
             "report": campaign / "inclusion_d6_v1/inclusion_report.json",
             "snapshot": campaign / "inclusion_d6_v1/source_snapshot.zip"}
    input_hashes = {key: _file_hash(path) for key, path in paths.items()}
    manifest, rule, report = (json.loads(paths[key].read_text(encoding="utf-8"))
                              for key in ("manifest", "rule", "report"))
    _verify(rule)
    _verify(report)
    if (rule["artifact_hash"] != APPROVED_RULE_HASH
            or report["artifact_hash"] != APPROVED_REPORT_HASH):
        raise ValueError("inputs are not the approved frozen D6 rule/report")
    if (input_hashes["manifest"] != rule["inputs_sha256"]["campaign_manifest.json"]
            or input_hashes["snapshot"] != rule["source_snapshot_sha256"]):
        raise ValueError("campaign bytes or D6 snapshot differ from the frozen rule")
    with zipfile.ZipFile(paths["snapshot"]) as archive:
        inventory = rule["source_inventory"]
        if (len(archive.namelist()) != len(inventory)
                or set(archive.namelist()) != set(inventory)
                or any(hashlib.sha256(archive.read(name)).hexdigest() != digest
                       for name, digest in inventory.items())):
            raise ValueError("D6 source snapshot members do not match the inventory")
    source_inventory = {relative: _file_hash(root / relative) for relative in SOURCE_FILES}
    projection = build_projection(manifest, rule, report, source_inventory)
    content = (json.dumps(projection, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    if (any(_file_hash(paths[key]) != digest for key, digest in input_hashes.items())
            or any(_file_hash(root / name) != digest for name, digest in source_inventory.items())):
        raise RuntimeError("projection input/source changed during export")
    _write_once(output, content)
    return projection

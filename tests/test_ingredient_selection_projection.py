"""P6 packaging is deterministic, test-isolated and does not make new decisions."""

import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.ingredient_selection import projection as p


def seal(value):
    value.pop("artifact_hash", None)
    value["artifact_hash"] = p._hash(value)
    return value


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def fixture():
    names = [f"label_{i:03d}" for i in range(165)]
    identity = {"protocol_id": "phase3-d1-v3", "class_order": names,
                "class_order_hash": p._hash(names),
                "metadata_sha256": {"train": "1" * 64, "val": "2" * 64}}
    manifest = {"status": "completed", "campaign_identity": identity,
                "campaign_identity_hash": p._hash(identity)}
    snapshot = io.BytesIO()
    with zipfile.ZipFile(snapshot, "w") as archive:
        archive.writestr("analysis.py", b"frozen analysis")
    common = {"schema_version": 1, "policy_id": p.POLICY_ID,
              "campaign_identity_hash": p._hash(identity),
              "class_order_hash": p._hash(names), "test_split_accessed": False,
              "exports_projection": False}
    rule = seal({**common,
                 "source_inventory": {"analysis.py": hashlib.sha256(b"frozen analysis").hexdigest()},
                 "source_snapshot_sha256": hashlib.sha256(snapshot.getvalue()).hexdigest(),
                 "inputs_sha256": {"metadata/train": "1" * 64, "metadata/val": "2" * 64,
                                   "campaign_manifest.json": hashlib.sha256(encoded(manifest)).hexdigest()}})
    rows = []
    for index, name in enumerate(names):
        outcome = ("included", "uncertain", "below_quality_floor")[index % 3]
        rows.append({"class_index": index, "class_name": name,
                     "decision": {"outcome": outcome, "included": outcome == "included",
                                  "reasons": [] if outcome == "included" else ["quality_lower_below_floor"],
                                  "axis_status": {"validity": "pass", "quality": outcome}},
                     "diagnostics": {"train_support": 1, "train_gain": 0}})
    report = seal({**common, "inclusion_rule_hash": rule["artifact_hash"],
                   "label_count": 165, "point_AP_parity_checked": True,
                   "labels": rows, "eligible_names": names[::3],
                   "outcome_counts": {"included": 55, "uncertain": 55, "below_quality_floor": 55},
                   "interpretation": "outcome-informed exploratory post-P4 amendment",
                   "limitations": ["Single seed/configuration; conditional on the reference selector."]})
    return manifest, rule, report, snapshot.getvalue()


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.rule, self.report, self.snapshot = fixture()

    def build(self):
        return p.build_projection(self.manifest, self.rule, self.report, {"export.py": "a" * 64})

    def test_mapping_preserves_base_order_and_independent_reasons(self):
        before = copy.deepcopy(self.report)
        result = self.build()
        p._verify(result)
        self.assertEqual(result["base_class_indices"], list(range(0, 165, 3)))
        self.assertEqual(result["class_order"], self.report["eligible_names"])
        self.assertEqual(result["class_order_hash"], p._hash(result["class_order"]))
        self.assertEqual(result["excluded_decisions"][0]["reasons"],
                         self.report["labels"][1]["decision"]["reasons"])
        self.assertEqual(result["excluded_decisions"][0]["axis_status"],
                         self.report["labels"][1]["decision"]["axis_status"])
        self.assertFalse(result["usage_contract"]["replaces_default"])
        self.assertFalse(result["metadata_exported"])
        self.assertEqual(self.report, before)
        self.assertEqual(result, self.build())

    def test_projection_preserves_empty_rows_and_full_prediction_column_mapping(self):
        result = self.build()
        names = result["base_vocabulary"]["class_order"]
        record_targets = [[names[1]], [names[0], names[3]], []]
        selected = set(result["class_order"])
        projected = [[name for name in row if name in selected] for row in record_targets]
        self.assertEqual(projected, [[], [names[0], names[3]], []])
        full_output = list(range(165))
        sliced = [full_output[index] for index in result["base_class_indices"]]
        self.assertEqual(sliced, list(range(0, 165, 3)))

    def test_rejects_changed_canonical_hash(self):
        self.report["eligible_names"].reverse()
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.build()

    def test_rejects_reordered_duplicate_or_missing_rows(self):
        original = copy.deepcopy(self.report)
        variants = [list(reversed(original["labels"])), original["labels"][:-1],
                    original["labels"][:-1] + [original["labels"][0]]]
        for rows in variants:
            with self.subTest(rows=len(rows)):
                self.report = seal({**original, "labels": rows})
                with self.assertRaisesRegex(ValueError, "class order"):
                    self.build()

    def test_rejects_inconsistent_eligibility_and_counts(self):
        for field, value in (("eligible_names", []), ("outcome_counts", {"included": 165})):
            with self.subTest(field=field):
                self.report = fixture()[2]
                self.report[field] = value
                seal(self.report)
                with self.assertRaisesRegex(ValueError, "membership/counts"):
                    self.build()
        self.report = fixture()[2]
        self.report["labels"][0]["decision"]["included"] = False
        seal(self.report)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.build()

    def test_rejects_identity_rule_metadata_or_isolation_mismatch(self):
        for key, value in (("test_split_accessed", True), ("exports_projection", True),
                           ("inclusion_rule_hash", "0" * 64), ("policy_id", "another"),
                           ("campaign_identity_hash", "0" * 64),
                           ("point_AP_parity_checked", False)):
            with self.subTest(key=key):
                self.report = fixture()[2]
                self.report[key] = value
                seal(self.report)
                with self.assertRaises(ValueError):
                    self.build()
        self.report = fixture()[2]
        self.rule["inputs_sha256"]["metadata/train"] = "0" * 64
        seal(self.rule)
        self.report["inclusion_rule_hash"] = self.rule["artifact_hash"]
        seal(self.report)
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.build()

    def prepare(self, directory):
        root = Path(directory)
        campaign = root / "analysis_outputs/campaign"
        d6 = campaign / "inclusion_d6_v1"
        d6.mkdir(parents=True)
        (campaign / "campaign_manifest.json").write_bytes(encoded(self.manifest))
        (d6 / "inclusion_rule.json").write_bytes(encoded(self.rule))
        (d6 / "inclusion_report.json").write_bytes(encoded(self.report))
        (d6 / "source_snapshot.zip").write_bytes(self.snapshot)
        for source in p.SOURCE_FILES:
            path = root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic exporter")
        return root, campaign

    def test_export_is_byte_identical_write_once_and_only_reads_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root, campaign = self.prepare(directory)
            output = root / p.RESOURCE_PATH
            allowed = {campaign / "campaign_manifest.json",
                       *(campaign / "inclusion_d6_v1" / name for name in
                         ("inclusion_rule.json", "inclusion_report.json", "source_snapshot.zip")),
                       *(root / name for name in p.SOURCE_FILES), output}
            original_open = Path.open

            def guarded(path, *args, **kwargs):
                if path not in allowed:
                    raise AssertionError(f"unexpected file access: {path}")
                return original_open(path, *args, **kwargs)

            with patch.object(p, "APPROVED_RULE_HASH", self.rule["artifact_hash"]), \
                 patch.object(p, "APPROVED_REPORT_HASH", self.report["artifact_hash"]), \
                 patch.object(Path, "open", guarded):
                result = p.export_projection(root, campaign)
                content = output.read_bytes()
                self.assertEqual(result, p.export_projection(root, campaign))
                self.assertEqual(output.read_bytes(), content)
            output.write_bytes(b"retain conflicting existing file")
            with patch.object(p, "APPROVED_RULE_HASH", self.rule["artifact_hash"]), \
                 patch.object(p, "APPROVED_REPORT_HASH", self.report["artifact_hash"]):
                with self.assertRaises(FileExistsError):
                    p.export_projection(root, campaign)
            self.assertEqual(output.read_bytes(), b"retain conflicting existing file")

    def test_export_rejects_unapproved_even_when_self_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root, campaign = self.prepare(directory)
            with self.assertRaisesRegex(ValueError, "approved"):
                p.export_projection(root, campaign)
            self.assertFalse((root / p.RESOURCE_PATH).exists())

    def test_export_rejects_changed_manifest_and_snapshot(self):
        for name in ("campaign_manifest.json", "inclusion_d6_v1/source_snapshot.zip"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root, campaign = self.prepare(directory)
                path = campaign / name
                path.write_bytes(path.read_bytes() + b" ")
                with patch.object(p, "APPROVED_RULE_HASH", self.rule["artifact_hash"]), \
                     patch.object(p, "APPROVED_REPORT_HASH", self.report["artifact_hash"]):
                    with self.assertRaisesRegex(ValueError, "frozen rule"):
                        p.export_projection(root, campaign)
                self.assertFalse((root / p.RESOURCE_PATH).exists())

    def test_rejects_source_or_input_changes_before_publication(self):
        for source in (True, False):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root, campaign = self.prepare(directory)
                original_build = p.build_projection

                def mutate(*args):
                    result = original_build(*args)
                    path = root / p.SOURCE_FILES[0] if source else campaign / "campaign_manifest.json"
                    path.write_bytes(path.read_bytes() + b" ")
                    return result

                with patch.object(p, "APPROVED_RULE_HASH", self.rule["artifact_hash"]), \
                     patch.object(p, "APPROVED_REPORT_HASH", self.report["artifact_hash"]), \
                     patch.object(p, "build_projection", mutate):
                    with self.assertRaisesRegex(RuntimeError, "changed during"):
                        p.export_projection(root, campaign)
                self.assertFalse((root / p.RESOURCE_PATH).exists())

    def test_import_does_not_load_training_or_numeric_stack(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import sys; import src.ingredient_selection.projection; "
                                 "assert not ({'numpy', 'torch', 'lightning'} & set(sys.modules))"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_committed_resource_matches_approved_evidence_and_current_exporter(self):
        root = Path(__file__).resolve().parents[1]
        result = json.loads((root / p.RESOURCE_PATH).read_text(encoding="utf-8"))
        p._verify(result)
        self.assertEqual(result["evidence"]["inclusion_rule_hash"], p.APPROVED_RULE_HASH)
        self.assertEqual(result["evidence"]["inclusion_report_hash"], p.APPROVED_REPORT_HASH)
        self.assertEqual(result["outcome_counts"],
                         {"included": 59, "uncertain": 60, "below_quality_floor": 46})
        self.assertEqual(result["label_count"], 59)
        self.assertEqual(result["class_order"], [result["base_vocabulary"]["class_order"][i]
                                               for i in result["base_class_indices"]])
        groups = result["groups_base_indices"]
        self.assertEqual(sorted(sum(groups.values(), [])), list(range(165)))
        self.assertEqual(groups["included"], result["base_class_indices"])
        self.assertEqual(len(result["excluded_decisions"]), 106)
        for name, digest in result["export_source_inventory"].items():
            self.assertEqual(p._file_hash(root / name), digest)

    def test_rejects_snapshot_member_mismatch_even_when_zip_hash_matches(self):
        self.rule["source_inventory"]["analysis.py"] = "0" * 64
        seal(self.rule)
        self.report["inclusion_rule_hash"] = self.rule["artifact_hash"]
        seal(self.report)
        with tempfile.TemporaryDirectory() as directory:
            root, campaign = self.prepare(directory)
            with patch.object(p, "APPROVED_RULE_HASH", self.rule["artifact_hash"]), \
                 patch.object(p, "APPROVED_REPORT_HASH", self.report["artifact_hash"]):
                with self.assertRaisesRegex(ValueError, "snapshot members"):
                    p.export_projection(root, campaign)
            self.assertFalse((root / p.RESOURCE_PATH).exists())


if __name__ == "__main__":
    unittest.main()

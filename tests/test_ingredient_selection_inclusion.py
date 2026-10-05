"""Synthetic integration checks for the versioned, validation-only D6 analysis."""

import csv
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.ingredient_selection import inclusion


def _hash_json(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


class InclusionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.campaign = self.root / "analysis_outputs" / "phase3-d1-v3"
        self.campaign.mkdir(parents=True)
        self.data = self.root / "data/input/yummly"
        self.images = self.data / "imgs/standard"
        self.images.mkdir(parents=True)
        self.names = [f"label_{index:03d}" for index in range(165)]
        self.ids = [f"val-{index}" for index in range(4)]
        self.targets = np.tile(np.asarray([0, 1, 0, 1], dtype=np.uint8)[:, None], (1, 165))
        self.logits = np.tile(np.asarray([-2, 2, -1, 1], dtype=np.float32)[:, None], (1, 165))
        for relative in inclusion.SOURCE_FILES + ("src/ingredient_selection/metrics.py",):
            source = self.root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# Synthetic source: {relative}\n", encoding="utf-8")
        for index in range(4):
            # Records zero and one share exact bytes but retain distinct targets.
            content = b"duplicate image bytes" if index < 2 else f"image-{index}".encode()
            (self.images / f"val-{index}.jpg").write_bytes(content)
        self.metadata = {}
        for split in ("train", "val"):
            records = [{"id": f"{split}-{index}",
                        "image": f"val-{index}.jpg" if split == "val" else "../../test/forbidden.jpg",
                        "ingredients_target": self.names if index % 2 else []}
                       for index in range(4)]
            self.metadata[split] = self.data / split / "ingredients_target_v5_metadata.json"
            _write_json(self.metadata[split], records)
        # A loader accidentally inspecting test metadata cannot parse this sentinel.
        (self.data / "test").mkdir()
        (self.data / "test/ingredients_target_v5_metadata.json").write_text("DO NOT READ", encoding="utf-8")
        self.identity = {
            "protocol_id": "phase3-d1-v3", "seed": 42,
            "execution": {"max_epochs": 40}, "class_order": self.names,
            "class_order_hash": _hash_json(self.names),
            "source_identity_hash": _hash_json({
                "training.py": hashlib.sha256(b"# synthetic frozen training source\n").hexdigest(),
            }),
        }
        self.evidence = []
        for index, name in enumerate(self.names):
            self.evidence.append({
                "class_index": index, "class_name": name,
                "trajectory_complete": True,
                "train_support": 2, "val_support": 2, "train_prevalence": .5,
                "train_initial_ap": .5, "train_initial_to_late_gain": .4,
                "train_early_to_late_gain": .3, "train_late_median_ap": .9,
                "train_late_iqr": 0, "train_near_to_late_shift": 0,
                "train_minus_val_late_gap": -.1, "cuisine_prior_ap": .5,
                "image_vs_cuisine_ap_advantage": .5,
                "val_late_median_ap": 1., "val_late_iqr": 0.,
                "provisional_outcome": "generalizable_candidate" if index == 0 else "uncertain",
                "profile_reasons": "fixture",
            })
        _write_csv(self.campaign / "profile_evidence.csv", self.evidence)
        (self.campaign / "pilot_profile_evidence.csv").write_text("synthetic,pilot\n", encoding="utf-8")
        self.metrics = [{
            "run_id": "phase3-d1-v3", "split": split, "audit_epoch": epoch,
            "class_index": index, "class_name": name, "records": 4,
            "support": 2, "average_precision": 1. if split == "val" else .9,
        } for split in ("train", "val") for epoch in range(0, 41, 2)
                       for index, name in enumerate(self.names)]
        _write_csv(self.campaign / "metrics_per_label_epoch.csv", self.metrics)
        (self.campaign / "audit_scores").mkdir()
        for epoch in inclusion.LATE_EPOCHS:
            self._write_scores(epoch)
        with zipfile.ZipFile(self.campaign / "source_snapshot.zip", "w") as archive:
            archive.writestr("training.py", b"# synthetic frozen training source\n")
        self._seal_inputs()

    def _write_scores(self, epoch, **changes):
        values = {"record_ids": np.asarray(self.ids), "targets": self.targets,
                  "logits": self.logits}
        values.update(changes)
        np.savez(self.campaign / f"audit_scores/validation_epoch_{epoch:02d}.npz", **values)

    def _seal_inputs(self):
        self.identity["metadata_sha256"] = {split: _hash_file(path)
                                            for split, path in self.metadata.items()}
        identity_hash = _hash_json(self.identity)
        _write_json(self.campaign / "campaign_manifest.json", {
            "status": "completed", "campaign_identity": self.identity,
            "campaign_identity_hash": identity_hash,
            "source_identity": {
                "files": {"training.py": hashlib.sha256(b"# synthetic frozen training source\n").hexdigest()},
                "sha256": _hash_json({"training.py": hashlib.sha256(b"# synthetic frozen training source\n").hexdigest()}),
            },
            "source_snapshot": {"sha256": _hash_file(self.campaign / "source_snapshot.zip")},
        })
        rule = {"campaign_identity_hash": identity_hash,
                "classifier_source_sha256": _hash_file(self.root / "src/ingredient_selection/metrics.py"),
                "pilot_evidence_sha256": _hash_file(self.campaign / "pilot_profile_evidence.csv")}
        rule["artifact_hash"] = _hash_json(rule)
        _write_json(self.campaign / "profile_rule.json", rule)
        report = {"campaign_identity_hash": identity_hash,
                  "profile_evidence_sha256": _hash_file(self.campaign / "profile_evidence.csv"),
                  "profile_rule_hash": rule["artifact_hash"]}
        _write_json(self.campaign / "p4_profile_report.json", report)
        _write_json(self.campaign / "validation_summary.json", {
            **report, "analysis_scope": "full", "test_split_accessed": False,
        })

    @staticmethod
    def _point_bootstrap(targets, scores, groups, *, seed):
        """Use exact fixture point AP without 165,000 integration bootstrap draws."""
        checkpoint_ap = []
        for checkpoint in scores:
            ordered_targets = targets[np.argsort(-checkpoint, kind="stable")]
            precision = np.cumsum(ordered_targets) / np.arange(1, len(targets) + 1)
            checkpoint_ap.append(float(np.sum(precision * ordered_targets) / targets.sum()))
        q = float(np.median(checkpoint_ap))
        prevalence = float(targets.mean())
        return {
            "status": "valid", "bootstrap_valid": True,
            "record_count": len(targets), "cluster_count": len(set(groups)),
            "positive_count": int(targets.sum()), "seed": seed,
            "requested_valid_draws": 1000, "max_draws": 10000,
            "valid_draws": 1000, "attempted_draws": 1000, "invalid_draws": 0,
            "checkpoint_ap": checkpoint_ap, "q": q,
            "iqr": float(np.quantile(checkpoint_ap, .75) - np.quantile(checkpoint_ap, .25)),
            "prevalence": prevalence, "q_minus_prevalence": q - prevalence,
            "q_lower": max(0., q - .1), "q_upper": min(1., q + .1),
            "difference_lower": q - prevalence - .1, "difference_upper": q - prevalence + .1,
        }

    def _review(self, *, progress=None, revision="synthetic-revision"):
        revision = subprocess.CompletedProcess(["git"], 0, stdout=revision + "\n", stderr="")
        with patch.object(inclusion.subprocess, "run", return_value=revision), \
                patch.object(inclusion, "paired_cluster_bootstrap", side_effect=self._point_bootstrap) as bootstrap:
            report = inclusion.review_inclusion(self.root, self.campaign, progress=progress)
        return report, bootstrap

    def test_load_recovers_aligned_exact_image_groups_without_test_access(self):
        original_open = Path.open
        observed = []

        def isolated_open(path, *args, **kwargs):
            if self.data / "test" in path.parents:
                raise AssertionError("D6 attempted to open the test split")
            observed.append(path)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", isolated_open):
            basis = inclusion.load_inputs(self.root, self.campaign)
        self.assertEqual(basis["scores"].shape, (5, 4, 165))
        np.testing.assert_array_equal(basis["targets"], self.targets)
        self.assertEqual(basis["names"], self.names)
        self.assertEqual(basis["groups"]["cluster_count"], 3)
        self.assertEqual(basis["groups"]["size_distribution"], {1: 2, 2: 1})
        self.assertEqual(basis["cluster_ids"][0], basis["cluster_ids"][1])
        self.assertEqual(len(basis["metrics"]), 6930)
        self.assertTrue(all(self.data / "test" not in path.parents for path in observed))
        self.assertFalse((self.campaign / "inclusion_d6_v1").exists())

    def test_metadata_byte_drift_is_rejected(self):
        with self.metadata["val"].open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "val metadata differs"):
            inclusion.load_inputs(self.root, self.campaign)

    def test_original_test_isolation_claim_must_be_false(self):
        path = self.campaign / "validation_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["test_split_accessed"] = True
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "test-isolation record changed"):
            inclusion.load_inputs(self.root, self.campaign)

    def test_npz_record_order_targets_extra_arrays_and_nonfinite_logits_are_rejected(self):
        cases = [
            {"record_ids": np.asarray(self.ids[::-1])},
            {"targets": 1 - self.targets},
            {"extra": np.zeros(1)},
            {"logits": np.full_like(self.logits, np.nan)},
        ]
        for changes in cases:
            with self.subTest(changes=list(changes)):
                self._write_scores(32, **changes)
                with self.assertRaisesRegex(ValueError, "alignment-mismatched validation scores"):
                    inclusion.load_inputs(self.root, self.campaign)
        self._write_scores(32)

    def test_duplicate_or_test_split_metric_rows_are_rejected(self):
        cases = [self.metrics + [self.metrics[0]],
                 [{**self.metrics[0], "split": "test"}] + self.metrics[1:]]
        for rows in cases:
            with self.subTest(rows=len(rows)):
                _write_csv(self.campaign / "metrics_per_label_epoch.csv", rows)
                with self.assertRaisesRegex(ValueError, "missing/duplicate/unexpected"):
                    inclusion.load_inputs(self.root, self.campaign)

    def test_image_path_escape_is_rejected_even_when_metadata_hash_is_valid(self):
        records = json.loads(self.metadata["val"].read_text(encoding="utf-8"))
        records[0]["image"] = "../../test/ingredients_target_v5_metadata.json"
        _write_json(self.metadata["val"], records)
        self._seal_inputs()
        with self.assertRaisesRegex(ValueError, "escaped the image root"):
            inclusion.load_inputs(self.root, self.campaign)

    def test_corrupt_d4_rule_or_evidence_is_rejected_before_output(self):
        path = self.campaign / "profile_rule.json"
        rule = json.loads(path.read_text(encoding="utf-8"))
        rule["classifier_source_sha256"] = "tampered"
        _write_json(path, rule)
        with self.assertRaisesRegex(ValueError, "invalid canonical artifact hash"):
            self._review()
        self._seal_inputs()
        self.evidence[0]["val_late_median_ap"] = .1
        _write_csv(self.campaign / "profile_evidence.csv", self.evidence)
        with self.assertRaisesRegex(ValueError, "D4 full evidence"):
            self._review()
        self.assertFalse((self.campaign / "inclusion_d6_v1").exists())

    def test_review_is_deterministic_write_once_and_preserves_d4(self):
        retained = {path.name: path.read_bytes() for path in self.campaign.iterdir() if path.is_file()}
        first, bootstrap = self._review()
        self.assertEqual(bootstrap.call_count, 165)
        self.assertEqual([call.kwargs["seed"] for call in bootstrap.call_args_list], list(range(42000, 42165)))
        self.assertEqual(first["outcome_counts"], {"included": 165})
        self.assertEqual(first["versus_d4"]["retained"], [self.names[0]])
        self.assertEqual(first["versus_d4"]["added"], self.names[1:])
        self.assertEqual(first["versus_d4"]["removed"], [])
        self.assertFalse(first["test_split_accessed"])
        self.assertFalse(first["exports_projection"])
        output = self.campaign / "inclusion_d6_v1"
        frozen = {path.name: path.read_bytes() for path in output.iterdir()}
        self.assertEqual(set(frozen), {"inclusion_rule.json", "source_snapshot.zip",
                                      "validation_image_groups.json", "inclusion_evidence.csv",
                                      "inclusion_report.json", "inclusion_decision_map.svg"})
        self.assertEqual(json.loads(frozen["inclusion_rule.json"])["original_d4_rule_hash"],
                         json.loads(retained["profile_rule.json"])["artifact_hash"])
        second, _ = self._review(revision="later-unrelated-commit")
        self.assertEqual(first, second)
        self.assertEqual(frozen, {path.name: path.read_bytes() for path in output.iterdir()})
        self.assertEqual(retained, {path.name: path.read_bytes() for path in self.campaign.iterdir() if path.is_file()})

    def test_corrupt_existing_d6_rule_is_never_overwritten(self):
        self._review()
        path = self.campaign / "inclusion_d6_v1/inclusion_rule.json"
        corrupted = json.loads(path.read_text(encoding="utf-8"))
        corrupted["quality_floor"] = .99
        _write_json(path, corrupted)
        with self.assertRaisesRegex(ValueError, "invalid canonical artifact hash"):
            self._review()
        corrupted.pop("artifact_hash")
        corrupted["artifact_hash"] = _hash_json(corrupted)
        _write_json(path, corrupted)
        with self.assertRaisesRegex(FileExistsError, "refusing to replace changed D6 artifact"):
            self._review()
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), corrupted)

    def test_train_support_gain_and_cuisine_are_reported_without_inclusion_veto(self):
        for row in self.evidence:
            row["train_initial_to_late_gain"] = 0.
            row["train_initial_ap"] = .1
            row["train_late_median_ap"] = .1
            row["train_minus_val_late_gap"] = -.9
            row["cuisine_prior_ap"] = 1.
            row["image_vs_cuisine_ap_advantage"] = 0.
        for row in self.metrics:
            if row["split"] == "train":
                row["average_precision"] = .1
        _write_csv(self.campaign / "profile_evidence.csv", self.evidence)
        _write_csv(self.campaign / "metrics_per_label_epoch.csv", self.metrics)
        self._seal_inputs()
        report, _ = self._review()
        self.assertEqual(report["outcome_counts"], {"included": 165})
        flags = report["labels"][0]["diagnostics"]["flags"]
        self.assertTrue(flags["final_train_support_below_500"])
        self.assertTrue(flags["train_gain_below_0_10"])
        self.assertTrue(flags["train_ap_below_0_35"])
        self.assertTrue(flags["cuisine_advantage_below_0_10"])

    def test_score_point_parity_failure_never_writes_report(self):
        self._write_scores(32, logits=-self.logits)
        with self.assertRaisesRegex(ValueError, "saved-score AP/trajectory parity failed"):
            self._review()
        self.assertFalse((self.campaign / "inclusion_d6_v1/inclusion_report.json").exists())

    def test_input_or_source_change_during_analysis_prevents_report_freeze(self):
        for relative in ("metrics_per_label_epoch.csv", "source", "image"):
            with self.subTest(changed=relative):
                target = (self.root / inclusion.SOURCE_FILES[0] if relative == "source"
                          else self.images / "val-0.jpg" if relative == "image"
                          else self.campaign / relative)
                original = target.read_bytes()

                def drift(message):
                    if "20/165" in message:
                        target.write_bytes(original + b"\n")

                with self.assertRaisesRegex(RuntimeError, "changed during execution"):
                    self._review(progress=drift)
                self.assertFalse((self.campaign / "inclusion_d6_v1/inclusion_report.json").exists())
                target.write_bytes(original)

    def test_training_snapshot_digest_and_inventory_are_verified(self):
        path = self.campaign / "source_snapshot.zip"
        original = path.read_bytes()
        with path.open("ab") as stream:
            stream.write(b"unexpected bytes")
        with self.assertRaisesRegex(ValueError, "snapshot/inventory hash"):
            inclusion.load_inputs(self.root, self.campaign)
        path.write_bytes(original)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("training.py", b"changed source")
        self._seal_inputs()
        with self.assertRaisesRegex(ValueError, "snapshot members differ"):
            inclusion.load_inputs(self.root, self.campaign)

    def test_source_inventory_must_match_frozen_campaign_identity(self):
        self.identity["source_identity_hash"] = "0" * 64
        self._seal_inputs()
        with self.assertRaisesRegex(ValueError, "snapshot/inventory hash"):
            inclusion.load_inputs(self.root, self.campaign)


if __name__ == "__main__":
    unittest.main()

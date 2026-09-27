import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.ingredient_selection.run_campaign import _campaign_identity
from src.ingredient_selection.analysis import analyze_campaign
from src.ingredient_selection.artifacts import write_json
from src.ingredient_selection.data import SelectorDataBundle, SelectorDataModule
from src.ingredient_selection.metrics import ProfileThresholds
from src.ingredient_selection.protocol import (
    SelectorProtocol,
    build_pilot_cohort,
    compute_pos_weight,
    ordered_values_hash,
    sha256_json,
)
from src.ingredient_selection.batching import resolve_batch_plan
from src.models.efficientnet import EfficientNetV2SSelector


class IngredientSelectionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        image_root = self.data_root / "imgs" / "standard"
        image_root.mkdir(parents=True)
        for name in ("train_positive.png", "train_negative.png", "val_positive.png", "val_negative.png"):
            Image.new("RGB", (8, 6), color=(100, 120, 140)).save(image_root / name)
        self.class_names = [f"ingredient_{index:03d}" for index in range(165)]
        train = [
            {"id": "train-positive", "image": "train_positive.png", "cuisine": "A",
             "ingredients_target": self.class_names},
            {"id": "train-negative", "image": "train_negative.png", "cuisine": "B",
             "ingredients_target": []},
        ]
        val = [
            {"id": "val-positive", "image": "val_positive.png", "cuisine": "A",
             "ingredients_target": self.class_names},
            {"id": "val-negative", "image": "val_negative.png", "cuisine": "B",
             "ingredients_target": []},
        ]
        for split, records in (("train", train), ("val", val)):
            directory = self.data_root / split
            directory.mkdir(parents=True)
            (directory / "ingredients_target_v5_metadata.json").write_text(
                json.dumps(records), encoding="utf-8"
            )
        # A malformed test file proves the maintained loader never opens it.
        test_dir = self.data_root / "test"
        test_dir.mkdir()
        (test_dir / "ingredients_target_v5_metadata.json").write_text("not-json", encoding="utf-8")
        self.bundle = SelectorDataBundle.load(self.data_root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_data_boundary_uses_only_train_and_validation(self):
        module = SelectorDataModule(self.bundle, batch_size=8, num_workers=0, seed=42)
        images, targets, record_ids = next(iter(module.val_dataloader()))

        self.assertEqual(self.bundle.class_names, tuple(self.class_names))
        self.assertEqual(tuple(images.shape), (2, 3, 384, 384))
        self.assertEqual(tuple(targets.shape), (2, 165))
        self.assertEqual(list(record_ids), ["val-positive", "val-negative"])

    def test_manifest_uses_the_same_budget_and_scheduler_as_training(self):
        protocol = SelectorProtocol()
        plan = resolve_batch_plan(protocol.batch_size, EfficientNetV2SSelector.MAX_ALLOWED_BATCH_SIZE)
        identity = _campaign_identity(
            self.bundle, protocol, "revision", self.bundle.train.supports.astype(int).tolist(),
            compute_pos_weight(self.bundle.train.targets), "head_hash", plan,
            {"sha256": "source_hash"}, 0,
        )
        self.assertEqual(identity["execution"]["max_epochs"], 40)
        self.assertEqual(identity["execution"]["planned_steps"], 40)
        self.assertEqual(identity["scheduler"]["warmup"]["total_iters"], 2)
        self.assertEqual(identity["scheduler"]["main"]["T_max"], 38)
        self.assertEqual(identity["execution"]["audit_epochs"], list(range(0, 41, 2)))

    def _write_campaign(self) -> Path:
        protocol = SelectorProtocol()
        plan = resolve_batch_plan(protocol.batch_size, EfficientNetV2SSelector.MAX_ALLOWED_BATCH_SIZE)
        output = self.root / "output"
        (output / "audit_scores").mkdir(parents=True)
        identity = {
            "protocol_id": protocol.protocol_id,
            "seed": protocol.seed,
            "class_order_hash": ordered_values_hash(self.bundle.class_names),
            "metadata_sha256": {
                "train": self.bundle.train.metadata_sha256,
                "val": self.bundle.val.metadata_sha256,
            },
            "model": {"type": "EfficientNetV2SSelector", "weights": "EfficientNet_V2_S_Weights.IMAGENET1K_V1", "input_size": [384, 384], "full_backbone_trainable": True},
            "loss": {"type": "BCEWithLogitsLoss", "reduction": "mean", "pos_weight_formula": "(N_train-P_c)/P_c"},
            "optimizer": {"type": "AdamW", "lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4, "amsgrad": False, "foreach": False, "fused": False, "parameter_groups": 1, "gradient_clipping": None},
            "scheduler": {"type": "SequentialLR", "milestones": [protocol.warmup_epochs], "interval": "epoch",
                          "warmup": {"type": "LinearLR", "start_factor": 0.1, "end_factor": 1.0, "total_iters": protocol.warmup_epochs},
                          "main": {"type": "CosineAnnealingLR", "T_max": protocol.cosine_epochs, "eta_min": protocol.minimum_learning_rate}},
            "execution": {"max_epochs": protocol.max_epochs, "batch_size": plan.physical_batch_size, "requested_batch_size": 128, "max_allowed_batch_size": EfficientNetV2SSelector.MAX_ALLOWED_BATCH_SIZE, "drop_last": False, "gradient_accumulation": plan.accumulate_grad_batches, "precision": "32-true", "early_stopping": False, "swa": False, "audit_epochs": list(protocol.audit_epochs)},
        }
        write_json(output / "campaign_manifest.json", {
            "campaign_identity": identity,
            "campaign_identity_hash": sha256_json(identity),
        })
        pilot = build_pilot_cohort(self.bundle.class_names, self.bundle.train.supports)
        write_json(output / "pilot_cohort.json", pilot)
        rows = []
        for class_index, class_name in enumerate(self.bundle.class_names):
            for split in ("train", "val"):
                for position, epoch in enumerate(protocol.audit_epochs):
                    rows.append({
                        "run_id": protocol.protocol_id,
                        "audit_epoch": epoch,
                        "split": split,
                        "class_index": class_index,
                        "class_name": class_name,
                        "support": 1,
                        "records": 2,
                        "prevalence": 0.5,
                        "average_precision": 0.1 + 0.7 * position / (len(protocol.audit_epochs) - 1) - (0.03 if split == "val" else 0),
                        "ap_valid": True,
                        "precision_at_0_5": 1.0,
                        "recall_at_0_5": 1.0,
                        "f1_at_0_5": 1.0,
                        "micro_f1_at_0_5": 1.0,
                        "learning_rate": 1e-4,
                    })
        pd.DataFrame(rows).to_csv(output / "metrics_per_label_epoch.csv", index=False)
        targets = self.bundle.val.targets
        logits = np.vstack([np.ones(165), -np.ones(165)]).astype(np.float32)
        np.savez_compressed(
            output / "audit_scores" / f"validation_epoch_{protocol.max_epochs:02d}.npz",
            record_ids=np.asarray(self.bundle.val.record_ids),
            targets=targets,
            logits=logits,
        )
        pd.DataFrame([
            {
                "class_index": index,
                "class_name": name,
                "valid": True,
                "valid_draws": 1000,
                "attempted_draws": 1000,
                "lower": 0.8,
                "upper": 1.0,
            }
            for index, name in enumerate(self.bundle.class_names)
        ]).to_csv(output / "final_validation_bootstrap.csv", index=False)
        return output

    def test_analysis_exposes_only_pilot_until_hashed_rule_exists(self):
        output = self._write_campaign()
        pilot_summary = analyze_campaign(output, self.bundle)
        pilot_evidence = pd.read_csv(output / "profile_evidence.csv")

        self.assertEqual(pilot_summary["analysis_scope"], "pilot_only")
        self.assertEqual(len(pilot_evidence), 24)
        self.assertFalse(pilot_summary["test_split_accessed"])

        manifest = json.loads((output / "campaign_manifest.json").read_text())
        pilot = json.loads((output / "pilot_cohort.json").read_text())
        gates = ProfileThresholds(1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.5, -1.0)
        rule = {
            "schema_version": 1,
            "protocol_id": SelectorProtocol().protocol_id,
            "campaign_identity_hash": manifest["campaign_identity_hash"],
            "pilot_artifact_hash": pilot["artifact_hash"],
            "gates": gates.to_dict(),
        }
        rule["artifact_hash"] = sha256_json(rule)
        write_json(output / "profile_rule.json", rule)

        full_summary = analyze_campaign(output, self.bundle)
        full_evidence = pd.read_csv(output / "profile_evidence.csv")
        self.assertEqual(full_summary["analysis_scope"], "full")
        self.assertEqual(len(full_evidence), 165)
        self.assertEqual(set(full_evidence["provisional_outcome"]), {"generalizable_candidate"})

    def test_analysis_rejects_a_tampered_rule(self):
        output = self._write_campaign()
        write_json(output / "profile_rule.json", {
            "protocol_id": SelectorProtocol().protocol_id,
            "artifact_hash": "invalid",
            "gates": ProfileThresholds(1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.5, 0.0).to_dict(),
        })
        with self.assertRaises(ValueError):
            analyze_campaign(output, self.bundle)

    def test_analysis_rejects_a_self_consistent_but_contradictory_campaign(self):
        output = self._write_campaign()
        manifest_path = output / "campaign_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["campaign_identity"]["seed"] = 7
        manifest["campaign_identity_hash"] = sha256_json(manifest["campaign_identity"])
        write_json(manifest_path, manifest)

        with self.assertRaisesRegex(ValueError, "seed"):
            analyze_campaign(output, self.bundle)

    def test_analysis_rejects_previous_twenty_epoch_budget(self):
        output = self._write_campaign()
        path = output / "campaign_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["campaign_identity"]["execution"]["max_epochs"] = 20
        manifest["campaign_identity_hash"] = sha256_json(manifest["campaign_identity"])
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "execution"):
            analyze_campaign(output, self.bundle)

    def test_analysis_rejects_previous_eighteen_epoch_cosine_schedule(self):
        output = self._write_campaign()
        path = output / "campaign_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["campaign_identity"]["scheduler"]["main"]["T_max"] = 18
        manifest["campaign_identity_hash"] = sha256_json(manifest["campaign_identity"])
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "scheduler"):
            analyze_campaign(output, self.bundle)


if __name__ == "__main__":
    unittest.main()

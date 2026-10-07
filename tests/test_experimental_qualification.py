"""No-CUDA orchestration, containment, artifact and initial-state guards."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.training import experimental_qualification as qualification
from src.ingredient_selection.runtime import PROJECTION_ID, resolve_projection


class ExperimentalQualificationTests(unittest.TestCase):
    def test_config_template_preserves_rng_and_declared_policy(self):
        def factory(contract, initialize_pretrained):
            self.assertFalse(initialize_pretrained)
            torch.rand(100)
            return SimpleNamespace(to_config=lambda: {"type": qualification.EfficientNetV2SExperiment,
                "num_classes": contract.num_classes, "experimental_contract": contract.to_config()})
        torch.manual_seed(55)
        before = torch.get_rng_state().clone()
        with patch.dict(qualification.MODEL_TYPES, {"efficientnet_v2_s": factory}):
            config = qualification.make_smoke_config("efficientnet_v2_s", "full", 32, False, Path("/isolated/trial_0"), 4)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(config.hp["batch_size"], 128)
        self.assertEqual(config.hp["physical_batch_size"], 32)
        self.assertEqual(config.trainer["limit_train_batches"], 16)
        self.assertIs(config.hp["optimizer"], torch.optim.AdamW)
        self.assertFalse(config.hp["weighted_loss"])

    def test_reference_preprocessing_uses_canonical_lightning_not_cached_adapter(self):
        class Wrapper(torch.nn.Module):
            def bind_ingredient_projection(self, value):
                pass
        wrapper = Wrapper()
        image = torch.zeros(3, 224, 224)
        def load_dm(config, model):
            self.assertIs(model, wrapper)
            self.assertTrue(hasattr(model, "bind_ingredient_projection"))
            return SimpleNamespace(val_dataset=[(image, torch.zeros(165))])
        with patch.object(qualification, "load_model_for_experiment", return_value=wrapper) as restored, \
                patch("src.dashboards.runtime.load_visualization_datamodule", side_effect=load_dm):
            qualification.verify_dashboard_preprocessing("config", "checkpoint", image)
        restored.assert_called_once_with("config", checkpoint_path="checkpoint")

    def test_real_projection_analysis_api_full_selected_and_order_rejection(self):
        projection = resolve_projection(PROJECTION_ID)
        full = torch.arange(330).reshape(2, 165)
        qualification.verify_analysis_identity(full, list(projection.base_class_order), False)
        selected = full[:, list(projection.base_class_indices)]
        qualification.verify_analysis_identity(selected, list(projection.class_order), True)
        with self.assertRaises(ValueError):
            qualification.verify_analysis_identity(full, list(reversed(projection.base_class_order)), False)
        with self.assertRaises(ValueError):
            qualification.verify_analysis_identity(selected, list(reversed(projection.class_order)), True)

    def _run_controller(self, folder, outcomes, changes=False):
        commands = []
        def fake_worker(command, **kwargs):
            commands.append(command)
            def option(name):
                return command[command.index(name) + 1]
            job = Path(option("--job"))
            job.mkdir()
            status = outcomes(option("--worker"), option("--adaptation"), int(option("--physical-batch")))
            qualification.write_record(job / "result.json", {"status": status, "error": "sentinel"})
            return SimpleNamespace(returncode=70 if status == "oom" else 0 if status == "passed" else 1)
        with patch.object(qualification, "ROOT", Path(folder)), \
                patch.object(qualification, "source_provenance", return_value={"source_inventory": {"a": "original"}}), \
                patch.object(qualification, "source_inventory", return_value={"a": "changed" if changes else "original"}), \
                patch.object(qualification.subprocess, "run", side_effect=fake_worker):
            result = qualification.run_portfolio("isolated", ["efficientnet_v2_s"])
        return json.loads((result / "qualification.json").read_text()), commands

    def test_descending_full_probes_then_fresh_full_and_selected_smokes(self):
        with tempfile.TemporaryDirectory() as folder:
            record, commands = self._run_controller(folder, lambda kind, adaptation, batch:
                "oom" if kind == "probe" and batch > 16 else "passed")
        self.assertEqual(record["status"], "passed")
        entry = record["models"]["efficientnet_v2_s"]
        self.assertEqual([p["physical_batch"] for p in entry["probes"]], [128, 64, 32, 16])
        self.assertEqual(entry["adaptation"], "full")
        self.assertEqual(entry["measured_physical_cap"], 16)
        self.assertEqual(len(commands), 6)
        self.assertNotIn("--selected", commands[-2])
        self.assertIn("--selected", commands[-1])
        self.assertEqual(len({command[command.index("--job") + 1] for command in commands}), 6)

    def test_frozen_fallback_only_after_all_declared_full_batches_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            record, commands = self._run_controller(folder, lambda kind, adaptation, batch:
                "oom" if kind == "probe" and adaptation == "full" else "passed")
        entry = record["models"]["efficientnet_v2_s"]
        self.assertEqual([p["physical_batch"] for p in entry["probes"][:5]], [128, 64, 32, 16, 8])
        self.assertTrue(all(p["adaptation"] == "full" for p in entry["probes"][:5]))
        self.assertEqual(entry["adaptation"], "frozen_encoder")
        self.assertEqual(entry["measured_physical_cap"], 128)

    def test_unrelated_worker_failure_and_source_drift_cannot_pass(self):
        for drift in (False, True):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(RuntimeError, "source inventory changed" if drift else "worker failed"):
                    self._run_controller(folder, lambda *args: "passed" if drift else "failed", changes=drift)
                record = json.loads((Path(folder) / "experiments/runtime_smoke/isolated/qualification.json").read_text())
                self.assertEqual(record["status"], "failed")

    def test_no_useful_fallback_retains_failure_without_topology_substitution(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "no useful permitted"):
                self._run_controller(folder, lambda *args: "oom")
            record = json.loads((Path(folder) / "experiments/runtime_smoke/isolated/qualification.json").read_text())
        entry = record["models"]["efficientnet_v2_s"]
        self.assertEqual(len(entry["probes"]), 10)
        self.assertNotIn("full_smoke", entry)

    def test_containment_and_no_overwrite(self):
        for name in ("", ".", "..", "../escape", "nested/run", "nested\\run"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                qualification.run_directory(name)
        with tempfile.TemporaryDirectory() as folder, patch.object(qualification, "ROOT", Path(folder)):
            run = qualification.run_directory("existing")
            run.mkdir(parents=True)
            sentinel = run / "sentinel"
            sentinel.write_text("preserved")
            with self.assertRaises(FileExistsError):
                qualification.run_portfolio("existing")
            self.assertEqual(sentinel.read_text(), "preserved")
            with self.assertRaisesRegex(ValueError, "isolated runtime-smoke"):
                qualification.execute_job(Path(folder) / "outside", "efficientnet_v2_s", "full", 8, False, "probe")

    def test_entire_artifact_hash_checked_not_filename_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            cached = Path(folder) / "checkpoints"
            cached.mkdir()
            path = cached / "efficientnet_v2_s-dd5fe13b.pth"
            path.write_bytes(b"original")
            expected = hashlib.sha256(b"original").hexdigest()
            with patch.object(qualification, "_ARTIFACTS", {"efficientnet_v2_s": (path.name, 8, expected)}), \
                    patch("torch.hub.get_dir", return_value=folder):
                self.assertEqual(qualification.verify_artifact("p2_s")["sha256"], expected)
                path.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "mismatched"):
                    qualification.verify_artifact("efficientnet_v2_s")

    def test_original_encoder_buffers_and_weights_must_match_before_training(self):
        encoder = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        model = torch.nn.Module()
        model.features = encoder
        model.classifier = torch.nn.Linear(2, 165)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        artifact = {"filename": "dummy.pth"}
        with patch("torch.load", return_value=state):
            self.assertEqual(qualification.original_encoder_matches(SimpleNamespace(model=model), artifact), 7)
            with torch.no_grad():
                encoder[1].running_mean.add_(1)
            with self.assertRaisesRegex(ValueError, "state changed"):
                qualification.original_encoder_matches(SimpleNamespace(model=model), artifact)


if __name__ == "__main__":
    unittest.main()

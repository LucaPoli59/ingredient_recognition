"""Synthetic CPU checkpoint and temporary-study provenance regression tests."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import optuna
import torch
from torch.utils.data import DataLoader

from src.lightning.benchmark_lgn import BenchmarkLGNM, BenchmarkTrainer
from src.models.experimental_contract import ExperimentalModelContract
from src.training.benchmark_campaign import open_study, _save_sampler, remaining_trials
from src.training.benchmark_protocol import BenchmarkSchedule, policy, policy_hash, LEGACY_PROTOCOL_ID
from src.training.benchmark_provenance import (
    ADDED, MODIFIED, inspect_patch, register_checkpoint_patch, validate_study_sources,
    write_source_snapshot, patch_directory,
)
from test_experimental_lightning import TinyExperimentalAdapter, synthetic_data, synthetic_datamodule


class RuntimePatchTests(unittest.TestCase):
    def legacy_study(self, root, resume=False):
        with mock.patch("src.training.benchmark_campaign.policy", return_value=policy(LEGACY_PROTOCOL_ID)), \
             mock.patch("src.training.benchmark_campaign.policy_hash", return_value=policy_hash(LEGACY_PROTOCOL_ID)):
            return open_study(root, "maxvit_t", resume=resume)

    def fixture(self, directory):
        repo = Path(directory) / "repo"
        files = MODIFIED | {"src/models/unchanged.py", "src/training/resources/phase6_protocol_v1.json"}
        for name in files:
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original fixture bytes\n")
        root = repo / "experiments/study"
        study = self.legacy_study(root)

        def objective(trial):
            trial.set_user_attr("completed_epochs", 40)
            trial.set_user_attr("optimizer_updates", 15000)
            return trial.suggest_float("x", 0., 1.)

        study.optimize(objective, n_trials=1)
        _save_sampler(root, study)
        self.snapshot(root, repo)
        (root / "policy.json").write_text(json.dumps(policy(LEGACY_PROTOCOL_ID)))
        for name in ("trial_0/trial_config.json", "trial_0/epoch_costs.jsonl",
                     "trial_0/checkpoints/last.ckpt", "trial_0/checkpoints/best.ckpt",
                     "validation/trial_0/prediction_manifest.json", "validation/trial_0/validation_predictions.npz"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"retained synthetic artifact")
        for name in MODIFIED | ADDED:
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("repaired fixture bytes\n")
        return root, repo

    def snapshot(self, destination, repo):
        with mock.patch("src.training.benchmark_provenance.subprocess.check_output", return_value="fixture-base\n"):
            return write_source_snapshot(destination, repo)

    def register(self, root, repo):
        with mock.patch("src.training.benchmark_provenance.subprocess.check_output", return_value="fixture-patch\n"):
            return register_checkpoint_patch(root, accept_missing_final=True, repo=repo)

    def test_registration_preserves_trial_journal_sampler_and_original_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root, repo = self.fixture(directory)
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with self.assertRaisesRegex(ValueError, "registration"):
                validate_study_sources(root, repo)
            identity = self.register(root, repo)
            self.assertEqual(identity["runtime_patch"], "final-checkpoint-v1")
            self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))
            self.assertEqual(self.register(root, repo), identity)
            resumed = self.legacy_study(root, resume=True)
            self.assertEqual(len(resumed.trials), 1)
            self.assertEqual(resumed.trials[0].state, optuna.trial.TrialState.COMPLETE)
            with mock.patch("src.training.benchmark_campaign.policy", return_value=policy(LEGACY_PROTOCOL_ID)):
                self.assertEqual(remaining_trials(resumed), 99)
            self.assertEqual((root / "journal.log").read_bytes(), before[root / "journal.log"])
            self.assertEqual((root / "sampler.bin").read_bytes(), before[root / "sampler.bin"])
            self.assertEqual((patch_directory(root) / "sampler_before.bin").read_bytes(), before[root / "sampler.bin"])

    def test_explicit_exception_acceptance_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root, repo = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "acceptance"):
                register_checkpoint_patch(root, repo=repo)
            self.assertFalse(patch_directory(root).exists())

    def test_unrelated_modified_or_added_source_rejected(self):
        for name in ("src/models/unchanged.py", "settings/unreviewed.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root, repo = self.fixture(directory)
                path = repo / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("unapproved change")
                with self.assertRaisesRegex(ValueError, "exceed"):
                    inspect_patch(root, repo)

    def test_registered_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, repo = self.fixture(directory)
            self.register(root, repo)
            (repo / next(iter(MODIFIED))).write_text("a further unregistered change")
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_study_sources(root, repo)

    def test_old_checkpoint_and_journal_prefix_remain_immutable(self):
        for name in ("trial_0/checkpoints/last.ckpt", "journal.log"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root, repo = self.fixture(directory)
                self.register(root, repo)
                (root / name).write_bytes(b"changed earlier evidence")
                with self.assertRaises(ValueError):
                    validate_study_sources(root, repo)

    def test_future_journal_append_and_live_sampler_do_not_rewrite_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root, repo = self.fixture(directory)
            identity = self.register(root, repo)
            journal = root / "journal.log"
            with journal.open("a") as stream:
                stream.write(json.dumps({"op_code": 6, "trial_id": 1, "state": 1}) + "\n")
            payload = b"future sampler state"
            (root / "sampler.bin").write_bytes(hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload)
            self.assertEqual(validate_study_sources(root, repo), identity)

    def test_partial_trial_and_changed_scientific_policy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, repo = self.fixture(directory)
            with (root / "journal.log").open("a") as stream:
                stream.write(json.dumps({"op_code": 6, "trial_id": 0, "state": 0}) + "\n")
            with self.assertRaisesRegex(ValueError, "completed first-release"):
                inspect_patch(root, repo)
        with tempfile.TemporaryDirectory() as directory:
            root, repo = self.fixture(directory)
            (root / "policy.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "policy"):
                inspect_patch(root, repo)

    def test_registered_zip_and_sampler_copy_corruption_rejected(self):
        for name in ("source_snapshot.zip", "sampler_before.bin"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root, repo = self.fixture(directory)
                self.register(root, repo)
                (patch_directory(root) / name).write_bytes(b"corrupt")
                with self.assertRaises(ValueError):
                    validate_study_sources(root, repo)


class LatestCheckpointPruningTests(unittest.TestCase):
    def test_actual_pruning_keeps_the_latest_full_checkpoint(self):
        torch.set_num_threads(2)
        model = TinyExperimentalAdapter(ExperimentalModelContract("efficientnet_v2_s", 2, "full", 42, .5))
        module = BenchmarkLGNM(model=model, lr=1e-3, batch_size=128, physical_batch_size=32,
                               max_physical_batch_size=32, output_class_order=["zucchini", "apple"],
                               optimizer=torch.optim.AdamW, loss_fn=torch.nn.BCEWithLogitsLoss,
                               weighted_loss=False, lr_scheduler=BenchmarkSchedule,
                               benchmark_policy_hash=policy_hash(), expected_validation_records=17)
        module.startup_model(synthetic_datamodule())
        trial = mock.Mock(number=1)
        trial.study.user_attrs = {"benchmark_policy_hash": policy_hash(), "benchmark_model_id": "efficientnet_v2_s"}
        trial.should_prune.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            trainer = BenchmarkTrainer(directory, grad_accum=4, accelerator="cpu", trial=trial,
                                       enable_progress_bar=False, enable_model_summary=False)
            loader = DataLoader(synthetic_data(17), batch_size=32)
            with self.assertRaises(optuna.TrialPruned):
                trainer.fit(module, loader, loader)
            saved = torch.load(trainer.last_checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual((saved["epoch"], saved["global_step"]), (0, 1))
            self.assertTrue(saved["optimizer_states"])
            self.assertTrue(saved["lr_schedulers"])


if __name__ == "__main__":
    unittest.main()

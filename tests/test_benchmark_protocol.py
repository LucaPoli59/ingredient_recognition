"""Focused Phase 6 metric/configuration tests; no food-image training or HPO."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from src.commons.exp_config import ExpConfig
from src.commons.config_enc_dec import encode_config
from src.lightning.benchmark_lgn import BenchmarkLGNM, BenchmarkTrainer, EpochPruning
from src.training.benchmark_protocol import (epoch_metrics, make_config, policy, policy_hash,
                                            BenchmarkSchedule, lr_factor, local_candidates)
from src.training.benchmark_campaign import open_study, remaining_trials, _save_sampler, select_trial
from src.training.experimental_runtime import load_model_for_experiment
from src.models.benchmark_anchors import BenchmarkAnchor, verified_dino_source
from src.models.experimental_contract import ExperimentalModelContract
from test_experimental_lightning import TinyExperimentalAdapter, synthetic_data, synthetic_datamodule


class BenchmarkProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def module(self, count=17):
        fixture = TinyExperimentalAdapter(ExperimentalModelContract("efficientnet_v2_s", 2, "full", 42, .5))
        module = BenchmarkLGNM(model=fixture, lr=1e-3, batch_size=128,
                               physical_batch_size=32, max_physical_batch_size=32,
                               output_class_order=["zucchini", "apple"], optimizer=torch.optim.AdamW,
                               loss_fn=torch.nn.BCEWithLogitsLoss, weighted_loss=False,
                               lr_scheduler=BenchmarkSchedule, benchmark_policy_hash=policy_hash(),
                               expected_validation_records=count)
        module.startup_model(synthetic_datamodule())
        return module

    def test_global_ap_differs_from_batch_average_and_handles_ties(self):
        y = np.array([[1], [0], [0], [1]])
        logits = np.array([[.9], [.8], [.7], [.1]])
        self.assertAlmostEqual(epoch_metrics(logits, y)["val_macro_ap"], .75)
        batch = np.mean([epoch_metrics(logits[:2], y[:2])["val_macro_ap"],
                         epoch_metrics(logits[2:], y[2:])["val_macro_ap"]])
        # Use another split to explicitly distinguish full ranking from batch ranking.
        self.assertAlmostEqual(batch, .75)
        logits = np.array([[.3], [.2], [.9], [.8]])
        self.assertNotEqual(epoch_metrics(logits, y)["val_macro_ap"], batch)
        self.assertEqual(epoch_metrics(np.zeros((4, 1)), y)["val_macro_ap"], .5)

    def test_metric_invalid_columns_and_inputs(self):
        y = np.array([[0, 1, 1], [0, 1, 0]])
        result = epoch_metrics(np.array([[1., 2., 3.], [1., 2., -3.]]), y)
        self.assertEqual(result["valid_ap_columns"], [2])
        self.assertEqual(result["val_macro_ap"], 1.)
        for x, target in [(np.zeros((2, 1)), np.zeros((2, 1))),
                          (np.array([[np.nan], [1.]]), np.array([[0], [1]])),
                          (np.ones((2, 1)), np.array([[0], [.5]]))]:
            with self.assertRaises(ValueError):
                epoch_metrics(x, target)

    def test_five_configs_and_selected_roundtrip_no_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in policy()["models"]:
                for selected in (False, True):
                    config = make_config(name, directory, selected=selected)
                    self.assertEqual(config.torch_model["num_classes"], 59 if selected else 165)
                    self.assertIs(config.lgn_model["lgn_model_type"], BenchmarkLGNM)
                    self.assertFalse(config.lgn_model["weighted_loss"])
                    self.assertEqual(config.lgn_model["output_class_order"], config.label_encoder["classes"])
                    config.save_to_file(Path(directory) / "config.json")
                    restored = ExpConfig.load_from_file(Path(directory) / "config.json")
                    self.assertEqual(encode_config(restored.torch_model), encode_config(config.torch_model))
                    self.assertEqual(restored.lgn_model["benchmark_policy_hash"], policy_hash())
                    restored.validate_ingredient_projection()

    def test_schedule_finite_horizon_and_local_panel(self):
        factors = [lr_factor(e) for e in range(40)]
        self.assertEqual(factors[:2], [.1, 1.])
        self.assertAlmostEqual(factors[-1], .001)
        self.assertTrue(all(a > b for a, b in zip(factors[1:-1], factors[2:])))
        self.assertEqual(len(set(local_candidates(1e-4, 1e-4))), 4)

    def test_pruning_one_based_no_prune_baseline_or_final(self):
        trial = mock.Mock(number=1)
        trial.should_prune.return_value = True
        callback = EpochPruning(trial)
        trainer = SimpleNamespace(current_epoch=4, callback_metrics={"val_macro_ap": .6})
        with self.assertRaises(optuna.TrialPruned):
            callback.on_train_epoch_end(trainer, None)
        trial.report.assert_called_with(.6, step=5)
        trial.number = 0
        callback.on_train_epoch_end(trainer, None)
        trial.number = 1
        trainer.current_epoch = 39
        callback.on_train_epoch_end(trainer, None)

    def test_production_trainer_ap_checkpoint_and_offline_full_light(self):
        with tempfile.TemporaryDirectory() as directory:
            module = self.module()
            loader = DataLoader(synthetic_data(17), batch_size=32)
            trainer = BenchmarkTrainer(directory, grad_accum=4, accelerator="cpu",
                                       enable_progress_bar=False, enable_model_summary=False)
            trainer.fit(module, loader, loader)
            self.assertEqual(trainer.global_step, 40)
            checkpoint = trainer.checkpoint_callback
            self.assertEqual(checkpoint.monitor, "val_macro_ap")
            self.assertEqual(checkpoint.mode, "max")
            self.assertTrue(Path(checkpoint.last_model_path).exists())
            self.assertTrue((Path(directory) / "logs/0/metrics.csv").is_file())
            self.assertEqual(len((Path(directory) / "epoch_costs.jsonl").read_text().splitlines()), 40)
            saved = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["epoch"], 0)  # Equal AP keeps the first epoch, not the last.
            clone = BenchmarkLGNM.load_from_config(dict(module.hparams), initialize_pretrained=False)
            clone.load_weights_from_checkpoint_data(saved)
            for a, b in zip(module.parameters(), clone.parameters()):
                torch.testing.assert_close(a, b)
            light = copy.deepcopy(saved)
            light.pop("hyper_parameters")
            clone.load_weights_from_checkpoint_data(light)
            wrong = copy.deepcopy(saved)
            wrong["experimental_training_config"]["benchmark_policy_hash"] = "wrong"
            with self.assertRaises(ValueError):
                clone.load_weights_from_checkpoint_data(wrong)

    def test_partial_validation_rejected(self):
        module = self.module(18)
        module._trainer = SimpleNamespace(sanity_checking=False)
        module._validation_parts = [(torch.zeros(17, 2), synthetic_data(17).tensors[1])]
        with self.assertRaisesRegex(ValueError, "complete validation"):
            module.on_validation_epoch_end()

    def test_resume_sampler_and_budget_without_journal_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "study"
            study = open_study(root, "maxvit_t")
            study.optimize(lambda t: t.suggest_float("x", 1e-5, 1e-3, log=True), n_trials=3)
            _save_sampler(root, study)
            before = (root / "journal.log").read_bytes()
            resumed = open_study(root, "maxvit_t", resume=True)
            self.assertEqual((root / "journal.log").read_bytes(), before)
            self.assertEqual(remaining_trials(resumed), 97)
            # Compare next TPE draw against a separately deserialized copy of the sampler.
            a = resumed.ask().suggest_float("x", 1e-5, 1e-3, log=True)
            b = study.ask().suggest_float("x", 1e-5, 1e-3, log=True)
            self.assertEqual(a, b)
            with self.assertRaisesRegex(ValueError, "interrupted/out-of-sync"):
                open_study(root, "maxvit_t", resume=True)
            with self.assertRaises(FileExistsError):
                open_study(root, "maxvit_t")

    def test_trial_selection_requires_complete_horizon_and_earliest_tie(self):
        study = optuna.create_study(direction="maximize")
        for score, epochs in [(1., 4), (.8, 40), (.8, 40)]:
            study.add_trial(optuna.trial.create_trial(value=score, user_attrs={"completed_epochs": epochs}))
        self.assertEqual(select_trial(study).number, 1)

    def test_real_resnet_offline_adapter_and_contract(self):
        contract = ExperimentalModelContract("resnet18_anchor", 2, "full", 42, .5)
        with mock.patch("torch.hub.download_url_to_file", side_effect=AssertionError("network")):
            model = BenchmarkAnchor(contract, initialize_pretrained=False)
            restored = BenchmarkAnchor.load_from_config(model.to_config(), initialize_pretrained=False)
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        self.assertEqual(model(torch.zeros(1, 3, 224, 224)).shape, (1, 2))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(model.transform_aug.horizontal_flip_probability, .5)
        changed = model.to_config()
        changed["anchor_provenance"]["sha256"] = "bad"
        with self.assertRaises(ValueError):
            BenchmarkAnchor.load_from_config(changed, initialize_pretrained=False)

    def test_anchor_adaptation_modes_reject_other_protocols(self):
        with self.assertRaises(ValueError):
            ExperimentalModelContract("dinov2_b14_anchor", 165, "full")
        with self.assertRaises(ValueError):
            ExperimentalModelContract("resnet18_anchor", 165, "frozen_encoder")


if __name__ == "__main__":
    unittest.main()

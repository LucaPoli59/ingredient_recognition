"""R1 CPU/synthetic implementation checks; no campaign or predictive test use."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import optuna
from optuna.trial import TrialState, create_trial
import torch
from torch.utils.data import DataLoader

from src.lightning.benchmark_lgn import BenchmarkLGNM, BenchmarkTrainer, EpochPruning
from src.models.experimental_contract import ExperimentalModelContract
from src.training.benchmark_campaign import open_study, remaining_trials, select_trial, _save_sampler
from src.training.benchmark_protocol import (policy, policy_hash, policy_for_hash, LEGACY_PROTOCOL_ID,
                                            ACTIVE_PROTOCOL_ID, BenchmarkSchedule)
from src.training.benchmark_stopping import PlateauState
from test_experimental_lightning import TinyExperimentalAdapter, synthetic_data, synthetic_datamodule


def incumbent(study, value=.8, epochs=40, updates=15000):
    study.add_trial(create_trial(value=value, user_attrs={"completed_epochs": epochs, "optimizer_updates": updates}))


class PlateauTests(unittest.TestCase):
    def test_policy_versions_only_declared_scientific_delta(self):
        old, new = policy(LEGACY_PROTOCOL_ID), policy()
        self.assertEqual(new["protocol_id"], ACTIVE_PROTOCOL_ID)
        self.assertEqual(new["hpo"]["started_trials_per_model"], 50)
        self.assertEqual(new["hpo"]["plateau"]["min_completed_epochs"], 20)
        self.assertEqual((new["hpo"]["plateau"]["patience"], new["hpo"]["plateau"]["min_delta"]), (10, 1e-4))
        expected = copy.deepcopy(old)
        expected["protocol_id"] = ACTIVE_PROTOCOL_ID
        expected["study_initialization"] = "fresh_no_trial_import"
        expected["hpo"]["started_trials_per_model"] = 50
        expected["hpo"]["plateau"] = new["hpo"]["plateau"]
        self.assertEqual(new, expected)
        self.assertEqual(policy_hash(LEGACY_PROTOCOL_ID), "4a5fe311e083594de1d872cd3eea3701e8bfa641bfa7753c9b8f3d40d45d756c")
        self.assertEqual(policy_for_hash(policy_hash(LEGACY_PROTOCOL_ID)), old)
        with self.assertRaises(ValueError):
            policy_for_hash("unknown")

    def test_patience_exact_tolerance_and_cumulative_improvement(self):
        state = PlateauState(10, 1e-4)
        state.observe(.5)
        for _ in range(9):
            state.observe(.5)
        self.assertEqual(state.bad_checks, 9)
        state.observe(.5 + 1e-4)
        self.assertEqual(state.bad_checks, 10)
        self.assertEqual(state.best_observed, .5 + 1e-4)
        self.assertEqual(state.patience_best, .5)
        state.observe(.5 + 1.5e-4)
        self.assertEqual(state.bad_checks, 0)
        with self.assertRaises(ValueError):
            state.observe(float("nan"))

    def test_floor_incumbent_protection_and_ineligible_complete(self):
        study = optuna.create_study(direction="maximize")
        state = PlateauState(10, 1e-4)
        for _ in range(20):
            state.observe(.6)
        rule = policy()["hpo"]["plateau"]
        self.assertFalse(state.should_prune(20, rule, study)[0])
        incumbent(study, 1., updates=14999)
        self.assertFalse(state.should_prune(20, rule, study)[0])
        incumbent(study, .6)
        self.assertFalse(state.should_prune(19, rule, study)[0])
        self.assertTrue(state.should_prune(20, rule, study)[0])
        state.best_observed = .600001
        self.assertFalse(state.should_prune(20, rule, study)[0])
        self.assertEqual(select_trial(study).number, 1)

    def test_hyperband_is_independent_and_reference_final_exempt(self):
        trial = mock.Mock(number=1)
        trial.should_prune.return_value = True
        trainer = SimpleNamespace(current_epoch=4, callback_metrics={"val_macro_ap": .9}, global_step=1875)
        with self.assertRaisesRegex(optuna.TrialPruned, "hyperband"):
            EpochPruning(trial).on_train_epoch_end(trainer, None)
        self.assertEqual(trial.set_user_attr.call_args.args[1]["reason"], "hyperband")
        for number, epoch in ((0, 20), (1, 40)):
            trial.number = number
            trial.should_prune.reset_mock()
            trainer.current_epoch = epoch - 1
            EpochPruning(trial).on_train_epoch_end(trainer, None)
            trial.should_prune.assert_not_called()

    def test_fresh_budget_includes_all_states_and_old_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"study"
            study = open_study(root, "maxvit_t")
            self.assertEqual(len(study.trials), 0)
            self.assertEqual(remaining_trials(study), 50)
            self.assertEqual(study.user_attrs["benchmark_policy_hash"], policy_hash())
            for index in range(50):
                state = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)[index % 3]
                study.add_trial(create_trial(state=state, value=.2 if state == TrialState.COMPLETE else None))
            self.assertEqual(remaining_trials(study), 0)
            _save_sampler(root, study)
            before = (root/"journal.log").read_bytes()
            restored = open_study(root, "maxvit_t", resume=True)
            self.assertEqual(len(restored.trials), 50)
            self.assertEqual((root/"journal.log").read_bytes(), before)
            identity = json.loads((root/"study_identity.json").read_text())
            identity["policy_hash"] = policy_hash(LEGACY_PROTOCOL_ID)
            (root/"study_identity.json").write_text(json.dumps(identity))
            with self.assertRaisesRegex(ValueError, "protocol/model/package"):
                open_study(root, "maxvit_t", resume=True)
            self.assertEqual((root/"journal.log").read_bytes(), before)

    def test_saved_v1_module_restores_its_own_policy_offline(self):
        model = TinyExperimentalAdapter(ExperimentalModelContract("efficientnet_v2_s", 2, "full", 42, .5))
        module = BenchmarkLGNM(model=model, lr=1e-3, batch_size=128, physical_batch_size=32,
            max_physical_batch_size=32, output_class_order=["zucchini", "apple"], optimizer=torch.optim.AdamW,
            loss_fn=torch.nn.BCEWithLogitsLoss, weighted_loss=False, lr_scheduler=BenchmarkSchedule,
            benchmark_policy_hash=policy_hash(LEGACY_PROTOCOL_ID), expected_validation_records=17)
        module.startup_model(synthetic_datamodule())
        restored = BenchmarkLGNM.load_from_config(dict(module.hparams), initialize_pretrained=False)
        self.assertEqual(restored.hparams["benchmark_policy_hash"], policy_hash(LEGACY_PROTOCOL_ID))

    def test_actual_plateau_prunes_after_checkpoint_with_cost_and_state(self):
        torch.set_num_threads(2)
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
        study.set_user_attr("benchmark_policy_hash", policy_hash())
        study.set_user_attr("benchmark_model_id", "efficientnet_v2_s")
        incumbent(study, 1.)
        trial = study.ask()
        model = TinyExperimentalAdapter(ExperimentalModelContract("efficientnet_v2_s", 2, "full", 42, .5))
        module = BenchmarkLGNM(model=model, lr=1e-3, batch_size=128, physical_batch_size=32,
            max_physical_batch_size=32, output_class_order=["zucchini", "apple"], optimizer=torch.optim.AdamW,
            loss_fn=torch.nn.BCEWithLogitsLoss, weighted_loss=False, lr_scheduler=BenchmarkSchedule,
            benchmark_policy_hash=policy_hash(), expected_validation_records=17)
        module.startup_model(synthetic_datamodule())
        fixed = {"val_macro_ap": .2, "val_micro_f1_at_0_5": .1, "valid_ap_columns": [0, 1], "records": 17}
        with tempfile.TemporaryDirectory() as directory:
            trainer = BenchmarkTrainer(directory, grad_accum=4, accelerator="cpu", trial=trial,
                                       enable_progress_bar=False, enable_model_summary=False)
            loader = DataLoader(synthetic_data(17), batch_size=32)
            with mock.patch("src.lightning.benchmark_lgn.epoch_metrics", return_value=fixed):
                with self.assertRaisesRegex(optuna.TrialPruned, "plateau at completed epoch 20"):
                    trainer.fit(module, loader, loader)
            study.tell(trial, state=TrialState.PRUNED)
            saved = torch.load(trainer.last_checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual((saved["epoch"], saved["global_step"]), (19, 20))
            self.assertTrue(saved["optimizer_states"])
            record = json.loads((Path(directory)/"pruning.json").read_text())
            self.assertEqual(record["reason"], "plateau")
            self.assertEqual((record["completed_epochs"], record["optimizer_updates"]), (20, 20))
            self.assertGreater(record["consumed_epoch_seconds"], 0)
            self.assertEqual(record, study.trials[-1].user_attrs["pruning"])
            self.assertEqual(study.trials[-1].state, TrialState.PRUNED)
            self.assertEqual(len((Path(directory)/"epoch_costs.jsonl").read_text().splitlines()), 20)


if __name__ == "__main__":
    unittest.main()

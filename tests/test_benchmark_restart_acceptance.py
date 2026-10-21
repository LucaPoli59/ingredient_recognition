"""R2 acceptance: real controller/studies, synthetic training/data boundaries only.

No campaign directory, pretrained construction, image inference or GPU fit.
The two pruning cases additionally run the actual tiny CPU Lightning trainer.
"""

from contextlib import ExitStack
import io
import json
from pathlib import Path
import runpy
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
from src.training import benchmark_campaign as campaign
from src.training import benchmark_queue
from src.training.benchmark_protocol import BenchmarkSchedule, policy, policy_hash
from test_experimental_lightning import TinyExperimentalAdapter, synthetic_data, synthetic_datamodule


REPO = Path(__file__).resolve().parents[1]


class RestartAcceptanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.allowed = Path(directory) / "benchmark_v2"
        self.root = self.allowed / "synthetic_acceptance" / "efficientnet_v2_s"
        self.stack.enter_context(mock.patch.object(campaign, "CAMPAIGN_ROOT", self.allowed))
        from src.ingredient_selection.runtime import resolve_projection
        projection = resolve_projection(policy()["selected_projection"])
        self.metadata_guard = self.stack.enter_context(mock.patch.object(type(projection), "verify_metadata"))
        self.environment = {"fixture": "synthetic CPU acceptance, not artifact qualification"}
        self.stack.enter_context(mock.patch.object(campaign, "environment_identity", return_value=self.environment))
        self.stack.enter_context(mock.patch("src.training.commons.load_datamodule", return_value=object()))
        self.stack.enter_context(mock.patch("src.training.experimental_runtime.prepare_experimental_config"))
        self.stack.enter_context(mock.patch("src.training.commons.set_torch_constants"))
        self.fit = self.stack.enter_context(mock.patch("src.training.commons.model_training", side_effect=self.training_double))
        self.export = self.stack.enter_context(mock.patch("src.training.benchmark_predictions.export_best_validation",
                                                        return_value={"synthetic_export_boundary": True}))
        self.mode = "complete"
        self.seen_trials = []
        self.cpu_trainers = []
        previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
        self.addCleanup(self.restore_numeric_settings, previous)

    @staticmethod
    def restore_numeric_settings(settings):
        precision, cuda_tf32, cudnn_tf32, benchmark = settings
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = benchmark

    def release(self, **kwargs):
        return campaign._execute(self.root, "efficientnet_v2_s", **kwargs)

    def study(self):
        return campaign.open_study(self.root, "efficientnet_v2_s", resume=True)

    def training_double(self, config, data, *, trainer_kwargs):
        trial = trainer_kwargs["trial"]
        self.seen_trials.append(trial.number)
        self.assertEqual(config.lgn_model["benchmark_policy_hash"], policy_hash())
        self.assertEqual(config.torch_model["num_classes"], 165)
        self.assertIsNone(config.datamodule["ingredient_projection"])
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cudnn.benchmark)
        if self.mode == "fail":
            raise RuntimeError("synthetic runtime failure")
        if self.mode in ("plateau", "hyperband"):
            return self.pruning_fit(config, trial)
        folder = Path(config.trainer["save_dir"])
        # Explicit controller doubles; these are NOT measured updates or real checkpoints.
        trainer = SimpleNamespace(current_epoch=39 if self.mode == "short" else 40,
                                  global_step=14999 if self.mode == "bad_updates" else 15000,
                                  checkpoint_callback=SimpleNamespace(best_model_score=.5,
                                      best_model_path=str(folder / "synthetic_best.ckpt")),
                                  last_checkpoint_path=str(folder / "synthetic_last.ckpt"))
        return trainer, None

    def pruning_fit(self, config, trial):
        model = TinyExperimentalAdapter(ExperimentalModelContract("efficientnet_v2_s", 2, "full", 42, .5))
        module = BenchmarkLGNM(model=model, lr=1e-3, batch_size=128, physical_batch_size=32,
            max_physical_batch_size=32, output_class_order=["zucchini", "apple"], optimizer=torch.optim.AdamW,
            loss_fn=torch.nn.BCEWithLogitsLoss, weighted_loss=False, lr_scheduler=BenchmarkSchedule,
            benchmark_policy_hash=policy_hash(), expected_validation_records=17)
        module.startup_model(synthetic_datamodule())
        trainer = BenchmarkTrainer(config.trainer["save_dir"], grad_accum=4, accelerator="cpu", trial=trial,
                                   enable_progress_bar=False, enable_model_summary=False)
        self.cpu_trainers.append(trainer)
        loader = DataLoader(synthetic_data(17), batch_size=32)
        def metrics(*_):
            # A strict improvement below min_delta must still save the AP-best checkpoint.
            score = .20005 if trainer.current_epoch >= 4 else .2
            return {"val_macro_ap": score, "val_micro_f1_at_0_5": .1,
                    "valid_ap_columns": [0, 1], "records": 17}
        # Force only the Hyperband decision, not its configured resources or the OR callback.
        with mock.patch.object(trial, "should_prune", side_effect=lambda: self.mode == "hyperband" and trainer.current_epoch >= 4), \
             mock.patch("src.lightning.benchmark_lgn.epoch_metrics", side_effect=metrics):
            trainer.fit(module, loader, loader)
        self.fail("synthetic pruning trace unexpectedly completed")

    def test_canonical_reference_resume_and_saved_artifact_identities(self):
        result = self.release()
        self.assertEqual((result["trials_started"], result["remaining"], result["best_trial"]), (1, 49, 0))
        study = self.study()
        self.assertEqual(study.trials[0].params, {"lr": 1e-4, "weight_decay": 1e-4})
        self.assertEqual(study.trials[0].state, TrialState.COMPLETE)
        self.assertIsInstance(study.sampler, optuna.samplers.TPESampler)
        self.assertIsInstance(study.pruner, optuna.pruners.HyperbandPruner)
        self.assertEqual((study.pruner._min_resource, study.pruner._max_resource, study.pruner._reduction_factor), (5, 40, 3))
        self.assertEqual(json.loads((self.root / "policy.json").read_text()), policy())
        stopping = json.loads((self.root / "trial_0/stopping_policy.json").read_text())
        self.assertEqual(stopping, study.trials[0].user_attrs["stopping_policy"])
        self.assertEqual(stopping["hpo"]["started_trials_per_model"], 50)
        self.assertEqual(stopping["required_optimizer_updates"], 15000)
        from src.training.benchmark_provenance import validate_study_sources, verify_snapshot
        source = validate_study_sources(self.root)
        self.assertEqual(source, study.trials[0].user_attrs["source_identity"])
        inventory = verify_snapshot(self.root)
        self.assertEqual(inventory["files"]["src/training/resources/phase6_protocol_v2.json"], policy_hash())
        journal = (self.root / "journal.log").read_bytes()
        result = self.release(resume=True, cost_reviewed=True)
        self.assertEqual((result["trials_started"], result["remaining"], result["best_trial"]), (2, 48, 0))
        self.assertEqual(self.seen_trials, [0, 1])
        self.assertTrue((self.root / "journal.log").read_bytes().startswith(journal))
        self.assertEqual(json.loads((self.root / "trial_1/stopping_policy.json").read_text()), stopping)

    def test_fiftieth_attempt_allowed_no_replacement_or_fifty_first(self):
        self.release()
        study = self.study()
        for index in range(48):
            state = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)[index % 3]
            study.add_trial(create_trial(state=state, value=.99 if state == TrialState.COMPLETE else None))
        campaign._save_sampler(self.root, study)
        before = self.mutable_bytes()
        with self.assertRaisesRegex(ValueError, "remaining fixed budget"):
            self.release(resume=True, cost_reviewed=True, additional_trials=2)
        self.assertEqual(self.mutable_bytes(), before)
        result = self.release(resume=True, cost_reviewed=True)
        self.assertEqual((result["trials_started"], result["remaining"], result["best_trial"]), (50, 0, 0))
        self.assertEqual(self.seen_trials, [0, 49])
        before = self.mutable_bytes()
        with self.assertRaisesRegex(ValueError, "remaining fixed budget"):
            self.release(resume=True, cost_reviewed=True)
        self.assertEqual(self.mutable_bytes(), before)
        self.assertEqual(self.fit.call_count, 2)

    def mutable_bytes(self):
        return {name: (self.root / name).read_bytes() for name in ("journal.log", "sampler.bin")}

    def test_runtime_failure_consumes_attempt_and_stops_without_retry(self):
        self.release()
        self.mode = "fail"
        with self.assertRaisesRegex(RuntimeError, "synthetic runtime failure"):
            self.release(resume=True, cost_reviewed=True, additional_trials=2)
        study = self.study()
        self.assertEqual([t.state for t in study.trials], [TrialState.COMPLETE, TrialState.FAIL])
        self.assertEqual(campaign.remaining_trials(study), 48)
        self.assertGreaterEqual(study.trials[-1].user_attrs["session_wall_seconds"], 0)
        self.assertFalse((self.root / "trial_2").exists())
        self.export.assert_called_once()
        failed = {p.relative_to(self.root): p.read_bytes() for p in (self.root / "trial_1").rglob("*") if p.is_file()}
        self.mode = "complete"
        result = self.release(resume=True, cost_reviewed=True)
        self.assertEqual(result["trials_started"], 3)
        self.assertEqual(self.seen_trials, [0, 1, 2])
        self.assertTrue(all((self.root / p).read_bytes() == value for p, value in failed.items()))

    def test_controller_records_actual_plateau_pruned_state_and_distinct_small_best(self):
        self.assert_actual_pruning("plateau", 20, 19)

    def test_controller_records_actual_hyperband_branch_without_plateau_delay(self):
        self.assert_actual_pruning("hyperband", 5, 4)

    def assert_actual_pruning(self, reason, epoch, bad_checks):
        self.release()
        self.mode = reason
        result = self.release(resume=True, cost_reviewed=True)
        study = self.study()
        trial = study.trials[1]
        self.assertEqual((trial.state, result["remaining"], result["best_trial"]), (TrialState.PRUNED, 48, 0))
        self.assertNotIn("completed_epochs", trial.user_attrs)
        self.assertGreater(trial.user_attrs["session_wall_seconds"], 0)
        record = json.loads((self.root / "trial_1/pruning.json").read_text())
        self.assertEqual(record, trial.user_attrs["pruning"])
        self.assertEqual((record["reason"], record["completed_epochs"], record["optimizer_updates"]), (reason, epoch, epoch))
        self.assertEqual(record["bad_checks"], bad_checks)
        self.assertAlmostEqual(record["patience_best_ap"], .2)
        self.assertAlmostEqual(record["best_observed_ap"], .20005)
        self.assertGreater(record["consumed_epoch_seconds"], 0)
        trainer = self.cpu_trainers[-1]
        best = torch.load(trainer.checkpoint_callback.best_model_path, map_location="cpu", weights_only=False)
        latest = torch.load(trainer.last_checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual((best["epoch"], latest["epoch"], latest["global_step"]), (4, epoch - 1, epoch))
        self.assertTrue(latest["optimizer_states"])
        self.assertTrue(latest["lr_schedulers"])
        if epoch > 5:
            self.assertTrue(any(not torch.equal(best["state_dict"][key], value)
                                for key, value in latest["state_dict"].items()))
        ledger = [json.loads(line) for line in (self.root / "trial_1/epoch_costs.jsonl").read_text().splitlines()]
        self.assertEqual([row["epoch"] for row in ledger], list(range(1, epoch + 1)))
        self.assertAlmostEqual(record["consumed_epoch_seconds"], sum(row["seconds"] for row in ledger))

    def test_incomplete_return_is_fail_not_eligible_complete(self):
        self.release()
        for mode in ("short", "bad_updates"):
            self.mode = mode
            with self.assertRaisesRegex(ValueError, "epoch/update horizon"):
                self.release(resume=True, cost_reviewed=True)
        study = self.study()
        self.assertEqual([t.state for t in study.trials], [TrialState.COMPLETE, TrialState.FAIL, TrialState.FAIL])
        self.assertEqual(campaign.select_trial(study).number, 0)
        self.assertEqual(campaign.remaining_trials(study), 47)

    def test_request_guards_reject_before_training_or_study_creation(self):
        for kwargs in ({"additional_trials": 0}, {"additional_trials": True}, {"additional_trials": 1.5},
                       {"additional_trials": 2}, {"resume": True},
                       {"additional_trials": 51, "cost_reviewed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.release(**kwargs)
            self.assertFalse(self.root.exists())
        for root in (self.allowed, self.allowed.parent / "benchmark_v1/study", self.allowed.parent / "outside"):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "isolated"):
                campaign._execute(root, "efficientnet_v2_s")
        self.fit.assert_not_called()

    def test_fresh_creation_refuses_overwrite_and_old_identity(self):
        self.release()
        before = self.mutable_bytes()
        with self.assertRaises(FileExistsError):
            self.release()
        self.assertEqual(self.mutable_bytes(), before)
        identity_path = self.root / "study_identity.json"
        identity = json.loads(identity_path.read_text())
        identity["policy_hash"] = "old policy is not imported"
        identity_path.write_text(json.dumps(identity))
        with self.assertRaisesRegex(ValueError, "protocol/model/package"):
            self.release(resume=True, cost_reviewed=True)
        self.assertEqual(self.mutable_bytes(), before)
        self.assertEqual(self.fit.call_count, 1)

    def test_resume_rejects_changed_policy_data_environment_and_sources(self):
        self.release()
        before = self.mutable_bytes()
        for name, mutate in (
                ("policy.json", lambda value: value["hpo"].update(started_trials_per_model=51)),
                ("data_identity.json", lambda value: value.update(base_class_order=["wrong"])),
                ("environment.json", lambda value: value.update(fixture="changed")),
                ("source_inventory.json", lambda value: value["files"].update({"src/training/benchmark_stopping.py": "wrong"}))):
            path = self.root / name
            original = path.read_bytes()
            value = json.loads(original)
            mutate(value)
            path.write_text(json.dumps(value))
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.release(resume=True, cost_reviewed=True)
            self.assertEqual(self.mutable_bytes(), before)
            path.write_bytes(original)
        self.assertEqual(self.fit.call_count, 1)

    def test_sampler_corruption_inflight_and_study_attribute_rejected_read_only(self):
        self.release()
        path = self.root / "sampler.bin"
        original = path.read_bytes()
        path.write_bytes(original + b"corrupt")
        journal = (self.root / "journal.log").read_bytes()
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.release(resume=True, cost_reviewed=True)
        self.assertEqual((self.root / "journal.log").read_bytes(), journal)
        path.write_bytes(original)
        study = self.study()
        study.set_user_attr("started_trial_budget", 51)
        before = self.mutable_bytes()
        with self.assertRaisesRegex(ValueError, "policy/model/budget"):
            self.release(resume=True, cost_reviewed=True)
        self.assertEqual(self.mutable_bytes(), before)
        study.set_user_attr("started_trial_budget", 50)
        study.ask()
        campaign._save_sampler(self.root, study)
        before = self.mutable_bytes()
        with self.assertRaisesRegex(ValueError, "interrupted/out-of-sync"):
            self.release(resume=True, cost_reviewed=True)
        self.assertEqual(self.mutable_bytes(), before)
        self.assertEqual(self.fit.call_count, 1)

    def test_missing_journal_is_not_silently_recreated(self):
        self.release()
        path = self.root / "journal.log"
        journal = path.read_bytes()
        path.rename(self.root / "retained_journal.log")
        sampler = (self.root / "sampler.bin").read_bytes()
        with self.assertRaisesRegex(ValueError, "missing.*recreated"):
            self.release(resume=True, cost_reviewed=True)
        self.assertFalse(path.exists())
        self.assertEqual((self.root / "retained_journal.log").read_bytes(), journal)
        self.assertEqual((self.root / "sampler.bin").read_bytes(), sampler)
        self.assertEqual(self.fit.call_count, 1)

    def test_all_five_studies_begin_empty_with_new_reproducible_samplers(self):
        draws = []
        for model_id in policy()["models"]:
            root = self.allowed / "fresh_sampler_fixtures" / model_id
            study = campaign.open_study(root, model_id)
            self.assertEqual(len(study.trials), 0)
            self.assertEqual(campaign.remaining_trials(study), 50)
            self.assertEqual(study.study_name, f"phase6-comparison-v2-{model_id}")
            self.assertEqual(study.user_attrs["benchmark_model_id"], model_id)
            trial = study.ask()
            self.assertEqual(trial.number, 0)
            draws.append(trial.suggest_float("lr", 1e-5, 1e-3, log=True))
        self.assertEqual(len(set(draws)), 1)
        self.fit.assert_not_called()

    def test_non_hpo_run_has_no_pruning_callback(self):
        trainer = BenchmarkTrainer(self.allowed / "standalone", accelerator="cpu",
                                   enable_progress_bar=False, enable_model_summary=False)
        self.assertFalse(any(isinstance(callback, EpochPruning) for callback in trainer.callbacks))

    def test_default_clis_only_print_and_first_trial_queue_stops_at_review(self):
        for script, args in (("train.py", []), ("first_trials.py", ["--run-name", "INSPECTION_ONLY"])):
            with self.subTest(script=script), mock.patch("sys.argv", [script, *args]), io.StringIO() as output, \
                 mock.patch("sys.stdout", output):
                runpy.run_path(str(REPO / "scripts/launch_exps/benchmark" / script), run_name="__main__")
                plan = json.loads(output.getvalue())
                if script == "train.py":
                    self.assertEqual(plan["hpo"]["started_trials_per_model"], 50)
                else:
                    self.assertEqual((plan["trials_per_model"], plan["next_state"]), (1, "awaiting_cost_review"))
        with mock.patch.object(benchmark_queue, "CAMPAIGN_ROOT", self.allowed), \
             mock.patch.object(benchmark_queue, "run_baselines") as baselines, \
             mock.patch.object(benchmark_queue, "execute", return_value={"synthetic": True}) as execute:
            state = benchmark_queue.first_trials("SYNTHETIC_QUEUE")
        self.assertEqual(state["status"], "awaiting_cost_review")
        self.assertEqual([call.args[1] for call in execute.call_args_list], policy()["models"])
        self.assertTrue(all(not call.kwargs for call in execute.call_args_list))
        baselines.assert_called_once()
        self.fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()

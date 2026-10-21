"""R3 temporary prepared studies; no pretrained construction, images or GPU fit."""

from contextlib import ExitStack
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from src.commons.exp_config import ExpConfig
from src.ingredient_selection.runtime import resolve_projection
from src.training import benchmark_campaign as campaign, benchmark_preparation as preparation, benchmark_queue
from src.training.benchmark_protocol import policy, policy_hash


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.allowed = Path(temporary) / "benchmark_v2"
        self.name = "synthetic-r3"
        self.stack.enter_context(mock.patch.object(campaign, "CAMPAIGN_ROOT", self.allowed))
        self.stack.enter_context(mock.patch.object(benchmark_queue, "CAMPAIGN_ROOT", self.allowed))
        projection = resolve_projection(policy()["selected_projection"])
        self.stack.enter_context(mock.patch.object(type(projection), "verify_metadata"))
        self.stack.enter_context(mock.patch.object(campaign, "environment_identity", side_effect=lambda name: {"synthetic_model": name}))
        self.costs = {"fixture": "synthetic costs, not a calendar estimate"}
        self.stack.enter_context(mock.patch.object(preparation, "cost_projection", return_value=self.costs))
        source = Path(temporary) / "synthetic_dino_source"
        (source / "dinov2").mkdir(parents=True)
        for name in ("dinov2/layers.py", "hubconf.py", "LICENSE"):
            (source / name).write_text("synthetic source fixture\n")
        self.stack.enter_context(mock.patch("src.models.benchmark_anchors.verified_dino_source", return_value=source))

    @property
    def root(self):
        return self.allowed / self.name

    def test_preparation_leaves_five_empty_studies_and_exact_config_snapshots(self):
        with mock.patch("optuna.study.Study.ask", side_effect=AssertionError("must not allocate")), \
             mock.patch("optuna.study.Study.optimize", side_effect=AssertionError("must not optimize")), \
             mock.patch("src.training.commons.model_training", side_effect=AssertionError("must not train")):
            manifest = preparation.prepare_campaign(self.name)
        self.assertEqual(manifest["status"], "prepared_not_launched")
        self.assertEqual(manifest["plan"]["full_hpo_attempt_allowance"], 250)
        before = self.all_bytes()
        self.assertEqual(preparation.inspect_campaign(self.name), manifest)
        self.assertEqual(self.all_bytes(), before)
        for name in policy()["models"]:
            directory = self.root / name
            study = campaign.open_study(directory, name, resume=True)
            self.assertEqual(len(study.trials), 0)
            self.assertEqual(campaign.remaining_trials(study), 50)
            config = ExpConfig.load_from_file(directory / "configuration_template.json")
            self.assertEqual(config.lgn_model["benchmark_policy_hash"], policy_hash())
            self.assertEqual(config.torch_model["num_classes"], 165)
            self.assertIsNone(config.datamodule["ingredient_projection"])
            self.assertEqual(len(config.lgn_model["output_class_order"]), 165)
            self.assertEqual(config.trainer["max_epochs"], 40)
            self.assertFalse(any(directory.glob("trial_*")))
        self.assertFalse((self.root / "campaign_queue.json").exists())
        self.assertFalse((self.root / "baselines").exists())
        with self.assertRaises(FileExistsError):
            preparation.prepare_campaign(self.name)
        self.assertEqual(self.all_bytes(), before)

    def all_bytes(self):
        return {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

    def test_prepared_launch_is_explicit_enqueues_reference_zero_and_cannot_repeat(self):
        preparation.prepare_campaign(self.name)
        root = self.root / "efficientnet_v2_s"
        before = self.all_bytes()
        with self.assertRaises(FileExistsError):
            campaign._execute(root, "efficientnet_v2_s")
        with self.assertRaisesRegex(ValueError, "empty studies require prepared"):
            campaign._execute(root, "efficientnet_v2_s", resume=True, cost_reviewed=True)
        with self.assertRaisesRegex(ValueError, "distinct"):
            campaign._execute(root, "efficientnet_v2_s", resume=True, prepared=True, cost_reviewed=True)
        self.assertEqual(self.all_bytes(), before)
        def fit(config, data, *, trainer_kwargs):
            trial = trainer_kwargs["trial"]
            self.assertEqual(trial.number, 0)
            self.assertEqual(trial.params, {"lr": 1e-4, "weight_decay": 1e-4})
            return SimpleNamespace(current_epoch=40, global_step=15000,
                checkpoint_callback=SimpleNamespace(best_model_path="synthetic_best", best_model_score=.5),
                last_checkpoint_path="synthetic_last"), None
        previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
        try:
            with mock.patch("src.training.commons.load_datamodule", return_value=object()), \
                 mock.patch("src.training.experimental_runtime.prepare_experimental_config"), \
                 mock.patch("src.training.commons.set_torch_constants"), \
                 mock.patch("src.training.commons.model_training", side_effect=fit), \
                 mock.patch("src.training.benchmark_predictions.export_best_validation", return_value={"synthetic": True}):
                result = campaign._execute(root, "efficientnet_v2_s", prepared=True)
        finally:
            torch.set_float32_matmul_precision(previous[0])
            torch.backends.cuda.matmul.allow_tf32 = previous[1]
            torch.backends.cudnn.allow_tf32 = previous[2]
            torch.backends.cudnn.benchmark = previous[3]
        self.assertEqual((result["trials_started"], result["remaining"]), (1, 49))
        with self.assertRaisesRegex(ValueError, "changed|started"):
            campaign._execute(root, "efficientnet_v2_s", prepared=True)

    def test_changed_config_or_journal_rejected_without_rewriting_evidence(self):
        preparation.prepare_campaign(self.name)
        root = self.root / "maxvit_t"
        for name in ("configuration_template.json", "journal.log", "sampler.bin"):
            path = root / name
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            before = self.all_bytes()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "prepared study inputs changed"):
                preparation.inspect_campaign(self.name)
            self.assertEqual(self.all_bytes(), before)
            path.write_bytes(original)

    def test_prepared_queue_validates_first_and_forwards_only_explicit_launch(self):
        preparation.prepare_campaign(self.name)
        with mock.patch.object(benchmark_queue, "run_baselines") as baselines, \
             mock.patch.object(benchmark_queue, "execute", return_value={"synthetic": True}) as execute:
            state = benchmark_queue.first_trials(self.name, prepared=True)
        self.assertEqual(state["status"], "awaiting_cost_review")
        self.assertEqual([call.args[1] for call in execute.call_args_list], policy()["models"])
        self.assertTrue(all(call.kwargs == {"prepared": True} for call in execute.call_args_list))
        baselines.assert_called_once()
        with self.assertRaisesRegex(ValueError, "already began"):
            benchmark_queue.first_trials(self.name, prepared=True)

    def test_cli_defaults_inspect_only_and_names_cannot_escape(self):
        script = preparation.REPO / "scripts/launch_exps/benchmark/prepare.py"
        with mock.patch("sys.argv", [str(script), "--run-name", self.name]), io.StringIO() as output, \
             mock.patch("sys.stdout", output), \
             mock.patch.object(preparation, "prepare_campaign", side_effect=AssertionError("must opt in")):
            runpy.run_path(str(script), run_name="__main__")
            plan = json.loads(output.getvalue())
        self.assertEqual(plan["policy_hash"], policy_hash())
        self.assertFalse(plan["training_authorized"])
        self.assertFalse(self.root.exists())
        for name in ("", ".", "..", "../old", "/tmp/other", "old\\study"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                preparation.campaign_plan(name)

    def test_preparation_failure_is_retained_and_cannot_be_overwritten(self):
        original = campaign.prepare_study
        def fail(root, model_id):
            if model_id == "maxvit_t":
                raise RuntimeError("synthetic preflight failure")
            return original(root, model_id)
        with mock.patch.object(campaign, "prepare_study", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "synthetic preflight failure"):
                preparation.prepare_campaign(self.name)
        failure = json.loads((self.root / "preparation_failure.json").read_text())
        self.assertFalse(failure["training_started"])
        self.assertEqual(len(failure["completed_studies"]), 3)
        with self.assertRaises(FileExistsError):
            preparation.prepare_campaign(self.name)


if __name__ == "__main__":
    unittest.main()

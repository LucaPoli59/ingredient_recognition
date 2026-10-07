import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "settings"):
    sys.path.insert(0, str(path))

import torch

from src.commons.config_enc_dec import encode_config
from src.commons.exp_config import ExpConfig
from src.models.resnet import Resnet18
from src.models.dinov2 import DinoV2B14
from src.lightning.lgn_models import BaseWithSchedulerLGNM
from src.training import selected_vocab as launcher


class SelectedVocabularyLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        presets = copy.deepcopy(launcher.PRESETS)
        for family, model in (("resnet", Resnet18), ("dinov2", DinoV2B14)):
            config = ExpConfig(tm_type=model, tm_num_classes=165, tm_pretrained=True,
                               lgn_model_type=BaseWithSchedulerLGNM, batch_size=128,
                               lr=3e-5 if family == "resnet" else 3.2e-4, weighted_loss=False,
                               optimizer=torch.optim.AdamW, weight_decay=1e-4,
                               hp_lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                               hp_lr_scheduler_params={"patience": 3},
                               dm_category="all", tr_max_epochs=40)
            if family == "dinov2":
                config.update_config(tm_freeze_backbone=True)
            path = self.root / presets[family]["source"]
            path.parent.mkdir(parents=True)
            config.save_to_file(path)
            presets[family]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.addCleanup(patch.stopall)
        patch.object(launcher, "PRESETS", presets).start()
        patch.object(launcher, "PROJECT_PATH", str(self.root)).start()
        patch.object(launcher, "EXPERIMENTS_PATH", str(self.root / "experiments")).start()
        patch.object(launcher, "YUMMLY_PATH", str(self.root / "data")).start()

    def test_both_presets_keep_full_hyperparameters_and_selected_identity(self):
        for family in launcher.PRESETS:
            config, contract = launcher.build_config(family)
            source = ExpConfig.load_from_file(self.root / launcher.PRESETS[family]["source"])
            for key in ("batch_size", "lr", "optimizer", "weight_decay", "momentum", "weighted_loss",
                        "lr_scheduler", "lr_scheduler_params", "use_swa"):
                self.assertEqual(config.hp[key], source.hp[key])
            self.assertEqual(config.torch_model["num_classes"], 59)
            self.assertEqual(config.trainer["max_epochs"], 40)
            self.assertEqual(config.datamodule["ingredient_projection"]["class_order"],
                             config.label_encoder["classes"])
            self.assertTrue(config.hp["log_per_ingredient_metrics"])
            self.assertEqual(contract["physical_batch"] * contract["accumulate_grad_batches"], 128)
        config, _ = launcher.build_config("dinov2")
        self.assertTrue(config.torch_model["freeze_backbone"])

    def test_config_round_trip_preserves_selected_order(self):
        config, _ = launcher.build_config("resnet")
        path = self.root / "roundtrip.json"
        config.save_to_file(path)
        restored = ExpConfig.load_from_file(path)
        self.assertEqual(encode_config(config.config), encode_config(restored.config))

    def test_rejects_changed_historical_source(self):
        path = self.root / launcher.PRESETS["resnet"]["source"]
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "source configuration"):
            launcher.build_config("resnet")

    def test_rejects_bad_names_workers_and_inexact_batches(self):
        for kwargs in ({"run_name": "../old"}, {"run_name": ""}, {"workers": -1},
                       {"physical_batch": 0}, {"physical_batch": 48}, {"physical_batch": 256}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                launcher.build_config("resnet", **kwargs)

    def test_dry_run_does_not_construct_models_prepare_data_or_train(self):
        with patch.object(launcher, "run_selected") as train, patch.object(launcher, "load_datamodule") as data:
            with patch("builtins.print"):
                launcher.main("dinov2", ["--dry-run"])
            train.assert_not_called()
            data.assert_not_called()
        self.assertFalse((self.root / "experiments" / "selected_v5").exists())

    def test_checkpoint_contract_and_batch_guards(self):
        config, contract = launcher.build_config("dinov2")
        callback = launcher.LaunchContract(contract)
        checkpoint = {}
        callback.on_save_checkpoint(None, None, checkpoint)
        callback.on_load_checkpoint(None, None, checkpoint)
        callback.on_fit_start(SimpleNamespace(accumulate_grad_batches=4), SimpleNamespace(batch_size=32))
        with self.assertRaises(ValueError):
            callback.on_fit_start(SimpleNamespace(accumulate_grad_batches=1), SimpleNamespace(batch_size=32))
        checkpoint["selected_vocab_launch"] = {"wrong": True}
        with self.assertRaises(ValueError):
            callback.on_load_checkpoint(None, None, checkpoint)

    def test_actual_legacy_lightning_batch_arithmetic_without_pretrained_download(self):
        for family in ("resnet", "dinov2"):
            config, contract = launcher.build_config(family, physical_batch=16)
            config.torch_model["pretrained"] = False
            hub_model = torch.nn.Module()
            hub_model.backbone = torch.nn.Linear(3, 4)
            hub_model.linear_head = torch.nn.Linear(4, 1000)
            model_type = config.torch_model["type"]
            with patch.object(torch.hub, "load", return_value=hub_model), \
                    patch.object(model_type, "max_allowed_batch_size", property(lambda model: 16)):
                module = config.hp["lgn_model_type"].load_from_config(config.hp)
            self.assertEqual(module.batch_size, 16)
            self.assertEqual(module.grad_accum, 8)
            self.assertEqual(module.num_classes, 59)
            launcher.LaunchContract(contract).on_fit_start(
                SimpleNamespace(accumulate_grad_batches=module.grad_accum), module)
            if family == "dinov2":
                self.assertFalse(any(p.requires_grad for p in hub_model.backbone.parameters()))
                self.assertTrue(all(p.requires_grad for p in hub_model.linear_head.parameters()))

    def test_resume_rejects_changed_contract_or_foreign_checkpoint(self):
        config, contract = launcher.build_config("resnet")
        checkpoint = {key: value[1] for key, value in encode_config(config.config).items()} | {
            "ingredient_projection": config.datamodule["ingredient_projection"],
            "selected_vocab_launch": contract,
        }
        manifest = json.loads(json.dumps({"contract": contract, "contract_hash": launcher._digest(contract),
                                          "status": "interrupted"}))
        launcher.validate_resume(config, contract, manifest, checkpoint)
        for bad in (manifest | {"contract_hash": "wrong"}, manifest | {"status": "completed"}):
            with self.assertRaises(ValueError):
                launcher.validate_resume(config, contract, bad, checkpoint)
        with self.assertRaises(ValueError):
            launcher.validate_resume(config, contract, manifest, checkpoint | {"selected_vocab_launch": None})
        full = ExpConfig(tm_type=Resnet18, tm_num_classes=165)
        with self.assertRaises(ValueError):
            launcher.validate_resume(config, contract, manifest,
                                     {key: value[1] for key, value in encode_config(full.config).items()} | {
                "ingredient_projection": None, "selected_vocab_launch": contract})

    def test_existing_run_refused_without_training(self):
        config, contract = launcher.build_config("resnet")
        Path(config.trainer["save_dir"]).parent.mkdir(parents=True)
        with patch.object(launcher, "model_training") as train, self.assertRaises(FileExistsError):
            launcher.run_selected(config, contract)
        train.assert_not_called()

    def test_mock_launch_uses_canonical_api_no_source_weights_and_restores_cap(self):
        # Runtime dependency inventory is read-only; use the actual source tree for it.
        config, contract = launcher.build_config("dinov2", workers=0)
        dm = SimpleNamespace(label_encoder=SimpleNamespace(to_config=lambda: config.label_encoder),
                             test_dataloader=lambda: None, predict_dataloader=lambda: None)
        original_property = DinoV2B14.max_allowed_batch_size

        def train(cfg, data, **kwargs):
            self.assertIs(cfg, config)
            self.assertIs(data, dm)
            self.assertIsNone(kwargs["ckpt_path"])
            self.assertEqual(DinoV2B14.max_allowed_batch_size.fget(None), 32)
            self.assertEqual(kwargs["trainer_kwargs"]["precision"], "16-mixed")
            with self.assertRaises(RuntimeError):
                dm.test_dataloader()
            return SimpleNamespace(global_step=40), object()

        with patch.object(launcher, "PROJECT_PATH", str(ROOT)), patch.object(launcher, "load_datamodule", return_value=dm), \
                patch.object(launcher, "model_training", side_effect=train), \
                patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_device_name", return_value="mock GPU"):
            launcher.run_selected(config, contract)
        self.assertIs(DinoV2B14.max_allowed_batch_size, original_property)
        manifest = json.loads((Path(config.trainer["save_dir"]).parent / "launch_manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["attempts"][0]["optimizer_updates"], 40)
        self.assertTrue(manifest["runtime_source_sha256"])


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.commons.exp_config import ExpConfig
from src.lightning.lgn_models import BaseLGNM, BaseWithSchedulerLGNM
from src.models.commons import BaseModel


class TinyModel(BaseModel):
    PRETTY_NAME = "Tiny"

    def __init__(self, num_classes=2, input_shape=(2, 2), **kwargs):
        super().__init__(num_classes=num_classes, input_shape=input_shape, **kwargs)
        self.linear = torch.nn.Linear(2, num_classes)

    def forward(self, values):
        return self.linear(values)

    @property
    def conv_target_layer(self):
        return self.linear

    @property
    def classifier_target_layer(self):
        return self.linear


def build_module(enabled, hparams_to_register=None):
    return BaseLGNM(
        TinyModel(),
        lr=0.01,
        batch_size=2,
        optimizer=torch.optim.SGD,
        loss_fn=torch.nn.BCEWithLogitsLoss,
        metrics={},
        hparams_to_register=hparams_to_register,
        log_per_ingredient_metrics=enabled,
    )


class PerIngredientLoggingTests(unittest.TestCase):
    def test_configuration_defaults_to_disabled_and_roundtrips_opt_in(self):
        self.assertFalse(ExpConfig().lgn_model["log_per_ingredient_metrics"])

        configured = ExpConfig(hp_log_per_ingredient_metrics=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            configured.save_to_file(path)
            restored = ExpConfig.load_from_file(path)

        self.assertTrue(restored.lgn_model["log_per_ingredient_metrics"])

    def test_disabled_mode_allocates_no_metric_state(self):
        module = build_module(False)

        self.assertIsNone(module.train_per_ingredient_metrics)
        self.assertIsNone(module.val_per_ingredient_metrics)
        self.assertIsNone(module.test_per_ingredient_metrics)

    def test_enabled_metrics_compute_over_the_full_epoch(self):
        module = build_module(True)
        metrics = module.val_per_ingredient_metrics

        metrics.update(
            torch.tensor([[10.0, -10.0], [10.0, 10.0]]),
            torch.tensor([[1, 1], [0, 1]]),
        )
        metrics.update(
            torch.tensor([[-10.0, -10.0]]),
            torch.tensor([[1, 0]]),
        )
        values = metrics.compute()

        self.assertAlmostEqual(values["precision"][0].item(), 0.5)
        self.assertAlmostEqual(values["recall"][0].item(), 0.5)
        self.assertAlmostEqual(values["f1"][0].item(), 0.5)
        self.assertAlmostEqual(values["precision"][1].item(), 1.0)
        self.assertAlmostEqual(values["recall"][1].item(), 0.5)
        self.assertAlmostEqual(values["f1"][1].item(), 2.0 / 3.0, places=6)

    def test_epoch_end_logs_stable_indexed_keys_and_resets_state(self):
        module = build_module(True)
        metrics = module.val_per_ingredient_metrics
        metrics.update(torch.tensor([[10.0, -10.0]]), torch.tensor([[1, 0]]))

        with patch.object(module, "log") as log:
            module._log_per_ingredient_metrics("val", metrics)

        self.assertEqual(log.call_count, 6)
        logged_names = {call.args[0] for call in log.call_args_list}
        self.assertEqual(
            logged_names,
            {
                "val_per_ingredient/precision/0",
                "val_per_ingredient/precision/1",
                "val_per_ingredient/recall/0",
                "val_per_ingredient/recall/1",
                "val_per_ingredient/f1/0",
                "val_per_ingredient/f1/1",
            },
        )
        self.assertTrue(all(metric._update_count == 0 for metric in metrics.values()))

    def test_empty_epoch_does_not_log_synthetic_zeroes(self):
        module = build_module(True)

        with patch.object(module, "log") as log:
            module.on_validation_epoch_end()

        log.assert_not_called()

    def test_scheduler_subclass_forwards_logging_flag(self):
        module = BaseWithSchedulerLGNM(
            TinyModel(),
            lr=0.01,
            batch_size=2,
            optimizer=torch.optim.SGD,
            loss_fn=torch.nn.BCEWithLogitsLoss,
            metrics={},
            log_per_ingredient_metrics=True,
        )

        self.assertTrue(module.log_per_ingredient_metrics)
        self.assertIsNotNone(module.val_per_ingredient_metrics)

    def test_legacy_model_config_defaults_logging_to_false(self):
        torch_model = TinyModel()
        config = {
            "lgn_model_type": BaseLGNM,
            "batch_size": 2,
            "lr": 0.01,
            "loss_fn": torch.nn.BCEWithLogitsLoss,
            "metrics": {},
            "optimizer": torch.optim.SGD,
            "torch_model": torch_model.to_config(),
        }

        restored = BaseLGNM.load_from_config(config)

        self.assertFalse(restored.log_per_ingredient_metrics)
        self.assertIsNone(restored.val_per_ingredient_metrics)

    def test_logging_flag_is_persisted_when_hpo_filters_hparams(self):
        module = build_module(True, hparams_to_register=["lr"])

        self.assertTrue(module.hparams["log_per_ingredient_metrics"])
        self.assertIn("lr", module.hparams)

    def test_lightning_fit_writes_epoch_level_ingredient_columns(self):
        from lightning.pytorch import Trainer
        from lightning.pytorch.loggers import CSVLogger
        from torch.utils.data import DataLoader, TensorDataset

        module = build_module(True)
        module.prepared = True
        module.loss_fn = torch.nn.BCEWithLogitsLoss()
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        dataloader = DataLoader(TensorDataset(features, targets), batch_size=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                logger=CSVLogger(temp_dir),
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                log_every_n_steps=1,
            )
            trainer.fit(module, train_dataloaders=dataloader, val_dataloaders=dataloader)
            metrics_path = next(Path(temp_dir).rglob("metrics.csv"))
            header = metrics_path.read_text(encoding="utf-8").splitlines()[0]

        self.assertIn("train_per_ingredient/f1/0", header)
        self.assertIn("val_per_ingredient/precision/1", header)


if __name__ == "__main__":
    unittest.main()

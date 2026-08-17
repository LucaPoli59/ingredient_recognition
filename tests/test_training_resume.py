import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import optuna
import torch

from src.commons.exp_config import HGeneratorConfig, _normalize_restored_trial_params
from src.lightning.lgn_trainers import BaseTrainer


def suggest_learning_rate(trial):
    return trial.suggest_float("lr", 1e-4, 1e-1, log=True)


class TrainingResumeTests(unittest.TestCase):
    def test_final_checkpoint_selection_accepts_missing_current_score(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoints_dir = Path(temp_dir) / "checkpoints"
            checkpoints_dir.mkdir()
            best_path = checkpoints_dir / "epoch=1.ckpt"
            last_path = checkpoints_dir / "last.ckpt"
            best_path.touch()
            last_path.touch()
            checkpoint_callback = SimpleNamespace(
                best_model_path=str(best_path),
                best_model_score=torch.tensor(1.0),
                current_score=None,
            )

            selected_path = BaseTrainer._select_final_checkpoint_path(
                checkpoint_callback,
                temp_dir,
            )

            self.assertEqual(selected_path, str(best_path))

    def test_final_checkpoint_selection_uses_last_for_current_best(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoints_dir = Path(temp_dir) / "checkpoints"
            checkpoints_dir.mkdir()
            best_path = checkpoints_dir / "epoch=39.ckpt"
            last_path = checkpoints_dir / "last.ckpt"
            best_path.touch()
            last_path.touch()
            checkpoint_callback = SimpleNamespace(
                best_model_path=str(best_path),
                best_model_score=torch.tensor(1.0),
                current_score=torch.tensor(1.0),
            )

            selected_path = BaseTrainer._select_final_checkpoint_path(
                checkpoint_callback,
                temp_dir,
            )

            self.assertEqual(selected_path, str(last_path))

    def test_hgenerator_reload_preserves_optuna_parameter_name(self):
        generator = HGeneratorConfig(hp_lr=suggest_learning_rate)
        distributions = {
            "lr": optuna.distributions.FloatDistribution(1e-4, 1e-1, log=True),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "hparam_gen_config.json"
            generator.save_to_file(config_path)
            restored_generator = HGeneratorConfig.load_from_file_with_dist(
                config_path,
                distributions,
            )

        trial = optuna.trial.FixedTrial({"lr": 0.01})
        generated_hparams = restored_generator.generate_hparams_on_trial(trial)

        self.assertEqual(generated_hparams["hp_lr"], 0.01)
        self.assertEqual(trial.params, {"lr": 0.01})

    def test_failed_trial_parameter_names_are_normalized_for_resume(self):
        restored_params = _normalize_restored_trial_params(
            {
                "hp_lr": 0.01,
                "hp_optimizer": "sgd",
                "tm_type": "resnet18Like",
                "tm_trns_aug": "no_aug",
            },
            {"lr", "optimizer", "tm_type", "trns_aug"},
        )

        self.assertEqual(
            restored_params,
            {
                "lr": 0.01,
                "optimizer": "sgd",
                "tm_type": "resnet18Like",
                "trns_aug": "no_aug",
            },
        )


if __name__ == "__main__":
    unittest.main()

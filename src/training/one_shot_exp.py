import os
import warnings
from typing import Tuple, Type, Optional, Dict, Any
import lightning as lgn
import torch

from settings.config import (EXPERIMENTS_PATH, DEF_BATCH_SIZE, DEF_LR)
from src.lightning.lgn_models import BaseLGNM
from src.lightning.lgn_trainers import TrainerInterface, BaseFasterTrainer
from src.models.dummy import DummyModel
from src.training.exp_config import ExpConfig
from src.training.commons import set_torch_constants, model_training, load_datamodule


def make_one_shot_exp(
        experiment_name: str,
        experiment_dir: str | None = None,
        max_epochs: int = 20,
        batch_size: Optional[int] = DEF_BATCH_SIZE,
        debug: bool = False,
        **config_kwargs
) -> Tuple[lgn.Trainer, lgn.LightningModule]:
    """Function that creates a one-shot experiment with the given configuration and run it. If the experiment is
    resumable, it will resume the last trial.

    Note: Other configuration parameters must be passed as keyword arguments, by following the design pattern
    of the ExpConfig class.
    """

    save_dir, to_resume = _setup_or_resume_dir(experiment_dir, experiment_name)
    if to_resume:
        return _resume_exp(str(os.path.join(save_dir, "checkpoints", "last.ckpt")))

    exp_config = ExpConfig(**config_kwargs)
    _assert_lgn_model_trainer_compatibility(exp_config.model["lgn_model_type"], exp_config.trainer["type"])
    exp_config.update_config(tr_save_dir=save_dir, tr_debug=debug, tr_max_epochs=max_epochs, batch_size=batch_size)
    return _run_new_exp(exp_config)


def _setup_or_resume_dir(experiment_dir: str | os.PathLike, experiment_name: str | os.PathLike
                         ) -> Tuple[str | os.PathLike, bool]:
    """
    Function that search if a resumable experiment is present or create a new one-shot experiment directory,
    returning the path of the last trial directory and a boolean that indicates if the experiment is resumable.
    """

    if not os.path.exists(os.path.join(experiment_dir, experiment_name)):
        os.makedirs(os.path.join(experiment_dir, experiment_name))
        return os.path.join(experiment_dir, experiment_name, f"trial_0"), False

    last_trial = _find_last_trial(experiment_dir, experiment_name)
    if last_trial == -1:  # No trials found
        return os.path.join(experiment_dir, experiment_name, f"trial_0"), False

    last_trial_path = os.path.join(experiment_dir, experiment_name, f"trial_{last_trial}")
    if os.path.exists(os.path.join(last_trial_path, "checkpoints", "last.ckpt")):
        return last_trial_path, True

    last_trial_path = os.path.join(experiment_dir, experiment_name, f"trial_{last_trial + 1}")
    return last_trial_path, False


def _find_last_trial(experiment_dir: str | os.PathLike, experiment_name: str | os.PathLike) -> int:
    """Function that search for trials in experiment_dir/experiment_name and return the last trial number"""
    exp_sub_files = os.listdir(os.path.join(experiment_dir, experiment_name))
    exp_vers = [file_name for file_name in exp_sub_files if "trial" in file_name]
    return len(exp_vers) - 1


def _assert_lgn_model_trainer_compatibility(model: Type[lgn.LightningModule], trainer: Type[TrainerInterface]):
    """Function that asserts if the model and the trainer are compatible with the training function."""
    if not issubclass(model, BaseLGNM):
        raise ValueError(f"Model must be a subclass of BaseLightning, got {model}")
    if not issubclass(trainer, TrainerInterface):
        raise ValueError(f"Trainer must be a subclass of TrainerInterface, got {trainer}")


def _run_new_exp(exp_config: ExpConfig) -> Tuple[lgn.Trainer, lgn.LightningModule]:
    """Function that runs a new experiment with the given configuration."""
    set_torch_constants()

    # Load the dataset
    data_module = load_datamodule(exp_config)
    exp_config.update_config(dm_label_encoder=data_module.label_encoder.to_config(),
                             num_classes=data_module.get_num_classes())

    return model_training(exp_config, data_module)


def _resume_exp(ckpt_path: str | os.PathLike) -> Tuple[lgn.Trainer, lgn.LightningModule]:
    """Function that resumes the experiment from the given checkpoint path."""
    set_torch_constants()
    warnings.filterwarnings("ignore", "Checkpoint directory .*. exists and is not empty.")
    checkpoint_data: Dict[str, Any] = torch.load(ckpt_path)
    return model_training(ExpConfig.load_from_ckpt_data(checkpoint_data), ckpt_path=ckpt_path)




if __name__ == "__main__":
    exp_dir, exp_name = os.path.join(EXPERIMENTS_PATH, "dummy"), "dummy_experiment"
    trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=20, batch_size=128, debug=False,
                                       torch_model_type=DummyModel, dm_category="all", tr_type=BaseFasterTrainer,
                                       lr=DEF_LR)



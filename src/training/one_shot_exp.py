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
from src.training.utils import decode_config, set_torch_constants


def make_one_shot_exp(
        experiment_name: str,
        experiment_dir: str | None = None,
        max_epochs: int = 20,
        batch_size: Optional[int] = DEF_BATCH_SIZE,
        debug: bool = False,
        **config_kwargs
) -> Tuple[Type[lgn.Trainer], Type[lgn.LightningModule]]:
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
    if not issubclass(model, BaseLGNM):
        raise ValueError(f"Model must be a subclass of BaseLightning, got {model}")
    if not issubclass(trainer, TrainerInterface):
        raise ValueError(f"Trainer must be a subclass of TrainerInterface, got {trainer}")


def _run_new_exp(exp_config: ExpConfig) -> Tuple[Type[lgn.Trainer], Type[lgn.LightningModule]]:
    set_torch_constants()

    # Load the dataset
    dm_config = exp_config.datamodule
    dm_type = dm_config["type"]
    data_module = dm_type.load_from_config(dm_config, image_size=exp_config.model["input_shape"],
                                           batch_size=exp_config.model["batch_size"])
    data_module.prepare_data()
    data_module.setup()
    exp_config.update_config(dm_label_encoder=data_module.label_encoder.to_config(),
                             num_classes=data_module.get_num_classes())

    return model_training(exp_config.config, data_module)


def _resume_exp(ckpt_path: str | os.PathLike) -> Tuple[Type[lgn.Trainer], Type[lgn.LightningModule]]:
    set_torch_constants()
    warnings.filterwarnings("ignore", "Checkpoint directory .*. exists and is not empty.")
    checkpoint_data: Dict[str, Any] = torch.load(ckpt_path)
    return model_training(ExpConfig.load_from_ckpt_data(checkpoint_data), ckpt_path=ckpt_path)


def model_training(exp_config: ExpConfig, data_module: Optional[Type[lgn.LightningDataModule]] = None,
                   ckpt_path: Optional[str | os.PathLike] = None
                   ) -> Tuple[Type[lgn.Trainer], Type[lgn.LightningModule]]:
    resuming = ckpt_path is None
    model_config, trainer_config = exp_config.model, exp_config.trainer

    lgn_model = model_config['lgn_model_type'].load_from_config(model_config)
    trainer = trainer_config['type'].load_from_config(trainer_config)

    if data_module is None:
        dm_config = exp_config.datamodule
        dm_type = dm_config['type']
        data_module = dm_type.load_from_config(dm_config, image_size=lgn_model.input_shape,
                                               batch_size=lgn_model.batch_size)

    debug = trainer.debug or True  # TODO: Remove or True (when the module is ready)
    if debug:
        print("Data Module, Models and Trainer loaded, " + ("training started" if resuming else "resume training"))

    trained_model = trainer.fit(
        model=lgn_model,
        datamodule=data_module,
        ckpt_path=ckpt_path
    )

    if debug:
        print("Training completed")
    return trainer, trained_model


if __name__ == "__main__":
    exp_dir, exp_name = os.path.join(EXPERIMENTS_PATH, "dummy"), "dummy_experiment"
    make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=5, batch_size=256, debug=True,
                      torch_model_type=DummyModel, dm_category="mexican", tr_type=BaseFasterTrainer, lr=DEF_LR)

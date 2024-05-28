import logging
import os
from typing import Tuple, Dict, Any, List, Set, Optional

import optuna
import torch
import lightning as lgn

from src.data_processing.data_handling import ImagesRecipesDataModule
from src.lightning.lgn_trainers import TrainerInterface
from src.training.exp_config import ExpConfig, HTunerExpConfig


def extract_name_trial_dir(save_dir: str) -> Tuple[str, str, str]:
    """Function that extract the path of the experiments directory, the experiment name and the experiment trial
    from the save_dir.  Example: save_dir = "experiments/food_classification/mexican/1"
    -> ("experiments/food_classification", "mexican", "1")"""
    exp_vers, exp_name = save_dir, os.path.dirname(save_dir)
    exps_dir = os.path.dirname(exp_name)
    return exps_dir, os.path.split(exp_name)[1], os.path.split(exp_vers)[1]


def register_hparams(elem: lgn.LightningModule | lgn.LightningDataModule,
                     hparams: List[Dict[str, Any] | str] | Set[Dict[str, Any] | str], log=True) -> None:
    """Function that register the hyperparameters to the elem """
    param_list = []
    param_dicts = []
    for param in hparams:
        if isinstance(param, dict):
            param_dicts.append(param)
        else:
            param_list.append(param)

    if len(param_list) > 0:
        elem.save_hyperparameters(*param_list, logger=log)
    for param_dict in param_dicts:
        elem.save_hyperparameters(param_dict, logger=log)


def set_torch_constants():
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # to remove warning messages
    torch.set_float32_matmul_precision('medium')  # For better performance with cuda
    torch.backends.cudnn.benchmark = True


def model_training(exp_config: ExpConfig, data_module: Optional[lgn.LightningDataModule] = None,
                   ckpt_path: Optional[str | os.PathLike] = None, torch_model_kwargs: Optional[Dict[str, Any]] = None,
                   lgn_model_kwargs: Optional[Dict[str, Any]] = None, trainer_kwargs: Optional[Dict[str, Any]] = None
                   ) -> Tuple[TrainerInterface, lgn.LightningModule]:
    """General function that trains a model with the given configuration."""
    if torch_model_kwargs is None:
        torch_model_kwargs = {}
    if lgn_model_kwargs is None:
        lgn_model_kwargs = {}
    if trainer_kwargs is None:
        trainer_kwargs = {}

    resuming = ckpt_path is None
    model_config, trainer_config = exp_config.model, exp_config.trainer

    lgn_model = model_config['lgn_model_type'].load_from_config(model_config, lgn_model_kwargs=lgn_model_kwargs,
                                                                torch_model_kwargs=torch_model_kwargs)
    trainer = trainer_config['type'].load_from_config(trainer_config, **trainer_kwargs)

    if data_module is None:
        dm_config = exp_config.datamodule
        dm_type = dm_config['type']
        data_module = dm_type.load_from_config(dm_config, image_size=lgn_model.input_shape,
                                               batch_size=lgn_model.batch_size)

    if trainer.debug:
        print("Data Module, Models and Trainer loaded, " + ("training started" if resuming else "resume training"))

    trained_model = trainer.fit(
        model=lgn_model,
        datamodule=data_module,
        ckpt_path=ckpt_path
    )

    if trainer.debug:
        print("Training completed")

    return trainer, trained_model


def load_datamodule(exp_config: ExpConfig | HTunerExpConfig) -> ImagesRecipesDataModule:
    dm_config = exp_config.datamodule
    dm_type = dm_config["type"]
    data_module = dm_type.load_from_config(dm_config, image_size=exp_config.model["input_shape"],
                                           batch_size=exp_config.model["batch_size"])
    data_module.prepare_data()
    data_module.setup()
    return data_module


def init_optuna_storage(path: os.PathLike | str) -> optuna.storages.JournalStorage:
    lock_file = optuna.storages.JournalFileOpenLock(path)
    return optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path=path, lock_obj=lock_file)
    )

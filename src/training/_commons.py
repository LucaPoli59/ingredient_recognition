import logging
import os
from typing import Tuple, Dict, Any, Optional

import optuna
import torch
import lightning as lgn

from settings.config import OPTUNA_JOURNAL_PATH
from src.commons.exp_config import ExpConfig, HTunerExpConfig
from src.data_processing.data_handling import ImagesRecipesDataModule


def set_torch_constants():
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # to remove warning messages
    torch.set_float32_matmul_precision('medium')  # For better performance with cuda
    torch.backends.cudnn.benchmark = True


def model_training(exp_config: ExpConfig, data_module: Optional[lgn.LightningDataModule] = None,
                   ckpt_path: Optional[str | os.PathLike] = None, torch_model_kwargs: Optional[Dict[str, Any]] = None,
                   lgn_model_kwargs: Optional[Dict[str, Any]] = None, trainer_kwargs: Optional[Dict[str, Any]] = None
                   ) -> Tuple[lgn.Trainer, lgn.LightningModule]:
    """General function that trains a model with the given configuration."""
    if torch_model_kwargs is None:
        torch_model_kwargs = {}
    if lgn_model_kwargs is None:
        lgn_model_kwargs = {}
    if trainer_kwargs is None:
        trainer_kwargs = {}

    resuming = ckpt_path is None
    model_config, trainer_config = exp_config.lgn_model, exp_config.trainer

    lgn_model = model_config['lgn_model_type'].load_from_config(model_config, lgn_model_kwargs=lgn_model_kwargs,
                                                                torch_model_kwargs=torch_model_kwargs)
    trainer = trainer_config['type'].load_from_config(trainer_config, **trainer_kwargs)

    if data_module is None:
        dm_config = exp_config.datamodule
        data_module = dm_config['type'].load_from_config(dm_config, batch_size=lgn_model.batch_size)

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


def init_optuna_storage(path: Optional[os.PathLike | str] = None) -> optuna.storages.JournalStorage:
    if path is None:
        path = OPTUNA_JOURNAL_PATH
    lock_file = optuna.storages.JournalFileOpenLock(path)
    return optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path=path, lock_obj=lock_file)
    )


def load_datamodule(exp_config: ExpConfig | HTunerExpConfig) -> ImagesRecipesDataModule:
    dm_config = exp_config.datamodule
    dm_type = dm_config["type"]
    data_module = dm_type.load_from_config(dm_config, batch_size=exp_config.lgn_model["batch_size"])
    data_module.prepare_data()
    data_module.setup()
    return data_module

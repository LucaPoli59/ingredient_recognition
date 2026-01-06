import copy
import json
import os
import random
import shutil
import warnings
from typing import Tuple, Type, Optional, List

import lightning as lgn
import numpy as np
import optuna
import torch
from tornado.gen import sleep

from config import HTUNING_TRIAL_CONFIG_FILE, HGEN_CONFIG_FILE
from src.dashboards.start_optuna import start_optuna

from src.commons.utils import extract_name_trial_dir
from settings.config import (EXPERIMENTS_PATH, DEF_BATCH_SIZE, OPTUNA_JOURNAL_FILENAME, HTUNER_CONFIG_FILE,
                             OPTUNA_JOURNAL_PATH, HTUNING_TRIAL_CONFIG_FILE)
from src.training.commons import set_torch_constants, model_training, init_optuna_storage, load_datamodule

from src.data_processing.images_recipes import ImagesRecipesBaseDataModule
from src.lightning.lgn_trainers import TrainerInterface, OptunaTrainer
from src.models.dummy import DummyModel
from src.commons.exp_config import ExpConfig, HGeneratorConfig, HTunerExpConfig


def silence_optuna_warnings():
    warnings.filterwarnings("ignore", "ExperimentalWarning: JournalStorage is experimental "
                                      "(supported from v3.1.0). The interface can change in the future.")

def make_htuning_exp(
        experiment_name: str,
        hgen_config: HGeneratorConfig,
        exp_config: Optional[HTunerExpConfig] = None,
        experiment_dir: str | None = None,
        max_epochs: int = 20,
        batch_size: Optional[int] = DEF_BATCH_SIZE,
        debug: bool = False,
        **config_kwargs
) -> Tuple[lgn.Trainer, lgn.LightningModule]:
    """Function that creates an experiment with hyperparameters tuning with the given configuration and run it.
    If the experiment is resumable, it will resume the last trial.

    The configuration can be passed as an ExpConfig object or as keyword arguments. If both are passed, the keyword
    arguments will be used to update the ExpConfig object.

    Note: Other configuration parameters must be passed as keyword arguments, by following the design pattern
    of the ExpConfig class.
    """
    warnings.filterwarnings("ignore", "Checkpoint directory .*. exists and is not empty.")
    if experiment_dir is None:
        experiment_dir = EXPERIMENTS_PATH

    if not debug:
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    set_torch_constants()

    save_dir, to_resume = _setup_or_resume_dir(experiment_dir, experiment_name)
    if to_resume:
        return _resume_exp(save_dir)

    if exp_config is None:
        exp_config = HTunerExpConfig(**config_kwargs)
    else:
        exp_config.update_config(**config_kwargs)

    _assert_lgn_model_trainer_compatibility(exp_config.lgn_model["lgn_model_type"], exp_config.trainer["type"])
    exp_config.update_config(tr_save_dir=save_dir, tr_debug=debug, tr_max_epochs=max_epochs, batch_size=batch_size,
                             ht_save_dir=save_dir)
    return _run_new_exp(exp_config, hgen_config)


def _setup_or_resume_dir(experiment_dir: str | os.PathLike, experiment_name: str | os.PathLike,
                         clear_inconsistent_state: bool = True
                         ) -> Tuple[str | os.PathLike, bool]:
    if not os.path.exists(os.path.join(experiment_dir, experiment_name)):
        os.makedirs(os.path.join(experiment_dir, experiment_name))
        return os.path.join(experiment_dir, experiment_name), False

    if not os.path.exists(OPTUNA_JOURNAL_PATH):
        if clear_inconsistent_state:
            shutil.rmtree(os.path.join(experiment_dir, experiment_name))
            os.makedirs(os.path.join(experiment_dir, experiment_name))
            return os.path.join(experiment_dir, experiment_name), False
        else:
            raise ValueError("Inconsistent state: experiment directory exists but no journal file found.")

    storage = init_optuna_storage()
    studies = optuna.get_all_study_names(storage)
    return os.path.join(experiment_dir, experiment_name), f"{experiment_dir}/{experiment_name}" in studies


def _restore_study(study_name: str, storage: optuna.storages.BaseStorage,
                   states_error: List[optuna.trial.TrialState], **study_kwargs
                   ) -> Tuple[optuna.study.Study, int]:
    """Function that uses an existing study to create a new one with the valid trials already saved, and the
    invalid ones re-enqueued."""
    old_study = optuna.load_study(study_name=study_name, storage=storage)
    old_trials = old_study.get_trials(deepcopy=True)
    optuna.delete_study(study_name=study_name, storage=storage)

    new_study = optuna.create_study(study_name=study_name, storage=storage, **study_kwargs)
    trials_completed = 0
    for trial in old_trials:
        if trial.state in states_error:
            new_study.enqueue_trial(trial.params)
        else:
            new_study.add_trial(trial)
            trials_completed += 1

    return new_study, trials_completed


def _resume_exp(save_dir: str | os.PathLike) -> Tuple[lgn.Trainer, lgn.LightningModule]:
    """Function that resumes the experiment from the given checkpoint path."""

    exp_config = HTunerExpConfig.load_from_file(os.path.join(save_dir, HTUNER_CONFIG_FILE))
    debug = exp_config.trainer['debug']

    storage = init_optuna_storage()
    exp_dir, exp_name, _ = extract_name_trial_dir(os.path.join(save_dir, "_"))
    study_name = f"{exp_dir}/{exp_name}"

    htuner_config = exp_config.htuner
    sampler = htuner_config["sampler"](**htuner_config["sampler_kwargs"])
    pruner = htuner_config["pruner"](**htuner_config["pruner_kwargs"])

    study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)
    if len(study.trials) == 0:
        raise ValueError("No trials found in the study.")

    exp_gen_config = HGeneratorConfig.load_from_file_with_dist(os.path.join(save_dir, HGEN_CONFIG_FILE),
                                                               study.trials[0].distributions)  # probabilmente sta roba non funziona
    data_module = load_datamodule(exp_config)
    exp_config.drop("lb")

    study, trials_completed = _restore_study(
        study_name, storage, [optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING],
        sampler=sampler, pruner=pruner)

    if debug:
        print("Resuming Optimization...")

    study.optimize(
        lambda trial: _objective_wrapper(trial, exp_config, data_module, exp_gen_config, check_for_resume=True),
        n_trials=htuner_config["n_trials"] - trials_completed
    )

    return save_best_trial(study, exp_config.trainer["save_dir"], exp_config=exp_config)


def _prepare_trial_dir(trial_path: str | os.PathLike, check_for_resume: bool) -> str | os.PathLike | None:
    if os.path.exists(trial_path) and os.path.exists(os.path.join(trial_path, "checkpoints", "last.ckpt")):
        if check_for_resume:
            return os.path.join(trial_path, "checkpoints", "last.ckpt")
        else:
            shutil.rmtree(trial_path)

    if not os.path.exists(trial_path):
        os.mkdir(trial_path)
    return None


def _assert_lgn_model_trainer_compatibility(model: Type[lgn.LightningModule], trainer: Type[TrainerInterface]):
    """Function that asserts if the model and the trainer are compatible with the training function."""
    if not issubclass(trainer, OptunaTrainer):
        raise ValueError(f"Model must be a subclass of OptunaTrainer, got {model}")


def _run_new_exp(exp_config: HTunerExpConfig, exp_gen_config: HGeneratorConfig
                 ) -> Tuple[lgn.Trainer, lgn.LightningModule]:
    """Function that runs a new experiment with the given configuration."""

    # Load the dataset
    data_module = load_datamodule(exp_config)
    exp_config.update_config(dm_label_encoder=data_module.label_encoder.to_config(),
                             tm_num_classes=data_module.get_num_classes())

    # Save the configuration to file
    exp_config.save_to_file(str(os.path.join(exp_config.trainer["save_dir"], HTUNER_CONFIG_FILE)))
    exp_config.drop("lb")  # From this point is useless to carry the label encoder in the config
    exp_gen_config.save_to_file(str(os.path.join(exp_config.trainer["save_dir"], HGEN_CONFIG_FILE)))

    htuner_config = exp_config.htuner
    sampler = htuner_config["sampler"](**htuner_config["sampler_kwargs"])
    pruner = htuner_config["pruner"](**htuner_config["pruner_kwargs"])
    direction = htuner_config["direction"]

    storage = init_optuna_storage()
    exp_dir, exp_name, _ = extract_name_trial_dir(os.path.join(exp_config.trainer["save_dir"], "_"))
    study_name = f"{exp_dir}/{exp_name}"
    study = optuna.create_study(sampler=sampler, direction=direction, study_name=study_name,
                                storage=storage, pruner=pruner)

    study.set_metric_names(["val_loss"])

    if exp_config.trainer['debug']:
        print("Starting Optimization...")

    study.optimize(
        lambda trial: _objective_wrapper(trial, exp_config, data_module, exp_gen_config, check_for_resume=False),
        n_trials=htuner_config["n_trials"]
    )

    return save_best_trial(study, exp_config.trainer["save_dir"], exp_config=exp_config)


def save_best_trial(study: optuna.study.Study, save_dir: str | os.PathLike, exp_config: Optional[HTunerExpConfig],
                    ) -> Tuple[TrainerInterface, lgn.LightningModule]:
    best_trial_path_in = os.path.join(save_dir, f"trial_{study.best_trial.number}")
    best_trial_path_out = os.path.join(save_dir, "trial_best")
    shutil.copytree(best_trial_path_in, best_trial_path_out)

    if exp_config is None:
        exp_config = HTunerExpConfig.load_from_file(
            file_path=os.path.join(best_trial_path_out, HTUNER_CONFIG_FILE))  # Questa parte andrà solo dopo il resume

    trainer = exp_config.trainer["type"].load_from_config(exp_config.trainer, trial=study.best_trial)
    model = exp_config.lgn_model["lgn_model_type"].load_from_config(exp_config.lgn_model)

    model = model.load_weights_from_checkpoint(os.path.join(best_trial_path_out, "best_model.ckpt"))
    return trainer, model


def _objective_wrapper(trial: optuna.Trial, exp_config: ExpConfig, data_module: ImagesRecipesBaseDataModule,
                       hparam_gen_config: HGeneratorConfig, check_for_resume: bool = False
                       ) -> float:
    trial_config = copy.deepcopy(exp_config)

    # generate the variable hyperparameters
    hparams = hparam_gen_config.generate_hparams_on_trial(trial)
    variable_hparams_names = [key for key, value in hparam_gen_config.lgn_model.items() if value is not None]

    if not exp_config.trainer['debug']: # todo remove not
        print(f"\nNew trial: {trial.number}, hparams: {hparams}\n\n")

    trial_path = os.path.join(trial_config.trainer["save_dir"], f"trial_{trial.number}")
    resume_path = _prepare_trial_dir(trial_path, check_for_resume)

    # Update the configuration with the generated hyperparameters and the trial
    trial_config.update_config(**hparams, tr_trial=trial, tr_save_dir=trial_path)
    trial_config.save_to_file(os.path.join(trial_path, HTUNING_TRIAL_CONFIG_FILE))

    # Run the experiment
    trainer, model = model_training(trial_config, data_module, ckpt_path=resume_path,
                                    lgn_model_kwargs={"hparams_to_register": variable_hparams_names})
    ckpt_callback = trainer.checkpoint_callback
    if ckpt_callback is None:
        raise ValueError("Checkpoint callback not found in the trainer.")
    ris = ckpt_callback.best_model_score

    return ris.item() if ris is not None else np.nan


if __name__ == "__main__":
    silence_optuna_warnings()
    exp_dir, exp_name = os.path.join(EXPERIMENTS_PATH, "dummy_htuning"), "dummy_experiment"
    debug = False
    random_int = random.randint(0, 100)
    exp_name = f"{exp_name}_{random_int}"
    print(exp_name)

    if debug:
        start_optuna()

    hgen_config = HGeneratorConfig(
        hp_lr=(lambda trial: trial.suggest_float("lr", 1e-5, 1e-1, log=True)),
    )

    make_htuning_exp(exp_name, hgen_config, experiment_dir=exp_dir, max_epochs=3, debug=debug,
                     torch_model_type=DummyModel, tr_type=OptunaTrainer, dm_category="mexican", hp_lr=None,
                     tr_limit_train_batches=25, ht_n_trials=3)


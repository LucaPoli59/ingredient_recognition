import os
import random
import warnings
from typing import Tuple, Type, Optional, Dict, Any
import lightning as lgn
import torch
import optuna

from settings.config import (EXPERIMENTS_PATH, DEF_BATCH_SIZE, DEF_LR)
from src.data_processing.data_handling import ImagesRecipesDataModule
from src.lightning.lgn_trainers import TrainerInterface, BaseFasterTrainer, OptunaTrainer
from src.models.dummy import DummyModel
from src.training.commons import ExpConfig, model_training, HGeneratorConfig, HTunerExpConfig, load_datamodule
from src.training.utils import set_torch_constants
from src.start_optuna import start_optuna


def make_htuning_exp(
        experiment_name: str,
        experiment_dir: str | None = None,
        max_epochs: int = 20,
        batch_size: Optional[int] = DEF_BATCH_SIZE,
        debug: bool = False,
        **config_kwargs
) -> Tuple[Type[lgn.Trainer], Type[lgn.LightningModule]]:
    """Function that creates an experiment with hyperparameters tuning with the given configuration and run it.
    If the experiment is resumable, it will resume the last trial.

    Note: Other configuration parameters must be passed as keyword arguments, by following the design pattern
    of the ExpConfig class.
    """

    save_dir, to_resume = _setup_or_resume_dir(experiment_dir, experiment_name)
    # if to_resume:
    #     return _resume_exp(str(os.path.join(save_dir, "checkpoints", "last.ckpt")))

    exp_config = HTunerExpConfig(**config_kwargs)
    _assert_lgn_model_trainer_compatibility(exp_config.model["lgn_model_type"], exp_config.trainer["type"])
    exp_config.update_config(tr_save_dir=save_dir, tr_debug=debug, tr_max_epochs=max_epochs, batch_size=batch_size,
                             ht_save_dir=os.path.dirname(save_dir))
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
    if not issubclass(model, OptunaTrainer):
        raise ValueError(f"Model must be a subclass of OptunaTrainer, got {model}")


def _run_new_exp(exp_config: HTunerExpConfig) -> Tuple[Type[lgn.Trainer], Type[lgn.LightningModule]]:
    """Function that runs a new experiment with the given configuration."""
    set_torch_constants()
    debug = exp_config.trainer['debug']

    # Load the dataset
    data_module = load_datamodule(exp_config)
    exp_config.update_config(dm_label_encoder=data_module.label_encoder.to_config(),
                             num_classes=data_module.get_num_classes())

    htuner_config = exp_config.htuner
    sampler = htuner_config["sampler"]()
    pruner = htuner_config["pruner"]()

    storage_type = htuner_config["storage"]
    if storage_type == optuna.storages.JournalStorage:
        log_file_path = os.path.join(htuner_config["save_path"], "journal.log")
        lock_file = optuna.storages.JournalFileOpenLock(log_file_path)
        storage = storage_type(file_path=log_file_path, lock_obj=lock_file)
    else:
        raise ValueError(f"Storage type {storage_type} not yet supported.")

    study_name = os.path.basename(exp_config.trainer["save_dir"]) # TODO: sistemare in base al save_dir che teroricamente sarà senza il trial folder
    study = optuna.create_study(sampler=sampler, direction="minimize", study_name=study_name,
                                storage=storage, pruner=pruner)

    if debug:
        start_optuna()
        print("Starting Optimization...")

    study.optimize(lambda trial: objective_wrapper(exp_config, data_module, trial, htuner_config["hparam_gen"]),
                   n_trials=htuner_config["n_trials"])

    return study.best_trial # TODO: guardare cos'è e cercare di ritornare il modello addestrato


def objective_wrapper(exp_config: ExpConfig, data_module: ImagesRecipesDataModule,
                      trial: optuna.Trial, hparam_gen_config: HGeneratorConfig) -> float:  # todo vedere per il resume
    # generate the variable hyperparameters
    hparams = hparam_gen_config.generate_hparams_on_trial(trial)

    # Update the configuration with the generated hyperparameters and the trial
    exp_config.update_config(**hparams, tr_trial=trial)

    # Run the experiment
    trainer, model = model_training(exp_config, data_module)
    ris = trainer.logged_metrics

    return ris["val_loss"].item()


def _resume_exp(ckpt_path: str | os.PathLike) -> Tuple[Type[lgn.Trainer], Type[lgn.LightningModule]]:
    """Function that resumes the experiment from the given checkpoint path."""
    set_torch_constants()
    warnings.filterwarnings("ignore", "Checkpoint directory .*. exists and is not empty.")
    checkpoint_data: Dict[str, Any] = torch.load(ckpt_path)
    return model_training(ExpConfig.load_from_ckpt_data(checkpoint_data), ckpt_path=ckpt_path)


if __name__ == "__main__":
    exp_dir, exp_name = os.path.join(EXPERIMENTS_PATH, "dummy_htuning"), "dummy_experiment"
    exp_name = f"{exp_name}_{random.randint(0, 100)}"
    print(exp_name)

    make_htuning_exp(exp_name, experiment_dir=exp_dir, max_epochs=5, batch_size=256, debug=True,
                     torch_model_type=DummyModel, dm_category="mexican", tr_type=BaseFasterTrainer, lr=DEF_LR)

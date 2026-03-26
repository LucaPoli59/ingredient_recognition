import os

import optuna
import pandas as pd
import numpy as np
import torch
import copy

from settings.config import *
from src.training.commons import init_optuna_storage
from src.commons.exp_config import HTunerExpConfig, HGeneratorConfig, ExpConfig, DEF_METRIC_F1L
from src.models.custom_schedulers import ConstantStartReduceOnPlateau
from src.data_processing.transformations import transform_aug_adv
from src.models.dummy import DummyBNModel
from src.models.resnet import ResnetLikeV1, Resnet18
from src.lightning.lgn_trainers import BaseTrainer
from training.one_shot_exp import make_one_shot_exp

if __name__ == "__main__":

    old_exp_dir = os.path.join(EXPERIMENTS_PATH, "basic")
    old_exp_name = "resnets_htuning{25k}"


    TRIAL_NAME_PREFIX = "trial_"
    best_trials = pd.read_csv(str(os.path.join(old_exp_dir, old_exp_name, "best_trials.csv")), index_col=0)
    best_trials['add_notes'] = ""

    new_prefab_trial = best_trials.iloc[0]  # adding a new trial with weighted loss True
    new_prefab_trial["hp_weighted_loss"] = True
    new_prefab_trial['add_notes'] = "Trial configuration made from trial 0 with weighted loss"
    new_prefab_trial.name = new_prefab_trial.name + 1000
    best_trials = pd.concat([best_trials, new_prefab_trial.to_frame().transpose()])

    exp_dir, exp_name = os.path.join(EXPERIMENTS_PATH, "basic"), "resnets_training_BM_F1_INGS"
    if not os.path.exists(os.path.join(exp_dir, exp_name)):
        os.makedirs(os.path.join(exp_dir, exp_name))

    debug = False

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    for trial_id in best_trials.index:
        trial_name = f"{TRIAL_NAME_PREFIX}{trial_id}"
        trial_source_id = trial_id if trial_id < 1000 else trial_id - 1000

        #todo FARE IL LOADING DEL DATAMODULE DAL GENERAL_CONFIG DELLO STUDIO, IN MODO DA GARANTIRE UNIFORMITA' TRA I LABEL ENCODERS

        trial_config = ExpConfig.load_from_file(str(os.path.join(old_exp_dir, old_exp_name, f"{TRIAL_NAME_PREFIX}{trial_source_id}", HTUNING_TRIAL_CONFIG_FILE)))
        trial_config.update_config(
            tr_type=BaseTrainer,  # revert to base trainer for one-shot training
            me_=(trial_config.metrics | {"f1_label": DEF_METRIC_F1L}), # add f1_label to metrics
            lgg_wandb_notes=f"Best trial from {old_exp_dir}/{old_exp_name} with id {trial_source_id}. "
                            f"Add. notes: {best_trials.at[trial_id, 'add_notes']}", # update wandb notes
            tr_limit_train_batches=1.0, # train on full dataset
        )

        print("\n\n\n\n----------------------\n\nRunning trial ", trial_name)


        trainer, model = make_one_shot_exp(experiment_name=exp_name, experiment_dir=exp_dir, exp_config=trial_config,
                                           debug=debug)



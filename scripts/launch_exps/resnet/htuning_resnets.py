from typing import Callable

import torch
import os

import torch
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

import src.models.custom_schedulers
from src.lightning.lgn_trainers import OptunaTrainer
from src.lightning.lgn_models import BaseWithSchedulerLGNM
from settings.config import (EXPERIMENTS_PATH, DEF_LR, HGEN_CONFIG_FILE)
from src.commons.exp_config import HGeneratorConfig
from src.data_processing.transformations import transform_aug_adv
from src.models.dummy import DummyBNModel, DummyModel
from src.models.resnet import ResnetLikeV1, Resnet18
from training.htuning_exp import make_htuning_exp

if __name__ == "__main__":
    exp_dir = os.path.join(EXPERIMENTS_PATH, "basic")
    exp_name = "resnets_htuning{25k}"


    # if not os.path.exists(exp_dir):
    #     os.makedirs(exp_dir)
    # if not os.path.exists(os.path.join(exp_dir, exp_name)):
    #     os.makedirs(os.path.join(exp_dir, exp_name))
    # else:
    #     print(f"WARNING: Experiment {exp_name} already exists in {exp_dir}")

    debug = False

    sampler = TPESampler
    sampler_kwargs = {
        "n_startup_trials": 7, # def 10
        "n_ei_candidates": 24, # def
        "multivariate": True, # def False  (for considering correlations between parameters)
        "group": True, # def False (increase efficiency for multivariate=True)
    }

    pruner = HyperbandPruner
    pruner_kwargs = {
        "min_resource": 2,  # def 5
        "max_resource": "auto",  # def
        "reduction_factor": 3,  # def
    }

    #PARAMS

    max_epochs = 40  # for each trial (40)
    n_trials = 100  # for the whole experiment

    #HPARAMS

    hgen_config = HGeneratorConfig(
        hp_lr=(lambda trial: trial.suggest_float("lr", 1e-5, 1e-0, log=True)),  # < 1e-3 is too low
        hp_lr_scheduler={
            "dist": (lambda trial: trial.suggest_categorical("lr_scheduler", [
                "ConstantStartReduceOnPlateau", "CosineAnnealingWarmRestarts", 'none'],)),  # < 1e-3 is too low
            "vocab": {
                "ConstantStartReduceOnPlateau":  src.models.custom_schedulers.ConstantStartReduceOnPlateau,
                "CosineAnnealingWarmRestarts":  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            },
            "effected_params": {
                "ConstantStartReduceOnPlateau": {
                    "hp_lr_scheduler_params": {
                        "warm_duration": 10,
                        "mode": "min",
                        "patience": 5,
                        "cooldown": 1,
                        "min_lr": DEF_LR / 1e3,
                        "factor": 0.05,
                    }
                },
                "CosineAnnealingWarmRestarts": {
                    "hp_lr_scheduler_params": {
                        "T_0": 5,
                        "T_mult": 2,
                        "eta_min": DEF_LR / 1e3,
                    }
                },
            },
        },
        hp_weighted_loss=(lambda trial: trial.suggest_categorical("weighted_loss", [True, False])),
        hp_optimizer={
            "dist": (lambda trial: trial.suggest_categorical("optimizer", ["adam", "sgd"])),
            "vocab": {
                "adam": torch.optim.Adam,
                "sgd": torch.optim.SGD,
            },
            "effected_params": {
                "sgd": {
                    "hp_momentum": 0.9,
                },
            },
        },
        hp_weight_decay=(lambda trial: trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)),

        tm_type={
            "dist": (lambda trial: trial.suggest_categorical("tm_type", ["dummy", "resnet18Like", "resnet18Pre"])),
            "vocab": {
                "dummy": DummyBNModel,
                "resnet18Like": ResnetLikeV1,
                "resnet18Pre": Resnet18,
            },
            "effected_params": {
                "resnet18Pre": {
                    "tm_pretrained": True
                }
            }
        },
        tm_trns_aug={
            "dist": (lambda trial: trial.suggest_categorical("trns_aug", ["no_aug", "aug_hard"])),
            "vocab": {
                "no_aug": None,
                "aug_hard": transform_aug_adv,
            },
            "effected_params": None
        }

    )


    make_htuning_exp(exp_name, hgen_config, experiment_dir=exp_dir, max_epochs=max_epochs, debug=debug, lgn_model_type=BaseWithSchedulerLGNM,
                     torch_model_type=DummyModel, tr_type=OptunaTrainer, dm_category="all", tr_limit_train_batches=0.5,
                     ht_n_trials=n_trials, ht_pruner=pruner, ht_pruner_kwargs=pruner_kwargs,
                     ht_sampler=sampler, ht_sampler_kwargs=sampler_kwargs, lgg_log_exp_config=True)
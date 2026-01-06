import os
from typing import Optional, Literal

import optuna
import json
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from deepdiff import DeepDiff
import itertools
import torch
from typing_extensions import LiteralString

from settings.config import *
from src.commons.exp_config import ExpConfig, DEF_METRIC_F1L
from src.lightning.lgn_trainers import BaseTrainer
from src.training.commons import set_torch_constants

if __name__ == "__main__":

    pio.templates.default = "plotly"

    idx = pd.IndexSlice

    set_torch_constants()

    old_exp_dir = os.path.join(EXPERIMENTS_PATH, "basic")
    old_exp_name = "resnets_htuning_sel{10k}"

    new_exp_dir = os.path.join(EXPERIMENTS_PATH, "basic")
    new_exp_name = "resnets_htuning_sel{10k}_test"

    TRIALS_TO_TEST = 3

    trial_df = pd.read_csv(str(os.path.join(old_exp_dir, old_exp_name, "best_trials.csv")), index_col=0).iloc[:TRIALS_TO_TEST]

    new_trial_id = 0
    for trial_id in trial_df.index:
        trial = trial_df.loc[trial_id]


        trial_config = ExpConfig.load_from_file(str(os.path.join(old_exp_dir, old_exp_name, f"trial_{trial_id}",HTUNING_TRIAL_CONFIG_FILE)))
        trial_config.update_config(
            tr_type=BaseTrainer,  # revert to base trainer for one-shot training
            me_=(trial_config.metrics | {"f1_label": DEF_METRIC_F1L}),
            tr_save_dir=os.path.join(new_exp_dir, new_exp_name, f"trial_{new_trial_id}"),
        )
        checkpoint_path = str(os.path.join(old_exp_dir, old_exp_name, f"trial_{trial_id}", "best_model.ckpt"))

        model_config, trainer_config, dm_config = trial_config.lgn_model, trial_config.trainer, trial_config.datamodule
        lgn_model = model_config['lgn_model_type'].load_from_config(model_config)
        trainer = trainer_config['type'].load_from_config(trainer_config)

        data_module = dm_config['type'].load_from_config(dm_config, batch_size=lgn_model.batch_size,
                                                                 transform_aug=lgn_model.transform_aug,
                                                                 transform_plain=lgn_model.transform_plain)
        data_module.prepare_data()
        data_module.setup()


        os.makedirs(trial_config.trainer['save_dir'], exist_ok=True)
        lgn_model.startup_model(data_module)
        metrics = trainer.validate(lgn_model, data_module, ckpt_path=checkpoint_path)

        print(f"\n\n\nTrial {new_trial_id} metrics: {metrics}\n\n\n")
        new_trial_id = new_trial_id + 1



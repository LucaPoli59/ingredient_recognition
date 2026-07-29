"""Launch Optuna tuning for frozen DINOv2 ViT-B/14 ingredient classification.

Search-space rationale
----------------------
This is a *linear-probing* study: DINOv2 stays frozen and the experiment learns
only the ingredient classifier head. DINOv2 is designed to yield transferable
features that can be evaluated with a linear classifier; keeping this regime
fixed makes all trials comparable and fits the available 8-GB GPU budget.

Learning rate is the most consequential variable for the randomly initialized
head. The existing successful DINO trial uses ``1e-3``, hence Optuna searches
only one compact log-scale decade around that anchor, ``[3e-4, 3e-3]``. This
avoids CNN-oriented ranges that include values too small to train the head in a
short trial or large enough to destabilize BCE multi-label optimization.

Weight decay is searched in ``[1e-6, 1e-4]``. Since only the small downstream
head is trainable, high decay values intended for full-network fine-tuning would
over-regularize the fit. The categorical batch choices are ``64`` and ``128``.
Both fit in the same physical micro-batch of 32; they compare gradient
accumulation of 2 versus 4 steps without introducing an OOM-prone memory search.
``weighted_loss`` is an additional categorical parameter: the ingredient vocabulary
is imbalanced, but class weights can improve recall while worsening calibration,
so both BCE variants merit comparison.

AdamW is fixed, while Optuna compares no scheduler with
``ReduceLROnPlateau``. This is the only scheduler choice useful within a
25-epoch, prunable study: it tests whether reducing LR after stalled validation
loss helps without introducing cosine-restart or warm-up parameters that would
make the search sparse. Extra augmentation is also excluded: DINOv2 already
uses its dedicated random-resized-crop and horizontal-flip preprocessing, while
an additional generic augmentation chain can apply a second crop.

Full backbone fine-tuning is intentionally not a trial option. It needs a much
smaller, usually discriminative backbone learning rate and separate optimizer
parameter groups; it should be a follow-up experiment seeded from the best
linear-probe configuration. The 24 trials, 25-epoch cap and Hyperband pruning
give TPE enough initial observations while terminating clearly weak runs early.
"""

import os

import torch
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from settings.config import EXPERIMENTS_PATH
from src.commons.exp_config import HGeneratorConfig
from src.lightning.lgn_models import BaseWithSchedulerLGNM
from src.lightning.lgn_trainers import OptunaTrainer
from src.models.dinov2 import DinoV2B14
from src.training.htuning_exp import make_htuning_exp


if __name__ == "__main__":
    exp_dir = os.path.join(EXPERIMENTS_PATH, "dinov2")
    exp_name = "dinov2_b14_reg_linear_probe_htuning_v1"

    # Default logical batch; Optuna compares 64 and 128 below. DinoV2B14 caps
    # the physical micro-batch at 32 and BaseLGNM accumulates automatically.
    batch_size = 128
    max_epochs = 25
    n_trials = 24

    sampler = TPESampler
    sampler_kwargs = {
        "n_startup_trials": 8,
        "n_ei_candidates": 24,
        "multivariate": True,
        "group": True,
    }

    pruner = HyperbandPruner
    pruner_kwargs = {
        "min_resource": 5,
        "max_resource": max_epochs,
        "reduction_factor": 3,
    }

    hgen_config = HGeneratorConfig(
        hp_batch_size=lambda trial: trial.suggest_categorical("batch_size", [64, 128]),
        # The existing DINO linear-probe baseline uses 1e-3.  Search one
        # compact decade around it instead of the broad CNN-oriented range.
        hp_lr=lambda trial: trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        # Only the new classifier head is trainable in this study; large decay
        # values appropriate for full networks would over-regularize it.
        hp_weight_decay=lambda trial: trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        # Class balancing is dataset-dependent, so retain it as a discrete
        # training choice alongside logical batch size and scheduler.
        hp_weighted_loss=lambda trial: trial.suggest_categorical("weighted_loss", [True, False]),
        hp_lr_scheduler={
            "dist": lambda trial: trial.suggest_categorical("lr_scheduler", ["none", "plateau"]),
            "vocab": {
                "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            },
            "effected_params": {
                "plateau": {
                    "hp_lr_scheduler_params": {
                        "mode": "min",
                        "factor": 0.2,
                        "patience": 3,
                        "min_lr": 1e-6,
                    }
                }
            },
        },
    )

    make_htuning_exp(
        exp_name,
        hgen_config,
        experiment_dir=exp_dir,
        max_epochs=max_epochs,
        batch_size=batch_size,
        debug=False,
        lgn_model_type=BaseWithSchedulerLGNM,
        tr_type=OptunaTrainer,
        tm_type=DinoV2B14,
        tm_pretrained=True,
        tm_freeze_backbone=True,
        dm_category="all",
        optimizer=torch.optim.AdamW,
        tr_limit_train_batches=1.0,
        ht_n_trials=n_trials,
        ht_sampler=sampler,
        ht_sampler_kwargs=sampler_kwargs,
        ht_pruner=pruner,
        ht_pruner_kwargs=pruner_kwargs,
        lgg_log_exp_config=True,
    )

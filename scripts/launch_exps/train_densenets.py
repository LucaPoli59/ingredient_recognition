import os
import lightning as lgn
import torch

from settings.config import (EXPERIMENTS_PATH, DEF_LR, DEF_LR_INIT)
from src.lightning.lgn_trainers import BaseFasterTrainer
from src.models.densenet import DensenetLikeV1, DensenetLikeV2, Densenet121, Densenet201
from src.training.one_shot_exp import make_one_shot_exp
from src.lightning.lgn_models import BaseWithSchedulerLGNM

if __name__ == "__main__":
    exp_dir = os.path.join(EXPERIMENTS_PATH, "basic")
    exp_name = "densenets_training"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(os.path.join(exp_dir, exp_name)):
        os.makedirs(os.path.join(exp_dir, exp_name))
    else:
        print(f"WARNING: Experiment {exp_name} already exists in {exp_dir}")

    lr = DEF_LR_INIT
    debug = False

    # print("Training DensenetLikeV1 ....")
    # trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=50, batch_size=128, debug=debug,
    #                                    tm_type=DensenetLikeV1, dm_category="all", tr_type=BaseFasterTrainer,
    #                                    lr=lr, lgn_model_type=BaseWithSchedulerLGNM, optimizer=torch.optim.SGD,
    #                                    momentum=0.9, weight_decay=5e-4)
    #
    # print("Training DensenetLikeV2 ....")
    # trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=50, batch_size=128, debug=debug,
    #                                    tm_type=DensenetLikeV2, dm_category="all", tr_type=BaseFasterTrainer,
    #                                    lr=lr, lgn_model_type=BaseWithSchedulerLGNM, optimizer=torch.optim.SGD,
    #                                    momentum=0.9, weight_decay=5e-4)

    print("Training Densenet121 ....")
    trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=50, batch_size=128, debug=debug,
                                       tm_type=Densenet121, tm_pretrained=False, dm_category="all",
                                       tr_type=BaseFasterTrainer, lr=lr, lgn_model_type=BaseWithSchedulerLGNM,
                                       optimizer=torch.optim.SGD, momentum=0.9, weight_decay=5e-4)

    print("Training Densenet201 ....")
    trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=50, batch_size=128, debug=debug,
                                       tm_type=Densenet201, tm_pretrained=False, dm_category="all",
                                       tr_type=BaseFasterTrainer, lr=lr, lgn_model_type=BaseWithSchedulerLGNM,
                                       optimizer=torch.optim.SGD, momentum=0.9, weight_decay=5e-4)

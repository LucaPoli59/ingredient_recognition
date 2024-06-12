import os
import lightning as lgn
import torch

from settings.config import (EXPERIMENTS_PATH, DEF_LR)
from src.lightning.lgn_trainers import BaseFasterTrainer
from src.models.resnet import ResnetLikeV1, ResnetLikeV2, Resnet18, Resnet50
from src.training.one_shot_exp import make_one_shot_exp
from src.lightning.lgn_models import BaseWithSchedulerLGNM

if __name__ == "__main__":
    exp_dir = os.path.join(EXPERIMENTS_PATH, "basic")
    exp_name = "resnets_training"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(os.path.join(exp_dir, exp_name)):
        os.makedirs(os.path.join(exp_dir, exp_name))
    else:
        print(f"WARNING: Experiment {exp_name} already exists in {exp_dir}")

    debug = False

    # print("Training ResnetLikeV1 ....")
    # trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=50, batch_size=128, debug=debug,
    #                                    tm_type=ResnetLikeV1, dm_category="all", tr_type=BaseFasterTrainer,
    #                                    lr=DEF_LR, lgn_model_type=BaseWithSchedulerLGNM, optimizer=torch.optim.SGD,
    #                                    momentum=0.9, weight_decay=5e-4)

    # print("Training ResnetLikeV2 ....")
    # trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=50, batch_size=128, debug=debug,
    #                                    tm_type=ResnetLikeV2, dm_category="all", tr_type=BaseFasterTrainer,
    #                                    lr=DEF_LR, lgn_model_type=BaseWithSchedulerLGNM, optimizer=torch.optim.SGD,
    #                                    momentum=0.9, weight_decay=5e-4)
    #
    # print("Training Resnet18 ....")
    # trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=50, batch_size=128, debug=debug,
    #                                    tm_type=Resnet18, tm_pretrained=False, dm_category="all",
    #                                    tr_type=BaseFasterTrainer, lr=DEF_LR, lgn_model_type=BaseWithSchedulerLGNM,
    #                                    optimizer=torch.optim.SGD, momentum=0.9, weight_decay=5e-4)
    #
    # print("Training Resnet50 ....")
    # trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=50, batch_size=128, debug=debug,
    #                                    tm_type=Resnet50, tm_pretrained=False, dm_category="all",
    #                                    tr_type=BaseFasterTrainer, lr=DEF_LR, lgn_model_type=BaseWithSchedulerLGNM,
    #                                    optimizer=torch.optim.SGD, momentum=0.9, weight_decay=5e-4)

    # print("Training Resnet18 pretrained ....")
    # trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=50, batch_size=128, debug=debug,
    #                                    tm_type=Resnet18, tm_pretrained=True, dm_category="all",
    #                                    tr_type=BaseFasterTrainer, lr=DEF_LR, lgn_model_type=BaseWithSchedulerLGNM,
    #                                    optimizer=torch.optim.SGD, momentum=0.9, weight_decay=5e-4)
    print("Training Resnet50 pretrained ....")
    trainer, model = make_one_shot_exp(exp_name, experiment_dir=exp_dir, max_epochs=50, batch_size=128, debug=debug,
                                       tm_type=Resnet50, tm_pretrained=True, dm_category="all",
                                       tr_type=BaseFasterTrainer, lr=DEF_LR, lgn_model_type=BaseWithSchedulerLGNM,
                                       optimizer=torch.optim.SGD, momentum=0.9, weight_decay=5e-4)
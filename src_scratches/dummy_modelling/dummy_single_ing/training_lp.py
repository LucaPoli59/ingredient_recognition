import os.path

import lightning as lgn
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from lightning.pytorch.profilers import SimpleProfiler
import subprocess
import wandb

from settings.config import *
from src.data_processing.images_recipes import ImagesRecipesDataset
from src.data_processing.labels_encoders import OneVSAllLabelEncoder, MultiLabelBinarizerRobust
from src.data_processing.transformations import _transform_core_base, transformations_wrapper, trasnform_aug_adv
from src.models.resnet import ResnetLikeV2
from src_scratches.dummy_modelling.dummy_single_ing.LightningModel import LightningModel

from _commons import *

if __name__ == "__main__":
    if not LAYER_WISE_PRETRAINING:
        raise RuntimeError("WRONG FILE FOR LAYER WISE PRETRAINING")

    # Load the dataset
    INPUT_SHAPE = 224
    TARGET_INGREDIENT = "salt"
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")
    if not os.path.exists(model_path):
        os.makedirs(os.path.join(model_path, "lightning_logs"))

    if os.path.exists(os.path.join(model_path, "dummy_single_ing", f"{RUN_ID}")):
        print(f"Run {RUN_ID} already exists")
        # exit(0)

    torch.set_float32_matmul_precision('medium')  # For better performance with cuda
    # print(os.environ["WANDB_MODE"])
    # os.environ["WANDB_MODE"] = "offline"
    # print(os.environ["WANDB_MODE"])
    if WANDB_OFFLINE:
        subprocess.run(["wandb", "offline"])
    else:
        # pass # change this when working locally
        subprocess.run(["wandb", "online"])

    if MODEL_TYPE in [DummyModel, DummyBNModel, ResnetLikeV1, ResnetLikeV2, ResnetLikeV1LVariant]:

        model = MODEL_TYPE(NUM_CLASSES, (INPUT_SHAPE, INPUT_SHAPE), lp_phase=0 if LAYER_WISE_PRETRAINING else None)
    elif MODEL_TYPE in [Resnet18, Resnet50, Densenet121, Densenet201]:
        model = MODEL_TYPE(NUM_CLASSES, (INPUT_SHAPE, INPUT_SHAPE), pretrained=MODEL_PRETRAINED)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not recognized")

    if NORMALIZE_IMGS:
        if MODEL_PRETRAINED:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            mean, std = pd.read_csv(os.path.join(YUMMLY_PATH, IMG_STATS_FILENAME), index_col=0).values
    else:
        mean, std = [0, 0, 0], [1, 1, 1]

    if AUGMENTING_IMGS:
        transform_train = trasnform_aug_adv((INPUT_SHAPE, INPUT_SHAPE))
    else:
        transform_train = _transform_core_base((INPUT_SHAPE, INPUT_SHAPE))

    transform_train = transformations_wrapper(transform_train, mean, std)

    transform_val = transformations_wrapper([v2.Resize((INPUT_SHAPE, INPUT_SHAPE))], mean, std)

    # Load the dataset

    # train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train,
    #                                              target_transform=encode_target)
    # val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val,
    #                                            target_transform=encode_target)

    if EASY_PROBLEM:
        label_encoder = OneVSAllLabelEncoder(target_ingredient=TARGET_INGREDIENT)
    else:
        label_encoder = MultiLabelBinarizerRobust()
    train_dataset = ImagesRecipesDataset(os.path.join(YUMMLY_PATH, "train"),
                                         transform=transform_train, label_encoder=label_encoder, category=CATEGORY)
    val_dataset = ImagesRecipesDataset(os.path.join(YUMMLY_PATH, "val"),
                                       transform=transform_val, label_encoder=label_encoder, category=CATEGORY)

    if N_SAMPLES is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(N_SAMPLES))
        val_samples = min(len(val_dataset), N_SAMPLES)
        val_dataset = torch.utils.data.Subset(val_dataset, range(val_samples))

    # BATCH_SIZE = 128
    NUM_WORKERS = os.cpu_count() if EPOCHS > 3 else 2

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True, persistent_workers=True)

    # Creation of the model

    wandb_logger = lgn.pytorch.loggers.WandbLogger(name=f"run_{RUN_ID}_{{{N_SAMPLES}-{BATCH_SIZE}}}",
                                                   project="dummy_single_ing", save_dir=model_path,
                                                   id=f"{RUN_ID}")
    wandb_logger.watch(model, log="all", log_freq=50)
    wandb_logger.experiment.config.update({"MODEL_TYPE": str(MODEL_TYPE), "NORM_IMG": NORMALIZE_IMGS,
                                           "AUGMENTING_IMGS": AUGMENTING_IMGS,
                                           "NOTES": "AFTER FIXING SHUFFLE ISSUES"})

    if LAYER_WISE_PRETRAINING:
        if not model.support_layer_pretrain:
            raise ValueError("The model does not support layer pretraining")
        n_phases = model.max_lp_phase + 1
        for phase_id in range(n_phases):
            epochs = min(EPOCHS // abs(phase_id - n_phases) + 10, EPOCHS)
            # wandb_logger = lgn.pytorch.loggers.WandbLogger(name=f"run_{RUN_ID}-{phase_id}_{{{N_SAMPLES}-{BATCH_SIZE}}}",
            #                                                project="dummy_single_ing", save_dir=model_path,
            #                                                id=f"{RUN_ID}-{phase_id}")
            # wandb_logger.watch(model, log="all", log_freq=10)
            # wandb_logger.experiment.config.update({"MODEL_TYPE": str(MODEL_TYPE), "NORM_IMG": NORMALIZE_IMGS,
            #                                        "AUGMENTING_IMGS": AUGMENTING_IMGS,
            #                                        "NOTES": "AFTER FIXING SHUFFLE ISSUES"})

            total_steps = len(train_dataloader) * EPOCHS
            # Lightning model
            lighting_model = LightningModel(model, LR, OPTIMIZER, LOSS_TYPE, ACCURACY_FN, N_SAMPLES, BATCH_SIZE,
                                            total_steps,
                                            momentum=MOMENTUM,
                                            weight_decay=WEIGHT_DECAY, swa=False, weighted_loss=WEIGHT_LOSS,
                                            easy_problem=EASY_PROBLEM, model_pretrained=MODEL_PRETRAINED,
                                            lr_scheduler=LR_SCHEDULER)
            bar_callback = lgn.pytorch.callbacks.RichProgressBar(leave=True)
            lr_monitor_callback = lgn.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")

            trainer = lgn.Trainer(
                max_epochs=epochs,
                accelerator="gpu",
                default_root_dir=model_path,
                logger=wandb_logger,
                precision="16-mixed",
                log_every_n_steps=len(train_dataloader),
                callbacks=[
                    bar_callback,
                    lr_monitor_callback,
                ],

                accumulate_grad_batches=5,
                gradient_clip_val=0.01,
                gradient_clip_algorithm="value",

                # fast_dev_run=True,
                # profiler=profiler,
                enable_model_summary=False,
            )
            print("starting phase ", phase_id)
            train_data = train_dataset if N_SAMPLES is None else train_dataset.dataset
            lighting_model.startup_model(train_data)

            trainer.fit(
                model=lighting_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )

            print(f"training phase {phase_id} done")
            model.lp_phase_step()

    total_steps = len(train_dataloader) * EPOCHS
    # Lightning model
    lighting_model = LightningModel(model, LR, OPTIMIZER, LOSS_TYPE, ACCURACY_FN, N_SAMPLES, BATCH_SIZE, total_steps,
                                    momentum=MOMENTUM,
                                    weight_decay=WEIGHT_DECAY, swa=SWA, weighted_loss=WEIGHT_LOSS,
                                    easy_problem=EASY_PROBLEM, model_pretrained=MODEL_PRETRAINED,
                                    lr_scheduler=LR_SCHEDULER)
    bar_callback = lgn.pytorch.callbacks.RichProgressBar(leave=True)
    device_stats_callback = lgn.pytorch.callbacks.DeviceStatsMonitor(cpu_stats=False)
    timer_callback = lgn.pytorch.callbacks.Timer()
    lr_monitor_callback = lgn.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = lgn.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        default_root_dir=model_path,
        logger=wandb_logger,
        precision="16-mixed",
        log_every_n_steps=len(train_dataloader),
        callbacks=[
            bar_callback,
            timer_callback,
            lr_monitor_callback,
            # device_stats_callback,
        ],

        accumulate_grad_batches=5,
        gradient_clip_val=0.01,
        gradient_clip_algorithm="value",

        # fast_dev_run=True,
        # profiler=profiler,
        enable_model_summary=False,
    )

    print("Starting training")
    train_data = train_dataset if N_SAMPLES is None else train_dataset.dataset
    lighting_model.startup_model(train_data)

    trainer.fit(
        model=lighting_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    if WANDB_OFFLINE:
        wandb.finish()
        wandb_path = os.path.join(model_path, "wandb")
        wandb_run_folder = [elem for elem in os.listdir(wandb_path) if elem.startswith("offline-run-") and elem.endswith(f"-{RUN_ID}")][0]
        subprocess.run(["wandb", "sync", str(os.path.join(wandb_path, wandb_run_folder))])
    # wandb.save(os.path.join(wandb_path, wandb_run_folder))

    ris = trainer.logged_metrics
    print("\n\n\n\n", ris)
    print(f"Training time {timer_callback.time_elapsed('train')}s")

    # Save the model
    # torch.save(model.state_dict(), "model.pth")

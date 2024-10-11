import shutil

import lightning as lgn
import pandas as pd
import torch
import torchmetrics.classification
import torchmetrics.functional.classification
import torchvision
import torchmetrics
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from lightning.pytorch.profilers import SimpleProfiler

from settings.config import *
from src.commons.visualizations import h_stack_imgs, gradcam
from src.data_processing.images_recipes import ImagesRecipesDataset
from src.data_processing.labels_encoders import OneVSAllLabelEncoder, MultiLabelBinarizer, MultiLabelBinarizerRobust
from src.models.resnet import ResnetLikeV2
from src_scratches.dummy_modelling.dummy.training import train
from src_scratches.dummy_modelling.dummy_single_ing.training import accuracy, encode_target
from src.data_processing.transformations import transform_aug_base, transformations_wrapper, trasnform_aug_adv
from src_scratches.dummy_modelling.dummy_single_ing.LightningModel import LightningModel

from _commons import *

if __name__ == "__main__":
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
        pass # change this when working locally
        # subprocess.run(["wandb", "online"])

    if MODEL_TYPE in [DummyModel, DummyBNModel, ResnetLikeV1, ResnetLikeV2]:
        model = MODEL_TYPE(NUM_CLASSES, (INPUT_SHAPE, INPUT_SHAPE))
    elif MODEL_TYPE in [Resnet18, Resnet50, Densenet121, Densenet201]:
        model = MODEL_TYPE(NUM_CLASSES, (INPUT_SHAPE, INPUT_SHAPE), pretrained=MODEL_PRETRAINED)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not recognized")

    if NORMALIZE_IMGS:
        mean, std = pd.read_csv(os.path.join(YUMMLY_PATH, IMG_STATS_FILENAME), index_col=0).values
    else:
        mean, std = [0, 0, 0], [1, 1, 1]

    if AUGMENTING_IMGS:
        transform_train = trasnform_aug_adv((INPUT_SHAPE, INPUT_SHAPE))
    else:
        transform_train = transform_aug_base((INPUT_SHAPE, INPUT_SHAPE))

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
                                                   project="dummy_single_ing", save_dir=model_path, id=f"{RUN_ID}")
    wandb_logger.watch(model, log="all", log_freq=10)
    wandb_logger.experiment.config.update({"MODEL_TYPE": str(MODEL_TYPE), "NORM_IMG": NORMALIZE_IMGS,
                                           "AUGMENTING_IMGS": AUGMENTING_IMGS,
                                           "NOTES": "AFTER FIXING SHUFFLE ISSUES"})

    total_steps = len(train_dataloader) * EPOCHS
    # Lightning model
    lighting_model = LightningModel(model, LR, OPTIMIZER, LOSS_TYPE, ACCURACY_FN, N_SAMPLES, BATCH_SIZE, total_steps,
                                    momentum=MOMENTUM,
                                    weight_decay=WEIGHT_DECAY, swa=SWA, weighted_loss=WEIGHT_LOSS,
                                    easy_problem=EASY_PROBLEM, model_pretrained=MODEL_PRETRAINED,
                                    lr_scheduler=LR_SCHEDULER)
    
    ckpt_path = os.path.join(model_path, "dummy_single_ing", f"{RUN_ID}", "checkpoints")
    ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
    lighting_model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model = lighting_model.model

    if DATA_SRC == "train":
        batch = next(iter(train_dataloader))
    else:
        batch = next(iter(val_dataloader))

    X, y = batch
    with torch.no_grad():
        y_pred = model(X)
    print("IMG", X[0][0][50:200],  "\n", X[0].shape, "\n")
    print("Y_PREDS", torch.argmax(torch.sigmoid(y_pred), dim=1), "\n")
    print("Y_TRUE", torch.argmax(y, dim=1), "\n")
    print("ACCURACY", ACCURACY_FN(torch.sigmoid(y_pred), y))

    img_idx = 0
    target_layer = model.conv_target_layer
    imgs_plot, gradcam_masks, targets, outputs = gradcam(model, target_layer, X[img_idx], targets=[0])
    plt.imshow(h_stack_imgs(X[img_idx].numpy(), imgs_plot[img_idx]))
    plt.show()


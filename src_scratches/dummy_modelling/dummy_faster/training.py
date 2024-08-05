import shutil

import lightning as lgn
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from lightning.pytorch.profilers import SimpleProfiler

from settings.config import *
from src.data_processing.data_handling import ImagesRecipesDataset
from src.data_processing.labels_encoders import MultiLabelBinarizerRobust
from src_scratches.dummy_modelling.dummy_faster.LightningModel import LightningModel
from src_scratches.dummy_modelling.dummy_faster.model import DummyModel


def multi_label_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    y_pred = torch.round(torch.sigmoid(y_pred))  # get the binary predictions
    # num of hits / num of classes mean over the batch
    return torch.mean((y_pred == y_true).sum(dim=1) / y_pred.size(1))


if __name__ == "__main__":
    # Load the dataset
    INPUT_SHAPE = 224
    CATEGORY = "mexican"
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    os.mkdir(os.path.join(model_path, "lightning_logs"))

    torch.set_float32_matmul_precision('medium')  # For better performance with cuda

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((INPUT_SHAPE, INPUT_SHAPE)),
        v2.TrivialAugmentWide(num_magnitude_bins=31),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # Load the dataset
    mlb = MultiLabelBinarizerRobust()
    train_dataset = ImagesRecipesDataset(os.path.join(YUMMLY_PATH, "train"), os.path.join(YUMMLY_RECIPES_PATH, "train.json"),
                                         transform=transform, label_encoder=mlb, category=CATEGORY)
    val_dataset = ImagesRecipesDataset(os.path.join(YUMMLY_PATH, "val"), os.path.join(YUMMLY_RECIPES_PATH, "val.json"),
                                       transform=transform, label_encoder=mlb, category=CATEGORY)

    BATCH_SIZE = 128
    EPOCHS = 2
    NUM_WORKERS = os.cpu_count() if EPOCHS > 3 else 2

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True, persistent_workers=True)

    # Creation of the model
    model = DummyModel(INPUT_SHAPE, len(mlb.classes) + 1)

    # Training hyperparameters
    LOSS_FN = torch.nn.BCEWithLogitsLoss()
    OPTIMIZER, LR = torch.optim.Adam, 1e-3
    ACCURACY_FN = multi_label_accuracy

    # Lightning model
    lighting_model = LightningModel(model, LR, OPTIMIZER, LOSS_FN, ACCURACY_FN)
    bar_callback = lgn.pytorch.callbacks.RichProgressBar(leave=True)
    device_stats_callback = lgn.pytorch.callbacks.DeviceStatsMonitor(cpu_stats=False)
    timer_callback = lgn.pytorch.callbacks.Timer()
    profiler = SimpleProfiler(dirpath=model_path, filename="profiler")

    trainer = lgn.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        default_root_dir=model_path,
        logger=True,
        precision="16-mixed",
        log_every_n_steps=len(train_dataloader),
        callbacks=[
            bar_callback,
            timer_callback,
            # device_stats_callback,
        ],

        accumulate_grad_batches=5,

        # fast_dev_run=True,
        # profiler=profiler,
        enable_model_summary=False,
    )

    print("Starting training")

    trainer.fit(
        model=lighting_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    ris = trainer.logged_metrics
    print("\n\n\n\n", ris)
    print(f"Training time {timer_callback.time_elapsed('train')}s")

    # Save the model
    # torch.save(model.state_dict(), "model.pth")

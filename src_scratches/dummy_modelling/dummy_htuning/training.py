import shutil

import lightning as lgn
import torch
from lightning.pytorch.profilers import SimpleProfiler

from settings.config import *
from src.data_processing.data_handling import ImagesRecipesDataModule
from src_scratches.dummy_modelling.dummy_htuning.model import DummyModel
from src_scratches.dummy_modelling.dummy_htuning.LightningModel import LightningModel


def multi_label_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    y_pred = torch.round(torch.sigmoid(y_pred))  # get the binary predictions
    # num of hits / num of classes mean over the batch
    return torch.mean((y_pred == y_true).sum(dim=1) / y_pred.size(1))


if __name__ == "__main__":
    # Load the dataset
    INPUT_SHAPE = (224, 224)
    CATEGORY = "mexican"
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    os.mkdir(os.path.join(model_path, "lightning_logs"))

    torch.set_float32_matmul_precision('medium')  # For better performance with cuda

    BATCH_SIZE = 128
    EPOCHS = 2
    NUM_WORKERS = os.cpu_count() if EPOCHS > 3 else 2
    LOSS_FN = torch.nn.BCEWithLogitsLoss()
    OPTIMIZER, LR = torch.optim.Adam, 1e-3
    ACCURACY_FN = multi_label_accuracy

    data_module = ImagesRecipesDataModule(IMAGES_PATH, RECIPES_PATH, category=CATEGORY,
                                          image_size=INPUT_SHAPE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    data_module.prepare_data()
    data_module.setup()
    print("Data module loaded")
    torch_model = DummyModel(INPUT_SHAPE, data_module.get_num_classes())
    lgn_model = LightningModel(torch_model, LR, OPTIMIZER, LOSS_FN, ACCURACY_FN)


    # Trainer
    bar_callback = lgn.pytorch.callbacks.RichProgressBar(leave=True)
    device_stats_callback = lgn.pytorch.callbacks.DeviceStatsMonitor(cpu_stats=False)
    timer_callback = lgn.pytorch.callbacks.Timer()

    trainer = lgn.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        default_root_dir=model_path,
        logger=True,
        precision="16-mixed",
        log_every_n_steps=25,
        callbacks=[
            bar_callback,
            timer_callback,
            # device_stats_callback,
        ],


        # fast_dev_run=True,
        # profiler=profiler,
        enable_model_summary=False,
    )

    print("Starting training")

    trainer.fit(
        model=lgn_model,
        datamodule=data_module
    )

    ris = trainer.logged_metrics
    print("\n\n\n\n", ris)
    print(f"Training time {timer_callback.time_elapsed('train')}s")

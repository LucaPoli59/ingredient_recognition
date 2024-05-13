import os
import shutil

import lightning as lgn
import numpy as np
import ray
import torch
from lightning.pytorch.profilers import SimpleProfiler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray import tune, train

from settings.config import *
from src.data_processing.data_handling import ImagesRecipesDataModule
from src_scratches.dummy_modelling.dummy_htuning_ray.model import DummyModel
from src_scratches.dummy_modelling.dummy_htuning_ray.LightningModel import LightningModel


def multi_label_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    y_pred = torch.round(torch.sigmoid(y_pred))  # get the binary predictions
    # num of hits / num of classes mean over the batch
    return torch.mean((y_pred == y_true).sum(dim=1) / y_pred.size(1))


def _trainable(config, data_module):
    model_config = config["hyper_parameters"]
    trainer_config = config["trainer_hyper_parameters"]

    input_shape, num_classes = tuple(model_config['input_shape']), model_config['num_classes']
    batch_size, lr = model_config['batch_size'], model_config['lr']
    loss_fn, accuracy_fn = model_config['loss_fn'], model_config['accuracy_fn']
    optimizer, model_name = model_config['optimizer'], model_config.get('model_name', None)

    torch_model = DummyModel(input_shape, num_classes)
    lgn_model = LightningModel(torch_model, lr, optimizer, loss_fn, accuracy_fn)

    debug, max_epochs, save_dir = trainer_config['debug'], trainer_config['max_epochs'], trainer_config['save_dir']

    ray_tune_callback = TuneReportCheckpointCallback(save_checkpoints=True, on="validation_end",
                                                     metrics=["val_loss", "val_acc"])

    trainer = lgn.Trainer(
        max_epochs=max_epochs,
        devices=1,
        accelerator="gpu",
        default_root_dir=save_dir,
        limit_train_batches=0.3,
        logger=True,
        precision="16-mixed",
        log_every_n_steps=25,

        # noinspection PyTypeChecker
        callbacks=[ray_tune_callback],

        # fast_dev_run=True,
        # profiler=profiler,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(
        model=lgn_model,
        datamodule=data_module
    )


if __name__ == "__main__":
    # Load the dataset
    INPUT_SHAPE = (224, 224)
    CATEGORY = "mexican"
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_logs")
    temp_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    os.mkdir(os.path.join(model_path, "lightning_logs"))

    torch.set_float32_matmul_precision('medium')  # For better performance with cuda
    ray.init(_temp_dir=temp_path)

    BATCH_SIZE = 256
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

    # tune.Tuner.restore(model_path, trainable=tune.with_parameters(_trainable, data_module=data_module))

    config = {
        "hyper_parameters": {
            "input_shape": INPUT_SHAPE,
            "num_classes": data_module.get_num_classes(),
            "batch_size": BATCH_SIZE,
            "lr": tune.loguniform(1e-5, 1e-1),
            "loss_fn": LOSS_FN,
            "accuracy_fn": ACCURACY_FN,
            "optimizer": OPTIMIZER,
            "model_name": "dummy_model"
        },
        "trainer_hyper_parameters": {
            "debug": False,
            "max_epochs": EPOCHS,
            "save_dir": model_path
        },
        "datamodule_hyper_parameters": {  # in questo caso non serve
        }
    }

    scheduler = tune.schedulers.ASHAScheduler(max_t=EPOCHS, grace_period=1, reduction_factor=2)
    # scheduler = tune.schedulers.MedianStoppingRule()
    # search_algo = HEBOSearch()
    # search_algo = HyperOptSearch()
    # scheduler = tune.schedulers.HyperBandForBOHB(max_t=EPOCHS, reduction_factor=2)
    # search_algo = TuneBOHB()
    reporter = tune.CLIReporter(
        parameter_columns=["lr"],
        metric_columns=["val_loss", "val_acc", "training_iteration"],
    )
    resource_for_trial = {"cpu": 8, "gpu": 1}

    tuner = tune.Tuner(
        trainable=tune.with_resources(tune.with_parameters(_trainable, data_module=data_module),
                                      resources=resource_for_trial),
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=10,
            scheduler=scheduler,
            # search_alg=search_algo,
            trial_dirname_creator=(lambda trial: f"trial_{trial.trial_id}"),
            trial_name_creator=(lambda trial: f"trial_{trial.trial_id}"),
            max_concurrent_trials=1,
        ),
        run_config=train.RunConfig(name="bo", progress_reporter=reporter, storage_path=model_path)
    )
    print("Tuner configured, training started")
    results = tuner.fit()

    results.get_dataframe().to_csv(os.path.join(model_path, "results.csv"))

    # ris = trainer.logged_metrics
    # print("\n\n\n\n", ris)
    # print(f"Training time {timer_callback.time_elapsed('train')}s")

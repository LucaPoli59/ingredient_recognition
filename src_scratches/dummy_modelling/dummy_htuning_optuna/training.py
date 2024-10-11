import shutil
import joblib
import lightning as lgn
import torch
import optuna
from optuna_integration import PyTorchLightningPruningCallback

from settings.config import *
from src.data_processing.images_recipes import ImagesRecipesBaseDataModule
from src_scratches.dummy_modelling.dummy_htuning_optuna.LightningModel import LightningModel
from src_scratches.dummy_modelling.dummy_htuning_optuna.model import DummyModel
from src.dashboards.start_optuna import start_optuna

def multi_label_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    y_pred = torch.round(torch.sigmoid(y_pred))  # get the binary predictions
    # num of hits / num of classes mean over the batch
    return torch.mean((y_pred == y_true).sum(dim=1) / y_pred.size(1))


def objective(config, data_module, trial: optuna.Trial):
    model_config = config["hyper_parameters"]
    trainer_config = config["trainer_hyper_parameters"]

    # Fixed hyperparameters
    input_shape, num_classes = tuple(model_config['input_shape']), model_config['num_classes']
    batch_size, loss_fn, accuracy_fn = model_config['batch_size'], model_config['loss_fn'], model_config['accuracy_fn']
    optimizer, model_name = model_config['optimizer'], model_config.get('model_name', None)
    debug, max_epochs, save_dir = trainer_config['debug'], trainer_config['max_epochs'], trainer_config['save_dir']


    # Variable hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1) if model_config['lr'] is None else model_config['lr']
    # lr = 1e-3

    torch_model = DummyModel(input_shape, num_classes)
    lgn_model = LightningModel(torch_model, lr, optimizer, loss_fn, accuracy_fn)

    optuna_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    trainer = lgn.Trainer(
        max_epochs=max_epochs,
        devices=1,
        accelerator="gpu",
        default_root_dir=save_dir,
        limit_train_batches=0.2,
        logger=True,
        precision="16-mixed",
        log_every_n_steps=25,

        # noinspection PyTypeChecker
        callbacks=[optuna_callback],

        # fast_dev_run=True,
        # profiler=profiler,
        enable_model_summary=False,
        enable_progress_bar=True,
    )

    trainer.save_checkpoint(os.path.join(save_dir, "checkpoints", "init.ckpt"))

    print(F"Trial {trial.number} started with lr={lr}")
    trainer.fit(
        model=lgn_model,
        datamodule=data_module
    )

    trainer.checkpoint_callback

    ris = trainer.logged_metrics
    print(f"Result of the trial: {ris}")
    return ris['val_loss'].item()


if __name__ == "__main__":
    # Load the dataset
    INPUT_SHAPE = (224, 224)
    CATEGORY = None
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_logs")
    log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "journal.log")
    study_name = "dummy_study"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    os.mkdir(os.path.join(model_path, "lightning_logs"))

    torch.set_float32_matmul_precision('medium')  # For better performance with cuda

    BATCH_SIZE = 128
    EPOCHS = 5
    NUM_WORKERS = os.cpu_count() if EPOCHS > 3 else 2
    LOSS_FN = torch.nn.BCEWithLogitsLoss()
    OPTIMIZER, LR = torch.optim.Adam, 1e-3
    ACCURACY_FN = multi_label_accuracy

    data_module = ImagesRecipesBaseDataModule(YUMMLY_PATH, category=CATEGORY,
                                              batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    data_module.prepare_data()
    data_module.setup()
    print("Data module loaded")

    config = {
        "hyper_parameters": {
            "input_shape": INPUT_SHAPE,
            "num_classes": data_module.get_num_classes(),
            "batch_size": BATCH_SIZE,
            "lr": None,
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

    sampler = optuna.samplers.RandomSampler()
    pruner = optuna.pruners.NopPruner()

    lock_file = optuna.storages.JournalFileOpenLock(log_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path=log_path, lock_obj=lock_file),
        # failed_trial_callback=
    )

    study = optuna.create_study(sampler=sampler, direction="minimize", study_name=study_name,
                                storage=storage, pruner=pruner)

    joblib.dump(study, os.path.join(model_path, "study.pkl"))

    start_optuna()
    study.optimize(lambda trial: objective(config, data_module, trial), n_trials=10)
    print(study.best_params)

    study.trials_dataframe().to_csv(os.path.join(model_path, "trials_results.csv"))


    # ris = trainer.logged_metrics
    # print("\n\n\n\n", ris)
    # print(f"Training time {timer_callback.time_elapsed('train')}s")

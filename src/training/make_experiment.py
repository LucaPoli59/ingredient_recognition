from typing import Tuple, Callable, Type, Optional
import torch
import os
import logging
import lightning as lgn
from lightning.pytorch.tuner import Tuner

from settings.config import IMAGES_PATH, RECIPES_PATH, FOOD_CATEGORIES, EXPERIMENTS_PATH, DEF_BATCH_SIZE, DEF_LR
from src.data_processing.data_handling import ImagesRecipesDataModule
from src.training.lgn_models import BaseLGNM, AdvancedLGNM
from src.models.dummy import DummyModel
from src.training.utils import multi_label_accuracy
from src.training.lgn_trainers import TrainerInterface, BaseFasterTrainer, BaseTrainer, LiteTrainer


def make_experiment(
        experiment_name: str,
        model_type: Type[torch.nn.Module],
        lgn_model_type: Type[lgn.LightningModule] = BaseLGNM,
        trainer_type: Type[TrainerInterface] = LiteTrainer,
        experiment_dir: str | None = None,
        image_dir_path: os.path = IMAGES_PATH,
        recipe_dir_path: os.path = RECIPES_PATH,
        category: str | None = None,
        input_shape: Tuple[int, int] = (224, 224),
        max_epochs: int = 20,
        batch_size: Optional[int] = DEF_BATCH_SIZE,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr: Optional[float] = DEF_LR,
        loss_fn: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float] = multi_label_accuracy,
        model_kwargs: dict | None = None,
        debug: bool = False,
        testing: bool = False
):
    category, model_kwargs = _check_inputs(category, input_shape, image_dir_path, recipe_dir_path, model_kwargs)
    # lr, batch_size, tune_lr, tune_batch_size = _assert_lr_batch_tuning(lr, batch_size)
    _assert_lgn_model_trainer_compatibility(lgn_model_type, trainer_type)
    _set_torch_constants()
    save_dir = _prepare_save_dir(experiment_dir, experiment_name)
    if debug:
        print("Experiment directory created")

    # Load the dataset
    data_module = ImagesRecipesDataModule(image_dir_path, recipe_dir_path, category=category,
                                          image_size=input_shape, batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup()
    if debug:
        print("Data module loaded")

    torch_model = model_type(input_shape, data_module.get_num_classes(), **model_kwargs)
    lgn_model = BaseLGNM(torch_model, lr, batch_size, optimizer, loss_fn, accuracy_fn)
    trainer = trainer_type(save_dir=save_dir, max_epochs=max_epochs, debug=False)  # TODO: Put debug=debug

    # # Tuning batch size and LR if not provided
    # tuner = Tuner(trainer)
    # if tune_batch_size:
    #     _tune_batch_size(tuner, lgn_model, data_module, debug)
    # if tune_lr:
    #     _tune_lr(tuner, lgn_model, data_module, debug)

    if debug:
        print("Trainer loaded, training started")

    # Training
    trained_model = trainer.fit(
        model=lgn_model,
        datamodule=data_module
    )

    if debug:
        print("Training completed")
        print(trained_model == torch_model)

    if testing:
        print("Testing the last model...")
        trainer.test(model=lgn_model, datamodule=data_module)
        print(f"Testing the best model...{trainer.model_checkpoint_callback.best_model_path}")
        trainer.test(model=trained_model, datamodule=data_module)


def _set_torch_constants():
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # to remove warning messages
    torch.set_float32_matmul_precision('medium')  # For better performance with cuda
    torch.backends.cudnn.benchmark = True

def _assert_lgn_model_trainer_compatibility(model: Type[lgn.LightningModule], trainer: Type[TrainerInterface]):
    if not issubclass(model, BaseLGNM):
        raise ValueError(f"Model must be a subclass of BaseLightning, got {model}")
    if not issubclass(trainer, TrainerInterface):
        raise ValueError(f"Trainer must be a subclass of TrainerInterface, got {trainer}")


def _prepare_save_dir(experiment_dir: str | os.PathLike, experiment_name: str | os.PathLike) -> str | os.PathLike:
    """Function that creates the directory structure for the experiments and returns the path of the new experiment
    directory with the version number.
    Example: experiment_dir = "experiments/food_classification", experiment_name = "mexican"
    -> "experiments/food_classification/mexican/version_0" """
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    if not os.path.exists(os.path.join(experiment_dir, experiment_name)):
        os.makedirs(os.path.join(experiment_dir, experiment_name))
        version = 0
    else:
        version = _find_next_version(experiment_dir, experiment_name)
    return os.path.join(experiment_dir, experiment_name, f"version_{version}")


def _find_next_version(experiment_dir: str | os.PathLike, experiment_name: str | os.PathLike) -> int:
    exp_vers_files = [exp for exp in os.listdir(str(os.path.join(experiment_dir, experiment_name)))]
    vers = [int(exp_vers.split("_")[1]) for exp_vers in exp_vers_files]
    return len(vers)


def _check_inputs(category: str, input_shape: Tuple[int, int],
                  images_path: str | os.PathLike, recipes_path: str | os.PathLike, model_kwargs: dict | None):
    if category is not None:
        category = category.lower()
        if category == 'all':
            category = None
        elif category not in FOOD_CATEGORIES:
            raise ValueError(f'Invalid category: {category}')

    if input_shape[0] <= 0 or input_shape[1] <= 0:
        raise ValueError(f'Invalid input shape: {input_shape}')
    for path in [images_path, recipes_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path does not exist: {path}')

    if model_kwargs is None:
        model_kwargs = {}

    return category, model_kwargs




if __name__ == "__main__":
    make_experiment("dummy_experiment", DummyModel, category="all", trainer_type=BaseFasterTrainer,
                    experiment_dir=os.path.join(EXPERIMENTS_PATH, "dummy"), batch_size=256, lr=DEF_LR,
                    max_epochs=20, debug=True, testing=True)

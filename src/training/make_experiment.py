from typing import Tuple, Callable, Type
import torch
import os
import logging
import lightning as lgn

from settings.config import IMAGES_PATH, RECIPES_PATH, FOOD_CATEGORIES, EXPERIMENTS_PATH
from src.data_processing.data_handling import ImagesRecipesDataModule
from src.training.lgn_models import BaseLightning
from src.training.lgn_trainers import BaseTrainer
from src.models.dummy import DummyModel
from src.training.utils import multi_label_accuracy


def make_experiment(
        experiment_name: str,
        model: Type[torch.nn.Module],
        experiment_dir: str | None = None,
        image_dir_path: os.path = IMAGES_PATH,
        recipe_dir_path: os.path = RECIPES_PATH,
        category: str | None = None,
        input_shape: Tuple[int, int] = (224, 224),
        max_epochs: int = 20,
        batch_size: int = 128,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 1e-3,
        loss_fn: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float] = multi_label_accuracy,
        model_kwargs: dict | None = None,
):
    category, model_kwargs = _check_inputs(category, input_shape, image_dir_path, recipe_dir_path, model_kwargs)
    _set_torch_constants()

    save_dir = _prepare_save_dir(experiment_dir, experiment_name)

    # Load the dataset
    data_module = ImagesRecipesDataModule(image_dir_path, recipe_dir_path, category=category,
                                          image_size=input_shape, batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup()

    # Model creation
    torch_model = model(input_shape, data_module.get_num_classes(), **model_kwargs)
    lightning_model = BaseLightning(torch_model, lr, optimizer, loss_fn, accuracy_fn)

    # Callback definition

    # Trainer
    trainer = BaseTrainer(
        save_dir=save_dir,
        max_epochs=max_epochs,
        len_train_dataloader=len(data_module.train_dataloader()),
        accelerator="gpu"
    )

    # Training
    trainer.fit(
        model=lightning_model,
        datamodule=data_module
    )


def _set_torch_constants():
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # to remove warning messages
    torch.set_float32_matmul_precision('medium')  # For better performance with cuda


def _prepare_save_dir(experiment_dir: str | os.PathLike, experiment_name: str | os.PathLike) -> str | os.PathLike:
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
    return max(vers) + 1


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
    make_experiment("dummy_experiment", DummyModel, category="mexican",
                    experiment_dir=os.path.join(EXPERIMENTS_PATH, "dummy"),
                    max_epochs=2)
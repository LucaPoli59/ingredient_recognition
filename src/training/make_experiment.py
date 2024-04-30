from typing import Tuple, Callable
import torch
import os
import logging
import pytorch_lightning as lgn

from settings.config import IMAGES_PATH, RECIPES_PATH, FOOD_CATEGORIES
from src.training.metrics_funcs import multi_label_accuracy
from src.data_processing.data_handling import ImagesRecipesDataModule
from src.training.lgn_models import BaseLightning


def make_experiment(
        experiment_name: str,
        model: torch.nn.Module,
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
):
    category = category.lower()
    _check_inputs(category, input_shape, image_dir_path, recipe_dir_path)
    _set_torch_constants()

    # Load the dataset
    data_module = ImagesRecipesDataModule(image_dir_path, recipe_dir_path, category=category,
                                          image_size=input_shape, batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup()

    # Model creation
    torch_model = model(input_shape, data_module.get_num_classes())
    lightning_model = BaseLightning(torch_model, lr, optimizer, loss_fn, accuracy_fn)

    # Callback definition

    # Trainer
    trainer = lgn.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        # fast_dev_run=True,
        # profiler=profiler,
        enable_model_summary=False,
    )

    # Training
    trainer.fit(
        model=lightning_model,
        datamodule=data_module
    )


def _set_torch_constants():
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # to remove warning messages
    torch.set_float32_matmul_precision('medium')  # For better performance with cuda


def _check_inputs(category, input_shape, images_path, recipes_path):
    if category not in FOOD_CATEGORIES or category is None or category == "all":
        raise ValueError(f'Invalid category: {category}')
    if input_shape[0] <= 0 or input_shape[1] <= 0:
        raise ValueError(f'Invalid input shape: {input_shape}')
    for path in [images_path, recipes_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path does not exist: {path}')


if __name__ == "__main__":
    pass

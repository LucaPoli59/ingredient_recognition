import json
import os
from typing import Tuple, Dict, Any, List
import torch
import logging
import lightning as lgn
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import warnings
import sys
import importlib


def _extract_name_trial_dir(save_dir: str) -> Tuple[str, str, str]:
    """Function that extract the path of the experiments directory, the experiment name and the experiment trial
    from the save_dir.  Example: save_dir = "experiments/food_classification/mexican/1"
    -> ("experiments/food_classification", "mexican", "1")"""
    exp_vers, exp_name = save_dir, os.path.dirname(save_dir)
    exps_dir = os.path.dirname(exp_name)
    return exps_dir, os.path.split(exp_name)[1], os.path.split(exp_vers)[1]


def multi_label_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    y_pred = torch.round(torch.sigmoid(y_pred))  # get the binary predictions
    # num of hits / num of classes mean over the batch
    return torch.mean((y_pred == y_true).sum(dim=1) / y_pred.size(1))


class CSVLoggerQuiet(CSVLogger):
    """Simple wrapper for lighting CSVLogger, that disable a warning message,
    useful when multiple loggers use the same directory"""

    def __init__(self, *args, **kwargs):
        warnings.filterwarnings(
            "ignore",
            f"Experiment logs directory .*. exists and is not empty. "
            f"Previous log files in this directory will be deleted when the new ones are saved")
        super().__init__(*args, **kwargs)


def register_hparams(elem: lgn.LightningModule | lgn.LightningDataModule, hparams: List[Dict[str, Any] | str],
                     log=True, ) -> None:
    """Function that register the hyperparameters to the elem """
    param_list = []
    param_dicts = []
    for param in hparams:
        if isinstance(param, dict):
            param_dicts.append(param)
        else:
            param_list.append(param)

    if len(param_list) > 0:
        elem.save_hyperparameters(*param_list, logger=log)
    for param_dict in param_dicts:
        elem.save_hyperparameters(param_dict, logger=log)


def str_to_class(class_str):
    class_str = class_str.removeprefix("<class '").removesuffix("'>")
    module_name, classname = class_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, classname)


def func_to_str(func):
    return f"<function '{func.__module__}.{func.__name__}'>"


def str_to_func(func_str):
    func_str = func_str.removeprefix("<function '").removesuffix("'>")
    module_name, func_name = func_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

import json
import logging
import os
from typing import Tuple, Dict, Any, List, Set

import numpy as np
import torch
import lightning as lgn
import inspect
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


def register_hparams(elem: lgn.LightningModule | lgn.LightningDataModule,
                     hparams: List[Dict[str, Any] | str] | Set[Dict[str, Any] | str], log=True) -> None:
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


def encode_config(config: Dict[str, Any]) -> Dict[str, Tuple[str, str | Dict]]:
    """Function that encodes the config dictionary, converting the values to a tuple of the type and the str value"""
    new_config = {}
    for key, value in config.items():
        if value is None:
            new_config[key] = ("None", "None")
        if isinstance(value, (int, float, bool, str)):
            new_config[key] = (type(value).__name__, value)
        elif inspect.isclass(value):
            new_config[key] = ("class", str(value))
        elif inspect.isfunction(value):
            new_config[key] = ("function", func_to_str(value))
        elif isinstance(value, (list, tuple)):
            new_config[key] = ("list", json.dumps(value))
        elif isinstance(value, np.ndarray):
            new_config[key] = ("ndarray", json.dumps(value.tolist()))
        elif isinstance(value, dict):
            new_config[key] = ("config", encode_config(value))

    return new_config


def enc_config_to_yaml(config: Dict[str, Tuple[str, str | Dict]]) -> Dict[str, Any]:
    """Function that converts the encoded config dictionary to a dictionary without the type field"""
    return {key: value[1] if value[0] != "config" else enc_config_to_yaml(value[1]) for key, value in config.items()}


def decode_config(config: Dict[str, Tuple[str, str | Dict]]) -> Dict[str, Any]:
    """Function that decodes the config dictionary, converting the values to the original type"""
    new_config = {}
    for key, (type_str, value) in config.items():
        match type_str:
            case "None":
                new_config[key] = None
            case "int" | "float" | "bool" | "str":
                new_config[key] = value
            case "class":
                new_config[key] = str_to_class(value)
            case "function":
                new_config[key] = str_to_func(value)
            case "list":
                new_config[key] = json.loads(value)
            case "ndarray":
                new_config[key] = np.array(json.loads(value))
            case "config":
                new_config[key] = decode_config(value)

    return new_config

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


def set_torch_constants():
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # to remove warning messages
    torch.set_float32_matmul_precision('medium')  # For better performance with cuda
    torch.backends.cudnn.benchmark = True

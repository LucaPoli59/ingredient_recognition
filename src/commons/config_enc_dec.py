import importlib
import inspect
import json
from typing import Dict, Any, Tuple

import numpy as np


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


def decode_config(config: Dict[str, Tuple[str, str | Dict]], raise_lambda=True) -> Dict[str, Any]:
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
                if str_func_is_lambda(value):
                    if raise_lambda:
                        raise ValueError("Lambda functions are not supported")
                    else:
                        new_config[key] = None
                else:
                    new_config[key] = str_to_func(value)
            case "list":
                new_config[key] = json.loads(value)
            case "ndarray":
                new_config[key] = np.array(json.loads(value))
            case "config":
                new_config[key] = decode_config(value, raise_lambda=raise_lambda)

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


def str_func_is_lambda(func_str):
    func_str = func_str.removeprefix("<function '").removesuffix("'>")
    module_name, func_name = func_str.rsplit(".", 1)
    return module_name == "__main__" and func_name == "<lambda>"

def is_lambda(func):
    return callable(func) and func.__name__ == "<lambda>"

import os
from typing import Tuple
import torch
import logging
from lightning.pytorch.loggers import CSVLogger


def _extract_name_version_dir(save_dir: str) -> Tuple[str, str, str]:
    """Function that extract the path of the experiments directory, the experiment name and the experiment version
    from the save_dir.  Example: save_dir = "experiments/food_classification/mexican/1"
    -> ("experiments/food_classification", "mexican", "1")"""
    exp_vers, exp_name = save_dir, os.path.dirname(save_dir)
    exps_dir = os.path.dirname(exp_name)
    return exps_dir, os.path.split(exp_name)[1], os.path.split(exp_vers)[1]


def multi_label_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    y_pred = torch.round(torch.sigmoid(y_pred))  # get the binary predictions
    # num of hits / num of classes mean over the batch
    return torch.mean((y_pred == y_true).sum(dim=1) / y_pred.size(1))


import warnings


class CSVLoggerQuiet(CSVLogger):
    """Simple wrapper for lighting CSVLogger, that disable a warning message,
    useful when multiple loggers use the same directory"""
    def __init__(self, *args, **kwargs):
        warnings.filterwarnings(
            "ignore",
            f"Experiment logs directory .*. exists and is not empty. "
            f"Previous log files in this directory will be deleted when the new ones are saved")
        super().__init__(*args, **kwargs)

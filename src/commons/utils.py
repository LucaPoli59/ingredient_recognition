import os
from typing import List, Dict, Any, Set, Tuple

import lightning as lgn
import torch
from torch import Tensor
from torchmetrics import Metric


def pred_digits_to_values(y_pred: torch.Tensor) -> torch.Tensor:
    """function that converts the predicted digits to the binary values
    Example: -5.36 -> 0, -1.3 -> 1"""
    return torch.round(torch.sigmoid(y_pred))


def multi_label_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    y_pred = pred_digits_to_values(y_pred)
    # num of hits / num of classes mean over the batch
    return torch.mean((y_pred == y_true).sum(dim=1) / y_pred.size(1))


class MyMLAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._input_format(preds, target)

        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        if preds[0] < 0:  # if the preds are digits
            preds = pred_digits_to_values(preds)

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total


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


def extract_name_trial_dir(save_dir: str) -> Tuple[str, str, str]:
    """Function that extract the path of the experiments directory, the experiment name and the experiment trial
    from the save_dir.  Example: save_dir = "experiments/food_classification/mexican/1"
    -> ("experiments/food_classification", "mexican", "1")"""
    exp_vers, exp_name = save_dir, os.path.dirname(save_dir)
    exps_dir = os.path.dirname(exp_name)
    return exps_dir, os.path.split(exp_name)[1], os.path.split(exp_vers)[1]



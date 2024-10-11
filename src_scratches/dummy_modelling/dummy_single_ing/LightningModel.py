import logging
from typing import Any, Tuple, Union, Sequence
import lightning as lgn
from lightning import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
import numpy as np

from src_scratches.dummy_modelling.dummy_single_ing._commons import WARMUP_INIT_LR_COEFF, WARMUP_EPOCHS
from src.models.custom_schedulers import WarmStartReduceOnPlateau, ConstantStartReduceOnPlateau

def compute_classes_weights(label_data, minority_inversion=True, standardize=True):
    """Computes the class weights for the dataset labels"""

    classes_occ = np.sum(label_data, axis=0, dtype=np.float32)
    classes_occ[classes_occ == 0] = np.NaN  # Put NaNs in the classes that are not present in the dataset

    class_weights = np.nansum(classes_occ) / classes_occ

    if not minority_inversion:
        class_weights = 1 / class_weights

    if standardize:
        class_weights = class_weights / class_weights[~np.isnan(class_weights)].min()

    return torch.tensor(np.nan_to_num(class_weights), dtype=torch.float32)


class LightningModel(lgn.LightningModule):
    def __init__(self, model, lr, optimizer, loss_fn, accuracy_fn, n_samples, batch_size, total_steps,
                 momentum=None, weight_decay=None, swa=False, weighted_loss=False, problem=True,
                 model_pretrained=False, lr_scheduler=None):
        super().__init__()
        self._prepared = False
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.swa = swa
        self.weighted_loss = weighted_loss
        self.problem = problem
        self.model_pretrained = model_pretrained
        self.lr_scheduler = lr_scheduler

        self.total_steps = total_steps

        self.momentum = momentum if momentum is not None else 0
        self.weight_decay = weight_decay if weight_decay is not None else 0

        self.save_hyperparameters(ignore=["model", "loss_fn", "accuracy_fn"])  # For logging purposes

        # logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)  # to remove warning messages

    def startup_model(self, train_data):
        if self._prepared:
            return
        self._startup(train_data)
        self._prepared = True

    def _startup(self, train_data):
        self._init_loss(train_data)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer == torch.optim.SGD:
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                            weight_decay=self.weight_decay)
        else:
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler is not None:
            if self.lr_scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.lr_scheduler = self.lr_scheduler(self.optimizer, mode="min", factor=0.05, patience=5, cooldown=1,
                                                  min_lr=1e-6)
                
            elif self.lr_scheduler == torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
                self.lr_scheduler = self.lr_scheduler(self.optimizer, T_0 = 5, T_mult=2, eta_min=1e-6)
            elif self.lr_scheduler == torch.optim.lr_scheduler.OneCycleLR:
                self.lr_scheduler = self.lr_scheduler(self.optimizer, max_lr=self.lr * 100, total_steps=self.total_steps)

            elif self.lr_scheduler == WarmStartReduceOnPlateau:
                self.lr_scheduler = self.lr_scheduler(self.optimizer, warm_start=self.lr, warm_stop=self.lr * WARMUP_INIT_LR_COEFF,
                                                      warm_duration=WARMUP_EPOCHS, warm_type="linear", mode="min",
                                                      patience=5, cooldown=1, min_lr=1e-6, factor=0.05)
            elif self.lr_scheduler == ConstantStartReduceOnPlateau:
                self.lr_scheduler = self.lr_scheduler(self.optimizer, initial_lr=self.lr, warm_duration=WARMUP_EPOCHS,
                                                      mode="min", patience=5, cooldown=1, min_lr=1e-6, factor=0.05)
            else:
                raise ValueError(f"LR scheduler {self.lr_scheduler} not recognized")
            return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler, 'monitor': 'train_loss'}
        else:
            return self.optimizer



    def _base_step(self, batch) -> Tuple[Any, Any]:
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        with torch.no_grad():
            acc = self.accuracy_fn(y_pred, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._base_step(batch)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._base_step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._base_step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc})

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        callbacks = []
        if self.swa:
            callbacks += [lgn.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=0.5 * self.lr)]

        return callbacks

    def setup(self, stage: str) -> None:
        if not self._prepared:
            raise ValueError(f"Model not prepared. Call startup_model before {stage}")

    def _init_loss(self, train_data):
        if not self.weighted_loss:
            self.loss_fn = self.loss_fn()
        else:
            weights = compute_classes_weights(train_data.label_data)
            if self.loss_fn == torch.nn.BCEWithLogitsLoss:
                self.loss_fn = self.loss_fn(pos_weight=weights)
            elif self.loss_fn == torch.nn.CrossEntropyLoss:
                self.loss_fn = self.loss_fn(weight=weights)
            else:
                raise ValueError(f"Loss type {self.loss_fn} not recognized")

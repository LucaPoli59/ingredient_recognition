import logging
from typing import Any, Tuple
import lightning as lgn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch


class BaseLightning(lgn.LightningModule):
    def __init__(self, model, lr, optimizer, loss_fn, accuracy_fn):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.optimizer = optimizer
        self.save_hyperparameters(ignore=["model", "loss_fn", "accuracy_fn"])  # For logging purposes

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
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

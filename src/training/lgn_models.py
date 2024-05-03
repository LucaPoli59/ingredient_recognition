from typing import Any, Tuple
import lightning as lgn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch


class BaseLGNM(lgn.LightningModule):
    def __init__(self, model, lr, batch_size, optimizer, loss_fn, accuracy_fn, model_name=None):
        super().__init__()
        self._model = model
        self._lr = lr
        self._batch_size = batch_size
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.optimizer = optimizer
        self.model_name = model_name

        save_hp = ["lr", "batch_size"] + (["model_name"] if model_name is not None else [])
        self.save_hyperparameters(*save_hp)  # For logging purposes

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.hparams.lr)
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

    @property
    def model(self):
        return self._model

    @property
    def p_lr(self):
        return self.hparams.lr

    @p_lr.setter
    def p_lr(self, lr):
        if lr < 0:
            raise ValueError("Learning rate must be positive")
        self.hparams.lr = lr

    @property
    def p_batch_size(self):
        return self.hparams.batch_size

    @p_batch_size.setter
    def p_batch_size(self, batch_size):
        if batch_size < 1:
            raise ValueError("Batch size must be greater than 0")
        self.hparams.batch_size = batch_size


class AdvancedLGNM(BaseLGNM): # For more advanced models (for hparams fine tuning)
    def __init__(self, model, lr, batch_size, optimizer, loss_fn, accuracy_fn, model_name=None):
        super().__init__(model, lr, batch_size, optimizer, loss_fn, accuracy_fn, model_name)



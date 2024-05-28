from typing import Any, Tuple, Dict, Optional
import lightning as lgn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch

from src.training.commons import register_hparams


class BaseLGNM(lgn.LightningModule):
    def __init__(self, model, lr, batch_size, optimizer, loss_fn, accuracy_fn, model_name=None,
                 hparams_to_register=None):
        super().__init__()
        self._model = model
        self._lr = lr
        self._batch_size = batch_size
        self.loss_fn = loss_fn  # loss class
        self.accuracy_fn = accuracy_fn  # Function pointer
        self.optimizer = optimizer  # Optimizer class

        self.input_shape = self._model.input_shape
        self.num_classes = self._model.num_classes
        self.torch_model_type = type(self._model)
        self.model_name = model_name

        hparams = ([{"lr": self._lr}, {"batch_size": self._batch_size}, {"optimizer": self.optimizer},
                   {"loss_fn": self.loss_fn}, {"lgn_model_type": self.__class__},
                   {"torch_model_type": self.torch_model_type}, {"accuracy_fn": self.accuracy_fn},
                   {"input_shape": self.input_shape}, {"num_classes": self.num_classes}]
                   + (["model_name"] if model_name is not None else []))

        if hparams_to_register is not None:
            hparams = [hparam for hparam in hparams if list(hparam.keys())[0] in hparams_to_register]

        register_hparams(self, hparams)

        self.loss_fn = loss_fn()  # After registering the loss_fn, we need to instantiate it

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
    def lr(self):
        return self.hparams.lr

    @lr.setter
    def lr(self, lr):
        if lr < 0:
            raise ValueError("Learning rate must be positive")
        self.hparams.lr = lr

    @property
    def batch_size(self):
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        if batch_size < 1:
            raise ValueError("Batch size must be greater than 0")
        self.hparams.batch_size = batch_size

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], torch_model_kwargs: Optional[Dict[str, Any]] = None,
                         lgn_model_kwargs: Optional[Dict[str, Any]] = None) -> "BaseLGNM":
        if torch_model_kwargs is None:
            torch_model_kwargs = {}
        if lgn_model_kwargs is None:
            lgn_model_kwargs = {}

        if not issubclass(config['lgn_model_type'], cls):
            raise ValueError(f"Invalid model type. Expected {cls} but got {config['lgn_model_type']}")

        input_shape, num_classes = tuple(config['input_shape']), config['num_classes']
        batch_size, lr = config['batch_size'], config['lr']
        torch_model_type = config['torch_model_type']
        loss_fn, accuracy_fn = config['loss_fn'], config['accuracy_fn']
        optimizer, model_name = config['optimizer'], config.get('model_name', None)

        torch_model = torch_model_type(input_shape, num_classes, **torch_model_kwargs)
        lgn_model = cls(torch_model, lr, batch_size, optimizer, loss_fn, accuracy_fn, model_name, **lgn_model_kwargs)
        return lgn_model

    def load_weights_from_checkpoint(self, checkpoint_path: str):
        self.load_state_dict(torch.load(checkpoint_path)['state_dict'])


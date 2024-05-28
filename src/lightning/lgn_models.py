from typing import Any, Dict, Optional, List, Type
import lightning as lgn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torchmetrics import Metric
import inspect

from src.commons.utils import register_hparams


class BaseLGNM(lgn.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float, batch_size: int, optimizer: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 metrics: Type[Metric] | Dict[str, Type[Metric] | Dict[str, Dict[str, Any] | Type[Metric]]],
                 model_name: Optional[str] = None,
                 hparams_to_register: Optional[List[str]] = None):
        super().__init__()
        self._model = model
        self._lr = lr
        self._batch_size = batch_size
        self.loss_fn = loss_fn  # loss class
        self.metrics = self._parse_metrics(metrics)  # dict to metrics class pointer and their parameters
        self.optimizer = optimizer  # Optimizer class

        self.input_shape = self._model.input_shape
        self.num_classes = self._model.num_classes
        self.torch_model_type = type(self._model)
        self.model_name = model_name

        hparams = ([{"lr": self._lr}, {"batch_size": self._batch_size}, {"optimizer": self.optimizer},
                    {"loss_fn": self.loss_fn}, {"lgn_model_type": self.__class__},
                    {"torch_model_type": self.torch_model_type}, {"metrics": self.metrics},
                    {"input_shape": self.input_shape}, {"num_classes": self.num_classes}]
                   + (["model_name"] if model_name is not None else []))

        if hparams_to_register is not None:
            hparams = [hparam for hparam in hparams if list(hparam.keys())[0] in hparams_to_register]

        register_hparams(self, hparams)

        self.loss_fn = loss_fn()  # After registering the loss_fn, we need to instantiate it
        for metric in self.metrics.values():
            num_labels = metric['init_params'].get('num_labels', None)
            metric['init_params']['num_labels'] = num_labels if num_labels is not None else self.num_classes
            metric['obj']: Metric = metric['obj'](**metric['init_params'])

    @staticmethod
    def _parse_metrics(metrics_in) -> Dict[str, Dict[str, Type[Metric] | Dict[str, Any]]]:
        if inspect.isfunction(metrics_in):
            raise ValueError("Metrics must be a class, not a function")
        if inspect.isclass(metrics_in):
            return {'metric': dict(obj=metrics_in, init_params={}, logging_params={})}
        if isinstance(metrics_in, dict):
            for name, metric in metrics_in.items():
                if inspect.isclass(metric):
                    metrics_in[name] = dict(obj=metric, init_params={}, logging_params={})
                elif isinstance(metric, dict):
                    if 'obj' not in metric:
                        raise ValueError("Metrics must have the 'obj' key")
                    if 'init_params' not in metric:
                        metric['init_params'] = {}
                    if 'logging_params' not in metric:
                        metric['logging_params'] = {}
                else:
                    raise ValueError("Metric type not recognized")
            return metrics_in

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.hparams.lr)
        return self.optimizer

    def _base_step(self, batch) -> float:
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)

        for metric in self.metrics.values():
            metric['obj'](y_pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._base_step(batch)

        self.log("train_loss", loss, prog_bar=True, on_epoch=False, on_step=True)
        for metric_name, metric in self.metrics.items():
            self.log(f"train_{metric_name}", metric['obj'], **metric['logging_params'])

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._base_step(batch)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        for metric_name, metric in self.metrics.items():
            self.log(f"val_{metric_name}", metric['obj'], **metric['logging_params'])

        return loss

    def test_step(self, batch, batch_idx):
        loss = self._base_step(batch)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        for metric_name, metric in self.metrics.items():
            self.log(f"test_{metric_name}", metric['obj'], **metric['logging_params'])

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
        loss_fn, metrics = config['loss_fn'], config['metrics']
        optimizer, model_name = config['optimizer'], config.get('model_name', None)

        torch_model = torch_model_type(input_shape, num_classes, **torch_model_kwargs)
        lgn_model = cls(torch_model, lr, batch_size, optimizer, loss_fn, metrics, model_name, **lgn_model_kwargs)
        return lgn_model

    def load_weights_from_checkpoint(self, checkpoint_path: str):
        self.load_state_dict(torch.load(checkpoint_path)['state_dict'])

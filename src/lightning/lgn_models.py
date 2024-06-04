from typing import Any, Dict, Optional, List, Type, Tuple
import lightning as lgn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torchmetrics import Metric, MetricCollection
import inspect

from src.commons.utils import register_hparams
from src.commons.visualizations import gradcam


class BaseLGNM(lgn.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float, batch_size: int, optimizer: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 metrics: Optional[Type[Metric] | Dict[str, Type[Metric] | Dict[str, Dict[str, Any] | Type[Metric]]]
                                   ] = None,
                 model_name: Optional[str] = None,
                 hparams_to_register: Optional[List[str]] = None):
        """
        Initialize the BaseLGNM class
        :param model: underlying model (torch model instance)
        :param lr: learning rate
        :param batch_size: batch size
        :param optimizer: optimized to use (provided as a class)
        :param loss_fn: loss function to use (provided as a class)
        :param metrics: metrics to use (provided as a class or a dict with the name as key and as value a dict with
                        the keys 'type' (metric class) and 'init_params' (dict) and 'logging_params' (dict))
                        (by default is empty)

                        (by default is the property gradcam_target_layer of the model)
        :param model_name:
        :param hparams_to_register:
        """
        super().__init__()
        self._model = model
        self._lr = lr
        self._batch_size = batch_size
        self.loss_fn = loss_fn  # loss class
        self.metrics_config = self._parse_metrics_config(metrics)  # dict to metrics class pointer and their parameters
        self.optimizer = optimizer  # Optimizer class

        self.input_shape = self._model.input_shape
        self.num_classes = self._model.num_classes
        self.torch_model_type = type(self._model)
        self.model_name = model_name

        self.gradcam_target_layer = getattr(self._model, "gradcam_target_layer", None)

        hparams = ([{"lr": self._lr}, {"batch_size": self._batch_size}, {"optimizer": self.optimizer},
                    {"loss_fn": self.loss_fn}, {"lgn_model_type": self.__class__},
                    {"torch_model_type": self.torch_model_type}, {"metrics": self.metrics_config},
                    {"input_shape": self.input_shape}, {"num_classes": self.num_classes}]
                   + (["model_name"] if model_name is not None else []))

        if hparams_to_register is not None:
            hparams = [hparam for hparam in hparams if list(hparam.keys())[0] in hparams_to_register]

        register_hparams(self, hparams)

        self.loss_fn = loss_fn()  # After registering the loss_fn, we need to instantiate it
        metrics = MetricCollection(self._init_metrics())  # Initialize the metrics
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    @staticmethod
    def _parse_metrics_config(metrics_in) -> Dict[str, Dict[str, Type[Metric] | Dict[str, Any]]]:
        if metrics_in is None:
            return {}
        if inspect.isfunction(metrics_in):
            raise ValueError("Metrics must be a class, not a function")
        if inspect.isclass(metrics_in):
            return {'metric': dict(type=metrics_in, init_params={}, logging_params={})}
        if isinstance(metrics_in, dict):
            for name, metric in metrics_in.items():
                if inspect.isclass(metric):
                    metrics_in[name] = dict(type=metric, init_params={}, logging_params={})
                elif isinstance(metric, dict):
                    if 'type' not in metric:
                        raise ValueError("Metrics must have the 'type' key")
                    if 'init_params' not in metric:
                        metric['init_params'] = {}
                    if 'logging_params' not in metric:
                        metric['logging_params'] = {}
                else:
                    raise ValueError("Metric type not recognized")
            return metrics_in

    def _init_metrics(self) -> Dict[str, Metric]:
        metrics = {}
        for metric_name, metric_conf in self.metrics_config.items():
            if 'num_labels' in metric_conf['init_params']:
                num_labels = metric_conf['init_params']['num_labels']
                metric_conf['init_params']['num_labels'] = num_labels if num_labels is not None else self.num_classes
            metrics[metric_name] = metric_conf['type'](**metric_conf['init_params'])
            # metrics[metric_name].persistent(True)
        return metrics

    def _log_metric_collections(self, metric_out, prefix):
        for metric_name, metric in metric_out.items():
            self.log(metric_name, metric, **self.metrics_config[metric_name.removeprefix(prefix)]['logging_params'])

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.hparams.lr)
        return self.optimizer

    def _base_step(self, batch, metrics) -> Tuple[float, Dict[str, torch.Tensor]]:
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        metrics_out = metrics(y_pred, y)

        return loss, metrics_out

    def training_step(self, batch, batch_idx):
        loss, metrics_out = self._base_step(batch, self.train_metrics)

        self.log("train_loss", loss, prog_bar=True, on_epoch=False, on_step=True)
        self._log_metric_collections(metrics_out, self.train_metrics.prefix)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics_out = self._base_step(batch, self.val_metrics)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self._log_metric_collections(metrics_out, self.val_metrics.prefix)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics_out = self._base_step(batch, self.test_metrics)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self._log_metric_collections(metrics_out, self.test_metrics.prefix)
        return loss

    def predict_step(self, batch, batch_idx) -> Any:
        X, y = batch
        if self.gradcam_target_layer is None:
            return X, self.model(X), y
        cam_imgs, cam_masks, targets, outputs = gradcam(self.model, self.gradcam_target_layer, X)
        return X, outputs, y, cam_imgs, targets  # todo controlla se va bene

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

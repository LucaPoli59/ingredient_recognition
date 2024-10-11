from typing import Any, Dict, Optional, List, Type, Tuple
from typing_extensions import Self
import lightning as lgn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torchmetrics import Metric, MetricCollection
import inspect

from src.commons.utils import register_hparams
from src.commons.visualizations import gradcam
from src.models.commons import BaseModel
from src.data_processing.common import BaseDataModule


class BaseLGNM(lgn.LightningModule):
    def __init__(self, model: BaseModel, lr: float, batch_size: int, optimizer: Type[torch.optim.Optimizer],
                 loss_fn: torch.nn.Module, weight_loss: bool = False, momentum: Optional[float] = None,
                 weight_decay: Optional[float] = None, use_swa: bool = False,
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
        :param weight_loss: if True, the loss function will be weighted
        :param momentum: momentum for the optimizer (by default is None) [works only with SGD]
        :param weight_decay: weight decay for the optimizer (by default is None)
        :param use_swa: if True, the model will use Stochastic Weight Averaging
        :param metrics: metrics to use (provided as a class or a dict with the name as key and as value a dict with
                        the keys 'type' (metric class) and 'init_params' (dict) and 'logging_params' (dict))
                        (by default is empty)

                        (by default is the property gradcam_target_layer of the model)
        :param model_name:
        :param hparams_to_register:
        """
        super().__init__()
        self.prepared = False
        self._model = model
        self._lr = lr
        self._batch_size = batch_size
        self.loss_fn = loss_fn  # loss class
        self.weight_loss = weight_loss
        self.optimizer = optimizer  # Optimizer class
        self.momentum_val = momentum
        self.weight_decay_val = weight_decay
        self.use_swa = use_swa
        self.metrics_config = self._parse_metrics_config(metrics)  # dict to metrics class pointer and their parameters
        self.model_name = model_name

        self.num_classes = self._model.num_classes
        self.torch_model_type = type(self._model)
        self.gradcam_target_layer = getattr(self._model, "gradcam_target_layer", None)

        hparams = ([{"lr": self._lr}, {"batch_size": self._batch_size}, {"optimizer": self.optimizer},
                    {"momentum": self.momentum_val}, {"weight_decay": self.weight_decay_val},
                    {"loss_fn": self.loss_fn}, {"weight_loss": self.weight_loss}, {"lgn_model_type": self.__class__},
                    {"use_swa": self.use_swa},
                    {"torch_model": self._model.to_config()}, {"metrics": self.metrics_config},
                    {"num_classes": self.num_classes}] + (["model_name"] if model_name is not None else []))

        if hparams_to_register is not None:
            hparams = [hparam for hparam in hparams if list(hparam.keys())[0] in hparams_to_register]

        register_hparams(self, hparams)

        metrics = MetricCollection(self._init_metrics())  # Initialize the metrics
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def startup_model(self, datamodule: BaseDataModule):
        if not self.prepared:
            self._startup(datamodule)
            self.prepared = True

    def _startup(self, datamodule: BaseDataModule):
        self._init_loss(datamodule)

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
        weight_decay = 0 if self.weight_decay_val is None else self.weight_decay_val
        if self.optimizer == torch.optim.SGD:
            momentum = 0 if self.momentum_val is None else self.momentum_val
            self.optimizer = self.optimizer(self.model.parameters(), lr=self._lr, momentum=momentum,
                                            weight_decay=weight_decay)
        else:
            self.optimizer = self.optimizer(self.model.parameters(), lr=self._lr, weight_decay=weight_decay)
        return self.optimizer

    def configure_callbacks(self) -> List[lgn.pytorch.callbacks.Callback] | lgn.pytorch.callbacks.Callback:
        callbacks = []
        if self.use_swa:
            callbacks += [lgn.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=0.5 * self.lr)]

        return callbacks

    def setup(self, stage: str) -> None:  # Called by the fit of super() before training loop
        if not self._prepared:
            raise ValueError(f"Model not prepared. Call startup_model before {stage}")

    def _init_loss(self, datamodule: BaseDataModule):
        if not self.weighted_loss:
            self.loss_fn = self.loss_fn()
        else:
            weights = datamodule.classes_weights
            if self.loss_fn == torch.nn.BCEWithLogitsLoss:
                self.loss_fn = self.loss_fn(pos_weight=weights)
            elif self.loss_fn == torch.nn.CrossEntropyLoss:
                self.loss_fn = self.loss_fn(weight=weights)
            else:
                raise ValueError(f"Loss type {self.loss_fn} not recognized")

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

    @property
    def lr_swa(self):
        return self.hparams.lr * 0.5

    @batch_size.setter
    def batch_size(self, batch_size):
        if batch_size < 1:
            raise ValueError("Batch size must be greater than 0")
        self.hparams.batch_size = batch_size

    @property
    def transform_plain(self):
        return self._model.transform_plain

    @property
    def transform_aug(self):
        return self._model.transform_aug

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], torch_model_kwargs: Optional[Dict[str, Any]] = None,
                         lgn_model_kwargs: Optional[Dict[str, Any]] = None) -> Self:
        if torch_model_kwargs is None:
            torch_model_kwargs = {}
        if lgn_model_kwargs is None:
            lgn_model_kwargs = {}

        if not issubclass(config['lgn_model_type'], cls):
            raise ValueError(f"Invalid LGN model type. Expected {cls} but got {config['lgn_model_type']}")

        batch_size, lr = config['batch_size'], config['lr']
        loss_fn, metrics = config['loss_fn'], config['metrics']
        momentum, weight_decay = config.get('momentum', None), config.get('weight_decay', None)
        use_swa = config.get('use_swa', False)
        optimizer, model_name = config['optimizer'], config.get('model_name', None)

        torch_model_config = config['torch_model']
        torch_model = torch_model_config['type'].load_from_config(torch_model_config, **torch_model_kwargs)

        lgn_model = cls(torch_model, lr, batch_size, optimizer, loss_fn, momentum=momentum, weight_decay=weight_decay,
                        use_swa=use_swa, metrics=metrics, model_name=model_name,
                        **lgn_model_kwargs)
        return lgn_model

    def load_weights_from_checkpoint(self, checkpoint_path: str):
        self.load_state_dict(torch.load(checkpoint_path)['state_dict'])


class BaseWithSchedulerLGNM(BaseLGNM):
    def __init__(self, model: BaseModel, lr: float, batch_size: int, optimizer: Type[torch.optim.Optimizer],
                 loss_fn: torch.nn.Module, weighted_loss: bool = False, momentum: Optional[float] = None, weight_decay: Optional[float] = None,
                 lr_scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]] = None,
                 lr_scheduler_params: Optional[Dict[str, Any]] = None,
                 metrics: Optional[Type[Metric] | Dict[str, Type[Metric] | Dict[str, Dict[str, Any] | Type[Metric]]]
                                   ] = None,
                 model_name: Optional[str] = None,
                 hparams_to_register: Optional[List[str]] = None):

        # by using a scheduler we can increase the starting LR
        super().__init__(model, lr, batch_size, optimizer, loss_fn, weighted_loss, momentum, weight_decay,
                         metrics, model_name, hparams_to_register)

        if lr_scheduler_params is None:
            lr_scheduler_params = {}
        self.lr_scheduler_params = lr_scheduler_params
        self.lr_scheduler = lr_scheduler

        hparams = [{"lr_scheduler": self.lr_scheduler}, {"lr_scheduler_params": self.lr_scheduler_params}]

        if hparams_to_register is not None:
            hparams = [hparam for hparam in hparams if list(hparam.keys())[0] in hparams_to_register]

        register_hparams(self, hparams)

    @property
    def lr_swa(self):
        return max(self.hparams.lr / 100, self.lr_scheduler_params['min_lr']) * 0.5

    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.optimizer = super().configure_optimizers()
        if self.lr_scheduler is None:
            return self.optimizer

        self.lr_scheduler = self.lr_scheduler(self.optimizer, **self.lr_scheduler_params)
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler, 'monitor': 'val_loss'}

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], torch_model_kwargs: Optional[Dict[str, Any]] = None,
                         lgn_model_kwargs: Optional[Dict[str, Any]] = None) -> Self:
        if lgn_model_kwargs is None:
            lgn_model_kwargs = {}
        lgn_model_kwargs['lr_scheduler'] = config.get('lr_scheduler', None)
        lgn_model_kwargs['lr_scheduler_params'] = config.get('lr_scheduler_params', None)

        return super().load_from_config(config, torch_model_kwargs, lgn_model_kwargs)

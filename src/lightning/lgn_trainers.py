import os
import random
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
import optuna
from typing_extensions import Never, Self
import lightning as lgn
from lightning.pytorch import callbacks
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import Logger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
from optuna_integration import PyTorchLightningPruningCallback

from settings.config import EXPERIMENTS_PATH, EXPERIMENTS_TRASH_PATH
from src.commons.utils import extract_name_trial_dir

from src.lightning.lgn_models import BaseLGNM
from src.lightning.custom_callbacks import (FullModelCheckpoint, LightModelCheckpoint, TensorBoardEncodeLogger,
                                            CSVLoggerEncode)


class TrainerInterface(ABC, lgn.Trainer):
    def __init__(self,
                 max_epochs: int,
                 save_dir: Optional[str | os.PathLike] = None,
                 debug: bool = False,
                 **kwargs):
        self._max_epochs = max_epochs
        self._debug = debug

        self._save_dir = save_dir if save_dir is not None else self._get_def_save_dir()
        self._precision = kwargs.pop("precision", self._get_precision())
        self._grad_accum = kwargs.pop("accumulate_grad_batches", self._get_grad_accum())
        self._check_val_freq = kwargs.pop("check_val_every_n_epoch", self._compute_check_val_freq())

        self._chkp_callback = kwargs.pop("model_checkpoint_callback", self._get_chkp_callback())
        self._callbacks = kwargs.pop("callbacks", self._get_callbacks())
        self._loggers = kwargs.pop("logger", self._get_loggers())
        self._profiler = kwargs.pop("profiler", self._get_profiler())

        grad_clip_val, grad_clip_algo = self._get_grad_clip()
        self._grad_clip_val = kwargs.pop("gradient_clip_val", grad_clip_val)
        self._grad_clip_algo = kwargs.pop("gradient_clip_algorithm", grad_clip_algo)

        self.hparams = {'max_epochs': self._max_epochs, 'save_dir': self._save_dir, 'debug': self._debug,
                        'type': self.__class__}

        super().__init__(
            max_epochs=self._max_epochs,
            accelerator="gpu",
            callbacks=self._callbacks,
            logger=self._loggers,
            profiler=self._profiler,

            accumulate_grad_batches=self._grad_accum,
            precision=self._precision,
            gradient_clip_val=self._grad_clip_val,
            gradient_clip_algorithm=self._grad_clip_algo,

            num_sanity_val_steps=1,
            enable_model_summary=self._debug,
            fast_dev_run=self._debug,

            default_root_dir=EXPERIMENTS_TRASH_PATH,

            **kwargs
        )

    @property
    def debug(self):
        return self._debug

    @abstractmethod
    def _get_def_save_dir(self) -> Optional[str | os.PathLike]:
        return None

    def _get_loggers(self) -> Optional[List[Logger]]:
        return None

    def _get_chkp_callback(self) -> Optional[callbacks.ModelCheckpoint]:
        return None

    def _get_callbacks(self) -> List[callbacks.Callback] | List[Never]:
        return [] if self._chkp_callback is None else [self._chkp_callback]

    def _get_profiler(self) -> Optional[SimpleProfiler | AdvancedProfiler]:
        return None

    def _get_precision(self) -> Optional[str]:
        return None

    def _get_grad_accum(self) -> int:
        return 1

    def _get_grad_clip(self) -> Tuple[Optional[float], Optional[str]]:
        return None, None

    def _compute_check_val_freq(self) -> int:
        if self._max_epochs < 30 or self._debug:
            return 1
        return int(
            self._max_epochs ** (1 / 2) - 4)  # Con questa formula si ottiene un numero di validazioni di circa 20

    @property
    def checkpoint_callback(self) -> Optional[FullModelCheckpoint]:
        return super().checkpoint_callback

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], **kwargs) -> Self:
        debug, max_epochs, save_dir = config['debug'], config['max_epochs'], config['save_dir']
        log_every_n_steps, limit_predict_batches = config['log_every_n_steps'], config['limit_predict_batches']
        return cls(max_epochs=max_epochs, save_dir=save_dir, debug=debug, log_every_n_steps=log_every_n_steps,
                   limit_predict_batches=limit_predict_batches, **kwargs)

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, datamodule=None,
            ckpt_path=None) -> None:
        """Slightingly overrided fit method that return the best model (in this case the last one)"""
        super().fit(model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders,
                    datamodule=datamodule, ckpt_path=ckpt_path)

        for file in os.listdir(EXPERIMENTS_TRASH_PATH):
            os.remove(os.path.join(EXPERIMENTS_TRASH_PATH, file))


class LiteTrainer(TrainerInterface):
    def __init__(self, max_epochs: int, debug: bool = False, **kwargs):
        super().__init__(max_epochs=max_epochs, save_dir=kwargs.pop("save_dir", None), debug=debug, **kwargs)

    def _get_def_save_dir(self) -> Optional[str | os.PathLike]:
        return None

    def _get_callbacks(self) -> List[callbacks.Callback]:
        if not self._debug:
            return [callbacks.RichProgressBar(leave=True), callbacks.Timer()] + super()._get_callbacks()
        else:
            return [callbacks.RichModelSummary()] + super()._get_callbacks()


class BaseTrainer(TrainerInterface):

    def __init__(self, max_epochs: int, save_dir: str | os.PathLike | None = None, debug: bool = False, **kwargs):
        super().__init__(max_epochs=max_epochs, save_dir=save_dir, debug=debug, min_epochs=int(max_epochs / 3),
                         **kwargs)

    def _get_def_save_dir(self) -> Optional[str | os.PathLike]:
        base_path = EXPERIMENTS_PATH
        # generate a random number to avoid overwriting
        free, path = False, None
        while not free:
            slot = random.randint(0, 1000000)
            path = os.path.join(base_path, f"rand_exp_{slot}")
            free = not os.path.exists(path)
        return path

    @property
    def checkpoint_callback(self) -> FullModelCheckpoint:
        chkpt = self.checkpoint_callbacks[0]
        if not isinstance(chkpt, FullModelCheckpoint):
            raise ValueError("Checkpoint callback is not a FullModelCheckpoint")
        return chkpt

    def _get_chkp_callback(self, save_freq: int = 1, save_top_k: int = 5, save_all: bool = True
                           ) -> callbacks.ModelCheckpoint:
        return FullModelCheckpoint(  # Only checkpoint saved since is needed at the end
            dirpath=os.path.join(self._save_dir, "checkpoints"), every_n_epochs=save_freq, save_top_k=save_top_k,
            monitor="val_loss", mode="min", save_last=True, save_weights_only=not save_all,
            filename="epoch={epoch}-vloss={val_loss:.3f}"
        )

    def _get_callbacks(self) -> List[callbacks.Callback]:
        return super()._get_callbacks() + [
            callbacks.RichProgressBar(leave=True, theme=RichProgressBarTheme(metrics_format=".5e")),
            callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
        ]

    def _get_loggers(self) -> List[Logger]:
        exp_dir, exp_name, trial = extract_name_trial_dir(self._save_dir)
        return [
            TensorBoardEncodeLogger(save_dir=exp_dir, name=exp_name, version=trial),
            CSVLoggerEncode(save_dir=exp_dir, name=exp_name, version=trial)
        ]

    def _get_profiler(self) -> SimpleProfiler:
        return SimpleProfiler(dirpath=self._save_dir, filename="profiler")

    def fit(self, model: BaseLGNM, train_dataloaders=None, val_dataloaders=None, datamodule=None,
            ckpt_path=None) -> lgn.LightningModule:
        """Slightingly overrided fit method that return the best model"""
        try:
            super().fit(model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders,
                        datamodule=datamodule, ckpt_path=ckpt_path)
        except (Exception, KeyboardInterrupt) as e:
            raise e
        else:
            if self.interrupted:
                raise KeyboardInterrupt("Training interrupted by user")
            if self._debug:
                return model

            # Find the best model and load it, otherwise leave the last one
            ckpt_c = self.checkpoint_callback
            best_model_path = ckpt_c.best_model_path if ckpt_c is not None else None
            # IF the path exist and is not the last one
            if (best_model_path is None or best_model_path == ""
                    or not os.path.exists(best_model_path) or ckpt_c.best_model_score is None
                    or ckpt_c.best_model_score.item() == ckpt_c.current_score.item()):
                # if the best model path is not correct we take the last one
                best_model_path = os.path.join(self._save_dir, "checkpoints", "last.ckpt")

            model.load_weights_from_checkpoint(best_model_path)
            os.rename(best_model_path, os.path.join(self._save_dir, "best_model.ckpt"))
            if os.path.exists(os.path.join(self._save_dir, "checkpoints", "last.ckpt")):
                os.remove(os.path.join(self._save_dir, "checkpoints", "last.ckpt"))
            return model


class BaseFasterTrainer(BaseTrainer):
    def __init__(self, max_epochs: int, save_dir: str | os.PathLike | None = None, debug: bool = False, **kwargs):
        super().__init__(max_epochs=max_epochs, save_dir=save_dir, debug=debug, benchmark=True, **kwargs)

    def _get_precision(self) -> str:
        return '16-mixed'

    def _get_grad_accum(self) -> int:
        return 5

    def _get_callbacks(self) -> List[callbacks.Callback]:
        return super()._get_callbacks() + [
            callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=self._debug,
                                    patience=int(self._max_epochs * 1 / self._check_val_freq / 2))
        ]


class OptunaTrainer(BaseTrainer):
    def __init__(self, max_epochs: int, trial: optuna.Trial, save_dir: str | os.PathLike | None = None,
                 debug: bool = False, limit_train_batches: float = 1.0, **kwargs):
        self.trial = trial
        super().__init__(max_epochs=max_epochs, save_dir=save_dir, debug=debug, limit_train_batches=limit_train_batches,
                         **kwargs)

    def _get_def_save_dir(self) -> Optional[str | os.PathLike]:
        raise ValueError("OptunaTrainer must have a save_dir (that depends on the trial)")

    def _get_precision(self) -> str:
        return '16-mixed'

    def _get_grad_accum(self) -> int:
        return 5

    def _get_chkp_callback(self, save_freq: int = 2, save_top_k: int = 2, save_all: bool = True
                           ) -> callbacks.ModelCheckpoint:
        return LightModelCheckpoint(  # Only checkpoint saved since is needed at the end
            dirpath=os.path.join(self._save_dir, "checkpoints"), every_n_epochs=save_freq, save_top_k=save_top_k,
            monitor="val_loss", mode="min", save_last=True, save_weights_only=not save_all,
            filename="epoch={epoch}-vloss={val_loss:.3f}"
        )

    def _get_callbacks(self) -> List[callbacks.Callback]:
        return super()._get_callbacks() + [PyTorchLightningPruningCallback(self.trial, monitor="val_loss")]

    def _get_profiler(self) -> None:
        return None

    @property
    def checkpoint_callback(self) -> LightModelCheckpoint:
        chkpt = self.checkpoint_callbacks[0]
        if not isinstance(chkpt, LightModelCheckpoint):
            raise ValueError("Checkpoint callback is not a LightModelCheckpoint")
        return chkpt

    @classmethod
    def load_from_config(cls, config: Dict[str, Any], trial: Optional[optuna.Trial] = None, **kwargs
                         ) -> Self:
        if kwargs is None:
            kwargs = {}

        kwargs["limit_train_batches"] = config.get("limit_train_batches", 1)  # Use get for backward compatibility
        if trial is None:
            kwargs["trial"] = config.get('trial', None)
        if trial is None:
            raise ValueError("Trail not found in config")

        return super().load_from_config(config, **kwargs)

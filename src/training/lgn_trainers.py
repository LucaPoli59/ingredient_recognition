import os
import random
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from typing_extensions import Never

import lightning as lgn
from lightning.pytorch import callbacks
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger, Logger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

from settings.config import EXPERIMENTS_PATH, EXPERIMENTS_TRASH_PATH
from src.training.utils import _extract_name_version_dir, CSVLoggerQuiet


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

    @abstractmethod
    def _get_def_save_dir(self) -> Optional[str | os.PathLike]:
        return None

    def _get_loggers(self) -> Optional[List[Logger]]:
        return None

    def _get_chkp_callback(self) -> Optional[callbacks.ModelCheckpoint]:
        return None

    def _get_callbacks(self) -> List[callbacks.Callback] | List[Never]:
        chkp_callback = self._get_chkp_callback()
        return [] if chkp_callback is None else [chkp_callback]

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
    def model_checkpoint_callback(self) -> callbacks.ModelCheckpoint:
        return self._chkp_callback

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, datamodule=None,
            ckpt_path=None) -> lgn.LightningModule:
        """Slightingly overrided fit method that return the best model (in this case the last one)"""
        super().fit(model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders,
                    datamodule=datamodule, ckpt_path=ckpt_path)

        # TODO: Remove cleaning of the trash folder

        return model


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

    def _get_chkp_callback(self) -> callbacks.ModelCheckpoint:
        return callbacks.ModelCheckpoint(  # Only checkpoint saved since is needed at the end
            dirpath=os.path.join(self._save_dir, "checkpoints"), every_n_epochs=1, save_top_k=5,
            monitor="val_loss", mode="min", save_last=True, save_weights_only=True,
            filename="epoch={epoch}-vloss={val_loss:.3f}"
        )

    def _get_callbacks(self) -> List[callbacks.Callback]:
        return [
            callbacks.RichProgressBar(leave=True, theme=RichProgressBarTheme(metrics_format=".5e")),
            callbacks.Timer(),
        ] + super()._get_callbacks()

    def _get_loggers(self) -> List[Logger]:
        exp_dir, exp_name, version = _extract_name_version_dir(self._save_dir)
        return [
            TensorBoardLogger(save_dir=exp_dir, name=exp_name, version=version),
            CSVLoggerQuiet(save_dir=exp_dir, name=exp_name, version=version)
        ]

    def _get_profiler(self) -> SimpleProfiler:
        return SimpleProfiler(dirpath=self._save_dir, filename="profiler")

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, datamodule=None,
            ckpt_path=None) -> lgn.LightningModule:
        """Slightingly overrided fit method that return the best model"""
        super().fit(model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders,
                    datamodule=datamodule, ckpt_path=ckpt_path)
        # Find the best model and load it, otherwise return the last one
        best_model_path = self._chkp_callback.best_model_path
        if (not self._debug and best_model_path is not None and best_model_path != ""
                and os.path.exists(best_model_path)):

            if self._chkp_callback.save_weights_only:
                # Retrieve hparams.yaml file
                hparams_path = os.path.join(self._save_dir, "hparams.yaml")
            else:
                hparams_path = None

            return model.load_from_checkpoint(best_model_path, hparams_file=hparams_path)
        return model


class BaseFasterTrainer(BaseTrainer):
    def __init__(self, max_epochs: int, save_dir: str | os.PathLike | None = None, debug: bool = False, **kwargs):
        super().__init__(max_epochs=max_epochs, save_dir=save_dir, debug=debug, benchmark=True, **kwargs)

    def _get_precision(self) -> str:
        return '16-mixed'

    def _get_grad_accum(self) -> int:
        return 5

    def _get_callbacks(self) -> List[callbacks.Callback]:
        return [callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=self._debug,
                                        patience=int(self._max_epochs * 1 / self._check_val_freq / 4))
                ] + super()._get_callbacks()

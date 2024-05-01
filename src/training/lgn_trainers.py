import os.path
import lightning as lgn
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import RichProgressBar, Timer, RichModelSummary, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from src.training.utils import _extract_name_version_dir, CSVLoggerQuiet






class LiteTrainer(lgn.Trainer):
    def __init__(self,
                 max_epochs: int,
                 accelerator: str = "gpu",
                 debug: bool = False):
        callbacks = []
        if not debug:
            callbacks.append(RichProgressBar(leave=True))
            callbacks.append(Timer())
        else:
            callbacks.append(RichModelSummary())

        super().__init__(
            max_epochs=max_epochs,
            accelerator=accelerator,
            callbacks=callbacks,
            fast_dev_run=debug,
            profiler=None if not debug else "simple",
            enable_model_summary=False,
            enable_checkpointing=False,
        )


class BaseTrainer(lgn.Trainer):
    def __init__(self,
                 save_dir: str | os.PathLike,
                 max_epochs: int,
                 len_train_dataloader: int,
                 accelerator: str = "gpu"):

        exps_dir, exp_name, exp_vers = _extract_name_version_dir(save_dir)
        checkpoint_path = os.path.join(exps_dir, exp_name, exp_vers, "checkpoints")

        callbacks = [
            RichProgressBar(leave=True),
            Timer(),
            ModelCheckpoint(dirpath=checkpoint_path, every_n_epochs=1, save_top_k=2, monitor="val_loss", mode="min"),
        ]

        loggers = [
            TensorBoardLogger(save_dir=exps_dir, name=exp_name, version=exp_vers),
            CSVLoggerQuiet(save_dir=exps_dir, name=exp_name, version=exp_vers)
        ]

        profiler = SimpleProfiler(dirpath=os.path.join(exps_dir, exp_name, exp_vers), filename="profiler")

        super().__init__(
            max_epochs=max_epochs,
            min_epochs=int(max_epochs / 2),
            accelerator=accelerator,
            log_every_n_steps=int(len_train_dataloader / 2),
            callbacks=callbacks,
            logger=loggers,
            enable_model_summary=False,
            fast_dev_run=False,
            profiler=profiler,
        )

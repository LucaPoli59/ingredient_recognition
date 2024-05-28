import warnings
from typing import Dict, Any, Optional
import lightning as lgn
from lightning.pytorch import callbacks

from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from src.training._config_enc_dec import encode_config, enc_config_to_yaml, decode_config


class CSVLoggerQuiet(CSVLogger):
    """Simple wrapper for lighting CSVLogger. This is useful when multiple loggers use the same directory"""

    def __init__(self, *args, **kwargs):
        warnings.filterwarnings(
            "ignore",
            f"Experiment logs directory .*. exists and is not empty. "
            f"Previous log files in this directory will be deleted when the new ones are saved")
        super().__init__(*args, **kwargs)


class CSVLoggerEncode(CSVLoggerQuiet):
    """Simple wrapper for lighting CSVLogger that encodes the hyperparameters in a more readable way"""

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        super().log_hyperparams(enc_config_to_yaml(encode_config(params)))


class FullModelCheckpoint(callbacks.ModelCheckpoint):
    """Custom ModelCheckpoint that saves also some information about the data used and trainer configuration"""

    def on_save_checkpoint(
            self, trainer: "lgn.pytorch.Trainer", pl_module: "lgn.pytorch.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint["trainer_hyper_parameters"] = trainer.hparams
        for field in ['hyper_parameters', 'datamodule_hyper_parameters', 'trainer_hyper_parameters']:
            checkpoint[field] = encode_config(checkpoint[field])

    def on_load_checkpoint(
            self, trainer: "lgn.pytorch.Trainer", pl_module: "lgn.pytorch.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        trainer.hparams = decode_config(checkpoint["trainer_hyper_parameters"])


class LightModelCheckpoint(callbacks.ModelCheckpoint):
    def __init__(self, *args, fields_to_skip: Optional[list[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if fields_to_skip is None:
            fields_to_skip = ["datamodule_hyper_parameters", "trainer_hyper_parameters", "hyper_parameters"]
        self.fields_to_skip = fields_to_skip

    def on_save_checkpoint(
            self, trainer: "lgn.pytorch.Trainer", pl_module: "lgn.pytorch.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        for field in self.fields_to_skip:
            checkpoint.pop(field, None)


class TensorBoardEncodeLogger(TensorBoardLogger):
    """Custom TensorBoardLogger when saving the hyperparameters it encodes them in a more readable way"""

    def log_hyperparams(self, params: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> None:
        super().log_hyperparams(enc_config_to_yaml(encode_config(params)), metrics)

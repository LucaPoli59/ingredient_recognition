import lightning as lgn
import torch
import os
from typing import Dict, Any, Optional, Type, Tuple

from settings.config import IMAGES_PATH, RECIPES_PATH, DEF_BATCH_SIZE, DEF_LR, DEF_UNKNOWN_TOKEN
from src.models.dummy import DummyModel
from src.lightning.lgn_models import BaseLGNM
from src.lightning.lgn_trainers import BaseTrainer
from src.data_processing.data_handling import ImagesRecipesDataModule
from src.training.utils import multi_label_accuracy, decode_config
from src.data_processing.labels_encoders import MultiLabelBinarizerRobust

DEF_EXP_CONFIG = {
    "hyper_parameters": {
        "torch_model_type": DummyModel,
        "lgn_model_type": BaseLGNM,
        "input_shape": (224, 224),
        "num_classes": None,
        "batch_size": DEF_BATCH_SIZE,
        "lr": DEF_LR,
        "loss_fn": torch.nn.BCEWithLogitsLoss,
        "accuracy_fn": multi_label_accuracy,
        "optimizer": torch.optim.Adam,
        "model_name": None
    },
    "trainer_hyper_parameters": {
        "type": BaseTrainer,
        "debug": False,
        "max_epochs": None,
        "save_dir": None
    },
    "datamodule_hyper_parameters": {  # in questo caso non serve
        "type": ImagesRecipesDataModule,
        "global_images_dir": IMAGES_PATH,
        "recipes_dir": RECIPES_PATH,
        "category": None,
        "recipe_feature_label": "ingredients_ok",
        "num_workers": os.cpu_count(),
        "label_encoder": {
            "type": MultiLabelBinarizerRobust,
            "classes": None,
            "encode_map": None,
            "fitted": False,
            "unknown_token": DEF_UNKNOWN_TOKEN
        }
    }
}


class ExpConfig:
    def __init__(self, **update_kwargs):
        """Takes some kwargs to update the default configuration.
        Begin Code:
            - "tr" or "trainer": for trainer_hyper_parameters
            - "dm" or "datamodule" or "data_module": for datamodule_hyper_parameters
            - "lb" or "label_encoder": for label_encoder
            - _: for hyper_parameters
            """
        self.config = DEF_EXP_CONFIG.copy()
        
        self.update_config(**update_kwargs)

    def update_config(self, **kwargs) -> None:
        """Function that takes some kwargs to update the default configuration.
        Each key in kwargs must be a key must begin with a code related to the configuration section, and followed by the
        _key of the parameter to update.
        Begin Code:
            - "tr" or "trainer": for trainer_hyper_parameters
            - "dm" or "datamodule" or "data_module": for datamodule_hyper_parameters
            - "lb" or "label_encoder": for label_encoder
            - _: for hyper_parameters

        Note: it ignores null values
        """
        config_map = { # map the prefix to the configuration section
            "tr": self.config["trainer_hyper_parameters"],
            "trainer": self.config["trainer_hyper_parameters"],
            "dm": self.config["datamodule_hyper_parameters"],
            "datamodule": self.config["datamodule_hyper_parameters"],
            "data_module": self.config["datamodule_hyper_parameters"],
            "lb": self.config["datamodule_hyper_parameters"]["label_encoder"],
            "label_encoder": self.config["datamodule_hyper_parameters"]["label_encoder"],
        }

        for key, value in ((key, value) for key, value in kwargs.items() if value is not None):
            split_key = key.split("_", 1)

            if len(split_key) == 1: # one of the case for hyper_parameters
                prefix_config = self.config["hyper_parameters"]
            else:
                prefix, key = split_key
                prefix_config = config_map.get(prefix, None)
                if prefix_config is None:
                    prefix_config = self.config["hyper_parameters"]
                    key = prefix + "_" + key # in case the prefix is not recognized with have tu use the starting key

            if key in prefix_config:
                prefix_config[key] = value
    @property
    def trainer(self) -> Dict[str, Any]:
        return self.config["trainer_hyper_parameters"]

    @property
    def datamodule(self) -> Dict[str, Any]:
        return self.config["datamodule_hyper_parameters"]

    @property
    def model(self) -> Dict[str, Any]:
        return self.config["hyper_parameters"]

    @property
    def label_encoder(self) -> Dict[str, Any]:
        return self.config["datamodule_hyper_parameters"]["label_encoder"]

    @classmethod
    def load_from_ckpt_data(cls, ckpt_data: Dict[str, Any]) -> "ExpConfig":
        config = {k: decode_config(v) for k, v in ckpt_data.items() if k in
                  ['trainer_hyper_parameters', 'hyper_parameters', 'datamodule_hyper_parameters']}
        config_edit = (
            {f"tr_{key}": value for key, value in config['trainer_hyper_parameters'].items()} |
            {f"dm_{key}": value for key, value in config['datamodule_hyper_parameters'].items()} |
            {key: value for key, value in config['hyper_parameters'].items()}
        )
        return cls(**config_edit)


    def __str__(self):
        return str(self.config)


def model_training(exp_config: ExpConfig, data_module: Optional[Type[lgn.LightningDataModule]] = None,
                   ckpt_path: Optional[str | os.PathLike] = None
                   ) -> Tuple[Type[lgn.Trainer], Type[lgn.LightningModule]]:
    resuming = ckpt_path is None
    model_config, trainer_config = exp_config.model, exp_config.trainer

    lgn_model = model_config['lgn_model_type'].load_from_config(model_config)
    trainer = trainer_config['type'].load_from_config(trainer_config)

    if data_module is None:
        dm_config = exp_config.datamodule
        dm_type = dm_config['type']
        data_module = dm_type.load_from_config(dm_config, image_size=lgn_model.input_shape,
                                               batch_size=lgn_model.batch_size)

    debug = trainer.debug or True  # TODO: Remove or True (when the module is ready)
    if debug:
        print("Data Module, Models and Trainer loaded, " + ("training started" if resuming else "resume training"))

    trained_model = trainer.fit(
        model=lgn_model,
        datamodule=data_module,
        ckpt_path=ckpt_path
    )

    if debug:
        print("Training completed")
    return trainer, trained_model

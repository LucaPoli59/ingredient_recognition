import lightning as lgn
import torch
import os
from typing import Dict, Any, Optional, Type, Tuple, List, TypeVar
import optuna

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
        "save_dir": None,
        "limit_train_batches": 1.0,
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

DEF_EXP_CONFIG_MAP = {  # map the prefix to the configuration section
    "tr": "trainer_hyper_parameters",
    "trainer": "trainer_hyper_parameters",
    "dm": "datamodule_hyper_parameters",
    "datamodule": "datamodule_hyper_parameters",
    "data_module": "datamodule_hyper_parameters",
    "lb": ("datamodule_hyper_parameters", "label_encoder"),
    "label_encoder": ("datamodule_hyper_parameters", "label_encoder"),
    "hp": "hyper_parameters"
}


class ExpConfig:
    """
    Class that represents the configuration of an experiment.
    It's used to initialize the default parameters and change only the necessary ones.
    """

    def __init__(self, def_configs: Optional[Dict[str, Any]] = None,
                 config_map: Optional[Dict[str, List[str] | str]] = None,
                 **update_kwargs):
        """Takes some kwargs to update the default configuration.
        Begin Code:
            - "tr" or "trainer": for trainer_hyper_parameters
            - "dm" or "datamodule" or "data_module": for datamodule_hyper_parameters
            - "lb" or "label_encoder": for label_encoder
            - "hp" or None: for hyper_parameters
            """
        if def_configs is None:
            def_configs = DEF_EXP_CONFIG.copy()
        self._config = def_configs

        if config_map is None:
            config_map = DEF_EXP_CONFIG_MAP
        self._config_map = config_map

        self.update_config(**update_kwargs)

    def update_config(self, **kwargs) -> None:
        """Function that takes some kwargs to update the default configuration.
        Each key in kwargs must be a key must begin with a code related to the configuration section, and followed by the
        _key of the parameter to update.
        Begin Code (with the default config map):
            - "tr" or "trainer": for trainer_hyper_parameters
            - "dm" or "datamodule" or "data_module": for datamodule_hyper_parameters
            - "lb" or "label_encoder": for label_encoder
            - "hp" or None: for hyper_parameters

        Note: it ignores null values
        """

        for key, value in ((key, value) for key, value in kwargs.items() if value is not None):
            split_key = key.split("_", 1)

            if len(split_key) == 1:  # one of the case for hyper_parameters
                prefix_config = self.idx(self._config, self._config_map["hp"])
            else:
                prefix, key = split_key
                prefix_config_key = self._config_map.get(prefix, None)
                if prefix_config_key is not None:
                    prefix_config = self.idx(self._config, prefix_config_key)
                else:
                    prefix_config = self.idx(self._config, self._config_map["hp"])
                    key = prefix + "_" + key  # in case the prefix is not recognized with have tu use the starting key

            prefix_config[key] = value

    @property
    def trainer(self) -> Dict[str, Any]:
        return self._config["trainer_hyper_parameters"]

    @property
    def datamodule(self) -> Dict[str, Any]:
        return self._config["datamodule_hyper_parameters"]

    @property
    def model(self) -> Dict[str, Any]:
        return self._config["hyper_parameters"]

    @property
    def label_encoder(self) -> Dict[str, Any]:
        return self._config["datamodule_hyper_parameters"]["label_encoder"]

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @staticmethod
    def idx(obj_dict, key: str | List[str]):
        if isinstance(key, str):
            return obj_dict[key]

        ris = obj_dict
        for k in key:
            ris = ris[k]
        return ris

    @property
    def config_map(self) -> Dict[str, str | List[str]]:
        return self._config_map

    @classmethod
    def load_from_ckpt_data(cls, ckpt_data: Dict[str, Any], loaded_config: Optional[Dict[str, Any]] = None
                            ) -> "ExpConfig":
        if loaded_config is None:
            loaded_config = {}

        config = {k: decode_config(v) for k, v in ckpt_data.items() if k in
                  ['trainer_hyper_parameters', 'hyper_parameters', 'datamodule_hyper_parameters']}
        config_edit = (
                {f"tr_{key}": value for key, value in config['trainer_hyper_parameters'].items()} |
                {f"dm_{key}": value for key, value in config['datamodule_hyper_parameters'].items()} |
                {key: value for key, value in config['hyper_parameters'].items()} |
                loaded_config
        )
        return cls(**config_edit)

    def __str__(self):
        return str(self._config)


class HTunerExpConfig(ExpConfig):
    """
    Class that provide a dictionary with the configuration of the experiment for the hyperparameters tuning.
    """
    def __init__(self, def_configs: Optional[Dict[str, Any]] = None,
                 config_map: Optional[Dict[str, List[str] | str]] = None,
                 **update_kwargs):
        super().__init__(def_configs, config_map, **update_kwargs)
        self._config = self._config | {
            "htuner_hyper_parameters": {
                "sampler": optuna.samplers.RandomSampler,
                "pruner": optuna.pruners.NopPruner,
                "storage": optuna.storages.JournalStorage,
                "save_path": None,
                "n_trials": 10,
            }
        }
        self._config_map = self._config_map | {
            "ht": "htuner_hyper_parameters",
            "htuner": "htuner_hyper_parameters"
        }

    @property
    def htuner(self) -> Dict[str, Any]:
        return self._config["htuner_hyper_parameters"]

    def load_from_ckpt_data(cls, ckpt_data: Dict[str, Any], loaded_config: Optional[Dict[str, Any]] = None
                            ) -> "HTunerExpConfig":
        pass # TODO: Fare quando si è capito come funziona il loading di Optuna, anche se gli htuner hyperparameters
        #  non dovrebbero essere salvati nei checkpoint delle trials


class HGeneratorConfig(ExpConfig):
    """
    Class that provide a dictionary with the generators of hyperparameters for the experiments with htuning.
    """

    def __init__(self, **update_kwargs):
        def_configs = {
            "hyper_parameters": {
                "input_shape": None,
                "batch_size": None,
                "lr": None,  # expected: (lambda trial: trial.suggest_float("lr", 1e-5, 1e-1))
                "optimizer": None,
            },
            "trainer_hyper_parameters": {
                "max_epochs": None,
            },
            "datamodule_hyper_parameters": {
            },
            "label_encoder": {
            }
        }

        config_map = {
            "tr": "trainer_hyper_parameters",
            "dm": "datamodule_hyper_parameters",
            "hp": "hyper_parameters",
            "lb": "label_encoder",
        }

        super().__init__(def_configs, config_map, **update_kwargs)

        self._inverted_map = self._invert_config_map()

    def _invert_config_map(self) -> Dict[str, str | List[str]]:
        inverted_map = {}
        for key, value in self._config_map.items():
            if key not in inverted_map:
                inverted_map[value] = key
        return inverted_map

    def generate_hparams_on_trial(self, trial: optuna.trial) -> Dict[str, Any]:
        """
        Function that uses the trail obj and the function saved in the configuration to generate the hyperparameters.
        """

        res_dict = {}
        for group_key in self._config[1:]:
            target_group_key = self._inverted_map[group_key]
            for key, gen in ((key, value) for key, value in self._config[group_key].items() if value is not None):
                res_dict[f"{target_group_key}_{key}"] = gen(trial)
        return res_dict

    def update_config(self, **kwargs):
        raise PermissionError("This method is not allowed for this class")


def model_training(exp_config: ExpConfig, data_module: Optional[Type[lgn.LightningDataModule]] = None,
                   ckpt_path: Optional[str | os.PathLike] = None
                   ) -> Tuple[Type[lgn.Trainer], Type[lgn.LightningModule]]:
    """General function that trains a model with the given configuration."""
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

def load_datamodule(exp_config: ExpConfig | HTunerExpConfig) -> ImagesRecipesDataModule:
    dm_config = exp_config.datamodule
    dm_type = dm_config["type"]
    data_module = dm_type.load_from_config(dm_config, image_size=exp_config.model["input_shape"],
                                           batch_size=exp_config.model["batch_size"])
    data_module.prepare_data()
    data_module.setup()
    return data_module

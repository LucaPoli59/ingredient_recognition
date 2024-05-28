import json
import torch
import os
from typing import Dict, Any, Optional, List
from torchmetrics import Accuracy, HammingDistance, Precision, Recall
import optuna

from src.commons.config_enc_dec import encode_config, decode_config
from settings.config import IMAGES_PATH, RECIPES_PATH, DEF_BATCH_SIZE, DEF_LR, DEF_UNKNOWN_TOKEN, DEF_N_TRIALS
from src.models.dummy import DummyModel
from src.lightning.lgn_models import BaseLGNM
from src.lightning.lgn_trainers import BaseTrainer
from src.data_processing.labels_encoders import MultiLabelBinarizerRobust
from src.data_processing.data_handling import ImagesRecipesDataModule
from src.commons.utils import MyMLAccuracy

DEF_METRIC_INIT_P = {
    "task": "multilabel",
    "num_labels": None,
    "average": "macro",
}
DEF_METRIC_LOGGING_P = {
    "prog_bar": True,
    "on_epoch": True,
    "on_step": False,
}

DEF_EXP_CONFIG = {
    "hyper_parameters": {
        "torch_model_type": DummyModel,
        "lgn_model_type": BaseLGNM,
        "input_shape": (224, 224),
        "num_classes": None,
        "batch_size": DEF_BATCH_SIZE,
        "lr": DEF_LR,
        "loss_fn": torch.nn.BCEWithLogitsLoss,
        "optimizer": torch.optim.Adam,
        "model_name": None,
        "metrics": {
            "acc_t": {'obj': Accuracy, 'init_params': DEF_METRIC_INIT_P, 'logging_params': DEF_METRIC_LOGGING_P},
            "precision": {'obj': Precision, 'init_params': DEF_METRIC_INIT_P, 'logging_params': DEF_METRIC_LOGGING_P},
            "recall": {'obj': Recall, 'init_params': DEF_METRIC_INIT_P, 'logging_params': DEF_METRIC_LOGGING_P},
            "hamming": {'obj': HammingDistance,
                        'init_params': {'num_labels': None, 'average': DEF_METRIC_INIT_P['average']},
                        'logging_params': DEF_METRIC_LOGGING_P},
            "acc_my" : {'obj': MyMLAccuracy, 'init_params': {}, 'logging_params': DEF_METRIC_LOGGING_P}
        }
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
    "hp": "hyper_parameters",
    "metrics": ("hyper_parameters", "metrics"),
    "me": ("hyper_parameters", "metrics"),
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
        self._inverted_map = self._invert_config_map()

        if update_kwargs:
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
        return self.datamodule["label_encoder"] if "label_encoder" in self.datamodule else {}

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @staticmethod
    def idx(obj_dict: Dict[str | List[str], Any], key: str | List[str]) -> Any:
        if isinstance(key, str):
            return obj_dict[key]

        ris = obj_dict
        for k in key:
            ris = ris[k]
        return ris

    @staticmethod
    def idx_set(obj_dict: Dict[str | List[str], Any], key: str | List[str], value: Any) -> Dict[str | List[str], Any]:
        if isinstance(key, str):
            obj_dict[key] = value
        else:
            dest = obj_dict
            for k in key[:-1]:
                dest = dest[k]

            dest[key[-1]] = value

        return obj_dict

    @property
    def config_map(self) -> Dict[str, str | List[str]]:
        return self._config_map

    def _invert_config_map(self) -> Dict[str, str]:
        inverted_map = {}
        for key, value in self._config_map.items():
            if isinstance(value, str) and value not in inverted_map:  # we use only direct mapping
                inverted_map[value] = key
        return inverted_map

    def _convert_dict_to_update_kwargs(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        output_dict = {}
        for class_key in (class_key for class_key in input_dict if class_key in self._config.keys()):
            for item_key, value in input_dict[class_key].items():
                output_dict[f"{self._inverted_map[class_key]}_{item_key}"] = value
        return output_dict

    @classmethod
    def load_from_ckpt_data(cls, ckpt_data: Dict[str, Any], loaded_config: Optional[Dict[str, Any]] = None
                            ) -> "ExpConfig":
        if loaded_config is None:
            loaded_config = {}

        exp_config = cls(**loaded_config)

        config = {k: decode_config(v) for k, v in ckpt_data.items() if k in exp_config._config.keys()}
        config_edit = exp_config._convert_dict_to_update_kwargs(config)
        exp_config.update_config(**config_edit)
        return exp_config

    def save_to_file(self, file_path: str | os.PathLike) -> None:
        with open(file_path, "w") as file:
            json.dump(encode_config(self._config), file, indent=4)

    @classmethod
    def load_from_file(cls, file_path: str | os.PathLike) -> "ExpConfig":
        with open(file_path, "r") as file:
            config = decode_config(json.load(file))

            exp_config = cls()
            exp_config.update_config(**exp_config._convert_dict_to_update_kwargs(config))
            return exp_config

    def _drop(self, key: str):
        if key not in self._config_map:
            raise KeyError(f"Key {key} not found in the configuration map.")
        self.idx_set(self._config, self._config_map[key], {})

    def drop(self, keys: List[str] | str):
        """Drop a category from the configuration (useful when it's not needed anymore)."""
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            self._drop(key)


    def __str__(self):
        return str(self._config)


class HTunerExpConfig(ExpConfig):
    """
    Class that provide a dictionary with the configuration of the experiment for the hyperparameters tuning.
    """

    def __init__(self, def_configs: Optional[Dict[str, Any]] = None,
                 config_map: Optional[Dict[str, List[str] | str]] = None,
                 **update_kwargs):
        super().__init__(def_configs, config_map)
        self._config = self._config | {
            "htuner_hyper_parameters": {
                "sampler": optuna.samplers.RandomSampler,
                "pruner": optuna.pruners.NopPruner,
                "direction": "minimize",  # "minimize" or "maximize
                "n_trials": DEF_N_TRIALS,
            }
        }
        self._config_map = self._config_map | {
            "ht": "htuner_hyper_parameters",
            "htuner": "htuner_hyper_parameters"
        }
        self._inverted_map = self._invert_config_map()
        self.update_config(**update_kwargs)

    @property
    def htuner(self) -> Dict[str, Any]:
        return self._config["htuner_hyper_parameters"]

class HGeneratorConfig(ExpConfig):
    """
    Class that provide a dictionary with the generators of hyperparameters for the experiments with htuning.
    """

    def __init__(self, **update_kwargs):
        def_configs = {
            "hyper_parameters": {
                "input_shape": None,
                "lr": None,  # expected: (lambda trial: trial.suggest_float("lr", 1e-5, 1e-1)) or optuna.distributions
                "optimizer": None,
            },
            "trainer_hyper_parameters": {
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

        for key, value in update_kwargs.items():
            if isinstance(value, optuna.distributions.BaseDistribution):
                update_kwargs[key] = self._convert_dist2lambda(key, value)

        super().__init__(def_configs, config_map, **update_kwargs)

    @staticmethod
    def _convert_dist2lambda(dist_name: str, dist: optuna.distributions.BaseDistribution):
        dist = json.loads(optuna.distributions.distribution_to_json(dist))
        match dist['name']:
            case 'CategoricalDistribution':
                return lambda trial: trial.suggest_categorical(dist_name, **dist['attributes'])
            case 'IntDistribution':
                return lambda trial: trial.suggest_int(dist_name, **dist['attributes'])
            case 'FloatDistribution':
                return lambda trial: trial.suggest_float(dist_name, **dist['attributes'])
            case _:
                raise ValueError(f"Invalid distribution type: {dist['name']}")

    def generate_hparams_on_trial(self, trial: optuna.trial) -> Dict[str, Any]:
        """
        Function that uses the trail obj and the function saved in the configuration to generate the hyperparameters.
        """

        res_dict = {}
        for group_key in self._config:
            target_group_key = self._inverted_map[group_key]
            for key, gen in ((key, value) for key, value in self._config[group_key].items() if value is not None):
                res_dict[f"{target_group_key}_{key}"] = gen(trial)
        return res_dict



import inspect
import json
import torch
import os
import copy
from typing import Dict, Any, Optional, List
from typing_extensions import Self

from torchmetrics import Accuracy, HammingDistance, Precision, Recall, F1Score
import optuna

from src.commons.config_enc_dec import encode_config, decode_config
from settings.config import YUMMLY_PATH, YUMMLY_RECIPES_PATH, DEF_BATCH_SIZE, DEF_LR, DEF_UNKNOWN_TOKEN, DEF_N_TRIALS, \
    METADATA_FILENAME, WANDB_PROJECT_NAME
from src.models.dummy import DummyModel
from src.lightning.lgn_models import BaseLGNM
from src.lightning.lgn_trainers import BaseTrainer
from src.data_processing.labels_encoders import MultiLabelBinarizerRobust
from src.data_processing.images_recipes import ImagesRecipesBaseDataModule
from src.commons.utils import MyMLAccuracy

DEF_METRIC_INIT_P = {
    "task": "multilabel",
    "num_labels": None,
    "average": "weighted",
}
DEF_METRIC_LOGGING_P = {
    "prog_bar": False,
    "on_epoch": True,
    "on_step": False,
}

DEF_METRIC_F1L_INIT_P = copy.deepcopy(DEF_METRIC_INIT_P)
DEF_METRIC_F1L_INIT_P["average"] = "none"
DEF_METRIC_F1L = {'type': F1Score, 'init_params': DEF_METRIC_F1L_INIT_P, 'logging_params': DEF_METRIC_LOGGING_P}

DEF_EXP_CONFIG = {
    "hyper_parameters": {
        "lgn_model_type": BaseLGNM,
        "batch_size": DEF_BATCH_SIZE,
        "lr": DEF_LR,
        "loss_fn": torch.nn.BCEWithLogitsLoss,
        "weighted_loss": True,
        "optimizer": torch.optim.Adam,
        "momentum": None,
        "weight_decay": None,
        "use_swa": False,
        "lr_scheduler": None,  #  this works only with BaeWithSchedulerLGNM
        "lr_scheduler_params": {
        },
        "torch_model": {
            "type": DummyModel,
            "input_shape": (224, 224),
            "num_classes": None,
            "trns_aug": None,
            "trns_bld_aug": None,  # trns_bld stands for transformation builder and is a function that returns a list of transformations
            "trns_bld_plain": None,
            "lp_phase": None,
        },
        "metrics": {
            "acc": {'type': Accuracy, 'init_params': DEF_METRIC_INIT_P, 'logging_params': DEF_METRIC_LOGGING_P},
            # "f1_label": DEF_METRIC_F1L,
            "precision": {'type': Precision, 'init_params': DEF_METRIC_INIT_P, 'logging_params': DEF_METRIC_LOGGING_P},
            "recall": {'type': Recall, 'init_params': DEF_METRIC_INIT_P, 'logging_params': DEF_METRIC_LOGGING_P},
            "hamming": {'type': HammingDistance, 'init_params': DEF_METRIC_INIT_P,
                        'logging_params': DEF_METRIC_LOGGING_P | {"prog_bar": True}},
        },
    },
    "trainer_hyper_parameters": {
        "type": BaseTrainer,
        "debug": False,
        "max_epochs": 20,
        "save_dir": None,
        "limit_train_batches": 1.0,
        "limit_predict_batches": 1,
        "log_every_n_steps": 50,
        "early_stop": None,  # used only if the trainer has the early stopping callback
    },
    "datamodule_hyper_parameters": {  # in questo caso non serve
        "type": ImagesRecipesBaseDataModule,
        "image_shape": (224, 224),
        "data_dir": YUMMLY_PATH,
        "metadata_filename": METADATA_FILENAME,
        "category": None,
        "feature_label": "ingredients_ok",
        "num_workers": os.cpu_count(),
        "label_encoder": {
            "type": MultiLabelBinarizerRobust,
            "classes": None,
            "encode_map": None,
            "fitted": False,
            "unknown_token": DEF_UNKNOWN_TOKEN
        }
    },
    "logging_hyper_parameters": {  # useds by BaseTrainer (and subclasses)
        "wandb_notes": None,
        "wandb_project": WANDB_PROJECT_NAME,
        "log_exp_config": False,
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
    "hyper_parameters": "hyper_parameters",
    "lgn_model": "hyper_parameters",
    "torch_model": ("hyper_parameters", "torch_model"),
    "tm": ("hyper_parameters", "torch_model"),
    "metrics": ("hyper_parameters", "metrics"),
    "me": ("hyper_parameters", "metrics"),
    "logging": "logging_hyper_parameters",
    "lgg": "logging_hyper_parameters",
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
        Begin Code: any key in the config_map
            """
        if def_configs is None:
            def_configs = copy.deepcopy(DEF_EXP_CONFIG)
        self._config: Dict[str, Dict[str, Any]] = def_configs

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
        Begin Code: any key in the config_map, the default is the hyper_parameters section


        Notes:
             - it ignores null values
             - if the key is empty (for ex: "dm_") it updates the entire section (recommended use for the metrics section)
        """

        for key, value in ((key, value) for key, value in kwargs.items() if value is not None):
            split_key = key.split("_", 1)

            #extract the related key
            if len(split_key) == 1:  # one of the case for hyper_parameters
                prefix_config = self.idx(self._config, self._config_map["hp"])
                prefix_config[key] = value
            else:
                prefix, key = split_key
                prefix_config_key = self._config_map.get(prefix, None)
                if key is None or key == "": # update the entire section
                    prefix_config_key = prefix_config_key if prefix_config_key is not None else self._config_map["hp"]
                    self.idx_set(self._config, prefix_config_key, value)
                else:  # update a single key
                    if prefix_config_key is not None:
                        prefix_config = self.idx(self._config, prefix_config_key)
                    else:
                        prefix_config = self.idx(self._config, self._config_map["hp"])
                        key = prefix + "_" + key  # in case the prefix is not recognized with have to use the starting key

                    prefix_config[key] = value

    @property
    def trainer(self) -> Dict[str, Any]:
        return self._config["trainer_hyper_parameters"]

    @property
    def datamodule(self) -> Dict[str, Any]:
        return self._config["datamodule_hyper_parameters"]

    @property
    def lgn_model(self) -> Dict[str, Any]:
        return self._config["hyper_parameters"]

    @property
    def hp(self) -> Dict[str, Any]:
        return self._config["hyper_parameters"]

    @property
    def metrics(self) -> Dict[str, Dict[str, Any]]:
        return self.idx(self._config, self._config_map["metrics"])

    @property
    def torch_model(self) -> Dict[str, Any]:
        return self.idx(self._config, self._config_map["torch_model"])

    @property
    def label_encoder(self) -> Dict[str, Any]:
        return self.datamodule["label_encoder"] if "label_encoder" in self.datamodule else {}

    @property
    def logging(self) -> Dict[str, Any]:
        return self._config["logging_hyper_parameters"]

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
                "sampler_kwargs": {},
                "pruner": optuna.pruners.NopPruner,
                "pruner_kwargs": {},
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


# TODO CAPIRE COME GESTIRE DISTRIBUZIONI INNESTATE
class HGeneratorConfig(ExpConfig):
    """
    Class that provide a dictionary with the generators of hyperparameters for the experiments with htuning.
    where each key is the name of the hyperparameter (identified by the prefix) and the value is a
    "gen_dict" dictionary with:

    - dist: the distribution to use (either optuna.distributions or a lambda function)
    - vocab (Dict[str, Any]): the vocabulary to use for the translation (in case of categorical distribution)
    - effected_params (Dict[str, Any]): the config dict with parameters value that depends on the hyperparameter generated

    It's also possible to use directly the optuna distribution (or the lambda function) instead of the gen_dict (coversion
    is done automatically).

    Example 1 (LR):

        hgen_config = HGeneratorConfig(
            hp_lr={
                "dist": (lambda trial: trial.suggest_float("hp_lr", 1e-3, 1e-1, log=True)),  # < 1e-3 is too low
                "vocab": None,
                "effected_params": None,
            },
        )
        Which is equal to
        hgen_config = HGeneratorConfig(
            hp_lr=optuna.distributions.LogUniformDistribution(1e-3, 1e-1)
        )

        In this case, we are generating a float value for the learning rate with a log scale between 1e-3 and 1e-1.

    Example 2 (LR Scheduler):
        hgen_config = HGeneratorConfig(
            hp_lr_scheduler={
                "dist": (lambda trial: trial.suggest_categorical("lr_scheduler", [
                    "ConstantStartReduceOnPlateau", "CosineAnnealingWarmRestarts", 'none'],)),  # < 1e-3 is too low
                "vocab": {
                    "ConstantStartReduceOnPlateau":  src.models.custom_schedulers.ConstantStartReduceOnPlateau,
                    "CosineAnnealingWarmRestarts":  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                },
                "effected_params": {
                    "ConstantStartReduceOnPlateau": {
                        "hp_lr_scheduler_params": {   # NOTE: use the prefix syntax to identify the group
                            "warm_duration": (lambda trial: trial.suggest_int("warm_duration", 0, 20, step=5)),
                            "mode": "min",
                            "patience": 5,
                            "cooldown": 1,
                            "min_lr": DEF_LR / 1e3,
                            "factor": 0.05,
                        }
                    },
                    "CosineAnnealingWarmRestarts": {
                        "hp_lr_scheduler_params": {
                            "T_0": 5,
                            "T_mult": 2,
                            "eta_min": DEF_LR / 1e3,
                        }
                    },
                },
            },
        )

        In this case, we are generating a categorical value for the learning rate scheduler, with the possibility to choose
        between two different schedulers. In case of the ConstantStartReduceOnPlateau, we have some parameters that are
        generated based on the value of the scheduler.
    """

    def __init__(self, **update_kwargs):
        def_configs = {
            "hyper_parameters": {
            },
            "trainer_hyper_parameters": {
            },
            "datamodule_hyper_parameters": {
            },
            "label_encoder": {
            },
            "torch_model": {
            },
        }

        config_map = {
            "tr": "trainer_hyper_parameters",
            "dm": "datamodule_hyper_parameters",
            "hp": "hyper_parameters",
            "lb": "label_encoder",
            "tm": "torch_model",
        }

        update_kwargs = self._standardize_generators(update_kwargs)
        super().__init__(def_configs, config_map, **update_kwargs)

    @classmethod
    def load_from_file_with_dist(cls, file_path: str | os.PathLike,
                                 distributions: Dict[str, optuna.distributions.BaseDistribution]) -> Self:
        with open(file_path, "r") as file:
            config = decode_config(json.load(file), raise_lambda=False)

            exp_config = cls()
            config_update_kwargs = exp_config._convert_dict_to_update_kwargs(config)

            for (hp_id, hp_dict), (dist_name, dist) in zip(config_update_kwargs.items(), distributions.items()):
                hp_prefix, hp_name = hp_id.split("_", 1)

                if hp_name != dist_name and hp_id != dist_name:
                    raise ValueError(f"Expected hp_name {hp_name} (found in config file) to be equal to dist_name {dist_name} (found in distributions)")

                config_update_kwargs[hp_id]['dist'] = cls._convert_dist2lambda(hp_id, dist)

            exp_config.update_config(**config_update_kwargs)
            return exp_config



    @staticmethod
    def _standardize_generators(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """preprocess function that assure that each hyperparams is associated with a gen_dict"""
        output_dict = {}
        for hp_name, gen_dict in input_dict.items():
            if not isinstance(gen_dict, dict):
                gen_dict = {
                    "dist": gen_dict,
                    "vocab": None,
                    "effected_params": None,
                }
            output_dict[hp_name] = gen_dict
        return output_dict

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


    def update_config(self, **kwargs) -> None:
        kwargs = self._standardize_generators(kwargs)
        super().update_config(**kwargs)

    def generate_hparams_on_trial(self, trial: optuna.trial) -> Dict[str, Any]:
        """
        Function that uses the trail obj and the function saved in the configuration to generate the hyperparameters.
        """

        res_params_dict, res_effected_params_dict = {}, {}
        for group_key in self._config:
            target_group_key = self._inverted_map[group_key]
            for key, gen_dict in ((key, value) for key, value in self._config[group_key].items() if value is not None):
                sampled_value = gen_dict["dist"](trial)  # sample the value from the distribution
                if sampled_value is None or sampled_value == "none":
                    real_value, eff_params = None, {}
                else:
                    # convert the value to the real value (if needed)
                    real_value = gen_dict["vocab"][sampled_value] if gen_dict["vocab"] is not None else sampled_value

                    # extract the effected parameters (if present)
                    if gen_dict["effected_params"] is None or sampled_value not in gen_dict["effected_params"]:
                        eff_params = {}
                    else:
                        eff_params = gen_dict["effected_params"][sampled_value]

                res_params_dict[f"{target_group_key}_{key}"] = real_value
                for eff_param_key, eff_param_value in eff_params.items(): # generate the effected parameters if needed
                    if not isinstance(eff_param_value, dict):
                        eff_param_value = eff_param_value(trial) if inspect.isfunction(eff_param_value) else eff_param_value
                    else:
                        eff_param_value = {k: v(trial) if inspect.isfunction(v) else v for k, v in eff_param_value.items()}

                    res_effected_params_dict[f"{eff_param_key}"] = eff_param_value


        return res_params_dict | res_effected_params_dict

    @property
    def torch_model(self) -> Dict[str, Any]:
        return self.idx(self._config, self._config_map["tm"])
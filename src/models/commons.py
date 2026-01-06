import json

import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Callable
from typing_extensions import Self

from src.data_processing.transformations import t_augmentations_builder
from settings.config import LP_MAX_PHASE
from src.data_processing.transformations import transform_aug_base, transform_plain_base, t_transform_builder, t_augmentations_builder, transform_core_base


class BaseModel(ABC, torch.nn.Module):
    PRETTY_NAME = "Base"
    DEF_TRNS_BLD_BASE = transform_core_base
    DEF_TRNS_BLD_AUG = transform_aug_base
    DEF_TRNS_BLD_PLAIN = transform_plain_base

    #trns_bld stands for transformation builder and is a function that returns a list of transformations
    def __init__(self, num_classes: int, input_shape: int | Tuple[int, int],
                 trns_aug: Optional[t_augmentations_builder] = None,
                 trns_bld_aug: Optional[t_transform_builder] = None,
                 trns_bld_plain: Optional[t_transform_builder] = None,
                 lp_phase: Optional[int] = None):
        """
        Base class for all vision models
        :param num_classes: number of output classes
        :param input_shape: input shape of the model
        :param trns_aug: function (without positional arguments) that returns a list of augmentations to add on top the default transformations
        :param trns_bld_aug: callable that returns a list of transformation (with some augmentations, for the training),
        if None, the default transformation builder will be used
        :param trns_bld_plain: callable that returns a list of transformation (without augmentations, for the validation),
        if None, the default transformation builder will be used
        :param lp_phase: layer pretraining phase (0 to LP_MAX_PHASE), -1 for out of LP

        Note: it's recommended to use either trns_aug or the pair trns_bld_aug, trns_bld_plain, but is possible to use
        trns_aug with trns_bld_plain (but not the other pair)
        """

        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        elif len(input_shape) > 2 or len(input_shape) < 1:
            raise ValueError("The input_shape must be a tuple of 2 integers")
        if input_shape[0] != input_shape[1]:
            raise ValueError("The input_shape must be a square")

        if lp_phase is not None and (not isinstance(lp_phase, int) or lp_phase < -1 or lp_phase > LP_MAX_PHASE):
            raise ValueError(f"The lp_phase must be an integer between 0 and {LP_MAX_PHASE}")  # -1 is for out of LP

        if trns_aug is not None and trns_bld_aug is not None:
            raise ValueError("It's mandatory to use either trns_aug or trns_bld_aug")

        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.trns_aug = trns_aug
        self.trns_bld_aug = trns_bld_aug if trns_bld_aug is not None else self.__class__.DEF_TRNS_BLD_AUG
        self.trns_bld_plain = trns_bld_plain if trns_bld_plain is not None else self.__class__.DEF_TRNS_BLD_PLAIN
        self._lp_phase = lp_phase

        self.max_lp_phase = LP_MAX_PHASE

    @classmethod
    def load_from_config(cls, config: str | Dict[str, Any]) -> Self:
        if isinstance(config, str):
            config = json.loads(config)

        params = cls._load_config(config)
        return cls(**params)

    def _unfreeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def support_layer_pretrain(self):
        return hasattr(self, "_lp_phase") and self._lp_phase is not None and self._lp_phase >= -1

    def to_config(self):
        trns_bld_aug = self.trns_bld_aug if self.trns_bld_aug != self.__class__.DEF_TRNS_BLD_AUG else None
        trns_bld_plain = self.trns_bld_plain if self.trns_bld_plain != self.__class__.DEF_TRNS_BLD_PLAIN else None

        return {
            "type": self.__class__,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "trns_aug": self.trns_aug,
            "trns_bld_aug": trns_bld_aug,
            "trns_bld_plain": trns_bld_plain,
            "lp_phase": self._lp_phase,
        }

    @classmethod
    def _load_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        if cls is not config['type']:
            raise ValueError(f"Config type {config['type']} does not match class type {cls}")

        params_names = "num_classes", "input_shape", "trns_aug", "trns_bld_aug", "trns_bld_plain", "lp_phase"
        params = {key: value for key, value in config.items() if key in params_names}
        return params

    @property
    @abstractmethod
    def conv_target_layer(self):
        pass

    @property
    @abstractmethod
    def classifier_target_layer(self):
        pass

    @property
    def transform_aug(self):
        if self.trns_aug is not None:
            return self.__class__.DEF_TRNS_BLD_BASE(self.input_shape, augmentations=self.trns_aug())
        return self.trns_bld_aug(self.input_shape)

    @property
    def transform_plain(self):
        if self.trns_aug is not None:
            return self.__class__.DEF_TRNS_BLD_BASE(self.input_shape, augmentations=None)
        return self.trns_bld_plain(self.input_shape)

    @property
    def device(self):
        return next(self.parameters()).device

    def _lp_init_layers(self, placeholder=torch.nn.Identity()):
        if not self.support_layer_pretrain:
            raise RuntimeError("The model does not support layer pretraining")

        put_placeholders = self._lp_phase >= 0
        for i in range(LP_MAX_PHASE + 1):
            self._lp_init_layer(i, as_placeholder=put_placeholders, placeholder=placeholder)

        if put_placeholders:
            phase_target = self._lp_phase  # we want to reach the phase set by the user
            self._lp_phase = 0
            for _ in range(phase_target):
                self.lp_phase_step()

    def lp_phase_step(self):
        if not self.support_layer_pretrain:
            raise RuntimeError("The model does not support layer pretraining")
        if self._lp_phase == -1:  # out of LP
            return

        if not self._lp_phase == LP_MAX_PHASE:
            last_trained_layers = self._lp_get_last_trained_layers()
            if last_trained_layers is not None:
                for layer in last_trained_layers:
                    for param in layer.parameters():  # freeze last block of layers
                        param.requires_grad = False

            self._lp_step_phase()
            self._lp_phase += 1
        else:
            self._unfreeze_all_layers()
            self._lp_post_reset()
            self._lp_phase = -1


    def _lp_get_last_trained_layers(self):
        if not self.support_layer_pretrain:
            raise RuntimeError("The model does not support layer pretraining")
        raise NotImplementedError("The method must be implemented in the child class")

    def _lp_step_phase(self):
        if not self.support_layer_pretrain:
            raise RuntimeError("The model does not support layer pretraining")
        raise NotImplementedError("The method must be implemented in the child class")

    def _lp_init_layer(self, phase, as_placeholder=True, placeholder=torch.nn.Identity()):
        if not self.support_layer_pretrain:
            raise RuntimeError("The model does not support layer pretraining")
        raise NotImplementedError("The method must be implemented in the child class")

    def _lp_post_reset(self):
        if not self.support_layer_pretrain:
            raise RuntimeError("The model does not support layer pretraining")
        raise NotImplementedError("The method must be implemented in the child class")

import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from typing_extensions import Self

from settings.config import LP_MAX_PHASE
from src.data_processing.transformations import transform_aug_base, transform_plain_base


class BaseModel(ABC, torch.nn.Module):
    def __init__(self, num_classes, input_shape, lp_phase=None, **kwargs):
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)
        elif len(input_shape) > 2 or len(input_shape) < 1:
            raise ValueError("The input_shape must be a tuple of 2 integers")
        if input_shape[0] != input_shape[1]:
            raise ValueError("The input_shape must be a square")

        if lp_phase is not None and (not isinstance(lp_phase, int) or lp_phase < -1 or lp_phase > LP_MAX_PHASE):
            raise ValueError(f"The lp_phase must be an integer between 0 and {LP_MAX_PHASE}")  # -1 is for out of LP

        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self._lp_phase = lp_phase
        self.init_params = dict(num_classes=self.num_classes, input_shape=self.input_shape) | kwargs

        self.max_lp_phase = LP_MAX_PHASE

    def _unfreeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def support_layer_pretrain(self):
        return hasattr(self, "_lp_phase") and self._lp_phase is not None and self._lp_phase >= -1

    def to_config(self):
        return {"type": self.__class__} | self.init_params

    @classmethod
    @abstractmethod
    def load_from_config(cls, config: Dict[str, Any]) -> Self:
        return cls(**config)

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
        return transform_aug_base(self.init_params["input_shape"])

    @property
    def transform_plain(self):
        return transform_plain_base(self.init_params["input_shape"])

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

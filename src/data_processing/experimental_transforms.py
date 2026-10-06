"""Versioned full-frame preprocessing for the 4A experimental portfolio."""

from __future__ import annotations

import json
import math

import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as functional

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TRANSFORM_ID = "4a-rgb-fit-pad-224-v1"


def fit_pad_geometry(width: int, height: int) -> tuple[int, int, int, int, int, int]:
    """Return resized W/H and left/top/right/bottom, using integer half-up rounding."""
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("image dimensions must be positive integers")
    if width >= height:
        resized_width, resized_height = 224, max(1, (448 * height + width) // (2 * width))
    else:
        resized_width, resized_height = max(1, (448 * width + height) // (2 * height)), 224
    left, top = (224 - resized_width) // 2, (224 - resized_height) // 2
    return resized_width, resized_height, left, top, 224 - resized_width - left, 224 - resized_height - top


class ExperimentalFitPad224(v2.Transform):
    """RGB, float conversion, fit/pad, optional flip, then ImageNet normalization."""

    def __init__(self, horizontal_flip_probability: float = 0.0):
        super().__init__()
        if isinstance(horizontal_flip_probability, bool) or not isinstance(horizontal_flip_probability, (int, float)):
            raise ValueError("flip probability must be a finite number in [0,1]")
        if not math.isfinite(horizontal_flip_probability) or not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("flip probability must be a finite number in [0,1]")
        self.horizontal_flip_probability = float(horizontal_flip_probability)

    def to_config(self) -> dict:
        return {
            "schema_version": 1,
            "transform_id": TRANSFORM_ID,
            "input_shape": [224, 224],
            "input": "single_rgb_uint8_or_float_0_1",
            "dtype": "float32",
            "resize": {"mode": "fit_long_side", "rounding": "integer_half_up",
                       "interpolation": "bilinear", "antialias": True},
            "padding": {"placement": "center_odd_right_bottom", "fill": list(MEAN)},
            "augmentation": {"horizontal_flip_probability": self.horizontal_flip_probability},
            "normalization": {"mean": list(MEAN), "std": list(STD)},
            "order": ["rgb", "float_0_1", "resize", "pad", "augmentation", "normalize"],
        }

    @classmethod
    def from_config(cls, config: dict) -> ExperimentalFitPad224:
        try:
            instance = cls(config["augmentation"]["horizontal_flip_probability"])
            if json.dumps(config, sort_keys=True, allow_nan=False) != json.dumps(instance.to_config(), sort_keys=True):
                raise ValueError("unsupported experimental transform configuration")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("invalid experimental transform configuration") from error
        return instance

    def forward(self, image) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        tensor = functional.to_image(image)
        if tensor.ndim != 3 or tensor.shape[0] not in (1, 3, 4):
            raise ValueError("expected one image with 1, 3 or 4 channels")
        if any(dimension <= 0 for dimension in tensor.shape[-2:]):
            raise ValueError("image dimensions must be positive")
        if tensor.shape[0] == 1:
            tensor = tensor.expand(3, *tensor.shape[-2:])
        else:
            tensor = tensor[:3]
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255
        elif tensor.is_floating_point():
            if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
                raise ValueError("floating image values must be finite and in [0,1]")
            tensor = tensor.to(torch.float32)
        else:
            raise ValueError("image dtype must be uint8 or floating [0,1]")
        width, height, left, top, _, _ = fit_pad_geometry(tensor.shape[-1], tensor.shape[-2])
        resized = functional.resize(tensor, [height, width], interpolation=v2.InterpolationMode.BILINEAR,
                                    antialias=True)
        mean = tensor.new_tensor(MEAN).view(3, 1, 1)
        canvas = mean.expand(3, 224, 224).clone()
        canvas[:, top:top + height, left:left + width] = resized
        probability = self.horizontal_flip_probability
        if probability > 0 and (probability == 1 or torch.rand(()) < probability):
            canvas = canvas.flip(-1)
        return (canvas - mean) / tensor.new_tensor(STD).view(3, 1, 1)

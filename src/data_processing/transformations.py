from typing import Tuple, Optional, Sequence, List

import torch
from torchvision.transforms import v2
from torchvision.models import Weights

from settings.config import DEF_IMAGE_SHAPE


def transformations_wrapper(operation_list: List[v2.Transform], mean: Sequence[float], std: Sequence[float]
                            ) -> v2.Transform:
    return v2.Compose([
        v2.ToImage(),
        *operation_list,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])


def _transform_core_base(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE,
                         augmentations: Optional[List[v2.Transform]] = None) -> List[v2.Transform]:
    if augmentations is None:
        augmentations = []

    return [
        v2.Resize(image_shape),
        *augmentations,
    ]


def transform_plain_base(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE) -> List[v2.Transform]:
    return _transform_core_base(image_shape)


def transform_aug_base(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE, num_magnitude_bins=31) -> List[v2.Transform]:
    return _transform_core_base(image_shape, [v2.TrivialAugmentWide(num_magnitude_bins=num_magnitude_bins)])


def _transform_core_imagenet(weights: Optional[Weights] = None, image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE,
                             random_crop: bool = False, augmentations: Optional[List[v2.Transform]] = None
                             ) -> List[v2.Transform] | v2.Transform:
    if augmentations is None:
        augmentations = []

    crop_size = image_shape
    if weights is None:
        return _transform_core_base(crop_size, augmentations=augmentations)

    base_transform = weights.transforms()

    resize_size = max(image_shape[0], base_transform.resize_size)
    resize_dim = (resize_size, resize_size)
    interpolation = base_transform.interpolation
    antialias = base_transform.antialias
    normalize_op = [v2.Normalize(mean=base_transform.mean, std=base_transform.std)]

    if not random_crop:
        resize_components = [v2.Resize(resize_dim, interpolation=interpolation, antialias=antialias),
                             v2.CenterCrop(crop_size)]
    else:
        resize_components = [v2.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=antialias)]

    return v2.Compose([
        v2.ToImage(),
        *resize_components,
        *augmentations,
        v2.ToDtype(torch.float32, scale=True),
        *normalize_op
    ])


def transform_aug_imagenet(weights: Optional[Weights] = None, image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE,
                           random_crop: bool = False) -> List[v2.Transform] | v2.Transform:
    return _transform_core_imagenet(weights, image_shape, random_crop, [v2.TrivialAugmentWide()])


def transform_plain_imagenet(weights: Optional[Weights] = None, image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE
                             ) -> List[v2.Transform] | v2.Transform:
    return _transform_core_imagenet(weights, image_shape, random_crop=False)


def trasnform_aug_adv(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE) -> List[v2.Transform]:
    return [
    v2.RandomResizedCrop(image_shape, scale=(0.3, 1.0), ratio=(0.5, 2.0)),
    v2.RandomHorizontalFlip(0.3),
    v2.RandomVerticalFlip(0.3),
    v2.RandomRotation(0.3),
    v2.RandomErasing(0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ]
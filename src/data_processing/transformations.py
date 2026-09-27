from typing import Tuple, Optional, Sequence, List, Callable, Any

from PIL import Image
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.models import Weights
from torchvision.transforms.v2 import functional as v2f

from settings.config import DEF_IMAGE_SHAPE

DINO_MEAN, DINO_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

t_transform = Callable[[Image.Image | np.ndarray | torch.Tensor], torch.Tensor]
t_transform_builder = Callable[[Any], List[v2.Transform]]
t_augmentations_builder = Callable[[], List[v2.Transform]]

def transformations_wrapper(operation_list: List[v2.Transform], mean: Sequence[float], std: Sequence[float]
                            ) -> v2.Transform:
    return v2.Compose([
        v2.ToImage(),
        *operation_list,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])


def transform_core_base(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE,
                        augmentations: Optional[List[v2.Transform]] = None) -> List[v2.Transform]:
    if augmentations is None:
        augmentations = []

    return [
        v2.Resize(image_shape),
        *augmentations,
    ]


def transform_plain_base(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE) -> List[v2.Transform]:
    return transform_core_base(image_shape)


def transform_aug_base(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE, num_magnitude_bins=31) -> List[v2.Transform]:
    return transform_core_base(image_shape, [v2.TrivialAugmentWide(num_magnitude_bins=num_magnitude_bins)])


def transform_core_imagenet(weights: Optional[Weights] = None, image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE,
                            random_crop: bool = False, augmentations: Optional[List[v2.Transform]] = None
                            ) -> List[v2.Transform] | v2.Transform:
    if augmentations is None:
        augmentations = []

    crop_size = image_shape
    if weights is None:
        return transform_core_base(crop_size, augmentations=augmentations)

    base_transform = weights.transforms()

    resize_size = max(image_shape[0], base_transform.resize_size[0])
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
    return transform_core_imagenet(weights, image_shape, random_crop, [v2.TrivialAugmentWide()])


def transform_plain_imagenet(weights: Optional[Weights] = None, image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE
                             ) -> List[v2.Transform] | v2.Transform:
    return transform_core_imagenet(weights, image_shape, random_crop=False)

def transform_aug_adv(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE) -> List[v2.Transform]:
    return [
    v2.RandomResizedCrop(image_shape, scale=(0.33, 1.0), ratio=(0.5, 2.0)),
    v2.RandomHorizontalFlip(0.3),
    v2.RandomVerticalFlip(0.3),
    v2.RandomRotation(0.2),
    ]

    # ]
    # v2.RandomResizedCrop(image_shape, scale=(0.3, 1.0), ratio=(0.5, 2.0)),
    # v2.RandomHorizontalFlip(0.3),
    # v2.RandomVerticalFlip(0.3),
    # v2.RandomRotation(0.3),
    # v2.RandomErasing(0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    # ]

def transform_aug_imagenet_adv(weights: Optional[Weights] = None, image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE,
                                 random_crop: bool = False) -> List[v2.Transform] | v2.Transform:
    augmentations = transform_aug_adv(image_shape)[1:]
    return transform_core_imagenet(weights, image_shape, random_crop, augmentations)


def transform_plain_dino(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE) -> List[v2.Transform]:
    """
    Creates a plain transformation pipeline for DinoV2. No augmentation is applied.
    """
    shortest_edge = 256  # Resize the shortest edge to 256
    return [
        v2.Resize(shortest_edge, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),  # Resize while maintaining aspect ratio
        v2.CenterCrop(image_shape),  # Center crop to target size
        v2.ToDtype(torch.float32, scale=True),  # Rescale [0, 255] to [0, 1]
        v2.Normalize(mean=DINO_MEAN, std=DINO_STD),  # Normalize using DinoV2 stats
    ]


def transform_aug_dino(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE) -> List[v2.Transform]:
    """
    Creates an augmented transformation pipeline for DinoV2.
    Augmentations include random resizing, cropping, flipping, and normalization.
    """
    return [
        v2.RandomResizedCrop(image_shape, scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=v2.InterpolationMode.BICUBIC),  # Random crop
        v2.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        v2.ToDtype(torch.float32, scale=True),  # Rescale [0, 255] to [0, 1]
        v2.Normalize(mean=DINO_MEAN, std=DINO_STD),  # Normalize using DinoV2 stats
    ]

def transform_core_dino(image_shape: Tuple[int, int] = DEF_IMAGE_SHAPE,
                        augmentations: Optional[List[v2.Transform]] = None, random_crop: bool = False) -> List[v2.Transform]:
    """
    Creates a base transformation pipeline for DinoV2.
    """

    if augmentations is None:
        augmentations = []

    shortest_edge = 256  # Resize the shortest edge to 256

    if not random_crop:
        resize_components = [v2.Resize(shortest_edge, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),  # Resize while maintaining aspect ratio
                             v2.CenterCrop(image_shape)]
    else:
        resize_components = [v2.RandomResizedCrop(image_shape, scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=v2.InterpolationMode.BICUBIC)]

    return [
        v2.ToImage(),
        *resize_components,
        *augmentations,
        v2.ToDtype(torch.float32, scale=True),  # Rescale [0, 255] to [0, 1]
        v2.Normalize(mean=DINO_MEAN, std=DINO_STD),  # Normalize using DinoV2 stats
    ]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def selector_fit_pad_geometry(width: int, height: int, size: int = 384
                              ) -> Tuple[int, int, int, int, int, int]:
    """Return the exact Phase 3 resized geometry and center padding.

    The short side uses integer arithmetic equivalent to round-half-up.  The
    returned values are ``(resized_width, resized_height, left, top, right,
    bottom)``.
    """
    if width <= 0 or height <= 0 or size <= 0:
        raise ValueError("width, height, and size must be positive")

    if width >= height:
        resized_width = size
        resized_height = max(1, (2 * size * height + width) // (2 * width))
    else:
        resized_height = size
        resized_width = max(1, (2 * size * width + height) // (2 * height))

    horizontal = size - resized_width
    vertical = size - resized_height
    left = horizontal // 2
    top = vertical // 2
    return resized_width, resized_height, left, top, horizontal - left, vertical - top


class ConvertToRGB(torch.nn.Module):
    """Convert PIL inputs to RGB and reject ambiguous tensor channel counts."""

    def forward(self, image: Image.Image | np.ndarray | torch.Tensor):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        tensor = v2f.to_image(image)
        channels = tensor.shape[-3]
        if channels == 1:
            return tensor.expand(3, *tensor.shape[-2:])
        if channels == 4:
            return tensor[:3]
        if channels != 3:
            raise ValueError(f"selector images must have 1, 3, or 4 channels, got {channels}")
        return tensor


class SelectorFullFrameFitPad(torch.nn.Module):
    """Resize a full image once and mean-pad it to the Phase 3 square input."""

    def __init__(self, size: int = 384, mean: Sequence[float] = IMAGENET_MEAN):
        super().__init__()
        if len(mean) != 3:
            raise ValueError("mean must contain exactly three RGB values")
        self.size = int(size)
        self.mean = tuple(float(value) for value in mean)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"expected a [3,H,W] image, got {tuple(image.shape)}")
        height, width = image.shape[-2:]
        resized_width, resized_height, left, top, _, _ = selector_fit_pad_geometry(
            width, height, self.size
        )
        resized = v2f.resize(
            image,
            [resized_height, resized_width],
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )
        fill = torch.as_tensor(self.mean, dtype=resized.dtype, device=resized.device).view(3, 1, 1)
        output = fill.expand(3, self.size, self.size).clone()
        output[:, top:top + resized_height, left:left + resized_width] = resized
        return output


def transform_selector_efficientnet_v2_s(
        image_shape: Tuple[int, int] = (384, 384), train: bool = False) -> v2.Compose:
    """Build the frozen 4B-D1 full-frame EfficientNetV2-S transform."""
    if tuple(image_shape) != (384, 384):
        raise ValueError("the frozen selector transform requires image_shape=(384, 384)")
    operations: List[torch.nn.Module] = [
        ConvertToRGB(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        SelectorFullFrameFitPad(size=384, mean=IMAGENET_MEAN),
    ]
    if train:
        operations.append(v2.RandomHorizontalFlip(p=0.5))
    operations.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return v2.Compose(operations)


def transform_aug_selector_efficientnet_v2_s(
        image_shape: Tuple[int, int] = (384, 384)) -> v2.Compose:
    return transform_selector_efficientnet_v2_s(image_shape=image_shape, train=True)


def transform_plain_selector_efficientnet_v2_s(
        image_shape: Tuple[int, int] = (384, 384)) -> v2.Compose:
    return transform_selector_efficientnet_v2_s(image_shape=image_shape, train=False)

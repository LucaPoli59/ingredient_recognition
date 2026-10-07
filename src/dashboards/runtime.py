"""Shared data contract for checkpoint-backed dashboard inference."""

import torch
from torchvision.transforms import Compose, Normalize, v2

from src.data_processing.experimental_transforms import ExperimentalFitPad224, MEAN, STD


def prepare_visualization_image(transform, image):
    """Run preprocessing once, then undo only its declared RGB normalization."""
    if isinstance(transform, ExperimentalFitPad224):
        mean, std = MEAN, STD
    elif (isinstance(transform, (Compose, v2.Compose)) and transform.transforms
          and isinstance(transform.transforms[-1], (Normalize, v2.Normalize))):
        normalization = transform.transforms[-1]
        mean, std = normalization.mean, normalization.std
    else:
        raise ValueError("dashboard display requires a transform with known final RGB normalization")
    tensor = transform(image)
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 3
            or not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
        raise ValueError("dashboard preprocessing must produce a finite floating RGB tensor")
    display = tensor * tensor.new_tensor(std).reshape(-1, 1, 1) + tensor.new_tensor(mean).reshape(-1, 1, 1)
    return tensor, display.clamp(0, 1)


def load_visualization_datamodule(exp_config, model):
    """Restore saved labels and model preprocessing before constructing datasets."""
    exp_config.validate_ingredient_projection()
    config = exp_config.datamodule
    datamodule = config["type"].load_from_config(
        config, batch_size=exp_config.lgn_model["batch_size"],
        transform_aug=model.transform_aug, transform_plain=model.transform_plain,
    )
    datamodule.prepare_data()
    if datamodule.get_num_classes() != model.num_classes:
        raise ValueError("dashboard encoder and model output dimensions disagree")
    model.bind_ingredient_projection(getattr(datamodule, "projection_config", None))
    if hasattr(model, "bind_output_encoder"):
        model.bind_output_encoder(datamodule.label_encoder)
    elif hasattr(model, "bind_output_class_order"):
        model.bind_output_class_order(list(datamodule.label_encoder.classes))
    datamodule.setup("fit")
    return datamodule

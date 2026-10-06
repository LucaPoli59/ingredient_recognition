"""Shared data contract for checkpoint-backed dashboard inference."""


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
    datamodule.setup("fit")
    return datamodule

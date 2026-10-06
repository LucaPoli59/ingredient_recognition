"""Narrow construction boundary for opt-in 4A models and legacy consumers."""

import copy

import torch

from src.lightning.experimental_lgn import ExperimentalLGNM, validate_experimental_encoder
from src.models.experimental_contract import ExperimentalModelContract, validate_checkpoint_contract
from src.training.batching import ExactBatchPlan


def is_experimental_config(exp_config):
    model_type = exp_config.lgn_model["lgn_model_type"]
    experimental = isinstance(model_type, type) and issubclass(model_type, ExperimentalLGNM)
    contract = exp_config.lgn_model.get("torch_model", {}).get("experimental_contract")
    if experimental != (contract is not None):
        raise ValueError("experimental model contracts require the explicit ExperimentalLGNM path")
    return experimental


def prepare_experimental_config(exp_config, data_module, lgn_model_kwargs=None):
    """Persist fitted order and exact batching before an external config is saved."""
    if not is_experimental_config(exp_config):
        return
    data_module.prepare_data()
    config = exp_config.lgn_model
    width = data_module.get_num_classes()
    contract = ExperimentalModelContract.from_config(exp_config.torch_model["experimental_contract"])
    names = validate_experimental_encoder(data_module.label_encoder.to_config(), width)
    if exp_config.torch_model.get("num_classes") not in (None, width) or contract.num_classes != width:
        raise ValueError("DataModule width differs from the experimental model configuration")
    if config.get("output_class_order") not in (None, names):
        raise ValueError("DataModule order differs from the experimental configuration")
    options = copy.deepcopy(lgn_model_kwargs or {})
    for name in ("physical_batch_size", "max_physical_batch_size", "exact_batch_plan"):
        if name in config:
            if name in options and options[name] != config[name]:
                raise ValueError("runtime batching overrides disagree with saved configuration")
            options[name] = config[name]
    if options.get("exact_batch_plan") is not None:
        plan = ExactBatchPlan.from_config(options["exact_batch_plan"])
        if plan.requested_batch_size != config["batch_size"] or options.get("physical_batch_size") not in (None, plan.physical_batch_size) or options.get("max_physical_batch_size") not in (None, plan.max_physical_batch_size):
            raise ValueError("saved exact plan and experimental batch settings disagree")
    else:
        plan = ExactBatchPlan.resolve(config["batch_size"], options.get("max_physical_batch_size"), options.get("physical_batch_size"))
    if getattr(data_module, "projection_config", None) != exp_config.datamodule.get("ingredient_projection"):
        raise ValueError("DataModule and experiment ingredient projections disagree")
    exp_config.update_config(dm_label_encoder=data_module.label_encoder.to_config(), tm_num_classes=width,
                             hp_output_class_order=names, hp_exact_batch_plan=plan.to_config())


def load_model_for_experiment(exp_config, *, checkpoint_path=None, lgn_model_kwargs=None, legacy_drop_fields=None):
    """Fresh approved initialization, or strict offline reconstruction of saved state."""
    exp_config.validate_ingredient_projection()
    config = copy.deepcopy(exp_config.lgn_model)
    model_type = config["lgn_model_type"]
    if is_experimental_config(exp_config):
        encoder = exp_config.label_encoder
        names = validate_experimental_encoder(encoder, config["torch_model"]["num_classes"]) if encoder.get("fitted") else None
        if names is not None:
            names = list(names)
            if config.get("output_class_order") not in (None, names):
                raise ValueError("experiment encoder and experimental output order disagree")
            config["output_class_order"] = names
        if checkpoint_path is not None and config.get("output_class_order") is None:
            raise ValueError("offline experimental restoration requires saved ordered classes")
        checkpoint = None
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            validate_checkpoint_contract(checkpoint, ExperimentalModelContract.from_config(config["torch_model"]["experimental_contract"]))
            if not isinstance(checkpoint.get("state_dict"), dict) or not checkpoint["state_dict"]:
                raise ValueError("offline experimental restoration requires complete nonempty state")
        model = model_type.load_from_config(config, lgn_model_kwargs=lgn_model_kwargs,
                                            initialize_pretrained=checkpoint_path is None)
        if checkpoint is not None:
            model.load_weights_from_checkpoint_data(checkpoint)
        return model
    model = model_type.load_from_config(config, lgn_model_kwargs=lgn_model_kwargs)
    if checkpoint_path is not None:
        model.load_weights_from_checkpoint(checkpoint_path, weights_only=False, drop_fields=legacy_drop_fields)
    return model

"""Explicit 4A Lightning path; legacy and selector batching remain unchanged."""

import copy
import math

import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler, IterableDataset

from src.commons.config_enc_dec import decode_config, encode_config
from src.data_processing.labels_encoders import MultiLabelBinarizer
from src.lightning.lgn_models import BaseWithSchedulerLGNM
from src.models.experimental_contract import (
    CHECKPOINT_CONTRACT_KEY, ExperimentalModelContract, validate_checkpoint_contract,
)
from src.training.batching import (
    ExactBatchPlan, finite_loader_consumed_records, planned_optimizer_updates,
    accumulation_loss_scale,
)

BATCH_CHECKPOINT_KEY = "experimental_batch_plan"
ORDER_CHECKPOINT_KEY = "experimental_output_class_order"
TRAINING_CHECKPOINT_KEY = "experimental_training_config"


def _class_order(value, num_classes):
    if not isinstance(value, (list, tuple)) or len(value) != num_classes:
        raise ValueError("experimental output class order must match the model width")
    if any(not isinstance(name, str) or not name for name in value) or len(set(value)) != len(value):
        raise ValueError("experimental output classes must be unique nonempty strings")
    return list(value)


def validate_experimental_encoder(config, num_classes):
    if config.get("type") is not MultiLabelBinarizer or config.get("fitted") is not True:
        raise ValueError("experimental targets require a fitted strict MultiLabelBinarizer")
    names = _class_order(config.get("classes"), num_classes)
    if config.get("encode_map") != {name: index for index, name in enumerate(names)}:
        raise ValueError("experimental encoder mapping differs from its ordered classes")
    return names


class ExperimentalLGNM(BaseWithSchedulerLGNM):
    """Mean-BCE, exact finite-loader accumulation and strict protocol persistence."""

    SUPPORTS_EXPERIMENTAL_CONTRACT = True

    def __init__(self, *args, exact_batch_plan=None, physical_batch_size=None,
                 max_physical_batch_size=None, output_class_order=None, **kwargs):
        self._saved_batch_plan = None if exact_batch_plan is None else ExactBatchPlan.from_config(exact_batch_plan)
        self._requested_physical = physical_batch_size
        self._requested_cap = max_physical_batch_size
        self.output_class_order = None
        self._consumed_records = None
        self._restored_batch_progress = None
        # HPO's variable-only log filter must not remove restoration identity.
        kwargs.pop("hparams_to_register", None)
        super().__init__(*args, **kwargs)
        if not isinstance(getattr(self.model, "experimental_contract", None), ExperimentalModelContract):
            raise ValueError("ExperimentalLGNM requires a versioned experimental adapter")
        if self.loss_fn is not torch.nn.BCEWithLogitsLoss:
            raise ValueError("experimental exact accumulation requires mean BCEWithLogitsLoss")
        if type(self.weighted_loss) is not bool:
            raise ValueError("experimental weighted_loss must be a boolean")
        self.hparams["exact_batch_plan"] = self.exact_batch_plan.to_config()
        self.bind_output_class_order(output_class_order)

    def _init_effective_batch_size(self):
        if self._saved_batch_plan is not None:
            plan = self._saved_batch_plan
            if plan.requested_batch_size != self._batch_size:
                raise ValueError("saved exact batch request differs from the Lightning configuration")
            if self._requested_physical not in (None, plan.physical_batch_size) or self._requested_cap not in (None, plan.max_physical_batch_size):
                raise ValueError("batch overrides differ from the saved exact plan")
        else:
            plan = ExactBatchPlan.resolve(self._batch_size, self._requested_cap, self._requested_physical)
        self.exact_batch_plan = plan
        self._effective_batch_size = plan.physical_batch_size
        self.grad_accum = plan.accumulate_grad_batches

    @property
    def batch_size(self):
        return self.exact_batch_plan.physical_batch_size

    @batch_size.setter
    def batch_size(self, value):
        if type(value) is not int or value != self.exact_batch_plan.requested_batch_size:
            raise ValueError("experimental batch plan is immutable; construct a new configured model")

    def bind_output_class_order(self, value):
        if value is None:
            return
        names = _class_order(value, self.num_classes)
        if self.output_class_order is not None and names != self.output_class_order:
            raise ValueError("experimental output class order differs from the saved configuration")
        self.output_class_order = names
        self.hparams["output_class_order"] = list(names)

    def bind_ingredient_projection(self, value=None):
        super().bind_ingredient_projection(value)
        if self._ingredient_projection is not None:
            self.bind_output_class_order(self._ingredient_projection["class_order"])

    def bind_output_encoder(self, encoder):
        self.bind_output_class_order(validate_experimental_encoder(encoder.to_config(), self.num_classes))

    def startup_model(self, datamodule):
        if datamodule.get_num_classes() != self.num_classes:
            raise ValueError("experimental encoder and model widths disagree")
        self.bind_output_encoder(datamodule.label_encoder)
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            self._init_loss(datamodule)
        super().startup_model(datamodule)

    def _init_loss(self, datamodule):
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            if self.weighted_loss and not torch.equal(self.loss_fn.pos_weight.cpu(), datamodule.classes_weights.cpu()):
                raise ValueError("restored positive weights differ from the training DataModule")
            return
        super()._init_loss(datamodule)

    def configure_optimizers(self):
        options = {"lr": self._lr, "weight_decay": self.weight_decay_val or 0}
        if self.optimizer is torch.optim.SGD:
            options["momentum"] = self.momentum_val or 0
        optimizer = self.optimizer([p for p in self.model.parameters() if p.requires_grad], **options)
        if self.lr_scheduler is None:
            return optimizer
        from src.models.custom_schedulers import ConstantStartReduceOnPlateau
        scheduler_options = dict(self.lr_scheduler_params)
        if self.lr_scheduler is ConstantStartReduceOnPlateau:
            scheduler_options["initial_lr"] = self._lr
        scheduler = self.lr_scheduler(optimizer, **scheduler_options)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def on_save_checkpoint(self, checkpoint):
        if self.output_class_order is None:
            raise ValueError("cannot save experimental checkpoint without ordered classes")
        super().on_save_checkpoint(checkpoint)
        checkpoint[CHECKPOINT_CONTRACT_KEY] = self.model.experimental_contract.to_config()
        checkpoint[BATCH_CHECKPOINT_KEY] = self.exact_batch_plan.to_config()
        checkpoint[ORDER_CHECKPOINT_KEY] = list(self.output_class_order)
        checkpoint[TRAINING_CHECKPOINT_KEY] = encode_config(dict(self.hparams))

    def on_load_checkpoint(self, checkpoint):
        validate_checkpoint_contract(checkpoint, self.model.experimental_contract)
        if ExactBatchPlan.from_config(checkpoint.get(BATCH_CHECKPOINT_KEY)) != self.exact_batch_plan:
            raise ValueError("checkpoint exact batch plan differs from the configured plan")
        if self.output_class_order is None or _class_order(checkpoint.get(ORDER_CHECKPOINT_KEY), self.num_classes) != self.output_class_order:
            raise ValueError("checkpoint output class order differs from the configured order")
        if checkpoint.get(TRAINING_CHECKPOINT_KEY) != encode_config(dict(self.hparams)):
            raise ValueError("checkpoint experimental training configuration differs")
        params = checkpoint.get("hyper_parameters")
        if params is not None:
            if isinstance(params.get("torch_model"), (tuple, list)):
                params = decode_config(params)
            if params.get("lgn_model_type") is not type(self):
                raise ValueError("checkpoint experimental Lightning class disagrees")
            if encode_config(params.get("torch_model", {})) != encode_config(self.model.to_config()):
                raise ValueError("checkpoint model configuration fields disagree")
            if params.get("exact_batch_plan") != self.exact_batch_plan.to_config() or params.get("output_class_order") != self.output_class_order:
                raise ValueError("checkpoint experimental identity fields disagree")
            if params.get("batch_size") != self.exact_batch_plan.requested_batch_size or params.get("weighted_loss") != self.weighted_loss:
                raise ValueError("checkpoint batch/loss configuration differs")
        super().on_load_checkpoint(checkpoint)
        self._restored_batch_progress = checkpoint.get("loops", {}).get("fit_loop", {}).get("epoch_loop.batch_progress")

    def on_train_start(self):
        saved = self._restored_batch_progress
        if saved is not None:
            if not isinstance(saved, dict) or not isinstance(saved.get("current"), dict):
                raise ValueError("invalid saved experimental batch progress")
            ready = saved.get("current", {}).get("ready")
            if type(ready) is not int or ready < 0 or (ready != 0 and not (ready == self.trainer.num_training_batches and saved.get("is_last_batch") is True)):
                raise ValueError("experimental training supports epoch-boundary resume only, not partial epochs")

    def on_train_epoch_start(self):
        trainer = self.trainer
        loader = trainer.train_dataloader
        if trainer.world_size != 1 or trainer.accumulate_grad_batches != self.grad_accum:
            raise ValueError("experimental exact batching requires one device and fixed matching accumulation")
        if not isinstance(loader, DataLoader) or isinstance(loader.dataset, IterableDataset):
            raise ValueError("experimental batching requires one finite conventional DataLoader")
        if type(loader.batch_sampler) is not BatchSampler or type(loader.sampler) not in (RandomSampler, SequentialSampler):
            raise ValueError("custom sampling is unsupported by the exact accumulation contract")
        if isinstance(loader.sampler, RandomSampler) and (loader.sampler.replacement or loader.sampler.num_samples != len(loader.dataset)):
            raise ValueError("replacement/subsampled loaders are unsupported")
        if loader.drop_last or loader.batch_size != self.batch_size or trainer.fast_dev_run:
            raise ValueError("experimental loader must preserve rows and the configured physical batch")
        consumed = finite_loader_consumed_records(len(loader.dataset), self.exact_batch_plan, trainer.limit_train_batches)
        expected_batches = math.ceil(consumed / self.batch_size)
        if trainer.num_training_batches != expected_batches or consumed <= 0:
            raise ValueError("actual training horizon differs from the exact accumulation contract")
        self._consumed_records = consumed
        self._processed_records = 0
        self._epoch_start_step = trainer.global_step
        self._expected_epoch_updates = planned_optimizer_updates(consumed, self.exact_batch_plan)
        if trainer.max_steps >= 0:
            self._expected_epoch_updates = min(self._expected_epoch_updates, trainer.max_steps - trainer.global_step)

    def training_step(self, batch, batch_idx):
        if self._consumed_records is None or self.loss_fn.reduction != "mean":
            raise ValueError("exact accumulation requires a verified horizon and mean loss")
        loss, metrics_out = self._base_step(batch, self.train_metrics, self.train_per_ingredient_metrics)
        self._processed_records += len(batch[0])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=len(batch[0]))
        self._log_metric_collections(metrics_out, self.train_metrics.prefix)
        return loss * accumulation_loss_scale(batch_idx, len(batch[0]), self._consumed_records, self.exact_batch_plan)

    def on_train_epoch_end(self):
        if self.trainer.global_step - self._epoch_start_step != self._expected_epoch_updates:
            raise ValueError("observed optimizer updates differ from the exact accumulation horizon")
        expected_records = min(self._consumed_records, self._expected_epoch_updates * self.exact_batch_plan.requested_batch_size)
        if self._processed_records != expected_records:
            raise ValueError("actual record consumption stopped inside an accumulation group")
        self._consumed_records = None
        super().on_train_epoch_end()

    @classmethod
    def load_from_config(cls, config, lgn_model_kwargs=None, *, initialize_pretrained=True):
        if config.get("lgn_model_type") is not cls:
            raise ValueError("experimental Lightning configuration type disagrees")
        options = copy.deepcopy(lgn_model_kwargs or {})
        fields = ("exact_batch_plan", "output_class_order", "physical_batch_size", "max_physical_batch_size")
        for name in fields:
            if name in config:
                if name in options and options[name] != config[name]:
                    raise ValueError("runtime overrides disagree with saved experimental fields")
                options[name] = config[name]
        model_config = config["torch_model"]
        ExperimentalModelContract.from_config(model_config.get("experimental_contract"))
        if config.get("num_classes", model_config["num_classes"]) != model_config["num_classes"]:
            raise ValueError("Lightning and torch model output widths disagree")
        model = model_config["type"].load_from_config(model_config, initialize_pretrained=initialize_pretrained)
        module = cls(model=model, lr=config["lr"], batch_size=config["batch_size"], optimizer=config["optimizer"],
                     loss_fn=config["loss_fn"], weighted_loss=config.get("weighted_loss", False),
                     momentum=config.get("momentum"), weight_decay=config.get("weight_decay"),
                     use_swa=config.get("use_swa", False), metrics=copy.deepcopy(config.get("metrics")),
                     lr_scheduler=config.get("lr_scheduler"), lr_scheduler_params=copy.deepcopy(config.get("lr_scheduler_params")),
                     log_per_ingredient_metrics=config.get("log_per_ingredient_metrics", False), **options)
        module.bind_ingredient_projection(config.get("ingredient_projection"))
        return module

    def load_weights_from_checkpoint(self, checkpoint_path, weights_only=True, drop_fields=None):
        if drop_fields:
            raise ValueError("experimental restoration cannot drop saved state fields")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=weights_only)
        self.load_weights_from_checkpoint_data(checkpoint)

    def load_weights_from_checkpoint_data(self, checkpoint):
        self.on_load_checkpoint(checkpoint)
        state = checkpoint["state_dict"]
        positive_weights = state.get("loss_fn.pos_weight")
        if self.weighted_loss != (positive_weights is not None):
            raise ValueError("checkpoint positive weights disagree with the loss configuration")
        if positive_weights is not None and (positive_weights.shape != (self.num_classes,) or not torch.isfinite(positive_weights).all() or (positive_weights < 0).any()):
            raise ValueError("invalid checkpoint positive weights")
        if not isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weights)
        self.load_state_dict(state, strict=True)

    @classmethod
    def load_from_checkpoint(cls, *args, **kwargs):
        raise ValueError("use load_model_for_experiment with complete saved experiment configuration")

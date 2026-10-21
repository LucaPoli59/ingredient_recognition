"""Opt-in epoch ranking/selection delta on the accepted exact training path."""

from pathlib import Path
import math
import json
import time

import lightning as lgn
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
import optuna
import torch

from src.lightning.experimental_lgn import ExperimentalLGNM
from src.lightning.custom_callbacks import CSVLoggerEncode
from src.training.benchmark_protocol import BenchmarkSchedule, MONITOR, epoch_metrics, policy, policy_hash


class BenchmarkLGNM(ExperimentalLGNM):
    def __init__(self, *args, benchmark_policy_hash=None, expected_validation_records=None, **kwargs):
        if benchmark_policy_hash != policy_hash():
            raise ValueError("benchmark policy identity is missing or differs")
        if type(expected_validation_records) is not int or expected_validation_records <= 0:
            raise ValueError("the complete validation record count is required")
        super().__init__(*args, **kwargs)
        if self.weighted_loss or self.optimizer is not torch.optim.AdamW or self.lr_scheduler is not BenchmarkSchedule or self.use_swa:
            raise ValueError("benchmark optimizer/loss/schedule differs from the frozen recipe")
        contract = self.model.experimental_contract
        physical = policy()["physical_batches"][contract.model_id]
        adaptation = "frozen_encoder" if contract.model_id == "dinov2_b14_anchor" else "full"
        if (contract.head_seed != 42 or contract.horizontal_flip_probability != .5
                or contract.adaptation != adaptation or self.exact_batch_plan.physical_batch_size != physical
                or self.exact_batch_plan.max_physical_batch_size != physical
                or self.exact_batch_plan.requested_batch_size != 128 or self.lr_scheduler_params):
            raise ValueError("benchmark seed/augmentation/batch/schedule differs")
        self.expected_validation_records = expected_validation_records
        self.hparams["benchmark_policy_hash"] = benchmark_policy_hash
        self.hparams["expected_validation_records"] = expected_validation_records
        self._validation_parts = []
        self._ap_columns = None

    @classmethod
    def load_from_config(cls, config, lgn_model_kwargs=None, *, initialize_pretrained=True):
        options = dict(lgn_model_kwargs or {})
        for key in ("benchmark_policy_hash", "expected_validation_records"):
            if key in options and options[key] != config.get(key):
                raise ValueError("benchmark runtime override disagrees")
            options[key] = config.get(key)
        return super().load_from_config(config, options, initialize_pretrained=initialize_pretrained)

    def on_validation_epoch_start(self):
        self._validation_parts = []

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        logits = self(x)
        loss = self.loss_fn(logits, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=len(x))
        self._validation_parts.append((logits.detach().float().cpu(), targets.detach().cpu()))
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self._validation_parts.clear()
            return
        if not self._validation_parts:
            raise ValueError("benchmark validation produced no records")
        logits, targets = (torch.cat(parts).numpy() for parts in zip(*self._validation_parts))
        self._validation_parts.clear()
        metrics = epoch_metrics(logits, targets)
        if metrics["records"] != self.expected_validation_records:
            raise ValueError("benchmark selection requires the complete validation population")
        if self._ap_columns is not None and self._ap_columns != metrics["valid_ap_columns"]:
            raise ValueError("the validation AP label mask changed")
        self._ap_columns = metrics["valid_ap_columns"]
        self.log(MONITOR, metrics[MONITOR], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_micro_f1_at_0_5", metrics["val_micro_f1_at_0_5"], on_step=False, on_epoch=True)
        self.log("val_ap_valid_labels", float(len(self._ap_columns)), on_step=False, on_epoch=True)


class EpochPruning(Callback):
    """Prune only after validation/checkpoint saving, with one-based epochs."""
    def __init__(self, trial):
        self.trial = trial

    def on_train_epoch_end(self, trainer, pl_module):
        value = float(trainer.callback_metrics[MONITOR])
        if not math.isfinite(value):
            raise ValueError("nonfinite HPO objective")
        epoch = trainer.current_epoch + 1
        self.trial.report(value, step=epoch)
        if self.trial.number != 0 and epoch < 40 and self.trial.should_prune():
            raise optuna.TrialPruned(f"Hyperband at completed epoch {epoch}")


class EpochCostLedger(Callback):
    """Actual complete-epoch cost, including validation/checkpoint callbacks."""
    def __init__(self, path):
        self.path = Path(path)

    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize()
        record = {"epoch": trainer.current_epoch + 1, "seconds": time.monotonic() - self.start,
                  "optimizer_updates": trainer.global_step, "val_macro_ap": float(trainer.callback_metrics[MONITOR])}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


class BenchmarkTrainer(lgn.Trainer):
    """One shared canonical trainer adapter, never a per-family training loop.

    Keep best and last full states distinct. No renaming/deleting historical
    checkpoints, global scratch cleanup, or implicit last-on-tie replacement.
    """
    def __init__(self, save_dir, grad_accum=1, trial=None, *, accelerator="gpu", **kwargs):
        self.debug = False
        save_dir = Path(save_dir)
        checkpoint = ModelCheckpoint(dirpath=save_dir / "checkpoints", monitor=MONITOR, mode="max",
                                     save_top_k=1, save_last=False, every_n_epochs=1,
                                     save_on_train_epoch_end=False, filename="epoch={epoch}-map={val_macro_ap:.8f}")
        # An independent unmonitored callback saves every epoch, even when AP falls.
        self._latest_checkpoint = ModelCheckpoint(
            dirpath=save_dir / "checkpoints", monitor=None, save_top_k=1, save_last=False,
            every_n_epochs=1, save_on_train_epoch_end=False, filename="last",
            enable_version_counter=False, save_weights_only=False)
        extra = kwargs.pop("callbacks", [])
        for name in ("limit_train_batches", "limit_val_batches"):
            if name in kwargs and kwargs[name] != 1.0:
                raise ValueError("benchmark fits require the complete train/validation populations")
        logger = kwargs.pop("logger", CSVLoggerEncode(str(save_dir), name="logs", version="0"))
        super().__init__(max_epochs=40, accelerator=accelerator, devices=1, precision="32-true",
                         accumulate_grad_batches=grad_accum, check_val_every_n_epoch=1,
                         num_sanity_val_steps=0, benchmark=False, deterministic="warn",
                         default_root_dir=str(save_dir / "scratch"), logger=logger,
                         callbacks=extra + [EpochCostLedger(save_dir / "epoch_costs.jsonl")]
                         + ([EpochPruning(trial)] if trial is not None else [])
                         + [checkpoint, self._latest_checkpoint],
                         **kwargs)

    @property
    def last_checkpoint_path(self):
        return self._latest_checkpoint.best_model_path

    @classmethod
    def load_from_config(cls, config, grad_accum=1, **kwargs):
        if (config["type"] is not cls or config["max_epochs"] != 40 or config["debug"]
                or config.get("limit_train_batches") != 1.0 or config.get("precision") != "32-true"
                or config.get("check_val_every_n_epoch") != 1):
            raise ValueError("benchmark trainer configuration differs from the frozen protocol")
        return cls(config["save_dir"], grad_accum, log_every_n_steps=config["log_every_n_steps"], **kwargs)

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, datamodule=None,
            ckpt_path=None, wandb_notes=None, wandb_log_config=None):
        if datamodule is not None:
            model.startup_model(datamodule)
        super().fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
        best = self.checkpoint_callback.best_model_path
        if not best:
            raise ValueError("completed benchmark fit has no retained AP-selected checkpoint")
        final = torch.load(self.last_checkpoint_path, map_location="cpu", weights_only=False)
        if (final["epoch"] + 1 != self.current_epoch or final["global_step"] != self.global_step
                or not final.get("optimizer_states") or not final.get("lr_schedulers")):
            raise ValueError("benchmark last checkpoint does not contain the final full training state")
        del final
        model.load_weights_from_checkpoint_data(torch.load(best, map_location="cpu", weights_only=False))
        return model

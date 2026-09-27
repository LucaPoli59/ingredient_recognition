"""Canonical Lightning integration for the frozen Phase 3 selector."""

from __future__ import annotations

from typing import Any
import time

import lightning as lgn
import numpy as np
import torch

from src.ingredient_selection.artifacts import SelectorArtifactStore
from src.ingredient_selection.data import SelectorDataModule
from src.ingredient_selection.batching import BatchPlan, accumulation_loss_factor
from src.ingredient_selection.metrics import final_validation_bootstrap, per_label_metrics
from src.ingredient_selection.protocol import SelectorProtocol
from src.models.efficientnet import EfficientNetV2SSelector


def configure_cuda_memory_budget() -> dict[str, Any]:
    """Avoid WDDM/shared-host-memory oversubscription during resource trials."""
    if not torch.cuda.is_available():
        raise RuntimeError("the selector requires CUDA")
    free, total = torch.cuda.mem_get_info()
    allowed = min(total - 512 * 1024 ** 2, free - 256 * 1024 ** 2)
    if allowed <= 0:
        raise RuntimeError("insufficient free CUDA memory for the resource gate")
    fraction = allowed / total
    torch.cuda.set_per_process_memory_fraction(fraction)
    return {"free_at_start_mib": free / (1024 ** 2),
            "allowed_mib": allowed / (1024 ** 2), "allocator_fraction": fraction,
            "free_memory_margin_mib": 256}


class SelectorLightningModule(lgn.LightningModule):
    """Exact loss/optimizer/scheduler wrapper declared by Phase 3-D1."""

    def __init__(
            self,
            model: torch.nn.Module,
            pos_weight: torch.Tensor,
            protocol: SelectorProtocol = SelectorProtocol(),
            batch_plan: BatchPlan | None = None,
            train_records: int | None = None,
    ):
        super().__init__()
        if pos_weight.shape != (protocol.num_classes,):
            raise ValueError("pos_weight does not match the frozen class count")
        self.model = model
        self.protocol = protocol
        self.batch_plan = batch_plan
        self.train_records = train_records
        self.register_buffer("pos_weight", pos_weight.detach().clone().float())
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="mean")
        self.save_hyperparameters({"protocol": protocol.to_dict()})

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch, batch_index: int) -> torch.Tensor:
        images, targets, _ = batch
        logits = self(images)
        loss = self.loss_fn(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite selector loss at batch {batch_index}")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=images.shape[0])
        if self.batch_plan is not None and self.train_records is not None:
            loss = loss * accumulation_loss_factor(
                batch_index, images.shape[0], self.train_records, self.batch_plan)
        return loss

    def configure_optimizers(self):
        p = self.protocol
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if len(parameters) != len(list(self.model.parameters())):
            raise AssertionError("the selector optimizer must contain every model parameter")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=p.learning_rate,
            betas=p.adam_betas,
            eps=p.adam_eps,
            weight_decay=p.weight_decay,
            amsgrad=False,
            foreach=False,
            fused=False,
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=p.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=p.cosine_epochs,
            eta_min=p.minimum_learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[p.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


class SelectorAuditCallback(lgn.Callback):
    """Run AP/F1 audits at fixed model states outside the training loop."""

    def __init__(
            self,
            store: SelectorArtifactStore,
            run_id: str,
            protocol: SelectorProtocol = SelectorProtocol(),
    ):
        super().__init__()
        self.store = store
        self.run_id = run_id
        self.protocol = protocol
        self._completed = store.existing_audit_epochs()

    @staticmethod
    def _collect(module: SelectorLightningModule, dataloader) -> tuple[list[str], np.ndarray, np.ndarray]:
        was_training = module.training
        module.eval()
        record_ids: list[str] = []
        logits_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for images, targets, batch_ids in dataloader:
                images = images.to(module.device, non_blocking=True)
                logits = module(images)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("audit logits contain non-finite values")
                logits_batches.append(logits.detach().cpu().numpy())
                target_batches.append(targets.numpy())
                record_ids.extend(map(str, batch_ids))
        module.train(was_training)
        return record_ids, np.concatenate(target_batches), np.concatenate(logits_batches)

    def _audit(self, trainer: lgn.Trainer, module: SelectorLightningModule, epoch: int) -> None:
        if epoch in self._completed:
            return
        datamodule = trainer.datamodule
        if not isinstance(datamodule, SelectorDataModule):
            raise TypeError("SelectorAuditCallback requires SelectorDataModule")
        optimizer = trainer.optimizers[0]
        learning_rate = float(optimizer.param_groups[0]["lr"])
        rows: list[dict[str, Any]] = []
        validation_payload = None
        for split, dataloader in (
                ("train", datamodule.audit_train_dataloader()),
                ("val", datamodule.val_dataloader()),
        ):
            record_ids, targets, logits = self._collect(module, dataloader)
            rows.extend(per_label_metrics(
                logits,
                targets,
                datamodule.bundle.class_names,
                run_id=self.run_id,
                split=split,
                audit_epoch=epoch,
                learning_rate=learning_rate,
                threshold=self.protocol.f1_threshold,
            ))
            if split == "val":
                validation_payload = (record_ids, targets, logits)
        assert validation_payload is not None
        self.store.write_validation_scores(epoch, *validation_payload)
        self.store.append_metric_rows(rows)
        if epoch == self.protocol.max_epochs:
            _, targets, logits = validation_payload
            self.store.write_bootstrap(final_validation_bootstrap(
                logits, targets, datamodule.bundle.class_names, self.protocol
            ))
        self._completed.add(epoch)

    def on_fit_start(self, trainer: lgn.Trainer, pl_module: SelectorLightningModule) -> None:
        self._audit(trainer, pl_module, 0)

    def on_train_epoch_end(self, trainer: lgn.Trainer, pl_module: SelectorLightningModule) -> None:
        completed_epoch = trainer.current_epoch + 1
        if completed_epoch in self.protocol.audit_epochs:
            self._audit(trainer, pl_module, completed_epoch)


def run_resource_gate(
        datamodule: SelectorDataModule,
        *,
        protocol: SelectorProtocol = SelectorProtocol(),
        batch_plan: BatchPlan,
        full_epoch: bool = False,
) -> dict[str, Any]:
    """Measure disposable Lightning training, including initialized AdamW state."""
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen selector resource gate requires CUDA")
    memory_budget = configure_cuda_memory_budget()
    lgn.seed_everything(protocol.seed, workers=True)
    pos_weight = torch.as_tensor(
        (len(datamodule.bundle.train.record_ids) - datamodule.bundle.train.supports)
        / datamodule.bundle.train.supports,
        dtype=torch.float32,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = {
        "device": torch.cuda.get_device_name(),
        "batch_shape": [batch_plan.physical_batch_size, 3, 384, 384],
        "batch_plan": batch_plan.to_dict(),
        "full_epoch": full_epoch,
        "memory_budget": memory_budget,
        "device_total_mib": torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
    }
    try:
        model = EfficientNetV2SSelector(num_classes=protocol.num_classes)
        module = SelectorLightningModule(model, pos_weight, protocol, batch_plan,
                                        len(datamodule.bundle.train.record_ids))
        trainer = lgn.Trainer(
            accelerator="gpu", devices=1, max_epochs=1, precision="32-true",
            deterministic=True, accumulate_grad_batches=batch_plan.accumulate_grad_batches,
            limit_train_batches=1.0 if full_epoch else 2 * batch_plan.accumulate_grad_batches,
            limit_val_batches=0, num_sanity_val_steps=0, logger=False,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        )
        trainer.fit(module, datamodule=datamodule)
        if full_epoch:
            module.eval()
            with torch.inference_mode():
                for images, _, _ in datamodule.val_dataloader():
                    if not torch.isfinite(module(images.to(module.device))).all():
                        raise FloatingPointError("non-finite capacity validation logits")
        torch.cuda.synchronize()
        result.update(passed=True, optimizer_steps=trainer.global_step,
                      completed_epochs=trainer.current_epoch)
    except torch.OutOfMemoryError as error:
        result.update(passed=False, failure="cuda_out_of_memory", message=str(error))
    result.update(
        peak_allocated_mib=torch.cuda.max_memory_allocated() / (1024 ** 2),
        peak_reserved_mib=torch.cuda.max_memory_reserved() / (1024 ** 2),
        elapsed_seconds=time.monotonic() - started,
    )
    return result

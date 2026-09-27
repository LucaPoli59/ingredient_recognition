"""Canonical Lightning integration for the frozen Phase 3 selector."""

from __future__ import annotations

from typing import Any

import lightning as lgn
import numpy as np
import torch

from src.ingredient_selection.artifacts import SelectorArtifactStore
from src.ingredient_selection.data import SelectorDataModule
from src.ingredient_selection.metrics import final_validation_bootstrap, per_label_metrics
from src.ingredient_selection.protocol import SelectorProtocol
from src.models.efficientnet import EfficientNetV2SSelector


class SelectorLightningModule(lgn.LightningModule):
    """Exact loss/optimizer/scheduler wrapper declared by Phase 3-D1."""

    def __init__(
            self,
            model: torch.nn.Module,
            pos_weight: torch.Tensor,
            protocol: SelectorProtocol = SelectorProtocol(),
    ):
        super().__init__()
        if pos_weight.shape != (protocol.num_classes,):
            raise ValueError("pos_weight does not match the frozen class count")
        self.model = model
        self.protocol = protocol
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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
) -> dict[str, Any]:
    """Measure a disposable real-data forward/BCE/backward/AdamW step."""
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen selector resource gate requires CUDA")
    lgn.seed_everything(protocol.seed, workers=True)
    model = EfficientNetV2SSelector(num_classes=protocol.num_classes).cuda()
    pos_weight = torch.as_tensor(
        (len(datamodule.bundle.train.record_ids) - datamodule.bundle.train.supports)
        / datamodule.bundle.train.supports,
        dtype=torch.float32,
    )
    module = SelectorLightningModule(model, pos_weight, protocol).cuda()
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    images, targets, _ = next(iter(datamodule.train_dataloader()))
    if images.shape != (protocol.batch_size, 3, protocol.image_size, protocol.image_size):
        raise AssertionError(f"resource gate received unexpected batch shape {tuple(images.shape)}")
    images = images.cuda()
    targets = targets.cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    loss = module.loss_fn(module(images), targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "device": torch.cuda.get_device_name(),
        "batch_shape": list(images.shape),
        "loss": float(loss.detach().cpu()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 ** 2),
        "device_total_mib": torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
        "passed": bool(torch.isfinite(loss)),
    }
    del optimizer, module, model, images, targets, loss
    torch.cuda.empty_cache()
    return result

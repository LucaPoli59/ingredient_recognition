"""Frozen Phase 6 configurations. Importing or printing them never starts training."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
import torch

POLICY_PATH = Path(__file__).with_name("resources") / "phase6_protocol_v1.json"
MONITOR = "val_macro_ap"


def policy():
    return json.loads(POLICY_PATH.read_text())


def policy_hash():
    return hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()


def epoch_metrics(logits, targets):
    """Non-interpolated global AP on raw ranks; never a mean of batch APs."""
    scores, y = np.asarray(logits), np.asarray(targets)
    if scores.ndim != 2 or scores.shape != y.shape or not len(scores):
        raise ValueError("nonempty record-by-class score/target matrices are required")
    if not np.isfinite(scores).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("scores must be finite and targets binary")
    counts = y.sum(axis=0)
    valid = (counts > 0) & (counts < len(y))
    if not valid.any():
        raise ValueError("AP requires at least one label with both target outcomes")
    aps = [float(average_precision_score(y[:, c], scores[:, c])) for c in np.flatnonzero(valid)]
    predicted = scores >= 0
    tp = np.logical_and(predicted, y == 1).sum()
    denominator = predicted.sum() + y.sum()
    return {MONITOR: float(np.mean(aps)), "val_micro_f1_at_0_5": float(2 * tp / denominator) if denominator else 0.,
            "valid_ap_columns": np.flatnonzero(valid).tolist(), "records": len(y)}


def lr_factor(epoch):
    # Used for epochs 1..40 at zero-based indices 0..39. No plateau extension.
    if epoch == 0:
        return .1
    if epoch == 1:
        return 1.
    return .001 + .999 * .5 * (1 + math.cos(math.pi * min(epoch - 1, 38) / 38))


class BenchmarkSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, lr_factor, last_epoch=last_epoch)


def make_config(model_id, save_dir, *, selected=False, lr=None, weight_decay=None):
    from src.commons.exp_config import ExpConfig
    from src.models.experimental_contract import ExperimentalModelContract
    from src.models.experimental_efficientnet import EfficientNetV2SExperiment
    from src.models.experimental_maxvit import MaxViTTExperiment
    from src.models.experimental_dica_net import DICANetSExperiment
    from src.models.benchmark_anchors import BenchmarkAnchor
    from src.lightning.benchmark_lgn import BenchmarkLGNM, BenchmarkTrainer
    from src.ingredient_selection.runtime import resolve_projection

    p = policy()
    if model_id not in p["models"] or type(selected) is not bool:
        raise ValueError("unknown protocol or invalid selected flag")
    projection = resolve_projection(p["selected_projection"])
    if projection.artifact_hash != p["selected_artifact_hash"]:
        raise ValueError("frozen selected artifact differs")
    order = list(projection.class_order if selected else projection.base_class_order)
    frozen = model_id == "dinov2_b14_anchor"
    contract = ExperimentalModelContract(model_id, len(order), "frozen_encoder" if frozen else "full", 42, .5)
    model_type = {"efficientnet_v2_s": EfficientNetV2SExperiment, "maxvit_t": MaxViTTExperiment,
                  "p2_s": DICANetSExperiment}.get(model_id, BenchmarkAnchor)
    model_config = {"type": model_type, "num_classes": len(order), "input_shape": (224, 224),
                    "trns_aug": None, "trns_bld_aug": None, "trns_bld_plain": None, "lp_phase": None,
                    "experimental_contract": contract.to_config()}
    if model_type is BenchmarkAnchor:
        model_config["anchor_provenance"] = BenchmarkAnchor.provenance(model_id)
    elif model_type is DICANetSExperiment:
        model_config["encoder_provenance"] = DICANetSExperiment._encoder_provenance()
    config = ExpConfig(
        hp_lgn_model_type=BenchmarkLGNM, tm_=model_config, hp_batch_size=128,
        hp_lr=lr if lr is not None else (1e-3 if frozen else 1e-4),
        hp_weight_decay=weight_decay if weight_decay is not None else 1e-4,
        hp_loss_fn=torch.nn.BCEWithLogitsLoss, hp_weighted_loss=False, hp_optimizer=torch.optim.AdamW,
        hp_use_swa=False, hp_metrics={}, hp_lr_scheduler=BenchmarkSchedule, hp_lr_scheduler_params={},
        hp_physical_batch_size=p["physical_batches"][model_id],
        hp_max_physical_batch_size=p["physical_batches"][model_id], hp_output_class_order=order,
        hp_benchmark_policy_hash=policy_hash(), hp_expected_validation_records=5996,
        tr_type=BenchmarkTrainer, tr_max_epochs=40, tr_save_dir=str(save_dir), tr_debug=False,
        tr_limit_train_batches=1.0, tr_precision="32-true", tr_check_val_every_n_epoch=1,
        dm_num_workers=2, dm_ingredient_projection=projection.to_config() if selected else None,
        lb_classes=order, lb_encode_map={name: i for i, name in enumerate(order)}, lb_fitted=True)
    return config


def local_candidates(lr, weight_decay):
    """Four additional one-axis configurations; transfer itself is not rerun."""
    return [(lr * .5, weight_decay), (lr * 2, weight_decay),
            (lr, weight_decay * .1), (lr, weight_decay * 10)]

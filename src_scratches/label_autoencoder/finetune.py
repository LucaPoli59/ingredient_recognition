"""Prepare by default; explicit --execute starts sequential CUDA backbone fine-tuning."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet18

from src.data_processing.experimental_transforms import ExperimentalFitPad224
from src_scratches.label_autoencoder import compare, image_models
from src_scratches.label_autoencoder.finetune_data import ARMS, RecipeImages, prepare


class ImageModel(nn.Module):
    def __init__(self, backbone, head, arm):
        super().__init__()
        if arm not in ARMS or (arm == "direct") != (head.kind == "direct"):
            raise ValueError("Head/arm mismatch")
        self.backbone, self.head, self.arm = backbone, head, arm
        self.backbone.requires_grad_(True)
        self.head.projection.requires_grad_(True)
        if arm != "direct":
            self.head.decoder.requires_grad_(arm == "ae32_trainable_decoder")

    def train(self, mode=True):
        super().train(mode)
        if self.arm == "ae32_frozen_decoder":
            self.head.decoder.eval()
        return self

    def forward(self, images):
        return self.head(self.backbone(images))


def optimizer_for(model, config):
    groups = [{"params": model.backbone.parameters(), "lr": config["backbone_lr"], "name": "backbone"},
              {"params": model.head.projection.parameters(), "lr": config["head_lr"], "name": "projection"}]
    if model.arm == "ae32_trainable_decoder":
        groups.append({"params": model.head.decoder.parameters(), "lr": config["decoder_lr"], "name": "decoder"})
    return torch.optim.Adam(groups)


def losses(logits, code, labels, target_code, latent_weight):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    mse = nn.functional.mse_loss(code, target_code) if code is not None else bce.new_zeros(())
    objective = bce + latent_weight * mse
    if not torch.isfinite(objective):
        raise FloatingPointError("Nonfinite objective")
    return objective, bce, mse


def epoch_pass(model, loader, device, latent_weight, optimizer=None):
    model.train(optimizer is not None)
    totals = np.zeros(3, dtype=np.float64)
    count, scores, codes = 0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels, target_code in loader:
            images = images.to(device, memory_format=torch.channels_last)
            labels, target_code = labels.to(device), target_code.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits, code = model(images)
            objective, bce, mse = losses(logits, code, labels, target_code, latent_weight)
            if optimizer is not None:
                objective.backward()
                optimizer.step()
            else:
                scores.append(logits.sigmoid().cpu().numpy())
                if code is not None:
                    codes.append(code.cpu().numpy())
            totals += np.array([objective.item(), bce.item(), mse.item()]) * len(labels)
            count += len(labels)
    if not count:
        raise ValueError("Empty loader")
    stats = dict(zip(("objective", "bce", "latent_mse"), (totals / count).tolist()))
    return stats, np.concatenate(scores) if scores else None, np.concatenate(codes) if codes else None


def metrics(targets, scores, support, labels):
    result, per_label = compare.evaluate(targets, scores, support, labels)
    precision, recall, thresholds = precision_recall_curve(targets.ravel(), scores.ravel())
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    threshold = float(thresholds[int(np.argmax(f1))])
    result["micro_f1_val_exact"] = compare.micro_f1(*compare.counts(targets, scores, threshold))
    result["val_exact_threshold"] = threshold
    # Per-label rows explicitly retain compare.evaluate's historical grid threshold.
    return result, per_label


def decoder_diagnostic(model, normalized_codes, labels, device, batch_size=1024):
    """Reconstruction from true-label codes: diagnostic only, never an image prediction."""
    model.eval()
    scores, loss = [], 0.0
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            codes = torch.from_numpy(normalized_codes[start:start + batch_size]).to(device)
            y = torch.from_numpy(labels[start:start + batch_size]).to(device)
            logits = model.head.decoder(codes * model.head.latent_std + model.head.latent_mean)
            loss += nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
            scores.append(logits.sigmoid().cpu().numpy())
    values = np.concatenate(scores)
    present = labels.sum(axis=0) > 0
    ap = compare.average_precision_score(labels[:, present], values[:, present], average="macro")
    return {"scope": "Validation ingredient reconstruction from fixed teacher codes, not image inference",
            "bce": loss / labels.size, "macro_ap": float(ap),
            "micro_f1_at_0_5": compare.micro_f1(*compare.counts(labels, values, 0.5))}


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    compare.write_json(temporary, value)
    temporary.replace(path)


def save_checkpoint(path, model, specification, contract, epoch, val_bce):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"format": "resnet18-ae32-finetune-v1", "arm": model.arm,
                "state_dict": cpu_state(model), "head_specification": specification,
                "contract": contract, "best_epoch": epoch, "best_val_bce": val_bce}, temporary)
    temporary.replace(path)


def empty_backbone():
    model = resnet18(weights=None)
    model.fc = nn.Identity()
    return model


def load_model(path, device=torch.device("cpu"), backbone_factory=empty_backbone):
    """Offline, self-contained inference restore; no source AE or pretrained file needed."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != "resnet18-ae32-finetune-v1":
        raise ValueError("Unsupported fine-tuned checkpoint")
    spec, state = saved["head_specification"], saved["state_dict"]
    decoder = None
    if saved["arm"] != "direct":
        decoder = nn.Sequential(nn.Linear(spec["latent_dim"], spec["decoder_hidden_dim"]), nn.ReLU(),
                                nn.Linear(spec["decoder_hidden_dim"], spec["label_count"]))
    head = image_models.ImageHead(spec["kind"], state["head.feature_mean"], state["head.feature_std"],
                                 spec["label_count"], decoder, state.get("head.latent_mean"), state.get("head.latent_std"))
    model = ImageModel(backbone_factory(), head, saved["arm"])
    model.load_state_dict(state, strict=True)
    return model.to(device=device, memory_format=torch.channels_last).eval(), saved


def predict_image(image_path, checkpoint, backbone_factory=empty_backbone):
    model, saved = load_model(checkpoint, backbone_factory=backbone_factory)
    transform = ExperimentalFitPad224.from_config(saved["contract"]["transform"])
    with Image.open(image_path) as image, torch.inference_mode():
        logits, normalized_code = model(transform(image)[None].to(memory_format=torch.channels_last))
        code = ((normalized_code * model.head.latent_std + model.head.latent_mean)[0].numpy()
                if normalized_code is not None else None)
    return {"labels": saved["contract"]["labels"], "probabilities": logits.sigmoid()[0].numpy(), "code": code}


def fit_arm(model, train_loader, val_loader, config, output, specification, contract, val_y, val_z,
            support, device):
    """Shared training core; CPU is used only by the bounded synthetic tests."""
    started = time.perf_counter()
    optimizer = optimizer_for(model, config)
    initial_state = cpu_state(model)
    initial, initial_scores, _ = epoch_pass(model, val_loader, device, config["latent_weight"])
    initial_metrics, _ = metrics(val_y, initial_scores, support, contract["labels"])
    is_ae = model.arm != "direct"
    reconstruction_before = decoder_diagnostic(model, val_z, val_y, device) if is_ae else None
    path = output / f"{model.arm}.pt"
    best, anchor, best_epoch, stale = initial["bce"], initial["bce"], 0, 0
    save_checkpoint(path, model, specification, contract, 0, best)
    history = [{"epoch": 0, "train_objective": None, "train_bce": None, "train_latent_mse": None,
                "val_bce": best, "val_latent_mse": initial["latent_mse"] if is_ae else None,
                "elapsed_seconds": time.perf_counter() - started}]
    compare.write_csv(output / f"{model.arm}_history.csv", history)
    for epoch in range(1, config["epochs"] + 1):
        trained, _, _ = epoch_pass(model, train_loader, device, config["latent_weight"], optimizer)
        validated, _, _ = epoch_pass(model, val_loader, device, config["latent_weight"])
        history.append({"epoch": epoch, "train_objective": trained["objective"], "train_bce": trained["bce"],
                        "train_latent_mse": trained["latent_mse"] if is_ae else None,
                        "val_bce": validated["bce"], "val_latent_mse": validated["latent_mse"] if is_ae else None,
                        "elapsed_seconds": time.perf_counter() - started})
        if validated["bce"] < best:
            best, best_epoch = validated["bce"], epoch
            save_checkpoint(path, model, specification, contract, epoch, best)
        if validated["bce"] < anchor - config["min_delta"]:
            anchor, stale = validated["bce"], 0
        else:
            stale += 1
        compare.write_csv(output / f"{model.arm}_history.csv", history)
        print(f"{model.arm} epoch={epoch}: val BCE={validated['bce']:.6f}; best={best_epoch}", flush=True)
        if stale >= config["patience"]:
            break
    # Check the final state too: selecting epoch zero must not conceal accidental updates.
    if model.arm == "ae32_frozen_decoder":
        if any(not torch.equal(value, initial_state[key]) for key, value in cpu_state(model).items()
               if key.startswith("head.decoder.")):
            raise AssertionError("Frozen decoder changed during training")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    _, scores, codes = epoch_pass(model, val_loader, device, config["latent_weight"])
    measured, per_label = metrics(val_y, scores, support, contract["labels"])
    compare.write_csv(output / f"{model.arm}_per_label_grid.csv", per_label)
    result = {"method": model.arm, "metrics": measured, "epoch_zero_metrics": initial_metrics,
              "epochs_ran": epoch, "best_epoch": best_epoch, "best_val_bce": best,
              "optimizer_steps": epoch * len(train_loader), "fit_seconds": time.perf_counter() - started,
              "checkpoint_sha256": compare.sha256(path),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    if is_ae:
        error = np.sum((codes.astype(np.float64) - val_z) ** 2)
        variation = np.sum((val_z.astype(np.float64) - val_z.mean(axis=0)) ** 2)
        delta = sum(float(((value.cpu() - initial_state["head.decoder." + key]).double() ** 2).sum())
                    for key, value in model.head.decoder.state_dict().items())
        base = sum(float((value.double() ** 2).sum()) for key, value in initial_state.items()
                   if key.startswith("head.decoder."))
        result["latent_diagnostics"] = {"normalized_mse": float(error / val_z.size),
                                       "variance_weighted_r2": float(1 - error / variation) if variation > 0 else None}
        result["decoder_retention"] = {"before": reconstruction_before,
            "after": decoder_diagnostic(model, val_z, val_y, device),
            "parameter_relative_l2_change": math.sqrt(delta / base) if base > 0 else None}
    return result


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def cuda_device():
    """Fail closed on a busy/unknown GPU; no waiting loop or CPU fallback."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "0"):
        raise RuntimeError("This launcher checks physical GPU 0; CUDA_VISIBLE_DEVICES must be unset or 0")
    snapshot = compare.gpu_snapshot()
    samples = snapshot.get("samples", [])
    if snapshot.get("error") or len(samples) != 3 or any(
        row["free_mib"] < 6144 or row["utilization_pct"] > 10 for row in samples
    ):
        raise RuntimeError("GPU busy/unknown: requires three samples with >=6144 MiB free and <=10% utilization")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no automatic CPU fallback")
    return torch.device("cuda:0"), snapshot


def write_report(output, summary):
    lines = ["# Backbone fine-tuning pilot", "", f"Status: {summary['status']}", "",
             "Warm-started, single-seed validation experiment; no final-test evaluation.",
             "F1 uses an exact validation threshold sweep; all arms select by ingredient BCE.", "",
             "| Arm | Macro AP | Micro F1 tuned | Rare AP | Best epoch | Decoder reconstruction AP before / after |",
             "| --- | --- | --- | --- | --- | --- |"]
    for arm in summary["arms"]:
        m, retention = arm["metrics"], arm.get("decoder_retention")
        retained = (f"{retention['before']['macro_ap']:.4f} / {retention['after']['macro_ap']:.4f}"
                    if retention else "—")
        lines.append(f"| {arm['method']} | {m['macro_ap']:.4f} | {m['micro_f1_val_exact']:.4f} | "
                     f"{m['frequency_slices']['rare']['macro_ap']:.4f} | {arm['best_epoch']} | {retained} |")
    lines.extend(["", "Decoder reconstruction uses true ingredient codes and is not an image-only score.",
                  "Preserving reconstruction and improving image prediction are separate observations.",
                  "Epoch zero remains eligible if fine-tuning fails to improve validation BCE.", ""])
    (output / "report.md").write_text("\n".join(lines))


def execute(prepared, output):
    if output.exists():
        raise FileExistsError("Use a new output directory; optimizer resume is not implemented")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device, snapshot = cuda_device()
    config, paths = prepared["config"], prepared["paths"]
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = copy.deepcopy(prepared["plan"])
    manifest.update(status="running", training_started=True, started_utc=datetime.now(timezone.utc).isoformat(),
                    gpu_snapshot=snapshot, device=str(device), precision="float32", batchnorm="train all backbone statistics and affine parameters",
                    versions={"torch": str(torch.__version__), "torchvision": torchvision.__version__, "numpy": np.__version__})
    manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=compare.ROOT, text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    summary = {"status": "running", "arms": []}
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=config["threads"]):
            ae_head = prepared["heads"]["ae32_aligned"]
            train_codes = image_models.teacher_codes(prepared["teacher"], prepared["train_y"])
            checked_mean, checked_std = image_models.normalization(train_codes)
            mean, std = ae_head.latent_mean.numpy(), ae_head.latent_std.numpy()
            if not (np.allclose(checked_mean, mean, rtol=1e-5, atol=1e-7)
                    and np.allclose(checked_std, std, rtol=1e-5, atol=1e-7)):
                raise ValueError("Teacher normalization differs from source training")
            latent = {"train": (train_codes - mean) / std,
                      "val": (image_models.teacher_codes(prepared["teacher"], prepared["val_y"]) - mean) / std}
            transform = ExperimentalFitPad224.from_config(manifest["contract"]["transform"])
            datasets = {split: RecipeImages(paths["data_dir"] / "imgs/standard", prepared["rows"][split],
                                           prepared[split + "_y"], latent[split], transform) for split in ("train", "val")}
            for arm in config["arms"]:
                summary["active_arm"] = arm
                write_json(output / "summary.json", summary)
                random.seed(config["seed"])
                np.random.seed(config["seed"])
                torch.manual_seed(config["seed"])
                source = "direct" if arm == "direct" else "ae32_aligned"
                model = ImageModel(image_models.load_backbone(paths["weights"]), copy.deepcopy(prepared["heads"][source]), arm)
                model.to(device=device, memory_format=torch.channels_last)
                loaders = {split: DataLoader(datasets[split], batch_size=config["batch_size"], shuffle=split == "train",
                           num_workers=config["workers"], pin_memory=False, drop_last=False, worker_init_fn=seed_worker,
                           generator=torch.Generator().manual_seed(config["seed"])) for split in ("train", "val")}
                result = fit_arm(model, loaders["train"], loaders["val"], config, output,
                                 prepared["specifications"][source], manifest["contract"], prepared["val_y"], latent["val"],
                                 prepared["train_y"].sum(axis=0).astype(int), device)
                summary["arms"].append(result)
                write_json(output / "summary.json", summary)
                write_report(output, summary)
                del model, loaders
                torch.cuda.empty_cache()
        summary["status"] = "complete"
        summary.pop("active_arm", None)
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error"] = str(exc)
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        manifest["status"] = summary["status"]
        write_json(output / "manifest.json", manifest)
        write_json(output / "summary.json", summary)
        write_report(output, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/backbone_finetune.json")
    parser.add_argument("--plan", type=Path, help="Save preparation only; never starts training")
    parser.add_argument("--execute", action="store_true", help="Explicitly start CUDA training after the resource gate")
    parser.add_argument("--output", type=Path, help="New training result directory; requires --execute")
    args = parser.parse_args(argv)
    if args.execute != (args.output is not None) or (args.execute and args.plan is not None):
        parser.error("Use --execute with --output, or preparation only with optional --plan")
    if args.plan is not None and args.plan.exists():
        raise FileExistsError("Refusing to overwrite a preparation record")
    prepared = prepare(args.config)
    if args.execute:
        return execute(prepared, args.output.resolve())
    if args.plan is not None:
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.plan, prepared["plan"])
    print(json.dumps({"status": "prepared_not_started", "arms": prepared["config"]["arms"],
                      "rows": {s: len(prepared[s + "_y"]) for s in ("train", "val")},
                      "labels": len(prepared["labels"]), "cuda_initialized": torch.cuda.is_initialized(),
                      "plan": str(args.plan) if args.plan else None}, indent=2))
    return prepared["plan"]


if __name__ == "__main__":
    main()

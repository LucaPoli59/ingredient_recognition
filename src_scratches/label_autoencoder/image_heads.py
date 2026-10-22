"""Image-only heads: direct ingredients versus an AE32 code and frozen decoder."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from src.data_processing.experimental_transforms import ExperimentalFitPad224
from src_scratches.label_autoencoder import compare
from src_scratches.label_autoencoder.image_features import atomic_json, load_backbone, load_features


ARMS = ("direct", "ae32_aligned", "ae32_mse")


def normalization(values):
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6)


class ImageHead(nn.Module):
    """All forward/inference inputs are image features; no ingredient input exists."""
    def __init__(self, kind, feature_mean, feature_std, label_count, decoder=None,
                 latent_mean=None, latent_std=None):
        super().__init__()
        if kind not in ARMS:
            raise ValueError(f"Unknown head {kind}")
        self.kind = kind
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))
        if kind == "direct":
            self.projection = nn.Linear(len(feature_mean), label_count)
        else:
            if decoder is None or latent_mean is None or latent_std is None:
                raise ValueError("AE head requires decoder and train-fitted latent normalization")
            self.register_buffer("latent_mean", torch.as_tensor(latent_mean, dtype=torch.float32))
            self.register_buffer("latent_std", torch.as_tensor(latent_std, dtype=torch.float32))
            self.projection = nn.Linear(len(feature_mean), len(latent_mean))
            self.decoder = copy.deepcopy(decoder).requires_grad_(False).eval()
            # Start near the train-average code; both AE objectives share this initialization.
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.projection.bias)

    def forward(self, features):
        projected = self.projection((features - self.feature_mean) / self.feature_std)
        if self.kind == "direct":
            return projected, None
        logits = self.decoder(projected * self.latent_std + self.latent_mean)
        return logits, projected


def teacher_codes(autoencoder, labels, batch_size=1024):
    """Used for training targets/diagnostics, never to construct image predictions."""
    autoencoder.eval()
    with torch.inference_mode():
        return np.concatenate([autoencoder.encoder(torch.from_numpy(labels[start:start + batch_size])).numpy()
                               for start in range(0, len(labels), batch_size)])


def predict(head, features, batch_size=1024):
    head.eval()
    scores, codes = [], []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            logits, code = head(torch.from_numpy(features[start:start + batch_size]))
            scores.append(logits.sigmoid().numpy())
            if code is not None:
                codes.append(code.numpy())
    return np.concatenate(scores), np.concatenate(codes) if codes else None


def validation_losses(head, features, labels, latent_targets, batch_size):
    head.eval()
    bce, mse = 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            target = torch.from_numpy(labels[start:start + batch_size])
            logits, code = head(torch.from_numpy(features[start:start + batch_size]))
            bce += nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="sum").item()
            if code is not None:
                latent = torch.from_numpy(latent_targets[start:start + batch_size])
                mse += nn.functional.mse_loss(code, latent, reduction="sum").item()
    return bce / labels.size, mse / latent_targets.size if head.kind != "direct" else None


def fit_head(head, train_x, train_y, train_z, val_x, val_y, val_z, args, output, contract):
    optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    x, y, z = map(torch.from_numpy, (train_x, train_y, train_z))
    initial_decoder = {k: v.clone() for k, v in head.decoder.state_dict().items()} if head.kind != "direct" else {}
    best, anchor, stale, best_epoch, best_state = math.inf, math.inf, 0, 0, None
    started = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_bce, total_mse, total_objective = 0.0, 0.0, 0.0
        for indices in torch.randperm(len(x), generator=generator).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits, code = head(x[indices])
            bce = nn.functional.binary_cross_entropy_with_logits(logits, y[indices])
            mse = nn.functional.mse_loss(code, z[indices]) if code is not None else torch.zeros(())
            objective = (mse if head.kind == "ae32_mse" else
                         bce + args.latent_weight * mse if head.kind == "ae32_aligned" else bce)
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite head objective")
            objective.backward()
            optimizer.step()
            total_bce += bce.item() * len(indices)
            total_mse += mse.item() * len(indices)
            total_objective += objective.item() * len(indices)
        val_bce, val_mse = validation_losses(head, val_x, val_y, val_z, args.batch_size)
        if not math.isfinite(val_bce) or (val_mse is not None and not math.isfinite(val_mse)):
            raise FloatingPointError("Nonfinite validation loss")
        history.append({"epoch": epoch, "train_bce": total_bce / len(x),
                        "train_normalized_latent_mse": total_mse / len(x) if head.kind != "direct" else None,
                        "train_objective": total_objective / len(x), "val_bce": val_bce,
                        "val_normalized_latent_mse": val_mse, "elapsed_seconds": time.perf_counter() - started})
        if val_bce < best:
            best, best_epoch = val_bce, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if val_bce < anchor - args.min_delta:
            anchor, stale = val_bce, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= args.patience:
            print(f"{head.kind} epoch={epoch}: val BCE={val_bce:.6f}, latent MSE={val_mse}, "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
        if stale >= args.patience:
            break
    elapsed = time.perf_counter() - started
    head.load_state_dict(best_state)
    if initial_decoder and any(not torch.equal(value, head.decoder.state_dict()[key])
                               for key, value in initial_decoder.items()):
        raise AssertionError("Frozen decoder changed")
    checkpoint = {"format_version": 1, "kind": head.kind, "state_dict": best_state,
                  "feature_dim": train_x.shape[1], "label_count": train_y.shape[1],
                  "latent_dim": train_z.shape[1], "decoder_hidden_dim": contract["decoder_hidden_dim"],
                  "labels": contract["labels"], "contract": contract,
                  "best_epoch": best_epoch, "best_val_bce": best}
    path = output / f"{head.kind}.pt"
    torch.save(checkpoint, path)
    compare.write_csv(output / f"{head.kind}_history.csv", history)
    restored, _ = load_head(path)
    scores, codes = predict(restored, val_x, args.batch_size)
    extra = {"fit_seconds": elapsed, "epochs_ran": epoch, "best_epoch": best_epoch,
             "best_val_bce": best, "stopped_early": epoch < args.epochs,
             "trainable_parameters": sum(p.numel() for p in head.parameters() if p.requires_grad),
             "frozen_parameters": sum(p.numel() for p in head.parameters() if not p.requires_grad),
             "checkpoint_sha256": compare.sha256(path), "optimizer_steps": epoch * math.ceil(len(x) / args.batch_size)}
    if codes is not None:
        squared_error = np.sum((codes.astype(np.float64) - val_z) ** 2)
        total_variation = np.sum((val_z.astype(np.float64) - val_z.mean(axis=0)) ** 2)
        extra["latent_diagnostics"] = {
            "normalized_mse": float(squared_error / val_z.size),
            "train_mean_code_normalized_mse": float(np.mean(val_z ** 2, dtype=np.float64)),
            "variance_weighted_r2": float(1 - squared_error / total_variation) if total_variation > 0 else None,
        }
    return scores, extra


def load_head(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state_dict"]
    decoder = None
    if saved["kind"] != "direct":
        decoder = nn.Sequential(nn.Linear(saved["latent_dim"], saved["decoder_hidden_dim"]), nn.ReLU(),
                                nn.Linear(saved["decoder_hidden_dim"], saved["label_count"]))
    model = ImageHead(saved["kind"], state["feature_mean"], state["feature_std"], saved["label_count"],
                      decoder, state.get("latent_mean"), state.get("latent_std"))
    model.load_state_dict(state, strict=True)
    return model.eval(), saved


def predict_image(image_path, head_path, weights_path):
    """A real deployment path accepting just an image and saved model artifacts."""
    head, saved = load_head(head_path)
    if compare.sha256(weights_path) != saved["contract"]["backbone_weights_sha256"]:
        raise ValueError("Backbone weights differ from head training")
    transform = ExperimentalFitPad224.from_config(saved["contract"]["transform"])
    backbone = load_backbone(weights_path)
    with Image.open(image_path) as image, torch.inference_mode():
        features = backbone(transform(image)[None].to(memory_format=torch.channels_last))
        logits, normalized_code = head(features)
        probabilities = logits.sigmoid().squeeze(0).numpy()
        code = ((normalized_code * head.latent_std + head.latent_mean).squeeze(0).numpy()
                if normalized_code is not None else None)
    return {"labels": saved["labels"], "probabilities": probabilities, "code": code}


def report(output, summary):
    lines = ["# Frozen-image-feature head pilot", "", f"Status: {summary['status']}", "",
             "All predictions use image features only. Validation selects checkpoints and thresholds;",
             "results are exploratory, from one seed, and are not final-test or end-to-end benchmark scores.", "",
             "| Head | Macro AP | Micro F1 @0.5 | Micro F1 tuned | Threshold | Rare AP | Code MSE |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm in summary["arms"]:
        m = arm["metrics"]
        mse = arm.get("latent_diagnostics", {}).get("normalized_mse")
        code = f"{mse:.4f}" if mse is not None else "—"
        def fmt(value):
            return "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {arm['method']} | {fmt(m['macro_ap'])} | {m['micro_f1_at_0_5']:.4f} | "
                     f"{m['micro_f1_val_tuned']:.4f} | {m['val_selected_threshold']:.2f} | "
                     f"{fmt(m['frequency_slices']['rare']['macro_ap'])} | {code} |")
    lines.extend(["", "Direct: image features -> 165 logits, ingredient BCE.",
                  "AE aligned: image features -> normalized 32-code -> frozen AE decoder; ingredient BCE + weighted code MSE.",
                  "AE MSE: same architecture/initialization, code MSE only. All arms select by validation ingredient BCE.",
                  "AE latent targets and both normalizations are fitted/computed from train only for optimization.",
                  "Validation ingredient targets are used for scoring, never as inference inputs.",
                  "The decoder, encoder weights and image backbone are never fine-tuned in this pilot.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=compare.DEFAULT_DATA)
    parser.add_argument("--metadata", default="ingredients_target_v5_metadata.json")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, default=Path(__file__).parent / "outputs/20261021_ingredients_cpu_seed42/ae_k32.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or len(set(args.arms)) != len(args.arms):
        parser.error("Positive counts and distinct arms are required")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must be in [0, 2**32)")
    for name in ("learning_rate", "min_delta", "latent_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.learning_rate == 0:
        parser.error("learning-rate must be positive")
    return args


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    train_y, val_y, labels, provenance = compare.load_labels(args.data_dir, args.metadata, "ingredients_target")
    train_x, val_x, cache = load_features(args.features, provenance)
    autoencoder, ae_saved = compare.load_autoencoder(args.autoencoder)
    if (ae_saved["latent_dim"] != 32 or ae_saved["labels"] != labels or
            ae_saved["train_sha256"] != provenance["train"]["sha256"]):
        raise ValueError("AE32 checkpoint vocabulary or train source does not match")
    autoencoder.requires_grad_(False).eval()
    feature_mean, feature_std = normalization(train_x)
    train_code = teacher_codes(autoencoder, train_y)
    latent_mean, latent_std = normalization(train_code)
    train_z = (train_code - latent_mean) / latent_std
    # Validation codes are diagnostic/loss targets only; predict() never consumes them.
    val_z = (teacher_codes(autoencoder, val_y) - latent_mean) / latent_std
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {"labels": labels, "decoder_hidden_dim": ae_saved["hidden_dim"],
                "train_sha256": provenance["train"]["sha256"], "val_sha256": provenance["val"]["sha256"],
                "autoencoder_sha256": compare.sha256(args.autoencoder),
                "feature_manifest_sha256": compare.sha256(args.features / "feature_manifest.json"),
                "backbone_weights_sha256": cache["identity"]["weights_sha256"],
                "transform": cache["identity"]["transform"]}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "contract": contract,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "sources": provenance, "scripts": {p.name: compare.sha256(p) for p in
                    (Path(__file__), Path(compare.__file__), Path(__file__).with_name("image_features.py"))},
                "device": "cpu", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "scheduler": os.sched_getscheduler(0) if hasattr(os, "sched_getscheduler") else None,
                "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                "threshold_grid": compare.THRESHOLDS.tolist(),
                "checkpoint_selection": "minimum validation ingredient BCE, shared across arms",
                "autoencoder_selected_epoch": ae_saved.get("best_epoch"),
                "pretraining_cost": "Additional ingredient-only pretraining; see the AE source run for full cost. Not an equal-total-compute comparison."}
    try:
        manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=compare.ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        manifest["git_head"] = None
    atomic_json(args.output / "manifest.json", manifest)
    summary = {"status": "running", "arms": []}
    support = train_y.sum(axis=0).astype(int)
    def record(method, scores, extra):
        metrics, per_label = compare.evaluate(val_y, scores, support, labels)
        summary["arms"].append({"method": method, "metrics": metrics, **extra})
        compare.write_csv(args.output / f"{method}_per_label.csv", per_label)
        atomic_json(args.output / "summary.json", summary)
        report(args.output, summary)
        print(f"{method}: macro AP={metrics['macro_ap']}, F1@.5={metrics['micro_f1_at_0_5']:.4f}, "
              f"F1 tuned={metrics['micro_f1_val_tuned']:.4f}", flush=True)
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=args.threads):
            record("prevalence", np.broadcast_to(train_y.mean(axis=0), val_y.shape), {})
            for kind in args.arms:
                torch.manual_seed(args.seed)
                head = ImageHead(kind, feature_mean, feature_std, len(labels), autoencoder.decoder, latent_mean, latent_std)
                if kind == "direct":
                    prevalence = np.clip(train_y.mean(axis=0), 1e-6, 1 - 1e-6)
                    with torch.no_grad():
                        head.projection.bias.copy_(torch.from_numpy(np.log(prevalence / (1 - prevalence))))
                scores, extra = fit_head(head, train_x, train_y, train_z, val_x, val_y, val_z, args, args.output, contract)
                record(kind, scores, extra)
        if torch.cuda.is_initialized():
            raise RuntimeError("Unexpected CUDA context in CPU head training")
        summary["status"] = "complete"
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error"] = str(exc)
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(args.output / "summary.json", summary)
        report(args.output, summary)
    return summary


if __name__ == "__main__":
    run(parse_args())

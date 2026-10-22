"""Ingredient-only PCA/AE reconstruction pilot; never loads images or test data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/input/yummly"
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_labels(data_dir: Path, metadata: str, field: str):
    """Fit alphabetical vocabulary on train only; retain every row, including empty sets."""
    if Path(metadata).name != metadata:
        raise ValueError("metadata must be a filename, not a path")
    targets, provenance = {}, {}
    for split in ("train", "val"):
        path = data_dir / split / metadata
        raw = path.read_bytes()
        records = json.loads(raw)
        if not isinstance(records, list) or not records:
            raise ValueError(f"{split}: expected a nonempty list of records")
        sets = []
        for i, record in enumerate(records):
            values = record.get(field) if isinstance(record, dict) else None
            if not isinstance(values, list) or any(
                not isinstance(label, str) or not label or label != label.strip()
                for label in values
            ):
                raise ValueError(f"{split} row {i}: malformed {field}")
            sets.append(set(values))
        targets[split] = sets
        provenance[split] = {"path": str(path.resolve()), "rows": len(sets),
                             "sha256": hashlib.sha256(raw).hexdigest()}
    labels = sorted(set().union(*targets["train"]))
    if not labels:
        raise ValueError("Training vocabulary is empty")
    index = {label: i for i, label in enumerate(labels)}
    matrices = {}
    for split, rows in targets.items():
        matrix = np.zeros((len(rows), len(labels)), dtype=np.float32)
        for i, row in enumerate(rows):
            unknown = row.difference(index)
            if unknown:
                raise ValueError(f"{split} row {i}: labels outside train vocabulary: {sorted(unknown)}")
            matrix[i, [index[label] for label in row]] = 1.0
        matrices[split] = matrix
    return matrices["train"], matrices["val"], labels, provenance


def gpu_snapshot():
    """Read driver telemetry without creating a CUDA context (WSL process lists may be empty)."""
    executable = shutil.which("nvidia-smi")
    if executable is None and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        return {"error": "nvidia-smi unavailable", "samples": []}
    samples = []
    try:
        for i in range(3):
            result = subprocess.run([
                executable, "--id=0",
                "--query-gpu=timestamp,name,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], check=True, capture_output=True, text=True, timeout=10)
            row = next(csv.reader(io.StringIO(result.stdout)))
            samples.append(dict(zip(
                ("timestamp", "name", "used_mib", "free_mib", "total_mib", "utilization_pct"),
                (row[0].strip(), row[1].strip(), *[float(x.strip()) for x in row[2:]]),
            )))
            if i < 2:
                time.sleep(1)
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as exc:
        return {"error": str(exc), "samples": samples}
    return {"samples": samples}


def resolve_device(requested: str, snapshot: dict) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    samples = snapshot.get("samples", [])
    if snapshot.get("error") or len(samples) != 3 or any(
        sample["free_mib"] < 2048 or sample["utilization_pct"] > 30 for sample in samples
    ):
        raise RuntimeError("GPU busy or telemetry unavailable: use --device cpu. "
                           "CUDA requires three samples with >=2048 MiB free and <=30% utilization.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    return torch.device("cuda:0")


class IngredientAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, input_dim))

    def forward(self, values):
        return self.decoder(self.encoder(values))


def predict_autoencoder(model, values, device, batch_size=1024):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size]).to(device)
            outputs.append(model(batch).sigmoid().cpu().numpy())
    return np.concatenate(outputs)


def validation_bce(model, values, device, batch_size):
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size]).to(device)
            total += nn.functional.binary_cross_entropy_with_logits(
                model(batch), batch, reduction="sum").item()
    return total / values.size


def train_autoencoder(train, val, latent_dim, args, device, output, labels, provenance):
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model = IngredientAutoencoder(train.shape[1], latent_dim, args.hidden_dim).to(device)
    # A train-only prior initialization avoids spending early epochs learning sparsity.
    prevalence = np.clip(train.mean(axis=0), 1e-6, 1 - 1e-6)
    with torch.no_grad():
        model.decoder[-1].bias.copy_(torch.from_numpy(np.log(prevalence / (1 - prevalence))).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    data = torch.from_numpy(train)  # CPU-resident, zero DataLoader workers.
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    patience_anchor = math.inf
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        order = torch.randperm(len(data), generator=generator)
        for indices in order.split(args.batch_size):
            batch = data[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(batch), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite AE loss at k={latent_dim}, epoch={epoch}")
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.numel()
        val_loss = validation_bce(model, val, device, args.batch_size)
        if not math.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation BCE")
        history.append({"epoch": epoch, "train_bce": total / train.size,
                        "val_bce": val_loss, "elapsed_seconds": time.perf_counter() - started})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if val_loss < patience_anchor - args.min_delta:
            patience_anchor, stale = val_loss, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= args.patience:
            print(f"AE k={latent_dim} epoch={epoch}: train BCE={total / train.size:.6f} "
                  f"val BCE={val_loss:.6f}, {time.perf_counter() - started:.1f}s", flush=True)
        if stale >= args.patience:
            break
    fit_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    checkpoint = {"format_version": 1, "state_dict": best_state, "labels": labels,
                  "input_dim": train.shape[1], "latent_dim": latent_dim, "hidden_dim": args.hidden_dim,
                  "seed": args.seed, "best_epoch": best_epoch, "best_val_bce": best_loss,
                  "train_sha256": provenance["train"]["sha256"], "loss": "unweighted_BCEWithLogits"}
    checkpoint_path = output / f"ae_k{latent_dim}.pt"
    torch.save(checkpoint, checkpoint_path)
    write_csv(output / f"ae_k{latent_dim}_history.csv", history)
    # Evaluate the reloaded artifact, not just the in-memory training instance.
    restored, _ = load_autoencoder(checkpoint_path, device)
    scores = predict_autoencoder(restored, val, device, args.batch_size)
    return scores, {"fit_seconds": fit_seconds, "epochs_ran": epoch, "best_epoch": best_epoch,
                    "best_val_bce": best_loss, "stopped_early": epoch < args.epochs,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "optimizer_steps": epoch * math.ceil(len(train) / args.batch_size)}


def load_autoencoder(path: Path, device=torch.device("cpu")):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = IngredientAutoencoder(checkpoint["input_dim"], checkpoint["latent_dim"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def pca_scores(values, mean, components):
    return (values - mean) @ components.T @ components + mean


def fit_pca(train, dimensions, output, labels):
    """One deterministic full SVD; nested leading components for all requested ranks."""
    started = time.perf_counter()
    pca = PCA(n_components=max(dimensions), svd_solver="full", whiten=False).fit(train)
    elapsed = time.perf_counter() - started
    for k in dimensions:
        np.savez_compressed(output / f"pca_k{k}.npz", mean=pca.mean_, components=pca.components_[:k],
                            explained_variance=pca.explained_variance_[:k],
                            explained_variance_ratio=pca.explained_variance_ratio_[:k],
                            labels=np.asarray(labels))
    return elapsed


def counts(targets, scores, threshold):
    positive = targets.astype(bool)
    predicted = scores >= threshold
    return (np.sum(predicted & positive, axis=0), np.sum(predicted & ~positive, axis=0),
            np.sum(~predicted & positive, axis=0))


def micro_f1(tp, fp, fn):
    denominator = 2 * np.sum(tp) + np.sum(fp) + np.sum(fn)
    return float(2 * np.sum(tp) / denominator) if denominator else 0.0


def select_threshold(targets, scores):
    # All methods receive the same bounded global grid. Smallest threshold wins ties.
    values = [micro_f1(*counts(targets, scores, threshold)) for threshold in THRESHOLDS]
    best = int(np.argmax(values))
    return float(THRESHOLDS[best]), values[best]


def frequency_slices(support):
    # Equal-count terciles, train support then alphabetic index for ties.
    order = np.argsort(support, kind="stable")
    return dict(zip(("rare", "middle", "frequent"), np.array_split(order, 3)))


def evaluate(targets, scores, train_support, labels):
    if scores.shape != targets.shape or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and match target shape")
    support = targets.sum(axis=0).astype(int)
    present = support > 0
    ap = np.full(len(labels), np.nan)
    if present.any():
        ap[present] = np.asarray(average_precision_score(targets[:, present], scores[:, present], average=None))
    threshold, tuned_f1 = select_threshold(targets, scores)
    tp, fp, fn = counts(targets, scores, threshold)
    recalls = np.divide(tp, tp + fn, out=np.zeros(len(labels), dtype=float), where=(tp + fn) > 0)
    slices = frequency_slices(train_support)
    result = {"macro_ap": float(np.mean(ap[present])) if present.any() else None,
              "micro_ap": float(average_precision_score(targets.ravel(), scores.ravel())) if present.any() else None,
              "micro_f1_at_0_5": micro_f1(*counts(targets, scores, 0.5)),
              "micro_f1_val_tuned": tuned_f1, "val_selected_threshold": threshold,
              "mse_raw": float(np.mean((targets - scores) ** 2, dtype=np.float64)),
              "mse_clipped": float(np.mean((targets - np.clip(scores, 0, 1)) ** 2, dtype=np.float64)),
              "score_fraction_outside_0_1": float(np.mean((scores < 0) | (scores > 1))),
              "labels_with_val_positives": int(present.sum()), "frequency_slices": {}}
    for name, indices in slices.items():
        valid = indices[present[indices]]
        result["frequency_slices"][name] = {
            "labels": len(indices), "train_support_min": int(train_support[indices].min()) if len(indices) else None,
            "train_support_max": int(train_support[indices].max()) if len(indices) else None,
            "macro_ap": float(np.mean(ap[valid])) if len(valid) else None,
            "macro_recall_val_tuned": float(np.mean(recalls[valid])) if len(valid) else None,
        }
    memberships = {int(index): name for name, indices in slices.items() for index in indices}
    per_label = [{"label": label, "train_support": int(train_support[i]), "val_support": int(support[i]),
                  "frequency_slice": memberships[i], "ap": float(ap[i]) if present[i] else None,
                  "recall_val_tuned": float(recalls[i]) if present[i] else None,
                  "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i])} for i, label in enumerate(labels)]
    return result, per_label


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, summary):
    lines = ["# Ingredient-only reconstruction pilot", "", f"Status: {summary['status']}", "",
             "Validation diagnostics with true ingredient sets as input; not image prediction.",
             "AE checkpoints and global thresholds were selected on this same validation set;",
             "tuned F1 is selection-biased, not an independent test estimate. One pilot seed.", "",
             "| Method | k | Macro AP | Micro F1 @0.5 | Micro F1 tuned | Threshold | Rare AP | MSE clipped |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm in summary["arms"]:
        m = arm["metrics"]
        def fmt(value):
            return "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {arm['method']} | {arm['latent_dim'] or '-'} | {fmt(m['macro_ap'])} | "
                     f"{m['micro_f1_at_0_5']:.4f} | {m['micro_f1_val_tuned']:.4f} | "
                     f"{m['val_selected_threshold']:.2f} | {fmt(m['frequency_slices']['rare']['macro_ap'])} | "
                     f"{m['mse_clipped']:.6f} |")
    lines.extend(["", "PCA AP uses raw inverse-transform scores; clipping is used only for clipped MSE.",
                  "PCA uses one shared full SVD; its wall time is not charged independently three times.",
                  "PCA minimizes squared reconstruction error; the nonlinear AE minimizes unweighted BCE.",
                  "This compares practical method choices, not nonlinearity alone or equal training budgets.",
                  "Train-frequency terciles use train support only. Missing validation positives are excluded",
                  "from macro AP/recall and recorded as null per-label. Test files are never opened.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metadata", default="ingredients_target_v5_metadata.json")
    parser.add_argument("--field", default="ingredients_target")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True, help="New directory; refuses existing output")
    args = parser.parse_args(argv)
    for name in ("hidden_dim", "epochs", "patience", "batch_size", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not args.dimensions or min(args.dimensions) < 1 or len(set(args.dimensions)) != len(args.dimensions):
        parser.error("dimensions must be distinct positive integers")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("min-delta must be finite and nonnegative")
    if args.seed < 0 or args.seed >= 2 ** 32:
        parser.error("seed must be in [0, 2**32)")
    return args


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    train, val, labels, provenance = load_labels(args.data_dir, args.metadata, args.field)
    if max(args.dimensions) >= min(train.shape):
        raise ValueError("Latent dimensions must be smaller than both train rows and vocabulary size")
    snapshot = gpu_snapshot()
    device = resolve_device(args.device, snapshot)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        git_head = None
    write_json(args.output / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_head": git_head,
        "script_sha256": sha256(Path(__file__)), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "torch": str(torch.__version__)},
        "device": str(device), "threads": args.threads, "gpu_before_run": snapshot,
        "data": provenance, "labels": labels, "threshold_grid": THRESHOLDS.tolist(),
        "scope": "train fit; validation checkpoint/threshold selection and reconstruction diagnostics; no test/images/metadata features",
    })
    summary = {"status": "running", "arms": [], "pca_shared_fit_seconds": None}
    write_json(args.output / "summary.json", summary)
    support = train.sum(axis=0).astype(int)

    def record(method, k, scores, extra):
        metrics, per_label = evaluate(val, scores, support, labels)
        summary["arms"].append({"method": method, "latent_dim": k, "metrics": metrics, **extra})
        write_csv(args.output / f"{method}_k{k or 0}_per_label.csv", per_label)
        write_json(args.output / "summary.json", summary)
        write_report(args.output, summary)
        print(f"{method} k={k}: macro AP={metrics['macro_ap']}, "
              f"micro F1@.5={metrics['micro_f1_at_0_5']:.4f}, "
              f"val-tuned F1={metrics['micro_f1_val_tuned']:.4f}", flush=True)

    started = time.perf_counter()
    try:
        with threadpool_limits(limits=args.threads):
            record("prevalence", None, np.broadcast_to(train.mean(axis=0), val.shape), {})
            summary["pca_shared_fit_seconds"] = fit_pca(train, args.dimensions, args.output, labels)
            for k in args.dimensions:
                with np.load(args.output / f"pca_k{k}.npz", allow_pickle=False) as saved:
                    scores = pca_scores(val, saved["mean"], saved["components"])
                    variance = float(saved["explained_variance_ratio"].sum())
                record("pca", k, scores, {"train_explained_variance_ratio": variance})
            for k in args.dimensions:
                scores, training = train_autoencoder(train, val, k, args, device, args.output, labels, provenance)
                record("ae", k, scores, training)
        summary["status"] = "complete"
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error"] = str(exc)
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "summary.json", summary)
        write_report(args.output, summary)
    return summary


if __name__ == "__main__":
    run(parse_args())

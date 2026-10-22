"""Post-hoc validation threshold check for completed image heads; no training."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from threadpoolctl import threadpool_limits
import torch

from src_scratches.label_autoencoder import compare, image_heads, image_features


def exact_micro_f1(targets, scores):
    precision, recall, thresholds = precision_recall_curve(targets.ravel(), scores.ravel())
    denominator = precision[:-1] + recall[:-1]
    values = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
    # Thresholds ascend; the first maximum retains the smallest threshold on ties.
    best = int(np.argmax(values))
    threshold = float(thresholds[best])
    checked = float(f1_score(targets, scores >= threshold, average="micro", zero_division=0))
    if not np.isclose(values[best], checked, atol=1e-12):
        raise AssertionError("Threshold sweep and direct F1 disagree")
    return {"threshold": threshold, "micro_f1": checked, "candidate_thresholds": len(thresholds)}


def run(run_dir):
    output = run_dir / "threshold_diagnostic.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    original = json.loads((run_dir / "summary.json").read_text())
    if original["status"] != "complete":
        raise ValueError("A completed run is required")
    args = manifest["arguments"]
    train_y, val_y, labels, provenance = compare.load_labels(Path(args["data_dir"]), args["metadata"], "ingredients_target")
    if labels != manifest["contract"]["labels"] or any(
        provenance[split]["sha256"] != manifest["contract"][split + "_sha256"] for split in ("train", "val")
    ):
        raise ValueError("Diagnostic data differs from completed run")
    cache = Path(args["features"])
    if not cache.is_absolute():
        cache = compare.ROOT / cache
    _, val_x, _ = image_features.load_features(cache, provenance)
    torch.set_num_threads(1)
    report = {"scope": "Post-hoc validation-only diagnostic; prompted by MSE-head grid optimum at lower boundary 0.05. No training or test evaluation.",
              "script_sha256": compare.sha256(Path(__file__)),
              "run_manifest_sha256": compare.sha256(run_dir / "manifest.json"), "arms": []}
    with threadpool_limits(limits=1):
        for arm in original["arms"]:
            name = arm["method"]
            if name == "prevalence":
                scores = np.broadcast_to(train_y.mean(axis=0), val_y.shape)
            elif name in image_heads.ARMS:
                path = run_dir / f"{name}.pt"
                if compare.sha256(path) != arm["checkpoint_sha256"]:
                    raise ValueError("Checkpoint changed since original evaluation")
                model, _ = image_heads.load_head(path)
                scores, _ = image_heads.predict(model, val_x)
            else:
                raise ValueError(f"Unknown arm {name}")
            result = {"method": name, "original_grid_f1": arm["metrics"]["micro_f1_val_tuned"],
                      "original_grid_threshold": arm["metrics"]["val_selected_threshold"],
                      **exact_micro_f1(val_y, scores)}
            if result["micro_f1"] + 1e-12 < result["original_grid_f1"]:
                raise AssertionError("Exact sweep cannot underperform an eligible grid threshold")
            report["arms"].append(result)
            print(result, flush=True)
    compare.write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run(parser.parse_args().run_dir)

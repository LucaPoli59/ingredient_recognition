"""Bounded, read-only feasibility probe; not the proposed comparison CLI."""

import ast
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "inventory.json"
NAMES = ("resnets_htuning", "dinov2_htuning_v1")


def decode(value):
    """Decode data only; never import or execute serialized classes/functions."""
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        kind, item = value
        if kind == "config":
            return decode(item)
        if kind == "None":
            return None
        if kind in ("list", "tuple") and isinstance(item, str):
            return ast.literal_eval(item)
        return item
    return value


def config(path):
    return decode(json.loads(path.read_text()))


def rel(path):
    return str(path.relative_to(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    result = {}
    for name in NAMES:
        exp = ROOT / "experiments/basic_v5" / name
        trials = []
        for trial in sorted(exp.glob("trial_*")):
            if not re.fullmatch(r"trial_\d+", trial.name):
                continue
            cfg = config(trial / "trial_config.json")
            hp = cfg["hyper_parameters"]
            with (trial / "metrics.csv").open() as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                rows = list(reader)
            series = {key: [(float(r["epoch"]) if r.get("epoch") else None,
                             float(r["step"]) if r.get("step") else None, float(r[key]))
                            for r in rows if r.get(key)] for key in columns}
            events = sorted(trial.glob("events.out.tfevents.*"))
            tags = defaultdict(set)
            event_validation = []
            for event in events:
                accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0}).Reload()
                values = accumulator.Scalars("val_loss") if "val_loss" in accumulator.Tags()["scalars"] else []
                event_validation.append({"path": rel(event), "points": [[v.step, v.value] for v in values]})
                for kind, values in accumulator.Tags().items():
                    if isinstance(values, list):
                        tags[kind].update(values)
            checkpoints = sorted(trial.rglob("*.ckpt"))
            trials.append({
                "number": int(trial.name.split("_")[1]),
                "model": hp["torch_model"],
                "hyperparameters": {k: hp.get(k) for k in ("lr", "weighted_loss", "optimizer", "weight_decay", "lr_scheduler", "batch_size")},
                "data": {k: cfg["datamodule_hyper_parameters"].get(k) for k in ("metadata_filename", "feature_label", "category")},
                "config_sha256": digest(trial / "trial_config.json"),
                "metrics_sha256": digest(trial / "metrics.csv"),
                "csv_rows": len(rows), "columns": columns,
                "metric_points": {k: len(v) for k, v in series.items()},
                "epoch_range": [min(x[2] for x in series["epoch"]), max(x[2] for x in series["epoch"])],
                "best_val_loss": min(series["val_loss"], key=lambda x: x[2]),
                "best_val_loss_checkpoint_cadence": min((x for x in series["val_loss"] if int(x[0]) % 2 == 1), key=lambda x: x[2]),
                "last_val_loss": series["val_loss"][-1],
                "tensorboard_validation": event_validation,
                "tensorboard_files": len(events), "tensorboard_tags": {k: sorted(v) for k, v in tags.items()},
                "checkpoints": [{"path": rel(p), "bytes": p.stat().st_size} for p in checkpoints],
            })
        best_cfg = config(exp / "trial_best/trial_config.json")
        best_number = int(best_cfg["trainer_hyper_parameters"]["save_dir"].split("trial_")[-1])
        runs = sorted((ROOT / "experiments/wandb").rglob(f"run-basic_v5-{name}-trial_*.wandb"))
        counts = Counter(int(p.stem.split("trial_")[-1]) for p in runs)
        result[name] = {
            "trials": trials, "numbered_trials": len(trials),
            "models": dict(Counter(t["model"]["type"] for t in trials)),
            "weighted_loss": dict(Counter(str(t["hyperparameters"]["weighted_loss"]) for t in trials)),
            "trial_best_alias": best_number,
            "trial_best_csv_equal": digest(exp / "trial_best/metrics.csv") == digest(exp / f"trial_{best_number}/metrics.csv"),
            "wandb_files": len(runs), "wandb_bytes": sum(p.stat().st_size for p in runs),
            "wandb_missing_trials": sorted(set(range(100)) - counts.keys()),
            "wandb_repeated_trials": {k: v for k, v in counts.items() if v > 1},
        }
        print(name, {k: v for k, v in result[name].items() if k != "trials"}, flush=True)
    return result


def inspect_journal():
    import optuna

    source = ROOT / "experiments/journal.log"
    result = {"path": rel(source), "sha256": digest(source), "studies": []}
    # Optuna receives a disposable copy, so even lock/recovery behavior cannot touch the source.
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "journal.log"
        shutil.copyfile(source, target)
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(target)))
        for study_name in optuna.get_all_study_names(storage):
            if not any(study_name.replace("\\", "/").endswith("basic_v5/" + name) for name in NAMES):
                continue
            study = optuna.load_study(study_name=study_name, storage=storage)
            result["studies"].append({
                "name_suffix": study_name.replace("\\", "/").split("experiments/")[-1],
                "states": dict(Counter(t.state.name for t in study.trials)),
                "best_number": study.best_trial.number, "best_value": study.best_value,
                "trials": [{"number": t.number, "state": t.state.name, "value": t.value,
                            "params": t.params, "intermediate_values": t.intermediate_values,
                            "distributions": {k: optuna.distributions.distribution_to_json(v) for k, v in t.distributions.items()},
                            "duration_seconds": t.duration.total_seconds() if t.duration else None}
                           for t in study.trials],
            })
            print("journal", result["studies"][-1]["name_suffix"], result["studies"][-1]["states"], "best", study.best_trial.number, study.best_value, flush=True)
    return result


def inspect_wandb(path):
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2 as pb

    ds = DataStore()
    # Equivalent scan initialization to SDK 0.28.0, but with a strictly read-only descriptor.
    ds._fname = str(path)
    ds._fp = path.open("rb")
    ds._index = 0
    ds._size_bytes = path.stat().st_size
    ds._opened_for_scan = True
    ds._read_header()
    kinds, keys, bins_count = Counter(), Counter(), Counter()
    series = defaultdict(list)
    clock_samples, sdk_versions = [], []
    try:
        while (raw := ds.scan_data()) is not None:
            record = pb.Record()
            record.ParseFromString(raw)
            kind = record.WhichOneof("record_type")
            kinds[kind] += 1
            if kind == "telemetry" and record.telemetry.cli_version:
                sdk_versions.append(record.telemetry.cli_version)
            if kind != "history":
                continue
            row, hist = {}, defaultdict(dict)
            for item in record.history.item:
                parts = list(item.nested_key) or [item.key]
                value = json.loads(item.value_json)
                if len(parts) == 2 and parts[0].startswith(("gradients/", "parameters/")):
                    hist[parts[0]][parts[1]] = value
                elif len(parts) == 1:
                    row[parts[0]] = value
                    if isinstance(value, dict) and value.get("_type") == "histogram":
                        hist[parts[0]] = value
            keys.update(row)
            if not hist:
                continue
            clocks = {k: row.get(k) for k in ("epoch", "trainer/global_step", "_step", "_timestamp", "_runtime")}
            if len(clock_samples) < 4:
                clock_samples.append(clocks)
            for key, value in hist.items():
                if value.get("_type") != "histogram":
                    continue
                bins, counts = value["bins"], value["values"]
                assert len(bins) == len(counts) + 1
                n = sum(counts)
                assert n > 0 and all(c >= 0 for c in counts)
                centers = [(a + b) / 2 for a, b in zip(bins, bins[1:])]
                mean = sum(c * x for c, x in zip(counts, centers)) / n
                rms = (sum(c * x * x for c, x in zip(counts, centers)) / n) ** 0.5
                bins_count[len(counts)] += 1
                series[key].append({**clocks, "count": n, "bin_min": bins[0], "bin_max": bins[-1],
                                    "mean_midpoint_estimate": mean, "rms_midpoint_estimate": rms})
    finally:
        ds._fp.close()
    result = {"path": rel(path), "bytes": path.stat().st_size, "record_counts": dict(kinds),
              "producer_wandb_versions": sorted(set(sdk_versions)),
              "scalar_keys": sorted(keys), "histogram_bins": dict(bins_count),
              "clock_samples": clock_samples,
              "series": {k: {"samples": len(v), "first": v[0], "last": v[-1],
                             "epochs": sorted({p["epoch"] for p in v if p["epoch"] is not None})} for k, v in series.items()}}
    print("wandb sample", path.name, "records", dict(kinds), "series", len(series), "bins", dict(bins_count), flush=True)
    return result


def inspect_checkpoint(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    result = {"path": rel(path), "keys": sorted(checkpoint), "epoch": checkpoint.get("epoch"),
              "global_step": checkpoint.get("global_step"),
              "state_dict_tensors": len(checkpoint["state_dict"]),
              "state_dict_elements": sum(t.numel() for t in checkpoint["state_dict"].values()),
              "amp_scaler": checkpoint.get("MixedPrecision", checkpoint.get("native_amp_scaling_state")),
              "lightning_version": checkpoint.get("pytorch-lightning_version")}
    print("checkpoint", result, flush=True)
    return result


if __name__ == "__main__":
    report = {"audit_date": "2026-09-14", "scope": "All 200 numbered basic_v5 CSV/config/TensorBoard inventories; journal on a disposable copy; one full W&B session and one selected checkpoint per experiment.",
              "versions": {name: importlib.metadata.version(name) for name in ("wandb", "tensorboard", "optuna", "torch", "lightning")}}
    report["experiments"] = inventory()
    report["journal"] = inspect_journal()
    report["wandb_samples"] = []
    report["checkpoint_samples"] = []
    for name in NAMES:
        matches = list((ROOT / "experiments/wandb").rglob(f"run-basic_v5-{name}-trial_72.wandb"))
        assert len(matches) == 1, "Select the intended trial_72 session explicitly."
        report["wandb_samples"].append(inspect_wandb(matches[0]))
        report["checkpoint_samples"].append(inspect_checkpoint(ROOT / "experiments/basic_v5" / name / "trial_best/best_model.ckpt"))
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Saved", rel(OUT), flush=True)

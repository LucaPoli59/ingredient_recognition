"""Command-line orchestration for local experiment comparison."""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

from .analysis.aggregation import aggregate_experiment, compare_experiments
from .analysis.curves import summarize_curve
from .analysis.distributions import aggregate_distributions
from .analysis.hyperparameters import analyze_hyperparameters
from .analysis.ingredients import analyze_ingredients, logging_status
from .comparability import trial_signature
from .discovery import ExperimentFiles, TrialFiles, discover_experiment
from .normalization import flatten_config, label_contract, select_metric_series
from .readers.checkpoints import inspect_checkpoint
from .readers.config import read_config
from .readers.csv_metrics import parse_ingredient_key, read_csv_metrics
from .readers.optuna import match_study, read_journal
from .readers.tensorboard import read_tensorboard_scalars
from .readers.wandb_local import read_trials as read_wandb_trials
from .reporting.html_report import write_html_report
from .reporting.json_report import write_json_report
from .schema import SCHEMA_VERSION, json_safe, utc_now


def _trial_hparams(config: dict[str, Any], optuna_trial: dict[str, Any] | None) -> dict[str, Any]:
    if optuna_trial and optuna_trial.get("params"):
        return optuna_trial["params"]
    flattened = flatten_config(config.get("hyper_parameters", {}))
    return {
        key: value for key, value in flattened.items()
        if not key.startswith("metrics.") and key not in {"num_classes"}
    }


def _read_trial(files: TrialFiles, shared_config: dict[str, Any], metric: str,
                direction: str, target: float | None,
                optuna_trial: dict[str, Any] | None,
                include_tensorboard: bool) -> dict[str, Any]:
    config = read_config(files.config) if files.config else shared_config
    csv_data = read_csv_metrics(files.csv)
    tensorboard = read_tensorboard_scalars(files.tensorboard) if include_tensorboard else {}
    selected = select_metric_series(csv_data, tensorboard, metric)
    curve = summarize_curve(selected["points"], direction, target=target)
    ingredient_series = dict(csv_data["ingredient_series"])
    for key, observations in tensorboard.items():
        parsed = parse_ingredient_key(key)
        if parsed is None:
            continue
        split, ingredient_metric, label_index, legacy = parsed
        canonical = f"{split}/{ingredient_metric}/{label_index}"
        if canonical not in ingredient_series:
            ingredient_series[canonical] = [
                observation | {"epoch": None, "legacy_key": legacy}
                for observation in observations
            ]
    state = optuna_trial.get("state") if optuna_trial else None
    optuna_value = optuna_trial.get("value") if optuna_trial else None
    if optuna_value is not None:
        objective, objective_source = optuna_value, "optuna_selected_checkpoint_objective"
    elif curve.get("status") == "available":
        objective, objective_source = curve["best"]["value"], "observed_curve_best_fallback"
    else:
        objective, objective_source = None, None
    return {
        "number": files.number,
        "path": str(files.path),
        "state": state,
        "objective": objective,
        "objective_source": objective_source,
        "optuna": optuna_trial,
        "hyperparameters": _trial_hparams(config, optuna_trial),
        "comparability_signature": trial_signature(config),
        "label_contract": label_contract(config),
        "ingredient_logging_status": logging_status(config, ingredient_series),
        "ingredient_series": ingredient_series,
        "curve": curve,
        "metric_observations": selected,
        "artifact_coverage": {
            "config": str(files.config) if files.config else None,
            "csv": str(files.csv) if files.csv else None,
            "csv_rows": csv_data["rows"],
            "tensorboard_files": [str(path) for path in files.tensorboard],
            "checkpoints": [str(path) for path in files.checkpoints],
        },
    }


def _experiment_labels(trials: list[dict[str, Any]]) -> tuple[list[str] | None, str]:
    contracts = [trial["label_contract"] for trial in trials]
    available = [contract for contract in contracts if contract["classes"] is not None]
    if not available:
        return None, "unavailable"
    hashes = {contract["sha256"] for contract in available}
    if len(hashes) != 1 or len(available) != len(contracts):
        return None, "inconsistent_or_partial"
    return available[0]["classes"], "available"


def _build_experiment(files: ExperimentFiles, metric: str, direction: str,
                      target: float | None, studies: list[dict[str, Any]],
                      include_tensorboard: bool,
                      wandb_root: Path | None, wandb_scope: str,
                      include_gradients: bool, preserve_raw_histograms: bool,
                      checkpoint_metadata: bool) -> dict[str, Any]:
    shared_config = read_config(files.shared_config)
    study = match_study(studies, files.path)
    optuna_by_number = {
        trial["number"]: trial for trial in study.get("trials", [])
    } if study else {}
    trials = [
        _read_trial(
            trial_files,
            shared_config,
            metric,
            direction,
            target,
            optuna_by_number.get(trial_files.number),
            include_tensorboard,
        )
        for trial_files in files.trials
    ]
    summary = aggregate_experiment(trials, direction)
    labels, labels_status = _experiment_labels(trials)
    ingredients = analyze_ingredients(trials, labels)
    ingredients["label_contract_status"] = labels_status

    if wandb_scope == "all":
        selected_numbers = [trial.number for trial in files.trials]
    elif wandb_scope == "best":
        selected = summary["study_selected_trial"]
        best_number = selected["number"] if selected else files.best_alias_trial
        selected_numbers = [best_number] if best_number is not None else []
    else:
        selected_numbers = []
    wandb = read_wandb_trials(
        wandb_root,
        files.path,
        selected_numbers,
        include_gradients=include_gradients,
        preserve_raw_histograms=preserve_raw_histograms,
    ) if selected_numbers else {
        "status": "not_requested",
        "reason": "wandb scope is none or no best trial was resolvable",
        "trials": {},
    }
    parameter_analysis = aggregate_distributions(wandb)

    checkpoint = {"status": "not_requested"}
    if checkpoint_metadata:
        checkpoint_path = files.path / "trial_best" / "best_model.ckpt"
        checkpoint = inspect_checkpoint(checkpoint_path if checkpoint_path.is_file() else None)

    return {
        "name": files.name,
        "path": str(files.path),
        "metric": metric,
        "direction": direction,
        "study": ({
            "name": study["name"],
            "direction": study["direction"],
            "states": study["states"],
            "best_number": study["best_number"],
            "best_value": study["best_value"],
            "source": study["source"],
        } if study else {"status": "unavailable", "reason": "matching Optuna study not found"}),
        "trial_best_alias": files.best_alias_trial,
        "trials": trials,
        "summary": summary,
        "hyperparameter_analysis": analyze_hyperparameters(trials),
        "ingredient_analysis": ingredients,
        "wandb": wandb,
        "parameter_analysis": parameter_analysis,
        "selected_checkpoint": checkpoint,
        "coverage": {
            "numbered_trials": len(trials),
            "trials_with_metric": sum(trial["curve"]["status"] == "available" for trial in trials),
            "trials_with_config": sum(trial["artifact_coverage"]["config"] is not None for trial in trials),
            "trials_with_csv": sum(trial["artifact_coverage"]["csv"] is not None for trial in trials),
            "ingredient_logging_statuses": {
                status: sum(trial["ingredient_logging_status"] == status for trial in trials)
                for status in sorted({trial["ingredient_logging_status"] for trial in trials})
            },
        },
    }


def build_report(experiment_paths: Sequence[str | Path], metric: str = "val_loss",
                 direction: str = "min", target: float | None = None,
                 wandb_root: str | Path | None = None,
                 optuna_journal: str | Path | None = None,
                 wandb_scope: str = "best", include_tensorboard: bool = True,
                 include_gradients: bool = False,
                 preserve_raw_histograms: bool = False,
                 checkpoint_metadata: bool = False) -> dict[str, Any]:
    if len(experiment_paths) < 1:
        raise ValueError("At least one experiment directory is required")
    if direction not in {"min", "max"}:
        raise ValueError("direction must be 'min' or 'max'")
    discovered = [discover_experiment(path) for path in experiment_paths]
    journal_path = Path(optuna_journal).expanduser().resolve() if optuna_journal else None
    studies = read_journal(journal_path)
    resolved_wandb = Path(wandb_root).expanduser().resolve() if wandb_root else None
    experiments = [
        _build_experiment(
            files,
            metric,
            direction,
            target,
            studies,
            include_tensorboard,
            resolved_wandb,
            wandb_scope,
            include_gradients,
            preserve_raw_histograms,
            checkpoint_metadata,
        )
        for files in discovered
    ]
    return json_safe({
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "inputs": {
            "experiments": [str(files.path) for files in discovered],
            "wandb_root": str(resolved_wandb) if resolved_wandb else None,
            "optuna_journal": str(journal_path) if journal_path else None,
        },
        "settings": {
            "metric": metric,
            "direction": direction,
            "target": target,
            "wandb_scope": wandb_scope,
            "include_tensorboard": include_tensorboard,
            "include_gradients": include_gradients,
            "preserve_raw_histograms": preserve_raw_histograms,
            "checkpoint_metadata": checkpoint_metadata,
        },
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "reader_policy": "Read-only artifacts; Optuna journal is opened only through a disposable copy.",
        },
        "experiments": experiments,
        "intra_experiment": {
            experiment["name"]: {
                "summary": experiment["summary"],
                "hyperparameters": experiment["hyperparameter_analysis"],
            }
            for experiment in experiments
        },
        "inter_experiment": compare_experiments(experiments),
        "limitations": [
            "Historical aggregate precision and recall retain their original batch-derived logging semantics.",
            "Trials from adaptive HPO are conditional observations, not independent causal replicates.",
            "Pruned trials are retained for coverage but excluded from best-trial and cohort objective summaries.",
            "Parameter statistics are midpoint estimates from W&B histogram bins; coordinate-wise values and exact updates are unavailable.",
            "Gradient histograms, when requested, may contain AMP-scaled and pre-clipping microbatch gradients.",
            "Different loss weighting policies are analyzed as separate objective cohorts.",
        ],
    })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Compare N local training experiments.")
    result.add_argument("--experiments", nargs="+", required=True, help="Experiment directories containing trial_<n>.")
    result.add_argument("--output", required=True, help="Directory for comparison.json and comparison.html.")
    result.add_argument("--metric", default="val_loss")
    result.add_argument("--direction", choices=("min", "max"), default="min")
    result.add_argument("--target", type=float, help="Optional target for reached/censored crossing time.")
    result.add_argument("--wandb-root", help="Root containing local run-*.wandb files.")
    result.add_argument("--optuna-journal", help="Optuna journal; read through a temporary copy.")
    result.add_argument("--wandb-scope", choices=("none", "best", "all"), default="best")
    result.add_argument("--no-tensorboard", action="store_true")
    result.add_argument("--include-gradients", action="store_true")
    result.add_argument("--preserve-raw-histograms", action="store_true")
    result.add_argument("--checkpoint-metadata", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = build_report(
        args.experiments,
        metric=args.metric,
        direction=args.direction,
        target=args.target,
        wandb_root=args.wandb_root,
        optuna_journal=args.optuna_journal,
        wandb_scope=args.wandb_scope,
        include_tensorboard=not args.no_tensorboard,
        include_gradients=args.include_gradients,
        preserve_raw_histograms=args.preserve_raw_histograms,
        checkpoint_metadata=args.checkpoint_metadata,
    )
    output = Path(args.output).expanduser().resolve()
    json_path, html_path = output / "comparison.json", output / "comparison.html"
    write_json_report(report, json_path)
    write_html_report(report, html_path)
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
    return 0

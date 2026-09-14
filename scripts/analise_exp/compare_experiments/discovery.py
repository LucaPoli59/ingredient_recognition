"""Discover numbered trials and aliases inside experiment directories."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

TRIAL_RE = re.compile(r"trial_(\d+)$")


@dataclass(frozen=True)
class TrialFiles:
    number: int
    path: Path
    config: Path | None
    csv: Path | None
    tensorboard: tuple[Path, ...]
    checkpoints: tuple[Path, ...]


@dataclass(frozen=True)
class ExperimentFiles:
    name: str
    path: Path
    trials: tuple[TrialFiles, ...]
    best_alias_trial: int | None
    shared_config: Path | None


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_experiment(path: str | Path) -> ExperimentFiles:
    experiment = Path(path).expanduser().resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment}")

    trials = []
    for directory in experiment.iterdir():
        match = TRIAL_RE.fullmatch(directory.name)
        if not directory.is_dir() or match is None:
            continue
        events = tuple(sorted(directory.glob("events.out.tfevents.*")))
        if not events:
            events = tuple(sorted(directory.rglob("events.out.tfevents.*")))
        trials.append(TrialFiles(
            number=int(match.group(1)),
            path=directory,
            config=_first_existing(directory, ("trial_config.json", "config.json")),
            csv=_first_existing(directory, ("metrics.csv",)),
            tensorboard=events,
            checkpoints=tuple(sorted(directory.rglob("*.ckpt"))),
        ))
    trials.sort(key=lambda item: item.number)
    if not trials:
        raise ValueError(f"No numbered trial_<n> directories found in {experiment}")

    alias_trial = None
    alias = experiment / "trial_best"
    alias_csv = _first_existing(alias, ("metrics.csv",)) if alias.is_dir() else None
    if alias_csv:
        alias_hash = _sha256(alias_csv)
        matches = [trial.number for trial in trials if trial.csv and _sha256(trial.csv) == alias_hash]
        if len(matches) == 1:
            alias_trial = matches[0]

    return ExperimentFiles(
        name=experiment.name,
        path=experiment,
        trials=tuple(trials),
        best_alias_trial=alias_trial,
        shared_config=_first_existing(experiment, ("hparam_config.json", "config.json")),
    )

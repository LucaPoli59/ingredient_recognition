"""Read Optuna journal studies through a disposable copy."""
from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def read_journal(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    try:
        import optuna
    except ImportError:
        return []

    studies = []
    with tempfile.TemporaryDirectory() as temporary:
        copied = Path(temporary) / "journal.log"
        shutil.copy2(path, copied)
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(str(copied))
        )
        for name in optuna.get_all_study_names(storage):
            study = optuna.load_study(study_name=name, storage=storage)
            complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
            trials = [{
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "distributions": {
                    key: optuna.distributions.distribution_to_json(value)
                    for key, value in trial.distributions.items()
                },
                "intermediate_values": {str(key): value for key, value in trial.intermediate_values.items()},
                "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
            } for trial in study.trials]
            studies.append({
                "name": name,
                "normalized_name": name.replace("\\", "/"),
                "direction": study.direction.name.lower(),
                "states": dict(Counter(trial.state.name for trial in study.trials)),
                "best_number": study.best_trial.number if complete else None,
                "best_value": study.best_value if complete else None,
                "trials": trials,
                "source": str(path),
            })
    return studies


def match_study(studies: list[dict[str, Any]], experiment: Path) -> dict[str, Any] | None:
    normalized = str(experiment.resolve()).replace("\\", "/")
    exact = [study for study in studies if study["normalized_name"] == normalized]
    if len(exact) == 1:
        return exact[0]
    suffix = "/".join(experiment.parts[-2:])
    matches = [study for study in studies if study["normalized_name"].endswith(suffix)]
    return matches[0] if len(matches) == 1 else None

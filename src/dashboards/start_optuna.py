import os
import subprocess
import warnings
from settings.config import OPTUNA_JOURNAL_PATH
from src.dashboards._commons import OPTUNA_PORT

def start_optuna(path: str | os.PathLike | None = None, new_console: bool = True) -> subprocess.Popen | None:

    if path is None:
        path = OPTUNA_JOURNAL_PATH
    if new_console:
        creationflags = subprocess.CREATE_NEW_CONSOLE
    else:
        creationflags = 0

    try:
        process = subprocess.Popen(["optuna-dashboard", path, "--port", str(OPTUNA_PORT)], creationflags=creationflags)
    except FileNotFoundError:
        # Case of remote development
        print("Process started on remote, open dashboard manually")
        process = None
    return process


if __name__ == "__main__":
    start_optuna()


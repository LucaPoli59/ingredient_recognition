import os
import subprocess
from settings.config import EXPERIMENTS_PATH


def start_optuna(path: str | os.PathLike | None = None) -> subprocess.Popen | None:
    if path is None:
        path = EXPERIMENTS_PATH

    try:
        process = subprocess.Popen(["optuna-dashboard", path], creationflags=subprocess.CREATE_NEW_CONSOLE)
    except FileNotFoundError:
        # Case of remote development
        print("Process started on remote, open dashboard manually")
        process = None
    return process


if __name__ == "__main__":
    start_optuna()


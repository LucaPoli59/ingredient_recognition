import subprocess

from settings.config import EXPERIMENTS_PATH


def start_optuna():

    subprocess.Popen(["optuna-dashboard", EXPERIMENTS_PATH], creationflags=subprocess.CREATE_NEW_CONSOLE)

if __name__ == "__main__":
    start_optuna()

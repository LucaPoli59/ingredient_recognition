from settings.config import EXPERIMENTS_PATH, PROJECT_PATH
import subprocess
import os

def start_tensorboard():
    subprocess.Popen(["tensorboard", "--logdir", EXPERIMENTS_PATH], creationflags=subprocess.CREATE_NEW_CONSOLE)

if __name__ == "__main__":
    start_tensorboard()

from settings.config import EXPERIMENTS_PATH, PROJECT_PATH
import subprocess
import os

def start_tensorboard():
    try:
        process = subprocess.Popen(["tensorboard", "--logdir", EXPERIMENTS_PATH], creationflags=subprocess.CREATE_NEW_CONSOLE)
    except FileNotFoundError:
        # Case of remote development
        print("Process started on remote, open dashboard manually")
        process = None
    return process

if __name__ == "__main__":
    start_tensorboard()

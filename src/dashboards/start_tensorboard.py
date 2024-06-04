from settings.config import EXPERIMENTS_PATH
import subprocess

from _commons import TENSORBOARD_PORT


def start_tensorboard(new_console: bool = True):
    if new_console:
        creationflags = subprocess.CREATE_NEW_CONSOLE
    else:
        creationflags = 0

    try:
        process = subprocess.Popen(["tensorboard", "--logdir", EXPERIMENTS_PATH, "--port", str(TENSORBOARD_PORT)],
                                   creationflags=creationflags)
    except FileNotFoundError:
        # Case of remote development
        print("Process started on remote, open dashboard manually")
        process = None
    return process


if __name__ == "__main__":
    start_tensorboard()

from settings.config import EXPERIMENTS_PATH
import subprocess

if __name__ == "__main__":
    subprocess.run(["tensorboard", "--logdir", EXPERIMENTS_PATH])

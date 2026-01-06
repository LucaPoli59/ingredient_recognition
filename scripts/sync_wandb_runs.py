import os
import subprocess

from settings.config import EXPERIMENTS_WANDB_PATH

if __name__ == "__main__":
    wandb_path = os.path.join(EXPERIMENTS_WANDB_PATH, "wandb")  # append wandb to the path cause the logger creates a redundant folder
    print("Syncing offline runs at path: ", wandb_path)

    # Get all the offline runs
    wandb_run_folder = [elem for elem in os.listdir(wandb_path) if elem.startswith("offline-run-") and os.path.isdir(os.path.join(wandb_path, elem))]

    # Remove the runs that are already synced
    wandb_run_folder = [elem for elem in wandb_run_folder if not any([file.endswith(".synced") for file in os.listdir(os.path.join(wandb_path, elem))])]

    if len(wandb_run_folder) == 0:
        print("No offline runs to sync founded")
    else:
        for wandb_run in wandb_run_folder:
            print(f"\n\n-------------------------------------\nSyncing run {wandb_run}")
            subprocess.run(["wandb", "sync", os.path.join(wandb_path, wandb_run)])
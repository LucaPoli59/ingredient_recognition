import os
import subprocess
import uuid

from settings.config import EXPERIMENTS_WANDB_PATH, WANDB_ENTITY, WANDB_PROJECT_NAME


def make_remote_run_id(wandb_run_path: str) -> str:
    """Create a new valid W&B run ID for an offline run bundle."""
    bundle_files = [
        filename for filename in os.listdir(wandb_run_path)
        if filename.startswith("run-") and filename.endswith(".wandb")
    ]
    if len(bundle_files) != 1:
        raise RuntimeError(f"Expected exactly one .wandb bundle in {wandb_run_path}")

    local_run_id = bundle_files[0][len("run-"):-len(".wandb")]
    suffix = f"-sync-{uuid.uuid4().hex[:8]}"
    return f"{local_run_id[:64 - len(suffix)]}{suffix}"


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
            run_path = os.path.join(wandb_path, wandb_run)
            remote_run_id = make_remote_run_id(run_path)
            print(f"\n\n-------------------------------------\nSyncing run {wandb_run}")
            print(f"Remote W&B run ID: {remote_run_id}")
            subprocess.run([
                "wandb", "beta", "sync",
                "--entity", WANDB_ENTITY,
                "--project", WANDB_PROJECT_NAME,
                "--id", remote_run_id,
                run_path,
            ], check=True)

import os
import shutil
import subprocess
import sys

from settings.config import EXPERIMENTS_PATH
from _commons import TENSORBOARD_PORT, _kill_process_on_port


def _build_tensorboard_command() -> list[str]:
    tensorboard_executable = shutil.which("tensorboard")
    if tensorboard_executable is None:
        tensorboard_executable = os.path.join(
            os.path.dirname(sys.executable), "tensorboard"
        )
    return [
        tensorboard_executable,
        "--logdir",
        os.fspath(EXPERIMENTS_PATH),
        "--port",
        str(TENSORBOARD_PORT),
    ]


def _build_wsl_console_command() -> list[str]:
    tensorboard_command = _build_tensorboard_command()
    if not os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"):
        return [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "tensorboard",
            "--",
            *tensorboard_command,
        ]
    wsl_arguments = ["wsl.exe"]
    distro = os.environ.get("WSL_DISTRO_NAME")
    if distro:
        wsl_arguments.extend(["-d", distro])
    wsl_arguments.extend(["--", *tensorboard_command])
    return [
        "cmd.exe",
        "/c",
        "start",
        "TensorBoard",
        "cmd.exe",
        "/k",
        *wsl_arguments,
    ]


def start_tensorboard(new_console: bool = True):
    is_wsl = os.name != "nt" and bool(os.environ.get("WSL_INTEROP"))
    tmux_fallback = is_wsl and not os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
    tmux_session = "tensorboard" if tmux_fallback else None
    if _kill_process_on_port(TENSORBOARD_PORT, tmux_session):
        print(f"Terminated the previous process on port {TENSORBOARD_PORT}.")
    if new_console and is_wsl:
        command = _build_wsl_console_command()
    else:
        command = _build_tensorboard_command()
    creationflags = (
        subprocess.CREATE_NEW_CONSOLE
        if new_console and os.name == "nt"
        else 0
    )

    try:
        process = subprocess.Popen(command, creationflags=creationflags)
        print(f"TensorBoard available at http://localhost:{TENSORBOARD_PORT}")
        if new_console and tmux_fallback:
            print("Logs: tmux attach -t tensorboard (detach with Ctrl+B, then D)")
    except FileNotFoundError:
        # Case of remote development
        print("Process started on remote, open dashboard manually")
        process = None
    return process


if __name__ == "__main__":
    start_tensorboard()

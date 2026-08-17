import os
import shutil
import subprocess
import sys
import warnings

from config import OPTUNA_JOURNAL_TRASH_FILENAME, OPTUNA_JOURNAL_TRASH_PATH
from settings.config import OPTUNA_JOURNAL_PATH
from src.dashboards._commons import OPTUNA_PORT, _kill_process_on_port

OPTUNA_DASHBOARD_PORT = 8055

def _build_dashboard_command(path: str | os.PathLike) -> list[str]:
    dashboard_executable = shutil.which("optuna-dashboard")
    if dashboard_executable is None:
        dashboard_executable = os.path.join(
            os.path.dirname(sys.executable), "optuna-dashboard"
        )
    return [
        dashboard_executable,
        os.fspath(path),
        "--port",
        str(OPTUNA_DASHBOARD_PORT),
    ]


def _build_wsl_console_command(path: str | os.PathLike) -> list[str]:
    dashboard_command = _build_dashboard_command(path)
    if not os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"):
        return [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "optuna-dashboard",
            "--",
            *dashboard_command,
        ]
    wsl_arguments = ["wsl.exe"]
    distro = os.environ.get("WSL_DISTRO_NAME")
    if distro:
        wsl_arguments.extend(["-d", distro])
    wsl_arguments.extend(["--", *dashboard_command])
    return [
        "cmd.exe",
        "/c",
        "start",
        "Optuna Dashboard",
        "cmd.exe",
        "/k",
        *wsl_arguments,
    ]


def start_optuna(path: str | os.PathLike | None = None, new_console: bool = True) -> subprocess.Popen | None:

    if path is None:
        path = OPTUNA_JOURNAL_PATH
        # path = OPTUNA_JOURNAL_TRASH_PATH
    is_wsl = os.name != "nt" and bool(os.environ.get("WSL_INTEROP"))
    tmux_fallback = is_wsl and not os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
    tmux_session = "optuna-dashboard" if tmux_fallback else None
    if _kill_process_on_port(OPTUNA_DASHBOARD_PORT, tmux_session):
        print(f"Terminated the previous process on port {OPTUNA_DASHBOARD_PORT}.")
    if new_console and is_wsl:
        command = _build_wsl_console_command(path)
    else:
        command = _build_dashboard_command(path)
    creationflags = (
        subprocess.CREATE_NEW_CONSOLE
        if new_console and os.name == "nt"
        else 0
    )

    try:
        process = subprocess.Popen(command, creationflags=creationflags)
        print(f"Optuna Dashboard available at http://localhost:{OPTUNA_DASHBOARD_PORT}")
        if new_console and tmux_fallback:
            print("Logs: tmux attach -t optuna-dashboard (detach with Ctrl+B, then D)")
        # process = subprocess.Popen(["optuna-dashboard", path, "--port", str(OPTUNA_PORT)], creationflags=creationflags)
    except FileNotFoundError:
        # Case of remote development
        print("Process started on remote, open dashboard manually")
        process = None
    return process


if __name__ == "__main__":
    start_optuna()


import base64
import os
import shutil
import signal
from typing import Optional, List
import subprocess
import dash
import numpy as np
from io import BytesIO
from PIL import Image

from settings.commons import list_intersection
from settings.config import HTUNER_CONFIG_FILE, DASH_PATH, DATA_PATH, PROJECT_PATH

DASH_PORT = 8050
OPTUNA_PORT = 8051
TENSORBOARD_PORT = 8052

DASH_PAGES_APP = os.path.join(DASH_PATH, 'pages')
DASH_CACHE = os.path.join(DASH_PATH, '_cache')
DASH_STATIC = os.path.join(DASH_PATH, 'static')
DASH_ASSETS = os.path.join(DASH_STATIC, 'assets')

for p in [DASH_CACHE, DASH_STATIC, DASH_ASSETS]:
    if not os.path.exists(p):
        os.mkdir(p)


def recursive_listdir(path: str, stop_at: str = "trial_",
                      stop_dir_contains: Optional[List[str] | str] = HTUNER_CONFIG_FILE,
                      ignore: Optional[List[str]] = None) -> List[str]:
    """
    Function to recursively list all files in a directory until a certain stop condition is met:
    - when we reach a directory starting with `stop_at`.
    - when we reach a directory that contains at least one of the elements in `stop_dir_contain`.
    :param path: path to the directory to list
    :param stop_at: name of the directory to stop at
    :param stop_dir_contains: list of elements that a directory can contain to stop (one at least)
    :param ignore: directories to ignore
    :return: list of absolute paths
    """
    if ignore is None:
        ignore = []
    if isinstance(stop_dir_contains, str):
        stop_dir_contains = [stop_dir_contains]

    if os.path.basename(path).startswith(stop_at):
        return [path]

    paths = []
    for elem in os.listdir(path):
        elem_path = os.path.join(path, elem)

        if elem in stop_dir_contains:
            return [path]

        if os.path.isdir(elem_path) and elem not in ignore:
            paths += recursive_listdir(elem_path, stop_at=stop_at, stop_dir_contains=stop_dir_contains, ignore=ignore)
    return paths


def open_img(path):
    print("load image")
    img_type = path.split(".")[-1]
    with open(path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
        img_data = f"data:image/{img_type};base64, {img_data}"
    return img_data


def img_from_ndarray(img: np.ndarray, ext="jpg") -> str:
    """Function that converts an image from a numpy array to a base64 string"""
    print(img)
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format=ext)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{ext};base64,{encoded}"


def dash_get_asset_url(path):
    path = os.path.normpath(dash.get_asset_url(os.path.relpath(path, PROJECT_PATH)))
    return path


def _kill_process_on_port(port: int, tmux_session: Optional[str] = None) -> bool:
    killed = False

    if os.name == "nt":
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            result = None

        if result is not None:
            pids = set()
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) < 5 or fields[0].upper() != "TCP":
                    continue
                if fields[3].upper() == "LISTENING" and fields[1].rsplit(":", 1)[-1] == str(port):
                    pids.add(fields[4])
            for pid in pids:
                taskkill = subprocess.run(
                    ["taskkill", "/PID", pid, "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                killed = killed or taskkill.returncode == 0
    else:
        fuser = shutil.which("fuser")
        if fuser is not None:
            result = subprocess.run(
                [fuser, "-k", f"{port}/tcp"],
                capture_output=True,
                text=True,
                check=False,
            )
            killed = result.returncode == 0
        else:
            lsof = shutil.which("lsof")
            if lsof is not None:
                result = subprocess.run(
                    [lsof, "-tiTCP:" + str(port), "-sTCP:LISTEN"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                for pid in result.stdout.split():
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        killed = True
                    except (ProcessLookupError, PermissionError, ValueError):
                        pass

    if tmux_session is not None:
        tmux = shutil.which("tmux")
        if tmux is not None:
            result = subprocess.run(
                [tmux, "kill-session", "-t", tmux_session],
                capture_output=True,
                text=True,
                check=False,
            )
            killed = killed or result.returncode == 0

    return killed

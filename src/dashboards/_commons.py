import base64
import os
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

# Create a symlink to the data folder (so the Dash app can access the data)
if not os.path.exists(os.path.join(DASH_ASSETS, "data")):
    try:
        os.symlink(DATA_PATH, os.path.join(DASH_ASSETS, "data"), target_is_directory=True)
    except PermissionError:
        print("PermissionError: run this with admin privileges")


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

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm


def tokenize_category(category):
    return f"____{category}"

def detokenize_category(token):
    return token.rsplit("____", 1)[1] if "____" in token else token

def remove_token(token):
    return token.replace("____", "") if "____" in token else token

def pad_array(array, pad_val=None, step=1, n_pads=1):
    """Pads an array with a value (repeated for n_pads) every n steps"""
    dim_multiplayer = (step + n_pads) / step
    if not round(len(array) * dim_multiplayer, 3).is_integer():
        raise ValueError(
            f"The step and n_nones combination is not valid : {len(array) * dim_multiplayer} / step must be an integer.")
    output = np.array([pad_val] * int(len(array) * dim_multiplayer))
    for i in range(step):
        output[i::step + n_pads] = array[i::step]
    return output


def show_image(img, cmap=None, title=None):
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    plt.show()

def list_intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))

# External code for plotting images after transformations
def torch_plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


# TQDM bar for scratch training
class TrainingTQDM(tqdm):
    def __init__(self, *args, **kwargs):

        if "postfix" not in kwargs:
            kwargs["postfix"] = [np.nan] * 4

        if "bar_format" not in kwargs:
            kwargs["bar_format"] = ("{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] "
                                    "[Train Loss: {postfix[0]:.4f}, Train Acc: {postfix[1]:.4f}, "
                                    "Val Loss: {postfix[2]:.4f}, Val Acc: {postfix[3]:.4f}]")

        super().__init__(*args, **kwargs)

    def update(self, n: int = 1, metrics: List[float | None] = None):
        if metrics is not None:
            for i, metric in enumerate(metrics):
                if metric is not None:
                    self.postfix[i] = metric
        super().update(n)


    def get_elapsed_s(self) -> float:
        return self.format_dict["elapsed"]

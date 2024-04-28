import numpy as np
from tqdm import tqdm
from typing import List


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

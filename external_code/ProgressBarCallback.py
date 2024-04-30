# import sys
# from typing import Any
#
# from lightning.pytorch.callbacks import TQDMProgressBar
# from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
#
#
# class ProgressBarCallback(TQDMProgressBar):
#     def init_validation_tqdm(self):
#         bar = super().init_validation_tqdm()
#         # if not sys.stdout.isatty():
#         #     bar.disable = True
#         bar.position = 0
#         bar.leave = True
#         return bar
#
#     def init_predict_tqdm(self):
#         bar = super().init_predict_tqdm()
#         if not sys.stdout.isatty():
#             bar.disable = True
#         return bar
#
#     def init_test_tqdm(self):
#         bar = super().init_test_tqdm()
#         if not sys.stdout.isatty():
#             bar.disable = True
#         return bar

from tqdm.auto import tqdm

import torch
from lightning.pytorch.callbacks import Callback


class ProgressBarCallback(Callback):
    """Global progress bar.
    """

    def __init__(self, global_progress: bool = True, leave_global_progress: bool = True):
        super().__init__()

        self.global_progress = global_progress
        self.global_desc = "Epoch: {epoch}/{max_epoch}"
        self.leave_global_progress = leave_global_progress
        self.global_pb = None

    def on_fit_start(self, trainer, pl_module):
        desc = self.global_desc.format(epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs)

        self.global_pb = tqdm(
            desc=desc,
            total=trainer.max_epochs,
            initial=trainer.current_epoch,
            leave=self.leave_global_progress,
            disable=not self.global_progress,
        )

    def on_fit_end(self, trainer, pl_module):
        self.global_pb.close()
        self.global_pb = None

    def on_epoch_end(self, trainer, pl_module):

        # Set description
        desc = self.global_desc.format(epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs)
        self.global_pb.set_description(desc)

        # Set logs and metrics
        logs = pl_module.logs
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                logs[k] = v.squeeze().item()
        self.global_pb.set_postfix(logs)

        # Update progress
        self.global_pb.update(1)
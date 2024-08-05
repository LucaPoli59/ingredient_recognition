import math

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmStartReduceOnPlateau(ReduceLROnPlateau):
    def __init__(self,
                 optimizer,
                 warm_start: float,
                 warm_stop: float,
                 warm_patience: int = 0,
                 warm_duration: int = 25,
                 warm_type: str = "linear",
                 mode: str = "min",
                 patience: int = 10,
                 cooldown=0,
                 factor=0.1,
                 threshold=1e-4,
                 threshold_mode='rel',
                 min_lr=0,
                 eps=1e-8,
                 verbose=False):
        """
        Workaround class as SequentialLR with ReduceLROnPlateau is not working in pytorch lightning currently.
        Otherwise, simply use WarmStart class together with any of the other pytorch schedulers.

        See Also
        https://github.com/PyTorchLightning/pytorch-lightning/issues/10759

        :param optimizer: The optimizer to apply the learning rate to
        :param warm_start: The initial learning rate
        :param warm_stop: The final learning rate
        :param warm_patience: The number of epochs to wait before starting the warm-up phase
        :param warm_duration: The number of epochs to warm-up the learning rate
        :param warm_type: The type of warm-up to apply. Either "linear" or "smooth"

        The others parameters are the same as ReduceLROnPlateau

        """
        assert warm_type in ("linear", "smooth")
        assert warm_duration > 0
        assert warm_patience >= 0

        self.warm_start = warm_start
        self.warm_stop = warm_stop
        self.warm_patience = warm_patience
        self.warm_duration = warm_duration
        self.warm_type = warm_type
        self.warm_ended = False
        self._last_lr = warm_start

        super().__init__(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Check if in warm-up phase
        if self.last_epoch > self.warm_patience and not self.warm_ended:
            self._warm_lr(self.last_epoch)

        # Check if out of warm-up phase
        if self.last_epoch > self.warm_patience + self.warm_duration:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                # Indicate to warm up that LRReduce should happen; prevent LR override
                if self.verbose and not self.warm_ended:
                    print(f"Ending warm-up phase after {epoch} epochs. "
                          f"Switching over to ReduceLROnPlateau")
                self.warm_ended = True

                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _warm_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            slope = (self.warm_stop - self.warm_start)
            x = (epoch - self.warm_patience) / self.warm_duration
            lower_bound = min(self.warm_start, self.warm_stop)
            upper_bound = max(self.warm_start, self.warm_stop)
            if self.warm_type == "linear":
                new_lr = slope * x + self.warm_start
            else:
                new_lr = slope * math.tanh(x) + self.warm_start
            param_group['lr'] = np.clip(new_lr, lower_bound, upper_bound)
            if self.verbose and not np.isclose(old_lr, new_lr):
                print('Epoch {:5d}: warming-up learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, i, new_lr))


class ConstantStartReduceOnPlateau(ReduceLROnPlateau):
    def __init__(self,
                 optimizer,
                 initial_lr: float,
                 warm_patience: int = 0,
                 warm_duration: int = 25,
                 warm_type: str = "linear",
                 mode: str = "min",
                 patience: int = 10,
                 cooldown=0,
                 factor=0.1,
                 threshold=1e-4,
                 threshold_mode='rel',
                 min_lr=0,
                 eps=1e-8):
        """
        Workaround class as SequentialLR with ReduceLROnPlateau is not working in pytorch lightning currently.
        Otherwise, simply use Costant LR together with any of the other pytorch schedulers.

        :param optimizer: The optimizer to apply the learning rate to
        :param initial_lr: The initial learning rate
        :param warm_patience: The number of epochs to wait before starting the warm-up phase
        :param warm_duration: The number of epochs to warm-up the learning rate
        :param warm_type: The type of warm-up to apply. Either "linear" or "smooth"

        The others parameters are the same as ReduceLROnPlateau

        """
        assert warm_type in ("linear", "smooth")
        assert warm_duration > 0
        assert warm_patience >= 0

        self.initial_lr = initial_lr
        self.warm_patience = warm_patience
        self.warm_duration = warm_duration
        self.warm_type = warm_type
        self.warm_ended = False
        self._last_lr = initial_lr

        super().__init__(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Check if out of warm-up phase
        if self.last_epoch > self.warm_patience + self.warm_duration:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                # Indicate to warm up that LRReduce should happen; prevent LR override
                if self.verbose and not self.warm_ended:
                    print(f"Ending warm-up phase after {epoch} epochs. "
                          f"Switching over to ReduceLROnPlateau")
                self.warm_ended = True

                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

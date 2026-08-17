import unittest

import torch

from src.models.custom_schedulers import (
    ConstantStartReduceOnPlateau,
    WarmStartReduceOnPlateau,
)


class CustomSchedulerTests(unittest.TestCase):
    @staticmethod
    def _make_optimizer(lr):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.SGD([parameter], lr=lr)

    def test_constant_start_reduces_without_parent_verbose_attribute(self):
        optimizer = self._make_optimizer(lr=0.1)
        scheduler = ConstantStartReduceOnPlateau(
            optimizer,
            initial_lr=0.1,
            warm_duration=1,
            patience=0,
            factor=0.5,
        )

        scheduler.step(1.0)
        scheduler.step(1.0)
        scheduler.step(1.0)

        self.assertFalse(scheduler.verbose)
        self.assertTrue(scheduler.warm_ended)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)

    def test_warm_start_reduces_after_warmup(self):
        optimizer = self._make_optimizer(lr=0.01)
        scheduler = WarmStartReduceOnPlateau(
            optimizer,
            warm_start=0.01,
            warm_stop=0.1,
            warm_duration=1,
            patience=0,
            factor=0.5,
        )

        scheduler.step(1.0)
        scheduler.step(1.0)
        scheduler.step(1.0)

        self.assertFalse(scheduler.verbose)
        self.assertTrue(scheduler.warm_ended)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)

    def test_legacy_state_without_verbose_remains_loadable(self):
        optimizer = self._make_optimizer(lr=0.1)
        scheduler = ConstantStartReduceOnPlateau(
            optimizer,
            initial_lr=0.1,
            warm_duration=1,
            patience=0,
            factor=0.5,
        )
        scheduler.step(1.0)
        scheduler.step(1.0)
        legacy_state = scheduler.state_dict()
        legacy_state.pop("verbose")

        restored_optimizer = self._make_optimizer(lr=0.1)
        restored_scheduler = ConstantStartReduceOnPlateau(
            restored_optimizer,
            initial_lr=0.1,
            warm_duration=1,
            patience=0,
            factor=0.5,
        )
        restored_scheduler.load_state_dict(legacy_state)
        restored_scheduler.step(1.0)

        self.assertFalse(restored_scheduler.verbose)
        self.assertAlmostEqual(restored_optimizer.param_groups[0]["lr"], 0.05)


if __name__ == "__main__":
    unittest.main()

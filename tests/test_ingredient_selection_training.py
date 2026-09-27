import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.ingredient_selection.protocol import SelectorProtocol
from src.training.ingredient_selection import SelectorAuditCallback, SelectorLightningModule


class _TinySelector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 165)
        self.dropout = torch.nn.Dropout(p=0.9)

    def forward(self, values):
        return self.linear(self.dropout(values))


class _AuditDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.features = torch.ones(4, 2)
        self.targets = torch.zeros(4, 165)

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.features[index], self.targets[index], str(index)


class IngredientSelectionTrainingTests(unittest.TestCase):
    @staticmethod
    def _configured():
        module = SelectorLightningModule(_TinySelector(), torch.ones(165))
        configured = module.configure_optimizers()
        return module, configured["optimizer"], configured["lr_scheduler"]["scheduler"]

    def test_optimizer_has_one_complete_parameter_group_and_frozen_adamw_values(self):
        module, optimizer, _ = self._configured()
        group = optimizer.param_groups[0]

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual({id(parameter) for parameter in group["params"]},
                         {id(parameter) for parameter in module.model.parameters()})
        self.assertEqual(group["betas"], (0.9, 0.999))
        self.assertEqual(group["eps"], 1e-8)
        self.assertEqual(group["weight_decay"], 1e-4)
        self.assertFalse(group["amsgrad"])
        self.assertFalse(group["foreach"])
        self.assertFalse(group["fused"])

    def test_complete_forty_epoch_learning_rate_sequence_and_transition(self):
        _, optimizer, scheduler = self._configured()
        used = []
        for _ in range(40):
            used.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        expected = [1e-5, 5.5e-5] + [
            1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * step / 38)) / 2
            for step in range(38)
        ]

        self.assertEqual(len(used), 40)
        for actual, planned in zip(used, expected):
            self.assertAlmostEqual(actual, planned, places=14)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6, places=14)

    def test_optimizer_and_scheduler_resume_at_same_next_epoch_value(self):
        _, optimizer, scheduler = self._configured()
        for _ in range(7):
            optimizer.step()
            scheduler.step()
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        expected_next = optimizer.param_groups[0]["lr"]

        _, restored_optimizer, restored_scheduler = self._configured()
        restored_optimizer.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)

        self.assertAlmostEqual(restored_optimizer.param_groups[0]["lr"], expected_next, places=14)
        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()
        self.assertAlmostEqual(
            restored_optimizer.param_groups[0]["lr"],
            optimizer.param_groups[0]["lr"],
            places=14,
        )

    def test_audit_collection_uses_eval_mode_and_restores_training_mode(self):
        module = SelectorLightningModule(_TinySelector(), torch.ones(165))
        module.train()
        dataloader = DataLoader(_AuditDataset(), batch_size=2, shuffle=False)

        first = SelectorAuditCallback._collect(module, dataloader)
        second = SelectorAuditCallback._collect(module, dataloader)

        self.assertTrue(module.training)
        self.assertEqual(first[0], ["0", "1", "2", "3"])
        self.assertTrue(torch.equal(torch.from_numpy(first[1]), torch.from_numpy(second[1])))
        self.assertTrue(torch.equal(torch.from_numpy(first[2]), torch.from_numpy(second[2])))


if __name__ == "__main__":
    unittest.main()

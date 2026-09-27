import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import lightning as lgn
from torch.utils.data import DataLoader, TensorDataset

from src.ingredient_selection.batching import (
    batch_candidates, resolve_batch_plan, accumulation_loss_factor, planned_optimizer_steps,
)
from src.ingredient_selection.provenance import snapshot_sources
from src.ingredient_selection.protocol import sha256_file
from src.training.ingredient_selection import SelectorLightningModule


class IngredientSelectionBatchingTests(unittest.TestCase):
    def test_resolution_uses_largest_exact_divisor_below_capacity(self):
        self.assertEqual(batch_candidates(128), [128, 64, 32, 16, 8, 4, 2, 1])
        for cap, physical, accumulation in [(128, 128, 1), (50, 32, 4), (24, 16, 8), (8, 8, 16)]:
            plan = resolve_batch_plan(128, cap)
            self.assertEqual((plan.physical_batch_size, plan.accumulate_grad_batches),
                             (physical, accumulation))
            self.assertEqual(physical * accumulation, 128)
        self.assertEqual(planned_optimizer_steps(47965, resolve_batch_plan(128, 16)), 375)

    def test_partial_final_group_matches_direct_sample_mean_gradient(self):
        for capacity in (128, 32, 16, 8):
            plan = resolve_batch_plan(128, capacity)
            records = 221  # 128 full + 93 remaining records.
            inputs = torch.arange(records, dtype=torch.float64) / records
            for start in (0, 128):
                stop = min(start + 128, records)
                weight = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
                reference = ((weight * inputs[start:stop] - 1) ** 2).mean()
                expected = torch.autograd.grad(reference, weight)[0]
                total = torch.tensor(0.0, dtype=torch.float64)
                for offset in range(start, stop, plan.physical_batch_size):
                    batch = inputs[offset:min(offset + plan.physical_batch_size, stop)]
                    loss = ((weight * batch - 1) ** 2).mean()
                    factor = accumulation_loss_factor(
                        offset // plan.physical_batch_size, len(batch), records, plan)
                    total = total + loss * factor / plan.accumulate_grad_batches
                actual = torch.autograd.grad(total, weight)[0]
                self.assertAlmostEqual(actual.item(), expected.item(), places=13)

    def test_snapshot_rejects_changed_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.py"
            source.write_text("first", encoding="utf-8")
            identity = {"files": {"model.py": sha256_file(source)}}
            source.write_text("second", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                snapshot_sources(root, root / "snapshot.zip", identity)

    def test_lightning_accumulation_matches_two_direct_bce_updates_with_partial_tail(self):
        torch.manual_seed(42)
        inputs = torch.arange(221, dtype=torch.float32).reshape(-1, 1) / 221
        targets = (torch.arange(221).reshape(-1, 1) % 3 == 0).float().expand(-1, 165)
        model = torch.nn.Linear(1, 165)
        reference = torch.nn.Linear(1, 165)
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for start in (0, 128):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                reference(inputs[start:start + 128]), targets[start:start + 128])
            loss.backward()
            optimizer.step()
        plan = resolve_batch_plan(128, 8)
        module = SelectorLightningModule(model, torch.ones(165), batch_plan=plan, train_records=221)
        module.configure_optimizers = lambda: torch.optim.SGD(model.parameters(), lr=0.1)
        loader = DataLoader(TensorDataset(inputs, targets, torch.arange(221)), batch_size=8)
        trainer = lgn.Trainer(accelerator="cpu", max_epochs=1, logger=False,
                              enable_checkpointing=False, enable_progress_bar=False,
                              enable_model_summary=False, accumulate_grad_batches=16)
        trainer.fit(module, train_dataloaders=loader)
        self.assertEqual(trainer.global_step, 2)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

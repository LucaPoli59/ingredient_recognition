import dataclasses
import json
import unittest

import torch

from src.training.batching import (
    ExactBatchPlan, accumulation_loss_scale, finite_loader_consumed_records,
    planned_optimizer_updates,
)


class ExperimentalBatchingTests(unittest.TestCase):
    def test_resolution_preserves_exact_requested_size(self):
        for requested, cap, physical, accumulation in (
                (128, None, 128, 1), (128, 256, 128, 1),
                (128, 50, 32, 4), (128, 24, 16, 8), (128, 8, 8, 16),
                (130, 50, 26, 5), (127, 64, 1, 127), (1, None, 1, 1)):
            with self.subTest(requested=requested, cap=cap):
                plan = ExactBatchPlan.resolve(requested, cap)
                self.assertEqual((plan.physical_batch_size, plan.accumulate_grad_batches),
                                 (physical, accumulation))
                self.assertEqual(plan.actual_effective_batch_size, requested)

    def test_explicit_smaller_divisor_is_preserved(self):
        plan = ExactBatchPlan.resolve(128, 64, physical_batch_size=8)
        self.assertEqual(plan.physical_batch_size, 8)
        self.assertEqual(plan.accumulate_grad_batches, 16)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.physical_batch_size = 16
        self.assertEqual(ExactBatchPlan.from_config(plan.to_config()), plan)

    def test_invalid_resolution_inputs_are_rejected(self):
        for invalid in (True, False, 0, -1, 128.0, "128"):
            for argument in ("requested_batch_size", "max_physical_batch_size", "physical_batch_size"):
                with self.subTest(argument=argument, invalid=invalid):
                    values = {"requested_batch_size": 128, argument: invalid}
                    with self.assertRaises(ValueError):
                        ExactBatchPlan.resolve(**values)
        for values in ((128, 50, 43), (128, 8, 16), (128, None, 256)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ExactBatchPlan.resolve(*values)

    def test_versioned_json_configuration_roundtrip(self):
        plan = ExactBatchPlan.resolve(128, 50)
        self.assertEqual(ExactBatchPlan.from_config(json.loads(json.dumps(plan.to_config()))), plan)
        self.assertEqual(plan.to_config()["schema_version"], 1)

    def test_configuration_corruption_is_rejected_without_recalculation(self):
        config = ExactBatchPlan.resolve(128, 8).to_config()
        mutations = (
            {"schema_version": 2}, {"schema_version": True}, {"schema_version": 1.0},
            {"requested_batch_size": 129}, {"physical_batch_size": 16},
            {"accumulate_grad_batches": 15}, {"max_physical_batch_size": 4},
            {"physical_batch_size": True}, {"unexpected": "field"},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                ExactBatchPlan.from_config(config | mutation)
        incomplete = config.copy()
        incomplete.pop("max_physical_batch_size")
        with self.assertRaises(ValueError):
            ExactBatchPlan.from_config(incomplete)
        with self.assertRaises(ValueError):
            ExactBatchPlan.from_config(None)

    def test_finite_loader_limits_count_consumed_records(self):
        plan = ExactBatchPlan.resolve(128, 8)
        for limit, records, updates in ((1.0, 221, 2), (16, 128, 1), (22, 176, 2),
                                         (100, 221, 2), (1, 8, 1), (0.5, 112, 1),
                                         (0, 0, 0), (0.0, 0, 0)):
            with self.subTest(limit=limit):
                actual = finite_loader_consumed_records(221, plan, limit)
                self.assertEqual(actual, records)
                self.assertEqual(planned_optimizer_updates(actual, plan), updates)
        self.assertEqual(finite_loader_consumed_records(0, plan), 0)

    def test_invalid_counts_and_limits_are_rejected(self):
        plan = ExactBatchPlan.resolve(128, 8)
        for invalid in (True, -1, 2.0, float("nan"), float("inf"), "16"):
            with self.subTest(limit=invalid), self.assertRaises(ValueError):
                finite_loader_consumed_records(221, plan, invalid)
        with self.assertRaises(ValueError):
            finite_loader_consumed_records(221, plan, 0.001)
        for invalid in (True, -1, 221.0):
            with self.subTest(records=invalid):
                with self.assertRaises(ValueError):
                    finite_loader_consumed_records(invalid, plan)
                with self.assertRaises(ValueError):
                    planned_optimizer_updates(invalid, plan)

    def test_partial_group_gradient_equals_direct_sample_mean(self):
        for capacity in (None, 32, 8):
            plan = ExactBatchPlan.resolve(128, capacity)
            inputs = torch.arange(221, dtype=torch.float64) / 221
            for start in (0, 128):
                stop = min(start + 128, len(inputs))
                weight = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
                direct = ((weight * inputs[start:stop] - 1) ** 2).mean()
                expected = torch.autograd.grad(direct, weight)[0]
                accumulated = torch.tensor(0.0, dtype=torch.float64)
                for offset in range(start, stop, plan.physical_batch_size):
                    batch = inputs[offset:min(offset + plan.physical_batch_size, stop)]
                    loss = ((weight * batch - 1) ** 2).mean()
                    scale = accumulation_loss_scale(
                        offset // plan.physical_batch_size, len(batch), len(inputs), plan)
                    accumulated = accumulated + loss * scale / plan.accumulate_grad_batches
                actual = torch.autograd.grad(accumulated, weight)[0]
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_partial_and_truncated_loader_updates_equal_direct_groups(self):
        plan = ExactBatchPlan.resolve(128, 8)
        inputs = torch.arange(221, dtype=torch.float64).reshape(-1, 1) / 221
        targets = (torch.arange(221).reshape(-1, 1) % 3 == 0).double()
        for limit in (1.0, 16, 22):
            with self.subTest(limit=limit):
                records = finite_loader_consumed_records(len(inputs), plan, limit)
                model = torch.nn.Linear(1, 1, dtype=torch.float64)
                with torch.no_grad():
                    model.weight.fill_(0.3)
                    model.bias.fill_(-0.1)
                reference = torch.nn.Linear(1, 1, dtype=torch.float64)
                reference.load_state_dict(model.state_dict())
                direct_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
                for start in range(0, records, plan.requested_batch_size):
                    stop = min(start + plan.requested_batch_size, records)
                    direct_optimizer.zero_grad()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        reference(inputs[start:stop]), targets[start:stop])
                    loss.backward()
                    direct_optimizer.step()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                updates = 0
                for index, start in enumerate(range(0, records, plan.physical_batch_size)):
                    stop = min(start + plan.physical_batch_size, records)
                    if index % plan.accumulate_grad_batches == 0:
                        optimizer.zero_grad()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        model(inputs[start:stop]), targets[start:stop])
                    scale = accumulation_loss_scale(index, stop - start, records, plan)
                    (loss * scale / plan.accumulate_grad_batches).backward()
                    if (index + 1) % plan.accumulate_grad_batches == 0 or stop == records:
                        optimizer.step()
                        updates += 1
                self.assertEqual(updates, planned_optimizer_updates(records, plan))
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_loss_scale_rejects_inconsistent_microbatches(self):
        plan = ExactBatchPlan.resolve(128, 8)
        self.assertEqual(accumulation_loss_scale(0, 8, 221, plan), 1.0)
        self.assertAlmostEqual(accumulation_loss_scale(27, 5, 221, plan), 16 * 5 / 93)
        self.assertEqual(accumulation_loss_scale(21, 8, 176, plan), 16 * 8 / 48)
        for index, batch, records in ((-1, 8, 221), (True, 8, 221), (0, True, 221),
                                       (0, 5, 221), (27, 8, 221), (28, 8, 221),
                                       (0, 8, 0), (0, 8, 221.0)):
            with self.subTest(index=index, batch=batch, records=records), self.assertRaises(ValueError):
                accumulation_loss_scale(index, batch, records, plan)


if __name__ == "__main__":
    unittest.main()

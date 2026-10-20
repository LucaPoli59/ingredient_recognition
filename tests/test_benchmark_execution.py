"""Focused checks for train-only priors and ordered validation export."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.benchmark_baselines import prior_scores, probability_logits, load_metadata_pair
from src.training.benchmark_predictions import collect_validation
from src.training.benchmark_protocol import epoch_metrics


class ImageFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., -1.]))
        self.output_class_order = ["a", "b"]

    @property
    def device(self):
        return self.weight.device

    def forward(self, inputs):
        return inputs.mean(dim=(1, 2, 3))[:, None] * self.weight


class BenchmarkExecutionTests(unittest.TestCase):
    def test_priors_use_only_train_and_unseen_cuisine_falls_back(self):
        y = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
        scores = prior_scores(y, ["a", "a", "b", "b"], ["a", "b", "unknown"])
        np.testing.assert_array_equal(scores["all_negative"], np.zeros((3, 2)))
        np.testing.assert_array_equal(scores["train_prevalence"], [[.5, .25]] * 3)
        np.testing.assert_array_equal(scores["train_cuisine"], [[.75, .25], [.25, .5], [.5, .25]])
        self.assertTrue(np.isfinite(probability_logits(scores["all_negative"])).all())

    def test_invalid_train_targets_rejected(self):
        with self.assertRaises(ValueError):
            prior_scores(np.array([[.5, 1]]), ["a"], ["a"])

    def test_metadata_reader_never_opens_test_or_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = {}
            for split in ("train", "val"):
                (root / split).mkdir()
                content = json.dumps([{"id": 0, "ingredients_target": ["a"]}]).encode()
                (root / split / "metadata.json").write_bytes(content)
                hashes[split] = hashlib.sha256(content).hexdigest()
            projection = SimpleNamespace(metadata_hashes=tuple(hashes.items()), metadata_filename="metadata.json")
            with mock.patch("src.training.benchmark_baselines.resolve_projection", return_value=projection):
                _, rows = load_metadata_pair(root)
            self.assertEqual(set(rows), {"train", "val"})
            (root / "val/metadata.json").write_text("[]")
            with mock.patch("src.training.benchmark_baselines.resolve_projection", return_value=projection):
                with self.assertRaisesRegex(ValueError, "frozen generation"):
                    load_metadata_pair(root)

    def test_predictions_preserve_empty_rows_order_metrics_and_mode(self):
        model = ImageFixture().train()
        images = torch.arange(5, dtype=torch.float32)[:, None, None, None].expand(5, 3, 2, 2)
        targets = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 0.], [0., 0.]])
        loader = DataLoader(TensorDataset(images, targets), batch_size=2)
        logits, truth, metrics = collect_validation(model, loader, expected_targets=targets.numpy(),
                                                     class_order=["a", "b"], device="cpu")
        np.testing.assert_array_equal(truth, targets)
        np.testing.assert_array_equal(logits[:, 0], np.arange(5))
        self.assertEqual(metrics, epoch_metrics(logits, targets.numpy()))
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "class order"):
            collect_validation(model, loader, expected_targets=targets.numpy(), class_order=["b", "a"], device="cpu")
        with self.assertRaisesRegex(ValueError, "record/target order"):
            collect_validation(model, loader, expected_targets=targets.roll(1, 0).numpy(), class_order=["a", "b"], device="cpu")
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()

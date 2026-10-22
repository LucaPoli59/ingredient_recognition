"""CPU-only contract tests using synthetic ingredient sets, never the real test split."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.metrics import average_precision_score, f1_score
import torch

from src_scratches.label_autoencoder import compare as experiment


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_split(self, split, rows):
        path = self.root / split / "labels.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{"ingredients_target": row} for row in rows]))
        return path

    def test_train_vocabulary_order_duplicates_empty_rows_and_test_isolation(self):
        self.write_split("train", [["z", "a", "a"], [], ["b"]])
        self.write_split("val", [["b", "z"], []])
        test = self.write_split("test", [])
        test.write_text("INVALID JSON: test must not be opened")
        train, val, labels, provenance = experiment.load_labels(self.root, "labels.json", "ingredients_target")
        self.assertEqual(labels, ["a", "b", "z"])
        assert_array_equal(train, [[1, 0, 1], [0, 0, 0], [0, 1, 0]])
        assert_array_equal(val, [[0, 1, 1], [0, 0, 0]])
        self.assertEqual(set(provenance), {"train", "val"})
        self.assertEqual(provenance["train"]["sha256"], experiment.sha256(self.root / "train/labels.json"))

    def test_validation_cannot_expand_train_vocabulary(self):
        self.write_split("train", [["salt"]])
        self.write_split("val", [["salt", "unknown"]])
        with self.assertRaisesRegex(ValueError, "outside train vocabulary"):
            experiment.load_labels(self.root, "labels.json", "ingredients_target")

    def test_malformed_targets_and_path_traversal_fail(self):
        self.write_split("train", [[" salt"]])
        self.write_split("val", [[]])
        with self.assertRaisesRegex(ValueError, "malformed"):
            experiment.load_labels(self.root, "labels.json", "ingredients_target")
        with self.assertRaisesRegex(ValueError, "filename"):
            experiment.load_labels(self.root, "../labels.json", "ingredients_target")

    def test_pca_is_centered_on_train_and_saved_without_pickle(self):
        train = np.random.default_rng(4).integers(0, 2, (100, 5)).astype(np.float32)
        experiment.fit_pca(train, [2, 5], self.root, list("abcde"))
        with np.load(self.root / "pca_k5.npz", allow_pickle=False) as saved:
            assert_allclose(saved["mean"], train.mean(axis=0))
            assert_allclose(experiment.pca_scores(train, saved["mean"], saved["components"]), train, atol=1e-6)
            components = saved["components"].copy()
            self.assertEqual(saved["labels"].tolist(), list("abcde"))
        with np.load(self.root / "pca_k2.npz", allow_pickle=False) as saved:
            assert_array_equal(saved["components"], components[:2])

    def test_metrics_match_reference_and_preserve_unbounded_pca_ranking(self):
        targets = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
        scores = np.array([[1.2, -0.1], [0.2, 0.8], [1.1, 0.7], [1.05, 0.1]], dtype=np.float32)
        metrics, rows = experiment.evaluate(targets, scores, np.array([10, 2]), ["a", "b"])
        self.assertAlmostEqual(metrics["macro_ap"], average_precision_score(targets, scores, average="macro"))
        reference = [f1_score(targets, scores >= t, average="micro", zero_division=0)
                     for t in experiment.THRESHOLDS]
        self.assertAlmostEqual(metrics["micro_f1_val_tuned"], max(reference))
        self.assertEqual(metrics["val_selected_threshold"], experiment.THRESHOLDS[np.argmax(reference)])
        self.assertLess(metrics["mse_clipped"], metrics["mse_raw"])
        self.assertEqual(rows[1]["frequency_slice"], "rare")

    def test_missing_validation_positives_are_explicit(self):
        targets = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        metrics, rows = experiment.evaluate(targets, targets.copy(), np.array([1, 2, 3]), list("abc"))
        self.assertEqual(metrics["macro_ap"], 1.0)
        self.assertEqual(metrics["labels_with_val_positives"], 2)
        self.assertIsNone(rows[2]["ap"])
        self.assertIsNone(rows[2]["recall_val_tuned"])

    def test_invalid_scores_fail(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            experiment.evaluate(np.ones((2, 3)), np.full((2, 3), np.nan), np.ones(3), list("abc"))

    def test_cpu_never_queries_cuda_and_busy_gpu_is_rejected(self):
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")):
            self.assertEqual(str(experiment.resolve_device("cpu", {})), "cpu")
            with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                experiment.resolve_device("cuda", {"samples": [dict(free_mib=362, utilization_pct=94)] * 3})

    def test_end_to_end_all_arms_reload_and_output_protection(self):
        rng = np.random.default_rng(123)
        for split, count in (("train", 96), ("val", 24)):
            matrix = rng.integers(0, 2, (count, 8))
            self.write_split(split, [[f"l{j}" for j in np.flatnonzero(row)] for row in matrix])
        output = self.root / "output"
        args = experiment.parse_args([
            "--data-dir", str(self.root), "--metadata", "labels.json", "--output", str(output),
            "--dimensions", "2", "3", "4", "--epochs", "5", "--hidden-dim", "12", "--threads", "1",
            "--batch-size", "16", "--learning-rate", "0.01",
        ])
        with patch.object(experiment, "gpu_snapshot", return_value={"samples": []}), \
                patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")):
            summary = experiment.run(args)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(len(summary["arms"]), 7)
        _, val, labels, _ = experiment.load_labels(self.root, "labels.json", "ingredients_target")
        for k in (2, 3, 4):
            model, checkpoint = experiment.load_autoencoder(output / f"ae_k{k}.pt")
            self.assertEqual(checkpoint["labels"], labels)
            scores = experiment.predict_autoencoder(model, val, torch.device("cpu"))
            arm = next(x for x in summary["arms"] if x["method"] == "ae" and x["latent_dim"] == k)
            self.assertAlmostEqual(arm["metrics"]["macro_ap"], average_precision_score(val, scores, average="macro"))
            self.assertGreater(arm["optimizer_steps"], 0)
            history = np.genfromtxt(output / f"ae_k{k}_history.csv", delimiter=",", names=True)
            self.assertLess(history["train_bce"][-1], history["train_bce"][0])
            self.assertAlmostEqual(checkpoint["best_val_bce"], float(min(history["val_bce"])))
        self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "complete")
        with self.assertRaises(FileExistsError):
            experiment.run(args)


if __name__ == "__main__":
    unittest.main()

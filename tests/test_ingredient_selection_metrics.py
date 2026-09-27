import unittest

import numpy as np
import pandas as pd

from src.ingredient_selection.metrics import (
    ProfileThresholds,
    bootstrap_average_precision,
    classify_profile,
    per_label_metrics,
    trajectory_evidence,
)
from src.ingredient_selection.protocol import SelectorProtocol


class IngredientSelectionMetricTests(unittest.TestCase):
    def test_fixed_threshold_diagnostics_do_not_change_average_precision(self):
        logits = np.asarray([[2.0, -0.2], [-1.0, 0.2], [0.1, 1.0]])
        targets = np.asarray([[1, 0], [0, 1], [1, 1]])
        low = per_label_metrics(
            logits, targets, ["a", "b"], run_id="run", split="val",
            audit_epoch=2, learning_rate=1e-4, threshold=0.5,
        )
        high = per_label_metrics(
            logits, targets, ["a", "b"], run_id="run", split="val",
            audit_epoch=2, learning_rate=1e-4, threshold=0.9,
        )

        self.assertEqual([row["average_precision"] for row in low],
                         [row["average_precision"] for row in high])
        self.assertNotEqual([row["f1_at_0_5"] for row in low],
                            [row["f1_at_0_5"] for row in high])

    def test_bootstrap_is_deterministic_and_rejects_single_class_targets(self):
        targets = np.asarray([0, 1] * 10)
        scores = np.linspace(0, 1, len(targets))
        first = bootstrap_average_precision(scores, targets, seed=42000, samples=50)
        second = bootstrap_average_precision(scores, targets, seed=42000, samples=50)
        invalid = bootstrap_average_precision(scores, np.ones_like(targets), seed=42000, samples=50)

        self.assertEqual(first, second)
        self.assertTrue(first["valid"])
        self.assertFalse(invalid["valid"])

    @staticmethod
    def _trajectory_table():
        protocol = SelectorProtocol()
        profiles = {
            "improving": ([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.60, 0.64, 0.66, 0.68, 0.70],
                          [0.04, 0.12, 0.20, 0.30, 0.39, 0.48, 0.52, 0.55, 0.57, 0.58, 0.59]),
            "flat": ([0.10] * 11, [0.10] * 11),
            "noisy": ([0.05, 0.15, 0.20, 0.25, 0.30, 0.45, 0.20, 0.70, 0.22, 0.75, 0.25],
                      [0.05, 0.12, 0.16, 0.20, 0.22, 0.25, 0.15, 0.50, 0.16, 0.52, 0.18]),
            "degrading": ([0.60, 0.55, 0.50, 0.44, 0.38, 0.32, 0.28, 0.24, 0.20, 0.17, 0.15],
                          [0.50, 0.45, 0.40, 0.35, 0.30, 0.27, 0.23, 0.20, 0.18, 0.16, 0.14]),
            "missing": ([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.60, 0.64, 0.66, 0.68, 0.70],
                        [0.04, 0.12, 0.20, 0.30, 0.39, 0.48, 0.52, 0.55, 0.57, 0.58, 0.59]),
        }
        rows = []
        for class_index, (name, (train_values, val_values)) in enumerate(profiles.items()):
            for split, values in (("train", train_values), ("val", val_values)):
                values = values[:6] + [values[5]] * 10 + values[6:]
                for epoch, ap in zip(protocol.audit_epochs, values):
                    if name == "missing" and split == "train" and epoch == protocol.late_window[1]:
                        continue
                    rows.append({
                        "run_id": "run",
                        "audit_epoch": epoch,
                        "split": split,
                        "class_index": class_index,
                        "class_name": name,
                        "support": 100 if name != "missing" else 2,
                        "prevalence": 0.1,
                        "average_precision": ap,
                    })
        return pd.DataFrame(rows)

    def test_trajectory_windows_resist_spikes_and_mark_missing_epochs(self):
        evidence = trajectory_evidence(self._trajectory_table()).set_index("class_name")

        self.assertTrue(evidence.loc["improving", "trajectory_complete"])
        self.assertGreater(evidence.loc["improving", "train_initial_to_late_gain"], 0.5)
        self.assertGreater(evidence.loc["noisy", "train_late_iqr"], 0.4)
        self.assertLess(evidence.loc["degrading", "train_initial_to_late_gain"], 0)
        self.assertFalse(evidence.loc["missing", "trajectory_complete"])

    def test_profile_assignment_covers_declared_evidence_outcomes(self):
        gates = ProfileThresholds(10, 0.10, 0.30, 0.25, 0.10, 0.10, 0.35, 0.05)
        base = {
            "trajectory_complete": True,
            "train_support": 100,
            "train_initial_to_late_gain": 0.5,
            "train_late_median_ap": 0.7,
            "val_late_median_ap": 0.6,
            "train_late_iqr": 0.02,
            "val_late_iqr": 0.02,
            "train_minus_val_late_gap": 0.1,
            "image_vs_cuisine_ap_advantage": 0.2,
        }
        self.assertEqual(classify_profile(base, gates)[0], "generalizable_candidate")
        self.assertEqual(classify_profile(base | {"train_support": 2}, gates)[0], "uncertain")
        self.assertEqual(classify_profile(base | {"train_initial_to_late_gain": 0.01}, gates)[0],
                         "no_sustained_optimization")
        self.assertEqual(classify_profile(base | {"val_late_median_ap": 0.1}, gates)[0],
                         "optimization_only")
        self.assertEqual(classify_profile(base | {"train_late_iqr": 0.3}, gates)[0], "uncertain")
        self.assertEqual(classify_profile(base | {"image_vs_cuisine_ap_advantage": 0.01}, gates)[0],
                         "context_predictable")
        self.assertEqual(classify_profile(base | {"trajectory_complete": False}, gates)[0], "uncertain")


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from src.ingredient_selection.inclusion_statistics import (
    _prepare_rankings, _weighted_aps, classify_inclusion, paired_cluster_bootstrap,
)


class InclusionStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.targets = np.array([1, 0, 1, 0, 0, 1, 0, 1])
        self.scores = np.array([
            [.9, .9, .8, .7, .5, .5, .2, .1],
            [.2, .4, .9, .7, .1, .5, .3, .8],
            [.8, .8, .2, .3, .4, .6, .1, .5],
            [.3, .2, .8, .6, .4, .7, .1, .9],
            [.1, .2, .3, .4, .5, .6, .7, .8],
        ])
        self.clusters = ["a", "a", "b", "c", "c", "c", "d", "e"]

    def test_weighted_ap_matches_sklearn_with_ties_zero_weights_and_replication(self):
        weights = np.array([[1, 1, 2, 0, 0, 0, 3, 1], [0, 0, 1, 2, 2, 2, 0, 3], [1] * 8])
        actual = _weighted_aps(_prepare_rankings(self.targets, self.scores), weights)
        for draw, row_weights in enumerate(weights):
            replicated = np.repeat(np.arange(8), row_weights)
            for checkpoint in range(5):
                expected = average_precision_score(self.targets, self.scores[checkpoint], sample_weight=row_weights)
                explicit = average_precision_score(self.targets[replicated], self.scores[checkpoint, replicated])
                self.assertAlmostEqual(actual[draw, checkpoint], expected, places=14)
                self.assertAlmostEqual(actual[draw, checkpoint], explicit, places=14)

    def test_cluster_bootstrap_matches_explicit_record_replication(self):
        actual = paired_cluster_bootstrap(self.targets, self.scores, self.clusters, seed=42007,
                                          samples=31, max_draws=300, batch_size=7)
        groups = [np.flatnonzero(np.array(self.clusters) == name) for name in dict.fromkeys(self.clusters)]
        generator = np.random.default_rng(42007)
        q_values, differences, attempts = [], [], 0
        while len(q_values) < 31:
            attempts += 1
            chosen = generator.integers(0, len(groups), size=len(groups))
            indices = np.concatenate([groups[index] for index in chosen])
            labels = self.targets[indices]
            if labels.min() == labels.max():
                continue
            ap = [average_precision_score(labels, scores[indices]) for scores in self.scores]
            q_values.append(float(np.median(ap)))
            differences.append(q_values[-1] - labels.mean())
        ap = [average_precision_score(self.targets, scores) for scores in self.scores]
        np.testing.assert_allclose(actual["checkpoint_ap"], ap, rtol=0, atol=1e-14)
        np.testing.assert_allclose([actual["q_lower"], actual["q_upper"]], np.quantile(q_values, [.025, .975]), atol=1e-14)
        np.testing.assert_allclose([actual["difference_lower"], actual["difference_upper"]],
                                   np.quantile(differences, [.025, .975]), atol=1e-14)
        self.assertAlmostEqual(actual["q"], float(np.median(ap)))
        self.assertAlmostEqual(actual["iqr"], np.quantile(ap, .75) - np.quantile(ap, .25))
        self.assertEqual(actual["attempted_draws"], attempts)
        self.assertEqual(actual["invalid_draws"], attempts - 31)
        self.assertTrue(actual["bootstrap_valid"])

    def test_batch_size_preserves_seeded_sequence_and_attempt_count(self):
        expected = paired_cluster_bootstrap(self.targets, self.scores, self.clusters, seed=6, samples=19, batch_size=1)
        for size in [2, 7, 32, 100]:
            actual = paired_cluster_bootstrap(self.targets, self.scores, self.clusters, seed=6, samples=19, batch_size=size)
            self.assertEqual(expected, actual)

    def test_invalid_draws_are_bounded_and_never_publish_partial_intervals(self):
        targets = [1, 1, 0, 0, 0]
        scores = np.tile([5, 4, 3, 2, 1], (5, 1))
        actual = paired_cluster_bootstrap(targets, scores, ["positive"] * 2 + ["negative"] * 3,
                                          seed=42, samples=10, max_draws=8, batch_size=3)
        self.assertEqual(actual["status"], "insufficient_valid_draws")
        self.assertEqual(actual["attempted_draws"], 8)
        self.assertGreater(actual["invalid_draws"], 0)
        self.assertEqual(actual["valid_draws"] + actual["invalid_draws"], 8)
        self.assertFalse(actual["bootstrap_valid"])
        self.assertIsNone(actual["q_lower"])
        self.assertIsNone(actual["difference_lower"])

    def test_no_class_contrast_returns_invalid_evidence(self):
        actual = paired_cluster_bootstrap([1] * 8, self.scores, self.clusters, seed=1)
        self.assertEqual(actual["status"], "insufficient_classes")
        self.assertEqual(actual["attempted_draws"], 0)
        self.assertIsNone(actual["q"])
        self.assertEqual(classify_inclusion(actual)["outcome"], "uncertain")

    def test_identical_scores_have_zero_paired_difference(self):
        actual = paired_cluster_bootstrap(self.targets, np.ones((5, 8)), self.clusters, seed=2, samples=25)
        self.assertEqual(actual["q"], actual["prevalence"])
        self.assertEqual(actual["difference_lower"], 0)
        self.assertEqual(actual["difference_upper"], 0)
        self.assertFalse(classify_inclusion(actual)["included"])

    def test_classifier_exact_boundaries_and_overlap(self):
        evidence = dict(q=.2, iqr=.03, q_lower=.2, q_upper=.3, difference_lower=.01,
                        difference_upper=.1, bootstrap_valid=True)
        self.assertTrue(classify_inclusion(evidence)["included"])
        equality = {**evidence, "difference_lower": 0}
        self.assertEqual(classify_inclusion(equality)["outcome"], "uncertain")
        self.assertIn("positive_baseline_advantage_not_cleared", classify_inclusion(equality)["reasons"])
        touching = {**evidence, "q": .19, "q_lower": .1, "q_upper": .2}
        self.assertEqual(classify_inclusion(touching)["outcome"], "uncertain")
        below = {**touching, "q_upper": .199999}
        self.assertEqual(classify_inclusion(below)["outcome"], "below_quality_floor")

    def test_classifier_retains_independent_failure_reasons(self):
        evidence = dict(q=.1, iqr=.04, q_lower=.05, q_upper=.15, difference_lower=-.03,
                        difference_upper=.02, bootstrap_valid=True)
        result = classify_inclusion(evidence)
        self.assertEqual(result["outcome"], "uncertain")
        self.assertEqual(result["axis_status"]["quality"], "below_floor")
        self.assertIn("quality_interval_below_floor", result["reasons"])
        self.assertIn("late_validation_iqr_above_gate", result["reasons"])
        self.assertIn("positive_baseline_advantage_not_cleared", result["reasons"])
        self.assertFalse(result["included"])
        self.assertEqual(classify_inclusion({**evidence, "iqr": .03})["outcome"], "below_quality_floor")

    def test_bad_inputs_fail_early(self):
        variants = [(self.targets, self.scores[:4], self.clusters),
                    ([2] + list(self.targets[1:]), self.scores, self.clusters),
                    (self.targets, self.scores, self.clusters[:-1]),
                    (self.targets, self.scores, [None] + self.clusters[1:]),
                    (self.targets, self.scores * np.nan, self.clusters)]
        for targets, scores, clusters in variants:
            with self.assertRaises(ValueError):
                paired_cluster_bootstrap(targets, scores, clusters, seed=42)
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap(self.targets, self.scores, self.clusters, seed=42, batch_size=0)


if __name__ == "__main__":
    unittest.main()

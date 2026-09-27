import copy
import unittest

import numpy as np
import torch

from src.ingredient_selection.protocol import (
    build_pilot_cohort,
    compute_pos_weight,
    validate_pilot_cohort,
)


class IngredientSelectionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.names = [f"ingredient_{index:03d}" for index in range(165)]
        self.supports = [(index % 17) + 1 for index in range(165)]

    def test_pilot_is_deterministic_stratified_and_keeps_source_indices(self):
        first = build_pilot_cohort(self.names, self.supports)
        second = build_pilot_cohort(self.names, self.supports)

        self.assertEqual(first, second)
        self.assertEqual(len(first["labels"]), 24)
        self.assertEqual(
            [sum(item["support_stratum"] == stratum for item in first["labels"]) for stratum in range(3)],
            [8, 8, 8],
        )
        for item in first["labels"]:
            self.assertEqual(self.names[item["class_index"]], item["class_name"])
            self.assertEqual(self.supports[item["class_index"]], item["train_positive_count"])

    def test_pilot_validation_rejects_tampering(self):
        artifact = build_pilot_cohort(self.names, self.supports)
        tampered = copy.deepcopy(artifact)
        tampered["labels"][0]["class_name"] = "changed"
        with self.assertRaises(ValueError):
            validate_pilot_cohort(tampered, self.names, self.supports)

    def test_pos_weight_uses_train_negative_to_positive_ratio(self):
        targets = np.asarray([[1, 0], [1, 1], [0, 0], [0, 0]], dtype=np.uint8)
        weights = compute_pos_weight(targets)
        self.assertTrue(torch.equal(weights, torch.tensor([1.0, 3.0])))


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path

import numpy as np

from settings.config import DEF_UNKNOWN_TOKEN, YUMMLY_TARGET_METADATA_FILENAME
from src.data_processing.labels_encoders import MultiLabelBinarizer, MultiLabelBinarizerRobust


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
YUMMLY_ROOT = REPOSITORY_ROOT / "data" / "input" / "yummly"


class MultiLabelEncoderContractTests(unittest.TestCase):
    def test_new_multilabel_encoder_has_no_unknown_output(self):
        encoder = MultiLabelBinarizer()
        encoder.fit(np.array([["salt", "pepper"], ["salt"]], dtype=object))

        self.assertNotIn(DEF_UNKNOWN_TOKEN, encoder.classes)
        self.assertEqual(encoder.num_classes, 2)
        with self.assertRaises(KeyError):
            encoder.transform([["not-in-training-vocabulary"]])

    def test_legacy_robust_encoder_keeps_unknown_output_and_roundtrips(self):
        encoder = MultiLabelBinarizerRobust()
        encoder.fit(np.array([["salt", "pepper"], ["salt"]], dtype=object))
        restored = MultiLabelBinarizerRobust.load_from_config(encoder.to_config())

        self.assertEqual(list(restored.classes), ["pepper", "salt", DEF_UNKNOWN_TOKEN])
        self.assertEqual(restored.num_classes, 3)
        self.assertEqual(restored.transform([["new-label"]]).tolist(), [[0.0, 0.0, 1.0]])

    def test_v5_vocabulary_is_shared_and_contains_no_unknown_target(self):
        self.assertEqual(YUMMLY_TARGET_METADATA_FILENAME, "ingredients_target_v5_metadata.json")
        records_by_split = {}
        for split in ("train", "val", "test"):
            with (YUMMLY_ROOT / split / YUMMLY_TARGET_METADATA_FILENAME).open(encoding="utf-8") as handle:
                records_by_split[split] = json.load(handle)

        encoder = MultiLabelBinarizer()
        train_targets = np.array(
            [record["ingredients_target"] for record in records_by_split["train"]],
            dtype=object,
        )
        encoder.fit(train_targets)

        self.assertEqual(encoder.num_classes, 165)
        self.assertNotIn(DEF_UNKNOWN_TOKEN, encoder.classes)
        for split in ("val", "test"):
            labels = [record["ingredients_target"] for record in records_by_split[split]]
            encoder.transform(labels)


if __name__ == "__main__":
    unittest.main()

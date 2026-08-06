import unittest
from pathlib import Path

from src_scratches.data_anlysis.metadata_field_audit import audit_records


class MetadataFieldAuditTests(unittest.TestCase):
    METADATA = Path("train/ingredients_target_v5_metadata.json")

    def test_target_counts_distinguish_occurrences_and_recipe_support(self):
        records = [
            {"id": 1, "name": "A", "cuisine": "X", "ingredients_target": ["salt", "salt", "pepper"]},
            {"id": 2, "name": "B", "cuisine": "X", "ingredients_target": ["salt"]},
        ]
        report, value_rows, record_rows, cuisine_rows, normalized_rows = audit_records(
            records,
            field="ingredients_target",
            split="train",
            metadata_path=self.METADATA,
            top_k=10,
            normalize_ingredients=False,
            include_pairs=False,
        )

        self.assertEqual(report["metadata"]["records"], 2)
        self.assertEqual(report["value_summary"]["unique_value_count"], 2)
        self.assertEqual(report["value_summary"]["total_value_occurrences"], 4)
        self.assertEqual(report["value_summary"]["total_recipe_assignments"], 3)
        self.assertEqual(report["field_quality"]["duplicate_items_within_records"], 1)
        self.assertEqual(value_rows[0]["value"], "salt")
        self.assertEqual(value_rows[0]["occurrence_count"], 3)
        self.assertEqual(value_rows[0]["recipe_support"], 2)
        self.assertEqual(record_rows[0]["unique_value_count"], 2)
        self.assertEqual(cuisine_rows[0]["record_count"], 2)
        self.assertEqual(normalized_rows, [])

    def test_raw_ingredients_can_have_a_normalized_summary(self):
        records = [
            {
                "id": 1,
                "name": "A",
                "cuisine": "X",
                "ingredients": ["2 cloves garlic", "salt"],
            },
            {"id": 2, "name": "B", "cuisine": "Y", "ingredients": ["1 clove garlic"]},
        ]
        report, _, _, _, normalized_rows = audit_records(
            records,
            field="ingredients",
            split="train",
            metadata_path=self.METADATA,
            top_k=10,
            normalize_ingredients=True,
            include_pairs=True,
        )

        self.assertEqual(report["value_summary"]["unique_value_count"], 3)
        self.assertEqual(report["normalized_ingredients"]["unique_value_count"], 2)
        self.assertEqual(report["pair_summary"]["enabled"], True)
        self.assertTrue(any(row["value"] == "garlic" for row in normalized_rows))

    def test_invalid_field_shape_is_reported(self):
        records = [{"id": 1, "ingredients_target": None}, {"id": 2, "ingredients_target": "salt"}]
        report, _, _, _, _ = audit_records(
            records,
            field="ingredients_target",
            split="train",
            metadata_path=self.METADATA,
            top_k=10,
            normalize_ingredients=False,
            include_pairs=False,
        )
        self.assertEqual(report["field_quality"]["missing_or_null_field_records"], 1)
        self.assertEqual(report["field_quality"]["non_list_field_records"], 1)
        self.assertEqual(report["field_quality"]["records_with_any_problem"], 2)


if __name__ == "__main__":
    unittest.main()

import unittest

from src.data_processing.foodon_lexicon import load_packaged_foodon
from src.data_processing.ingredient_target_generation import (
    associate_ingredient_line,
    derive_controlled_targets,
)


class FoodOnTargetGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lexicon = load_packaged_foodon()

    def test_packaged_index_is_pinned_food_product_branch(self):
        self.assertEqual(self.lexicon.concept_count, 14185)
        self.assertEqual(self.lexicon.unique_id("english muffin"), "FOODON_03305833")

    def test_direct_foodon_precedes_fallback_mapping(self):
        tomato_paste = associate_ingredient_line(self.lexicon, "tomato paste")
        chicken_stock = associate_ingredient_line(self.lexicon, "chicken stock")
        self.assertEqual(tomato_paste.status, "direct")
        self.assertEqual(
            [self.lexicon.canonical_label(concept_id) for concept_id in tomato_paste.external_ids],
            ["tomato paste"],
        )
        self.assertEqual(chicken_stock.status, "fallback")
        self.assertEqual(
            [self.lexicon.canonical_label(concept_id) for concept_id in chicken_stock.external_ids],
            ["chicken broth"],
        )

    def test_plural_retry_and_local_residual(self):
        muffin = associate_ingredient_line(self.lexicon, "English muffins")
        local = associate_ingredient_line(self.lexicon, "garam masala")
        self.assertEqual(muffin.external_ids, ("FOODON_03305833",))
        self.assertEqual(local.external_ids, ())
        self.assertEqual(local.local_targets, ("garam masala",))

    def test_support_is_counted_on_train_records_only(self):
        train = [
            {"ingredients": ["tomato paste", "garam masala"]},
            {"ingredients": ["tomato paste", "garam masala"]},
        ]
        held_out = [{"ingredients": ["english muffins", "garam masala"]}]
        result = derive_controlled_targets(
            train + held_out,
            train,
            self.lexicon,
            min_recipe_support=2,
            min_targets_per_recipe=1,
        )
        self.assertEqual(result.targets_by_record[0], ["garam masala", "tomato paste"])
        self.assertEqual(result.targets_by_record[2], ["garam masala"])
        self.assertEqual(result.retained_records, [0, 1, 2])
        self.assertNotIn("english muffin", result.train_support)


if __name__ == "__main__":
    unittest.main()

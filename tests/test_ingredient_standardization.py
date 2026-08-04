import unittest

from src.data_processing.ingredient_standardization import (
    derive_ingredients_target,
    normalize_ingredient_line,
)


class IngredientStandardizationTests(unittest.TestCase):
    def test_explicit_aliases_are_token_bounded(self):
        self.assertEqual(normalize_ingredient_line("3 cloves of garlic, minced"), "garlic")
        self.assertEqual(normalize_ingredient_line("1 cup shredded mozzarella cheese"), "cheese")
        self.assertEqual(normalize_ingredient_line("2 tablespoons all-purpose flour"), "flour")

    def test_known_legacy_substring_collisions_do_not_match(self):
        self.assertEqual(normalize_ingredient_line("1 pineapple"), "pineapple")
        self.assertEqual(normalize_ingredient_line("1 butternut squash"), "butternut squash")
        self.assertEqual(normalize_ingredient_line("1 pepperoni"), "pepperoni")
        self.assertEqual(normalize_ingredient_line("1 cup watercress"), "watercress")

    def test_support_counts_distinct_recipes_and_targets_are_sorted(self):
        recipes = [
            {"ingredients": ["2 apples", "1 cup flour", "salt"]},
            {"ingredients": ["apple", "flour", "pepper"]},
            {"ingredients": ["apple", "flour", "oil"]},
        ]
        result = derive_ingredients_target(recipes, min_recipe_support=3, min_targets_per_recipe=2)
        self.assertEqual(result.recipe_support["apple"], 3)
        self.assertEqual(result.targets_by_record[0], ["apple", "flour"])
        self.assertEqual(result.retained_records, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

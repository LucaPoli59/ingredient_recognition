import unittest

from src.data_processing.ingredient_standardization import (
    derive_ingredients_target,
    normalize_ingredient_line,
    normalize_ingredient_targets,
)


class IngredientStandardizationTests(unittest.TestCase):
    def test_explicit_aliases_are_token_bounded(self):
        self.assertEqual(normalize_ingredient_line("3 cloves of garlic, minced"), "garlic")
        self.assertEqual(normalize_ingredient_line("1 cup shredded mozzarella cheese"), "cheese")
        self.assertEqual(normalize_ingredient_line("2 tablespoons all-purpose flour"), "flour")
        self.assertEqual(normalize_ingredient_line("freshly ground black pepper"), "pepper")
        self.assertEqual(normalize_ingredient_line("cracked black pepper"), "pepper")
        self.assertEqual(normalize_ingredient_line("boneless skinless chicken breasts"), "chicken")

    def test_known_legacy_substring_collisions_do_not_match(self):
        self.assertEqual(normalize_ingredient_line("1 pineapple"), "pineapple")
        self.assertEqual(normalize_ingredient_line("1 butternut squash"), "butternut squash")
        self.assertEqual(normalize_ingredient_line("1 pepperoni"), "pepperoni")
        self.assertEqual(normalize_ingredient_line("1 cup watercress"), "watercress")

    def test_approved_conservative_normalizations(self):
        cases = {
            "bay leaves": "bay leaf",
            "cold water": "water",
            "basil leaves": "basil",
            "flat-leaf parsley": "parsley",
            "ginger root": "ginger",
            "garlic paste": "garlic",
            "spring onions": "green onion",
            "purple onion": "red onion",
            "cayenne": "cayenne pepper",
            "cider vinegar": "apple cider vinegar",
            "coarse salt": "salt",
            "cooking oil": "oil",
            "whole milk": "milk",
            "bananas": "banana",
            "almonds": "almond",
            "pecans": "pecan",
            "walnuts": "walnut",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_ingredient_line(source), expected)

    def test_approved_composite_and_family_mappings(self):
        cases = {
            "fresh coriander": "cilantro",
            "ground coriander": "cilantro",
            "coriander seeds": "coriander seeds",
            "chicken stock": "chicken broth",
            "low-sodium chicken broth": "chicken broth",
            "dark soy sauce": "soy sauce",
            "plain Greek yoghurt": "yogurt",
            "nonfat plain greek yogurt": "yogurt",
            "2% lowfat greek yogurt": "yogurt",
            "whole-milk greek-style yogurt": "yogurt",
            "toasted sesame oil": "sesame oil",
            "tomato paste": "tomato sauce",
            "tomato": "tomato",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_ingredient_line(source), expected)
        self.assertEqual(normalize_ingredient_targets("salt & pepper"), ("salt", "pepper"))
        self.assertEqual(normalize_ingredient_targets("coarse salt and freshly ground black pepper"), ("salt", "pepper"))
        self.assertEqual(normalize_ingredient_targets("sauce"), ())

    def test_approved_spice_onion_sugar_and_pepper_taxonomies(self):
        cases = {
            "ground cumin": "cumin",
            "freshly ground cinnamon": "cinnamon",
            "cumin seeds": "cumin seeds",
            "cinnamon sticks": "cinnamon sticks",
            "ground cardamom": "ground cardamom",
            "ground turkey": "ground turkey",
            "garlic powder": "garlic powder",
            "white onion": "onion",
            "red onion": "red onion",
            "green onions": "green onion",
            "light brown sugar": "sugar",
            "red pepper": "red bell pepper",
            "green pepper": "green bell pepper",
            "dried crushed red pepper": "chili",
            "red pepper flakes": "chili",
            "crushed red pepper flakes": "chili",
            "chili flakes": "chili",
            "red chili powder": "chili",
            "red bell pepper": "red bell pepper",
            "green bell pepper": "green bell pepper",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_ingredient_line(source), expected)

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

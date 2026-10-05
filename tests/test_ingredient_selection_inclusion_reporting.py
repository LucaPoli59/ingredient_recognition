import copy
import unittest
import xml.etree.ElementTree as ET

from src.ingredient_selection.inclusion_reporting import render_inclusion_plot


class InclusionReportingTests(unittest.TestCase):
    def setUp(self):
        outcomes = ["included", "uncertain", "below_quality_floor"]
        self.report = {
            "label_count": 3,
            "outcome_counts": {key: 1 for key in outcomes},
            "labels": [
                {"class_index": index, "class_name": 'Tomato & <basil> "leaf"' if index == 0 else str(index),
                 "statistics": {"q": .4 - .1 * index, "q_lower": .2 - .05 * index,
                                "q_upper": .5 - .1 * index, "prevalence": .05 + .1 * index},
                 "decision": {"outcome": outcome}}
                for index, outcome in enumerate(outcomes)
            ],
            "sensitivity_counts": {"0.15": {"included": 2, "uncertain": 1},
                                   "0.20": {key: 1 for key in outcomes},
                                   "0.25": {"included": 1, "below_quality_floor": 2}},
        }

    def test_deterministic_valid_svg_with_complete_scientific_labels(self):
        first = render_inclusion_plot(self.report)
        reversed_rows = copy.deepcopy(self.report)
        reversed_rows["labels"].reverse()
        self.assertEqual(first, render_inclusion_plot(reversed_rows))
        root = ET.fromstring(first)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        text = first.decode()
        for phrase in ["seed 42", "single run", "32, 34, 36, 38 and 40", "95%", "Q = 0.20",
                       "paired Q", "not the final P6 vocabulary", "Fixed floor sensitivity"]:
            self.assertIn(phrase, text)
        self.assertEqual(text.count("data-class-index="), 3)

    def test_label_text_is_escaped_and_recovers_in_xml_title(self):
        result = render_inclusion_plot(self.report)
        self.assertIn(b"Tomato &amp; &lt;basil&gt; &quot;leaf&quot;", result)
        root = ET.fromstring(result)
        titles = [element.text for element in root.iter("{http://www.w3.org/2000/svg}title")]
        self.assertTrue(any('Tomato & <basil> "leaf"' in title for title in titles))

    def test_bad_counts_and_intervals_cannot_be_drawn(self):
        invalid = copy.deepcopy(self.report)
        invalid["label_count"] = 165
        with self.assertRaises(ValueError):
            render_inclusion_plot(invalid)
        invalid = copy.deepcopy(self.report)
        invalid["labels"][0]["statistics"]["q_upper"] = .1
        with self.assertRaises(ValueError):
            render_inclusion_plot(invalid)


if __name__ == "__main__":
    unittest.main()

"""Pure-standard-library checks for blinded P5 review packets and agreement."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.ingredient_selection.observability import (
    _json_bytes,
    _sample_pairs,
    score_pilot,
)


class ObservabilityReviewTests(unittest.TestCase):
    def test_deterministic_unique_blind_sampling(self):
        records = [
            {"id": index, "image": f"{index}.jpg",
             "ingredients_target": (["alpha"] if index % 2 else ["beta"])}
            for index in range(80)
        ]
        pairs = _sample_pairs(("alpha", "beta"), records)
        self.assertEqual(pairs, _sample_pairs(("alpha", "beta"), list(reversed(records))))
        self.assertEqual(len(pairs), 16)
        self.assertEqual(len({row["image"] for row in pairs}), 16)
        self.assertEqual(sum(row["recipe_label_present"] for row in pairs), 12)

    def test_agreement_and_response_guards(self):
        pairs = [
            {"pair_id": str(index), "ingredient": "alpha", "recipe_label_present": index < 2}
            for index in range(4)
        ]
        packet = {"pairs": pairs}
        packet["packet_hash"] = hashlib.sha256(_json_bytes(packet)).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet_path, a_path, b_path = (root / name for name in ("packet.json", "a.json", "b.json"))
            packet_path.write_bytes(_json_bytes(packet))
            def response(reviewer, name, categories):
                return {"schema_version": 1, "reviewer_id": reviewer,
                        "reviewer_name": name, "packet_hash": packet["packet_hash"],
                        "responses": {str(index): {"category": category}
                                      for index, category in enumerate(categories)}}
            a_path.write_bytes(_json_bytes(response("A", "first", [
                "direct", "direct", "contextual", "not_inferable"])))
            b = response("B", "second", [
                "direct", "contextual", "contextual", "not_inferable"])
            b_path.write_bytes(_json_bytes(b))
            result = score_pilot(packet_path, a_path, b_path)
            self.assertEqual(result["observed_agreement"], 0.75)
            self.assertAlmostEqual(result["cohen_kappa"], 7 / 11)
            self.assertEqual(result["per_label_positive_recipe_pairs"]["alpha"]["agreed_direct"], 1)
            image = root / "image.jpg"
            image.write_bytes(b"original")
            packet["pairs"][0]["image"] = image.name
            packet["pairs"][0]["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
            packet["packet_hash"] = hashlib.sha256(_json_bytes(
                {key: value for key, value in packet.items() if key != "packet_hash"})).hexdigest()
            packet_path.write_bytes(_json_bytes(packet))
            for path in (a_path, b_path):
                payload = json.loads(path.read_text())
                payload["packet_hash"] = packet["packet_hash"]
                path.write_bytes(_json_bytes(payload))
            image.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "image changed"):
                score_pilot(packet_path, a_path, b_path, root)
            del b["responses"]["3"]
            b["packet_hash"] = packet["packet_hash"]
            b_path.write_bytes(_json_bytes(b))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                score_pilot(packet_path, a_path, b_path)
            packet["pairs"][0]["ingredient"] = "tampered"
            packet_path.write_bytes(_json_bytes(packet))
            with self.assertRaisesRegex(ValueError, "modified"):
                score_pilot(packet_path, a_path, b_path)


if __name__ == "__main__":
    unittest.main()

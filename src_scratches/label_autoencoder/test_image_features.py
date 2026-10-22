"""Synthetic, CPU-only checks for feature-cache identity, resume and split isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from src_scratches.label_autoencoder import image_features as features


class TinyBackbone(nn.Module):
    def forward(self, batch):
        return batch.mean(dim=(1, 2, 3), keepdim=False)[:, None].expand(-1, 512).contiguous()


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        store = self.root / "imgs/standard"
        store.mkdir(parents=True)
        for split, count, offset in (("train", 5, 0), ("val", 3, 5)):
            rows = []
            for i in range(count):
                name = f"{split}_{i}.png"
                Image.new("RGB", (30 + i, 40), (20 * (i + offset), 50, 70)).save(store / name)
                rows.append({"image": name, "ingredients_target": ["a"]})
            (self.root / split).mkdir()
            (self.root / split / "labels.json").write_text(json.dumps(rows))
        # A deliberately malformed test split proves it is never read.
        (self.root / "test").mkdir()
        (self.root / "test/labels.json").write_text("DO NOT READ")
        self.weights = self.root / "dummy_weights"
        self.weights.write_bytes(b"test-backbone-identity")
        self.args = features.parse_args([
            "--data-dir", str(self.root), "--metadata", "labels.json", "--weights", str(self.weights),
            "--output", str(self.root / "cache"), "--batch-size", "2", "--chunk-size", "2", "--threads", "1",
        ])

    def test_complete_cache_matches_images_without_cuda(self):
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")):
            manifest = features.extract(self.args, TinyBackbone())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["cross_split_identical_image_hashes"], 0)
        train, val, _ = features.load_features(self.args.output, manifest["identity"]["metadata"])
        self.assertEqual(train.shape, (5, 512))
        self.assertEqual(val.shape, (3, 512))
        self.assertTrue(np.isfinite(train).all())
        self.assertGreater(float(val[0, 0]), float(train[0, 0]))
        with self.assertRaises(FileExistsError):
            features.extract(self.args, TinyBackbone())

    def test_resume_only_computes_uncommitted_rows(self):
        class InterruptingBackbone(TinyBackbone):
            calls = 0
            def forward(self, batch):
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt()
                return super().forward(batch)
        self.args.threads = 2
        with self.assertRaises(KeyboardInterrupt):
            features.extract(self.args, InterruptingBackbone())
        manifest = json.loads((self.args.output / "feature_manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(len(manifest["chunks"]), 1)
        first_hash = manifest["chunks"][0]["features_sha256"]
        self.args.resume = True
        self.args.threads = 1
        manifest = features.extract(self.args, TinyBackbone())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["chunks"][0]["features_sha256"], first_hash)
        self.assertEqual(manifest["sessions"][-1]["positions"], {"train": 2, "val": 0})
        self.assertEqual([session["threads"] for session in manifest["sessions"]], [2, 1])
        # Completed resume validates and returns without using the model.
        features.extract(self.args, InterruptingBackbone())

    def test_legacy_migration_is_explicit_and_rejects_unknown_source(self):
        manifest = features.extract(self.args, TinyBackbone())
        identity = dict(manifest["identity"])
        identity.pop("feature_algorithm")
        identity.update(schema_version=1, threads=2, extractor_sha256=features.LEGACY_EXTRACTOR_SHA256)
        self.assertEqual(features.migrate_identity_v1(identity), manifest["identity"])
        identity["extractor_sha256"] = "unknown-source"
        with self.assertRaisesRegex(ValueError, "Unknown legacy"):
            features.migrate_identity_v1(identity)

    def test_changed_source_or_corrupt_feature_cache_fails(self):
        manifest = features.extract(self.args, TinyBackbone())
        self.args.resume = True
        self.weights.write_bytes(b"different-weights")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            features.extract(self.args, TinyBackbone())
        source = manifest["identity"]["metadata"]
        source["val"]["sha256"] = "different-metadata"
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            features.load_features(self.args.output, source)
        source["val"]["sha256"] = features.sha256(self.root / "val/labels.json")
        chunk = self.args.output / manifest["chunks"][0]["features"]
        chunk.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Corrupt cache"):
            features.load_features(self.args.output, source)

    def test_cross_split_identical_images_fail(self):
        store = self.root / "imgs/standard"
        (store / "val_0.png").write_bytes((store / "train_0.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            features.extract(self.args, TinyBackbone())
        manifest = json.loads((self.args.output / "feature_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")


if __name__ == "__main__":
    unittest.main()

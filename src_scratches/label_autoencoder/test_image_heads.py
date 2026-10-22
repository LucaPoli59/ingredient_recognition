"""Synthetic training/reload tests for image heads, without real data or GPU use."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
import torch
from torch import nn

from src_scratches.label_autoencoder import compare, image_features, image_heads
from src_scratches.label_autoencoder.test_image_features import TinyBackbone


class ImageHeadTests(unittest.TestCase):
    def test_normalization_uses_supplied_train_only_and_handles_constants(self):
        train = np.array([[1, 10, 0], [3, 10, 2]], dtype=np.float32)
        mean, std = image_heads.normalization(train)
        assert_allclose(mean, [2, 10, 1])
        assert_allclose(std, [1, 1e-6, 1])

    def test_decoder_stays_frozen_but_gradient_reaches_code_projection(self):
        torch.manual_seed(1)
        decoder = nn.Sequential(nn.Linear(32, 12), nn.ReLU(), nn.Linear(12, 6))
        head = image_heads.ImageHead("ae32_aligned", np.zeros(5), np.ones(5), 6, decoder, np.zeros(32), np.ones(32))
        before = {k: v.clone() for k, v in head.decoder.state_dict().items()}
        logits, code = head(torch.randn(10, 5))
        nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
        self.assertEqual(code.shape, (10, 32))
        self.assertGreater(float(head.projection.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in head.decoder.parameters()))
        optimizer = torch.optim.SGD([p for p in head.parameters() if p.requires_grad], lr=0.1)
        optimizer.step()
        for key, value in before.items():
            self.assertTrue(torch.equal(value, head.decoder.state_dict()[key]))
        with self.assertRaises(TypeError):
            head(torch.randn(10, 5), torch.ones(10, 6))

    def test_ae_objectives_start_with_identical_parameters(self):
        decoder = nn.Sequential(nn.Linear(32, 12), nn.ReLU(), nn.Linear(12, 6))
        models = []
        for kind in ("ae32_aligned", "ae32_mse"):
            torch.manual_seed(42)
            models.append(image_heads.ImageHead(kind, np.zeros(5), np.ones(5), 6, decoder, np.zeros(32), np.ones(32)))
        for key, value in models[0].state_dict().items():
            self.assertTrue(torch.equal(value, models[1].state_dict()[key]))

    def test_full_three_arm_run_reload_inference_and_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "imgs/standard"
            store.mkdir(parents=True)
            for split, count, offset in (("train", 24, 0), ("val", 12, 24)):
                rows = []
                for i in range(count):
                    image = f"{split}_{i}.png"
                    Image.new("RGB", (30, 40), ((i + offset) * 6, 70, 40)).save(store / image)
                    rows.append({"image": image, "ingredients_target": [f"l{i % 6}", f"l{(i + 2) % 6}"]})
                (root / split).mkdir()
                (root / split / "labels.json").write_text(json.dumps(rows))
            weights = root / "dummy_weights"
            weights.write_bytes(b"synthetic-backbone")
            extraction = image_features.parse_args(["--data-dir", str(root), "--metadata", "labels.json",
                "--weights", str(weights), "--output", str(root / "cache"), "--threads", "1"])
            image_features.extract(extraction, TinyBackbone())
            train_y, val_y, labels, provenance = compare.load_labels(root, "labels.json", "ingredients_target")
            torch.manual_seed(4)
            autoencoder = compare.IngredientAutoencoder(6, 32, 12)
            ae_path = root / "ae32.pt"
            saved = {"input_dim": 6, "latent_dim": 32, "hidden_dim": 12, "labels": labels,
                     "state_dict": autoencoder.state_dict(), "train_sha256": provenance["train"]["sha256"]}
            torch.save(saved, ae_path)
            args = image_heads.parse_args(["--data-dir", str(root), "--metadata", "labels.json",
                "--features", str(root / "cache"), "--autoencoder", str(ae_path),
                "--output", str(root / "result"), "--threads", "1", "--epochs", "4", "--batch-size", "8"])
            with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")):
                result = image_heads.run(args)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["arms"]), 4)
            _, val_x, _ = image_features.load_features(root / "cache", provenance)
            for arm in result["arms"][1:]:
                path = args.output / f"{arm['method']}.pt"
                model, checkpoint = image_heads.load_head(path)
                probabilities, code = image_heads.predict(model, val_x)
                metrics, _ = compare.evaluate(val_y, probabilities, train_y.sum(axis=0), labels)
                self.assertAlmostEqual(metrics["macro_ap"], arm["metrics"]["macro_ap"])
                self.assertEqual(checkpoint["labels"], labels)
                history = np.genfromtxt(args.output / f"{arm['method']}_history.csv", delimiter=",", names=True)
                self.assertAlmostEqual(float(min(history["val_bce"])), checkpoint["best_val_bce"])
                if arm["method"] != "direct":
                    self.assertEqual(code.shape, (len(val_y), 32))
                    for key, value in autoencoder.decoder.state_dict().items():
                        self.assertTrue(torch.equal(value, model.decoder.state_dict()[key]))
                with patch.object(image_heads, "load_backbone", return_value=TinyBackbone()):
                    prediction = image_heads.predict_image(store / "val_0.png", path, weights)
                assert_allclose(prediction["probabilities"], probabilities[0], atol=1e-6)
                self.assertEqual(prediction["labels"], labels)
                self.assertEqual(prediction["code"] is None, arm["method"] == "direct")
            with self.assertRaises(FileExistsError):
                image_heads.run(args)
            args.output = root / "wrong_checkpoint_run"
            saved["train_sha256"] = "wrong"
            torch.save(saved, ae_path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                image_heads.run(args)


if __name__ == "__main__":
    unittest.main()

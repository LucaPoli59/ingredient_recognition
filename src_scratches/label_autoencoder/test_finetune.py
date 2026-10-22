"""Bounded synthetic CPU tests; no real-image training or CUDA initialization."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from src_scratches.label_autoencoder import compare, finetune, finetune_data, image_models
from src.data_processing.experimental_transforms import ExperimentalFitPad224


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 512)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, pixels):
        return self.bn(self.projection(pixels.mean(dim=(2, 3))))


def fixture(root):
    store = root / "imgs/standard"
    store.mkdir(parents=True)
    inventories = {}
    for split, count, offset in (("train", 12, 0), ("val", 6, 12)):
        rows, inventory = [], []
        for i in range(count):
            name = f"{split}_{i}.png"
            Image.new("RGB", (12, 16), ((i + offset) * 11, 50 + i, 90)).save(store / name)
            inventory.append({"image": name, "sha256": compare.sha256(store / name)})
            rows.append({"image": name, "ingredients_target": [f"l{i % 6}", f"l{(i + 1) % 6}"]})
        (root / split).mkdir()
        compare.write_json(root / split / "labels.json", rows)
        inventories[split] = inventory
    # Malformed test metadata must never be opened.
    (root / "test").mkdir()
    (root / "test/labels.json").write_text("not valid json")
    train, val, labels, provenance = compare.load_labels(root, "labels.json", "ingredients_target")
    torch.manual_seed(5)
    teacher = compare.IngredientAutoencoder(6, 32, 8)
    ae_path = root / "ae.pt"
    torch.save({"input_dim": 6, "hidden_dim": 8, "latent_dim": 32, "labels": labels,
                "state_dict": teacher.state_dict(), "train_sha256": provenance["train"]["sha256"]}, ae_path)
    mean, std = image_models.normalization(image_models.teacher_codes(teacher, train))
    weights = root / "backbone.pt"
    weights.write_bytes(b"synthetic backbone; never used by preparation")
    transform = ExperimentalFitPad224()
    cache = root / "cache"
    cache.mkdir()
    chunks = []
    for split, records in inventories.items():
        path = cache / f"{split}.json"
        compare.write_json(path, records)
        chunks.append({"split": split, "start": 0, "stop": len(records), "images": path.name,
                       "images_sha256": compare.sha256(path)})
    compare.write_json(cache / "feature_manifest.json", {"status": "complete", "cross_split_identical_image_hashes": 0,
        "chunks": chunks, "identity": {"metadata": provenance, "weights_sha256": compare.sha256(weights),
        "transform": transform.to_config(),
        "transform_source_sha256": compare.sha256(compare.ROOT / "src/data_processing/experimental_transforms.py")}})
    contract = {"labels": labels, "decoder_hidden_dim": 8,
                "train_sha256": provenance["train"]["sha256"], "val_sha256": provenance["val"]["sha256"],
                "autoencoder_sha256": compare.sha256(ae_path), "backbone_weights_sha256": compare.sha256(weights),
                "feature_manifest_sha256": compare.sha256(cache / "feature_manifest.json"), "transform": transform.to_config()}
    run = root / "source"
    run.mkdir()
    arms = []
    for name in ("direct", "ae32_aligned"):
        head = image_models.ImageHead(name, np.zeros(512), np.ones(512), 6, teacher.decoder, mean, std)
        torch.save({"kind": name, "state_dict": head.state_dict(), "feature_dim": 512, "label_count": 6,
                    "latent_dim": 32, "decoder_hidden_dim": 8, "labels": labels, "contract": contract}, run / f"{name}.pt")
        arms.append({"method": name, "checkpoint_sha256": compare.sha256(run / f"{name}.pt")})
    compare.write_json(run / "summary.json", {"status": "complete", "arms": arms})
    compare.write_json(run / "manifest.json", {"contract": contract})
    config = json.loads((Path(__file__).parent / "configs/backbone_finetune.json").read_text())
    config.update(data_dir=str(root), metadata="labels.json", expected_labels=6, source_run=str(run),
                  feature_cache=str(cache), autoencoder=str(ae_path), weights=str(weights),
                  workers=0, threads=1, epochs=2, batch_size=6)
    path = root / "config.json"
    compare.write_json(path, config)
    return path


class FineTuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_import_preserves_cuda_visibility_without_initialization(self):
        result = subprocess.run([sys.executable, "-c",
            "import os, torch; from src_scratches.label_autoencoder import finetune; "
            "assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'; assert not torch.cuda.is_initialized()"],
            cwd=compare.ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preparation_does_not_read_image_bytes_train_or_touch_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            with patch.object(finetune, "execute", side_effect=AssertionError("training")), \
                 patch.object(compare, "gpu_snapshot", side_effect=AssertionError("GPU probe")), \
                 patch.object(Image, "open", side_effect=AssertionError("image read")), \
                 patch.object(torch.cuda, "init", side_effect=AssertionError("CUDA")):
                plan = finetune.main(["--config", str(config), "--plan", str(root / "plan.json")])
            self.assertFalse(plan["training_started"])
            self.assertEqual(plan["data"]["train"]["rows"], 12)
            self.assertFalse(plan["test_split_accessed"])
            self.assertFalse((root / "results").exists())
            with self.assertRaises(FileExistsError):
                finetune.main(["--config", str(config), "--plan", str(root / "plan.json")])
            with self.assertRaises(SystemExit):
                finetune.main(["--config", str(config), "--output", str(root / "results")])

    def test_checkpoint_metadata_and_image_inventory_tampering_are_rejected(self):
        for target in ("source/direct.pt", "train/labels.json", "cache/train.json"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = fixture(root)
                path = root / target
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaises(ValueError):
                    finetune_data.prepare(config)

    def test_dataset_checks_actual_image_bytes_and_returns_targets_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = finetune_data.prepare(fixture(root))
            dataset = finetune_data.RecipeImages(root / "imgs/standard", prepared["rows"]["val"],
                prepared["val_y"], np.zeros((6, 32), dtype=np.float32), ExperimentalFitPad224())
            image, labels, code = dataset[0]
            self.assertEqual(image.shape, (3, 224, 224))
            self.assertEqual(labels.shape, (6,))
            self.assertEqual(code.shape, (32,))
            path = root / "imgs/standard" / prepared["rows"]["val"][0]["image"]
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Image bytes changed"):
                dataset[0]

    def test_shared_initialization_gradients_optimizer_groups_and_decoder_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = finetune_data.prepare(fixture(Path(directory)))
            torch.manual_seed(10)
            backbone = TinyBackbone()
            models = [finetune.ImageModel(copy.deepcopy(backbone), copy.deepcopy(prepared["heads"]["ae32_aligned"]), arm)
                      for arm in finetune_data.ARMS[1:]]
            for key, value in models[0].state_dict().items():
                self.assertTrue(torch.equal(value, models[1].state_dict()[key]))
            for model in models:
                before = finetune.cpu_state(model)
                model.train()
                optimizer = finetune.optimizer_for(model, prepared["config"])
                logits, code = model(torch.randn(6, 3, 12, 16))
                # BCE alone must reach backbone through the frozen decoder.
                nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
                self.assertGreater(float(model.backbone.projection.weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.head.projection.weight.grad.abs().sum()), 0)
                decoder_ids = {id(p) for p in model.head.decoder.parameters()}
                optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(optimizer_ids), len(set(optimizer_ids)))
                expected = model.arm == "ae32_trainable_decoder"
                self.assertEqual(bool(decoder_ids.intersection(optimizer_ids)), expected)
                optimizer.step()
                changes = [not torch.equal(value, before["head.decoder." + key]) for key, value in model.head.decoder.state_dict().items()]
                self.assertEqual(any(changes), expected)
                self.assertFalse(torch.equal(model.backbone.projection.weight, before["backbone.projection.weight"]))
                self.assertFalse(torch.equal(model.backbone.bn.running_mean, before["backbone.bn.running_mean"]))
                with self.assertRaises(TypeError):
                    model(torch.randn(6, 3, 12, 16), torch.ones(6, 6))

    def test_tiny_training_all_arms_checkpoint_reload_image_only_and_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = finetune_data.prepare(fixture(root))
            config = prepared["config"]
            head = prepared["heads"]["ae32_aligned"]
            codes = {s: (image_models.teacher_codes(prepared["teacher"], prepared[s + "_y"]) - head.latent_mean.numpy())
                     / head.latent_std.numpy() for s in ("train", "val")}
            teacher_before = finetune.cpu_state(prepared["teacher"])
            output = root / "result"
            output.mkdir()
            for arm in finetune_data.ARMS:
                torch.manual_seed(config["seed"])
                source = "direct" if arm == "direct" else "ae32_aligned"
                model = finetune.ImageModel(TinyBackbone(), copy.deepcopy(prepared["heads"][source]), arm)
                loaders = {s: DataLoader(finetune_data.RecipeImages(root / "imgs/standard", prepared["rows"][s],
                    prepared[s + "_y"], codes[s], ExperimentalFitPad224()), batch_size=6, shuffle=s == "train",
                    generator=torch.Generator().manual_seed(42)) for s in ("train", "val")}
                with patch.object(torch.cuda, "init", side_effect=AssertionError("CUDA")):
                    result = finetune.fit_arm(model, loaders["train"], loaders["val"], config, output,
                        prepared["specifications"][source], prepared["plan"]["contract"], prepared["val_y"], codes["val"],
                        prepared["train_y"].sum(axis=0).astype(int), torch.device("cpu"))
                    restored, saved = finetune.load_model(output / f"{arm}.pt", backbone_factory=TinyBackbone)
                    pixels = next(iter(loaders["val"]))[0]
                    with torch.inference_mode():
                        assert_allclose(model(pixels)[0].numpy(), restored(pixels)[0].numpy(), rtol=0, atol=0)
                    prediction = finetune.predict_image(root / "imgs/standard/val_0.png", output / f"{arm}.pt", TinyBackbone)
                self.assertEqual(prediction["probabilities"].shape, (6,))
                self.assertEqual(prediction["code"] is None, arm == "direct")
                self.assertEqual(saved["best_epoch"], result["best_epoch"])
                self.assertGreaterEqual(result["metrics"]["micro_f1_val_exact"] + 1e-12,
                                        result["metrics"]["micro_f1_val_tuned"])
                if arm == "ae32_frozen_decoder":
                    self.assertEqual(result["decoder_retention"]["parameter_relative_l2_change"], 0)
                    self.assertEqual(result["decoder_retention"]["before"], result["decoder_retention"]["after"])
                for key, value in prepared["teacher"].state_dict().items():
                    self.assertTrue(torch.equal(value, teacher_before[key]))
            self.assertFalse(torch.cuda.is_initialized())

    def test_epoch_zero_remains_eligible_and_gpu_gate_precedes_cuda_use(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
             patch.object(compare, "gpu_snapshot", return_value={"samples": [{"free_mib": 100, "utilization_pct": 90}] * 3}), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried too early")):
            with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                finetune.cuda_device()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = finetune_data.prepare(fixture(root))
            model = finetune.ImageModel(TinyBackbone(), prepared["heads"]["direct"], "direct")
            scores = np.full_like(prepared["val_y"], 0.1)
            good = ({"bce": 0.1, "latent_mse": 0.0, "objective": 0.1}, scores, None)
            bad = ({"bce": 0.2, "latent_mse": 0.0, "objective": 0.2}, scores, None)
            config = {**prepared["config"], "epochs": 3, "patience": 1}
            with patch.object(finetune, "epoch_pass", side_effect=[good, bad, bad, good]):
                result = finetune.fit_arm(model, [None], [None], config, root, prepared["specifications"]["direct"],
                    prepared["plan"]["contract"], prepared["val_y"], np.zeros((6, 32), dtype=np.float32),
                    prepared["train_y"].sum(axis=0).astype(int), torch.device("cpu"))
            self.assertEqual(result["best_epoch"], 0)
            self.assertEqual(result["epochs_ran"], 1)


    def test_sequential_launcher_with_synthetic_cpu_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = fixture(root)
            original_fit = finetune.fit_arm
            initial_states = {}

            def recorded_fit(model, *args, **kwargs):
                initial_states[model.arm] = finetune.cpu_state(model)
                return original_fit(model, *args, **kwargs)

            with patch.object(finetune, "cuda_device", return_value=(torch.device("cpu"), {"synthetic": True})), \
                 patch.object(image_models, "load_backbone", side_effect=lambda _: TinyBackbone()), \
                 patch.object(finetune, "fit_arm", side_effect=recorded_fit), \
                 patch.object(torch.cuda, "empty_cache"), \
                 patch.object(torch.cuda, "init", side_effect=AssertionError("CUDA")):
                result = finetune.main(["--config", str(config_path), "--execute", "--output", str(root / "run")])
            self.assertEqual(result["status"], "complete")
            self.assertEqual([arm["method"] for arm in result["arms"]], list(finetune_data.ARMS))
            for key, value in initial_states["ae32_frozen_decoder"].items():
                self.assertTrue(torch.equal(value, initial_states["ae32_trainable_decoder"][key]), key)
            manifest = json.loads((root / "run/manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["test_split_accessed"])
            self.assertTrue((root / "run/report.md").is_file())
            with self.assertRaises(FileExistsError):
                finetune.execute(finetune_data.prepare(config_path), root / "run")
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()

"""Synthetic CPU/no-network diagnostics, including the production Dash callback.

Weights-none fixtures prove interface behavior, not pretrained localization quality.
Dash registration is isolated; this is not browser/CUDA acceptance (Phase 5.5).
"""

import codecs
import copy
import importlib.util
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dash
import jsonpickle
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import v2

from src.commons.exp_config import ExpConfig
from src.commons.visualizations import gradcam, feature_factorization
from src.dashboards.runtime import prepare_visualization_image
from src.data_processing.experimental_transforms import ExperimentalFitPad224, MEAN, STD
from src.data_processing.labels_encoders import MultiLabelBinarizer
from src.data_processing.transformations import transformations_wrapper
from src.ingredient_selection.runtime import PROJECTION_ID, resolve_projection
from src.lightning.experimental_lgn import ExperimentalLGNM
from src.lightning.custom_callbacks import LightModelCheckpoint
from src.models.experimental_dica_net import DICANetSExperiment
from src.models.resnet import Resnet18
from src.training.experimental_runtime import load_model_for_experiment


def _image():
    return Image.fromarray(np.random.default_rng(543).integers(0, 256, (81, 137, 3), dtype=np.uint8))


def _encoder(names):
    encoder = MultiLabelBinarizer(classes=list(names))
    encoder.fit()
    return encoder


def _load_page():
    # Execute the real production functions without registering a second Dash app,
    # scanning real experiments, or resolving a running app's assets.
    from src.dashboards import _commons
    path = Path(__file__).resolve().parents[1] / "src/dashboards/dash/pages/model_visualization.py"
    spec = importlib.util.spec_from_file_location("phase543_visualization_test_page", path)
    page = importlib.util.module_from_spec(spec)
    with patch.object(dash, "register_page"), \
            patch.object(dash, "callback", side_effect=lambda *a, **kw: lambda fn: fn), \
            patch.object(_commons, "recursive_listdir", return_value=[]), \
            patch.object(_commons, "dash_get_asset_url", return_value="/assets/synthetic.png"):
        spec.loader.exec_module(page)
    return page


class DICANetDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.page = _load_page()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        torch.manual_seed(543)
        self.addCleanup(self.rng.__exit__, None, None, None)
        for target in ("torch.hub.download_url_to_file",
                       "torchvision.models.EfficientNet_V2_S_Weights.get_state_dict"):
            mocked = patch(target, side_effect=AssertionError("network forbidden"))
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_real_gradcam_full_frozen_uses_complete_readout_and_input_gradients(self):
        for adaptation, width in (("full", 165), ("frozen_encoder", 59)):
            with self.subTest(adaptation=adaptation):
                model = DICANetSExperiment(width, adaptation=adaptation, initialize_pretrained=False)
                model.train()
                if adaptation == "frozen_encoder":
                    self.assertTrue(all(not m.training for m in model.model.features.modules()))
                model.eval()
                tensor, display = prepare_visualization_image(model.transform_plain, _image())
                with torch.no_grad():
                    expected = model(tensor[None])
                buffers = {n: b.clone() for n, b in model.model.features.named_buffers()}
                captured, readouts = {}, []
                def capture_input(_module, inputs):
                    captured["input"] = inputs[0]
                    inputs[0].retain_grad()
                def capture_features(_module, _inputs, output):
                    captured["features"] = output
                    output.retain_grad()
                handles = [model.register_forward_pre_hook(capture_input),
                           model.conv_target_layer.register_forward_hook(capture_features),
                           model.classifier_target_layer.register_forward_hook(
                               lambda _m, ins, out: readouts.append((tuple(x.shape for x in ins), out)))]
                hook_count = len(model.conv_target_layer._forward_hooks)
                try:
                    # Both automatic and explicit targets; even an enclosing no_grad
                    # must not disable the diagnostic's input-gradient path.
                    for targets in (None, [3]):
                        with torch.no_grad():
                            images, masks, actual_targets, outputs = gradcam(
                                model, model.conv_target_layer, tensor, targets=targets, imgs_show=display)
                        torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
                        self.assertEqual(list(actual_targets), [int(expected.argmax())] if targets is None else targets)
                        self.assertEqual(images.shape, (1, 224, 224, 3))
                        self.assertEqual(masks.shape, (1, 224, 224))
                        self.assertTrue(np.isfinite(images).all() and np.isfinite(masks).all())
                        self.assertEqual(tuple(captured["features"].shape), (1, 1280, 7, 7))
                        for tensor_with_grad in captured.values():
                            self.assertIsNotNone(tensor_with_grad.grad)
                            self.assertTrue(torch.isfinite(tensor_with_grad.grad).all())
                            self.assertGreater(tensor_with_grad.grad.abs().sum().item(), 0)
                        self.assertEqual(readouts[-1][0], (torch.Size([1, 160, 14, 14]), torch.Size([1, 1280, 7, 7])))
                        self.assertIs(readouts[-1][1], outputs)
                        for parameter in (model.model.readout.ingredient_queries,
                                          model.model.readout.pooled_context_readout.weight):
                            self.assertGreater(parameter.grad.abs().sum().item(), 0)
                        self.assertEqual(len(model.conv_target_layer._forward_hooks), hook_count)
                    for name, value in model.model.features.named_buffers():
                        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
                    if adaptation == "frozen_encoder":
                        self.assertTrue(all(p.grad is None for p in model.model.features.parameters()))
                    else:
                        self.assertGreater(next(model.model.features.parameters()).grad.abs().sum().item(), 0)
                finally:
                    for handle in handles:
                        handle.remove()

    def test_factorization_explicitly_rejects_context_only_substitution(self):
        model = DICANetSExperiment(59, initialize_pretrained=False).eval()
        self.assertFalse(model.supports_feature_factorization)
        self.assertIs(model.classifier_target_layer, model.model.readout)
        with self.assertRaisesRegex(NotImplementedError, "both feature scales"):
            _ = model.factorization_classifier_layer
        before = len(model.conv_target_layer._forward_hooks)
        with self.assertRaisesRegex(NotImplementedError, "complete classifier"):
            feature_factorization(model, model.conv_target_layer, model.model.readout.pooled_context_readout,
                                  torch.zeros(1, 3, 224, 224))
        self.assertEqual(len(model.conv_target_layer._forward_hooks), before)

    def _callback(self, model, encoder, folder, target=None):
        image_path, cache_path = Path(folder) / "image.png", Path(folder) / "model.pt"
        _image().save(image_path)
        torch.save(model, cache_path)
        transform = model.transform_plain
        if isinstance(transform, list):
            # The DataModule wraps legacy operation lists with saved statistics.
            # These are declared synthetic statistics, not inferred recipe data.
            transform = transformations_wrapper(transform, MEAN, STD)
        serialized_transform = codecs.encode(pickle.dumps(transform), "base64").decode()
        with patch.object(self.page, "MODEL_CACHE_PATH", str(cache_path)):
            return self.page.make_inference(1, target, {"curr": 0}, [{"img": str(image_path)}], .5,
                                            serialized_transform, jsonpickle.encode(encoder), device="cpu")

    def test_production_callback_offline_restored_full_selected_order_and_factorization_skip(self):
        projection = resolve_projection(PROJECTION_ID)
        for selected, adaptation in ((False, "full"), (True, "frozen_encoder")):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as folder:
                names = list(projection.class_order if selected else projection.base_class_order)
                encoder = _encoder(names)
                data = SimpleNamespace(label_encoder=encoder, get_num_classes=lambda: len(names),
                                       projection_config=projection.to_config() if selected else None)
                adapter = DICANetSExperiment(len(names), adaptation=adaptation, initialize_pretrained=False)
                module = ExperimentalLGNM(model=adapter, output_class_order=names, batch_size=128,
                                          physical_batch_size=8, weighted_loss=False, lr=1e-4,
                                          optimizer=torch.optim.AdamW, loss_fn=torch.nn.BCEWithLogitsLoss)
                module.startup_model(data)
                config = ExpConfig(hp_=copy.deepcopy(dict(module.hparams)), lb_classes=names,
                                   lb_encode_map={name: i for i, name in enumerate(names)}, lb_fitted=True,
                                   dm_ingredient_projection=data.projection_config)
                checkpoint = {"state_dict": module.state_dict(),
                              "hyper_parameters": copy.deepcopy(dict(module.hparams))}
                module.on_save_checkpoint(checkpoint)
                LightModelCheckpoint().on_save_checkpoint(None, module, checkpoint)
                checkpoint_path = Path(folder) / "light.ckpt"
                torch.save(checkpoint, checkpoint_path)
                restored = load_model_for_experiment(config, checkpoint_path=checkpoint_path).eval()
                self.assertEqual(restored.output_class_order, names)
                self.assertEqual(restored._ingredient_projection, data.projection_config)
                self.assertFalse(restored.model.supports_feature_factorization)
                tensor, _ = prepare_visualization_image(restored.transform_plain, _image())
                with torch.no_grad():
                    expected = torch.sigmoid(restored(tensor[None]))[0].numpy()
                    target_rows = torch.zeros(1, len(names))
                    predictions = restored.predict_step((tensor[None], target_rows), 0)
                    self.assertEqual(len(predictions), 3)
                    torch.testing.assert_close(predictions[0], tensor[None])
                    torch.testing.assert_close(predictions[1], restored(tensor[None]))
                    self.assertIs(predictions[2], target_rows)
                with patch.object(self.page, "feature_factorization", side_effect=AssertionError("must skip")) as factor:
                    result = self._callback(restored.model, encoder, folder, names[3] if selected else None)
                factor.assert_not_called()
                self.assertEqual(result[-1], "info", result[-2])
                self.assertEqual(result[4:6], (False, True))
                self.assertIn("unavailable", result[-2])
                self.assertEqual(len(result[0].data), 1)
                self.assertEqual(len(result[1].data), 0)  # Replaces any stale previous model's factor plot.
                self.assertIn("both feature scales", result[1].layout.annotations[0].text.replace("<br>", " "))
                indices = np.argsort(expected)[::-1][:20]
                self.assertEqual([row["Ingredients"] for row in result[3]], [names[i] for i in indices])
                np.testing.assert_allclose([row["Confidence"] for row in result[3]], expected[indices], atol=6e-7)
                expected_target = names[3] if selected else names[int(expected.argmax())]
                self.assertIn(expected_target, result[0].layout.annotations[-1].text)

    def test_production_callback_supported_legacy_factorization_and_unrelated_failure(self):
        model = Resnet18(num_classes=7, pretrained=False).eval()
        encoder = _encoder([f"label-{i}" for i in range(7)])
        self.assertTrue(model.supports_feature_factorization)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(self.page, "feature_factorization", wraps=feature_factorization) as factor:
                result = self._callback(model, encoder, folder)
            self.assertEqual(result[-1], "success", result[-2])
            factor.assert_called_once()
            self.assertEqual(len(result[1].data), 1)
            self.assertEqual(len(result[3]), 7)
            with patch.object(self.page, "feature_factorization", side_effect=RuntimeError("unexpected factor error")), \
                    self.assertLogs(self.page.logger, level="ERROR") as logs:
                failed = self._callback(model, encoder, folder)
            self.assertEqual(failed[-1], "danger")
            self.assertTrue(all(item is dash.no_update for item in failed[:4]))
            self.assertIn("unexpected factor error", "\n".join(logs.output))

    def test_diagnostic_hooks_released_and_index_errors_not_swallowed(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 6, 1), torch.nn.ReLU(),
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(6, 3)).eval()
        image = torch.rand(1, 3, 16, 16)
        for method in ("cam", "factor"):
            with self.subTest(method=method), patch.object(model[-1], "forward", side_effect=IndexError("sentinel")):
                before = (len(model[0]._forward_hooks), len(model[0]._backward_hooks))
                with self.assertRaisesRegex(IndexError, "sentinel"):
                    if method == "cam":
                        gradcam(model, model[0], image, targets=[0])
                    else:
                        feature_factorization(model, model[0], model[-1], image, n_components=2)
                self.assertEqual((len(model[0]._forward_hooks), len(model[0]._backward_hooks)), before)


class VisualizationTransformTests(unittest.TestCase):
    def test_experimental_non_square_rgb_grayscale_rgba_exact_geometry(self):
        transform = ExperimentalFitPad224()
        for mode in ("RGB", "L", "RGBA"):
            with self.subTest(mode=mode):
                image = _image().convert(mode)
                tensor, display = prepare_visualization_image(transform, image)
                torch.testing.assert_close(tensor, transform(image), rtol=0, atol=0)
                self.assertEqual(tuple(display.shape), (3, 224, 224))
                torch.testing.assert_close(display[:, 0, 0], torch.tensor(MEAN))
                torch.testing.assert_close((display - torch.tensor(MEAN)[:, None, None]) /
                                           torch.tensor(STD)[:, None, None], tensor)

    def test_legacy_v1_v2_composes_and_stochastic_transform_run_once(self):
        class CountFlip:
            def __init__(self):
                self.calls = 0
            def __call__(self, tensor):
                self.calls += 1
                return tensor.flip(-1) if self.calls % 2 else tensor
        for api in (transforms, v2):
            with self.subTest(api=api):
                counter = CountFlip()
                to_tensor = transforms.ToTensor() if api is transforms else v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
                transform = api.Compose([to_tensor, counter, api.Normalize(MEAN, STD)])
                tensor, display = prepare_visualization_image(transform, _image())
                self.assertEqual(counter.calls, 1)
                torch.testing.assert_close(display, transforms.ToTensor()(_image()).flip(-1))
                torch.testing.assert_close((display - torch.tensor(MEAN)[:, None, None]) /
                                           torch.tensor(STD)[:, None, None], tensor)

    def test_unknown_normalization_and_invalid_outputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "known final RGB normalization"):
            prepare_visualization_image(v2.ToImage(), _image())
        for invalid in (torch.zeros(3, 4, 4, dtype=torch.uint8), torch.full((3, 4, 4), float("nan")),
                        torch.zeros(4, 4), torch.zeros(4, 4, 4)):
            transform = ExperimentalFitPad224()
            with self.subTest(shape=invalid.shape), patch.object(transform, "forward", return_value=invalid), \
                    self.assertRaisesRegex(ValueError, "finite floating RGB"):
                prepare_visualization_image(transform, _image())


if __name__ == "__main__":
    unittest.main()

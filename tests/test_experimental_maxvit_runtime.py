"""Real MaxViT CPU runtime checks with random weights and synthetic images only.

No test downloads weights or reads food metadata/images. Save payloads exercise
the production full/light hooks; shared Lightning Trainer mechanics are tested
separately in test_experimental_lightning.py.
"""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torchvision.models import MaxVit_T_Weights

from src.commons.exp_config import ExpConfig
from src.commons.visualizations import feature_factorization, gradcam
from src.data_processing.labels_encoders import MultiLabelBinarizer
from src.ingredient_selection.runtime import PROJECTION_ID, resolve_projection
from src.lightning.custom_callbacks import FullModelCheckpoint, LightModelCheckpoint
from src.lightning.experimental_lgn import (
    ExperimentalLGNM, BATCH_CHECKPOINT_KEY, ORDER_CHECKPOINT_KEY,
    TRAINING_CHECKPOINT_KEY,
)
from src.models.experimental_contract import CHECKPOINT_CONTRACT_KEY
from src.models.experimental_maxvit import MaxViTTExperiment
from src.training.experimental_runtime import load_model_for_experiment


def _fixture(selected=False, frozen=False, weighted=False):
    """Registered vocabulary resource is read; no split metadata is accessed."""
    projection = resolve_projection(PROJECTION_ID)
    names = list(projection.class_order if selected else projection.base_class_order)
    encoder = MultiLabelBinarizer(classes=names)
    encoder.fit()
    data = SimpleNamespace(
        label_encoder=encoder,
        classes_weights=torch.linspace(1., 2., len(names)),
        get_num_classes=lambda: len(names),
        projection_config=projection.to_config() if selected else None,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(537)
        model = MaxViTTExperiment(
            num_classes=len(names), initialize_pretrained=False,
            adaptation="frozen_encoder" if frozen else "full",
        )
    module = ExperimentalLGNM(
        model=model, lr=1e-4, batch_size=128, physical_batch_size=8,
        optimizer=torch.optim.AdamW, loss_fn=torch.nn.BCEWithLogitsLoss,
        weighted_loss=weighted, output_class_order=names,
    )
    module.startup_model(data)
    config = ExpConfig(
        hp_=copy.deepcopy(dict(module.hparams)),
        lb_classes=names, lb_encode_map={name: i for i, name in enumerate(names)},
        lb_fitted=True, dm_ingredient_projection=data.projection_config,
    )
    return module, config, data


def _checkpoint(module, variant):
    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": copy.deepcopy(dict(module.hparams)),
        "datamodule_hyper_parameters": {
            "ingredient_projection": copy.deepcopy(module._ingredient_projection),
        },
    }
    module.on_save_checkpoint(checkpoint)
    if variant == "full":
        FullModelCheckpoint().on_save_checkpoint(SimpleNamespace(hparams={}), module, checkpoint)
    else:
        LightModelCheckpoint().on_save_checkpoint(None, module, checkpoint)
    return checkpoint


def _image():
    return torch.linspace(0., 1., 3 * 224 * 224).reshape(1, 3, 224, 224)


class MaxViTRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_full_default_and_light_selected_restore_complete_offline_state(self):
        for variant, selected, frozen, weighted in (
                ("full", False, False, False), ("light", True, True, True)):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
                module, config, data = _fixture(selected, frozen, weighted)
                # Nondefault learned values ensure a fresh random construction
                # cannot accidentally look like a successful state restoration.
                with torch.no_grad():
                    module.model.classifier_target_layer.bias.add_(.137)
                    batch_norm = next(m for m in module.model.modules() if isinstance(m, torch.nn.BatchNorm2d))
                    batch_norm.running_mean.add_(.25)
                    batch_norm.num_batches_tracked.add_(7)
                module.eval()
                with torch.no_grad():
                    expected_logits = module(_image())
                config_path = Path(folder) / "config.json"
                config.save_to_file(config_path)
                config = ExpConfig.load_from_file(config_path)
                path = Path(folder) / "model.ckpt"
                payload = _checkpoint(module, variant)
                for key in (CHECKPOINT_CONTRACT_KEY, BATCH_CHECKPOINT_KEY,
                            ORDER_CHECKPOINT_KEY, TRAINING_CHECKPOINT_KEY):
                    self.assertIn(key, payload)
                self.assertEqual("hyper_parameters" in payload, variant == "full")
                torch.save(payload, path)
                # Both the enum path and the underlying downloader are disabled.
                # Complete saved state must be sufficient for reconstruction.
                with mock.patch.object(MaxVit_T_Weights, "get_state_dict", side_effect=AssertionError("network forbidden")) as weights, \
                        mock.patch("torch.hub.download_url_to_file", side_effect=AssertionError("network forbidden")) as download:
                    restored = load_model_for_experiment(config, checkpoint_path=path)
                weights.assert_not_called()
                download.assert_not_called()
                self.assertFalse(restored.model.initialize_pretrained)
                restored.startup_model(data)
                restored.eval()
                self.assertEqual(restored.output_class_order, list(data.label_encoder.classes))
                self.assertEqual(restored._ingredient_projection, data.projection_config)
                self.assertEqual(restored.exact_batch_plan, module.exact_batch_plan)
                self.assertEqual(restored.grad_accum, 16)
                self.assertEqual(restored.model.experimental_contract, module.model.experimental_contract)
                self.assertEqual(restored.model.to_config()["experimental_contract"]["weights_enum"],
                                 "MaxVit_T_Weights.IMAGENET1K_V1")
                self.assertEqual(set(restored.state_dict()), set(module.state_dict()))
                for name, value in module.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
                with torch.no_grad():
                    torch.testing.assert_close(restored(_image()), expected_logits, rtol=0, atol=0)

    def test_saved_relative_position_buffer_and_complete_head_are_required(self):
        module, _, _ = _fixture(selected=True, frozen=True)
        original = _checkpoint(module, "light")
        state = original["state_dict"]
        relative_key = next(key for key in state if key.endswith("relative_position_index"))
        head_key = next(key for key in state if key.endswith("classifier.2.weight"))
        for key, damage in ((relative_key, "missing"), (relative_key, "shape"),
                            (relative_key, "values"), (relative_key, "dtype"), (head_key, "missing")):
            with self.subTest(key=key, damage=damage):
                payload = dict(original, state_dict=dict(state))
                if damage == "missing":
                    del payload["state_dict"][key]
                elif damage == "shape":
                    payload["state_dict"][key] = state[key].reshape(-1)[:1]
                elif damage == "values":
                    payload["state_dict"][key] = state[key] + 1
                else:
                    payload["state_dict"][key] = state[key].float()
                with self.assertRaises(RuntimeError):
                    module.load_weights_from_checkpoint_data(payload)

    def test_normalization_policy_is_checked_in_config_and_checkpoint(self):
        module, _, _ = _fixture(selected=True, frozen=True)
        original = _checkpoint(module, "light")
        for field, value in (("batch_norm_momentum", .99), ("batch_norm_eps", 1e-5),
                             ("pretrained_running_statistics", "reset")):
            with self.subTest(field=field):
                config = copy.deepcopy(module.model.to_config())
                config["experimental_contract"]["architecture"]["normalization"][field] = value
                # Unsupported semantics fail before constructing a network.
                with mock.patch("src.models.experimental_maxvit.maxvit_t") as constructor:
                    with self.assertRaises(ValueError):
                        MaxViTTExperiment.load_from_config(config, initialize_pretrained=False)
                    constructor.assert_not_called()
                altered_contract = copy.deepcopy(original[CHECKPOINT_CONTRACT_KEY])
                altered_contract["architecture"]["normalization"][field] = value
                with self.assertRaises(ValueError):
                    module.on_load_checkpoint(dict(original, **{CHECKPOINT_CONTRACT_KEY: altered_contract}))

    def test_saved_output_and_projection_identity_cannot_be_reinterpreted(self):
        module, config, _ = _fixture(selected=True, frozen=True)
        original = _checkpoint(module, "light")
        wrong_order = dict(original, **{ORDER_CHECKPOINT_KEY: list(reversed(module.output_class_order))})
        wrong_projection = dict(original, ingredient_projection=None)
        for payload in (wrong_order, wrong_projection):
            with self.subTest(projection=payload.get("ingredient_projection")), self.assertRaises(ValueError):
                module.on_load_checkpoint(payload)
        changed_config = copy.deepcopy(config)
        changed_config.label_encoder["classes"] = list(reversed(module.output_class_order))
        with self.assertRaises(ValueError):
            load_model_for_experiment(changed_config, checkpoint_path="must-not-be-opened.ckpt")

    def test_gradcam_and_feature_factorization_use_real_frozen_forward(self):
        module, _, data = _fixture(selected=True, frozen=True)
        model = module.model.eval()
        with torch.no_grad():
            target = int(model(_image()).argmax(dim=1).item())
        seen_features, seen_concepts = [], []
        feature_hook = model.conv_target_layer.register_forward_hook(
            lambda _module, _inputs, output: seen_features.append(tuple(output.shape)))
        classifier_hook = model.factorization_classifier_layer.register_forward_hook(
            lambda _module, inputs, _output: seen_concepts.append(tuple(inputs[0].shape)))
        try:
            images, masks, targets, logits = gradcam(
                model, model.conv_target_layer, _image(), targets=[target],
                reshape_transform=model.gradcam_reshape_transform,
            )
            self.assertEqual(tuple(images.shape), (1, 224, 224, 3))
            self.assertEqual(tuple(masks.shape), (1, 224, 224))
            self.assertEqual(targets, [target])
            self.assertEqual(tuple(logits.shape), (1, 59))
            self.assertTrue(np.isfinite(images).all())
            self.assertTrue(np.isfinite(masks).all())
            self.assertGreater(float(masks.max()), 0.)
            self.assertIn((1, 512, 7, 7), seen_features)
            self.assertIsNotNone(model.classifier_target_layer.weight.grad)
            self.assertTrue(all(parameter.grad is None for parameter in model.model.stem.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in model.model.blocks.parameters()))
            with torch.no_grad():
                factors = feature_factorization(
                    model, model.conv_target_layer, model.factorization_classifier_layer,
                    _image(), label_encoder=data.label_encoder, n_components=2, top_k=2,
                    reshape_transform=model.gradcam_reshape_transform,
                )
            self.assertEqual(factors.shape[0], 1)
            self.assertEqual(factors.shape[-1], 3)
            self.assertTrue(np.isfinite(factors).all())
            self.assertGreaterEqual(factors.shape[1], 224)
            self.assertGreaterEqual(factors.shape[2], 224)
            self.assertIn((2, 512), seen_concepts)
            self.assertEqual(seen_features.count((1, 512, 7, 7)), 2)
        finally:
            feature_hook.remove()
            classifier_hook.remove()


if __name__ == "__main__":
    unittest.main()

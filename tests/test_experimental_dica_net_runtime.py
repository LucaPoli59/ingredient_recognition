"""Synthetic DICA-Net full/light canonical save/restore, no network or recipe data."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torchvision.models import EfficientNet_V2_S_Weights

from src.commons.exp_config import ExpConfig
from src.commons.config_enc_dec import decode_config
from src.data_processing.labels_encoders import MultiLabelBinarizer
from src.ingredient_selection.runtime import PROJECTION_ID, resolve_projection
from src.lightning.custom_callbacks import FullModelCheckpoint, LightModelCheckpoint
from src.lightning.experimental_lgn import (
    ExperimentalLGNM, BATCH_CHECKPOINT_KEY, ORDER_CHECKPOINT_KEY, TRAINING_CHECKPOINT_KEY,
)
from src.models.experimental_contract import CHECKPOINT_CONTRACT_KEY
from src.models.experimental_dica_net import DICANetSExperiment
from src.training.experimental_runtime import load_model_for_experiment


def _fixture(selected=False, frozen=False, weighted=False):
    projection = resolve_projection(PROJECTION_ID)
    names = list(projection.class_order if selected else projection.base_class_order)
    encoder = MultiLabelBinarizer(classes=names)
    encoder.fit()
    data = SimpleNamespace(
        label_encoder=encoder, classes_weights=torch.linspace(1., 2., len(names)),
        get_num_classes=lambda: len(names), projection_config=projection.to_config() if selected else None,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(542)
        model = DICANetSExperiment(num_classes=len(names), initialize_pretrained=False,
                                  adaptation="frozen_encoder" if frozen else "full")
    module = ExperimentalLGNM(model=model, lr=1e-4, batch_size=128, physical_batch_size=8,
                             optimizer=torch.optim.AdamW, loss_fn=torch.nn.BCEWithLogitsLoss,
                             weighted_loss=weighted, output_class_order=names)
    module.startup_model(data)
    config = ExpConfig(hp_=copy.deepcopy(dict(module.hparams)), lb_classes=names,
                       lb_encode_map={name: i for i, name in enumerate(names)}, lb_fitted=True,
                       dm_ingredient_projection=data.projection_config)
    return module, config, data


def _checkpoint(module, variant):
    checkpoint = {
        "state_dict": module.state_dict(), "hyper_parameters": copy.deepcopy(dict(module.hparams)),
        "datamodule_hyper_parameters": {"ingredient_projection": copy.deepcopy(module._ingredient_projection)},
    }
    module.on_save_checkpoint(checkpoint)
    if variant == "full":
        FullModelCheckpoint().on_save_checkpoint(SimpleNamespace(hparams={}), module, checkpoint)
    else:
        LightModelCheckpoint().on_save_checkpoint(None, module, checkpoint)
    return checkpoint


class DICANetRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_full_light_full_selected_full_frozen_restore_offline_complete_state(self):
        cases = (("full", False, False, False), ("full", True, True, True),
                 ("light", False, True, True), ("light", True, False, False))
        image = torch.linspace(-1, 1, 3*224*224).reshape(1, 3, 224, 224)
        for variant, selected, frozen, weighted in cases:
            with self.subTest(case=(variant, selected, frozen, weighted)), tempfile.TemporaryDirectory() as folder:
                module, config, data = _fixture(selected, frozen, weighted)
                with torch.no_grad():
                    module.model.model.readout.ingredient_queries.add_(.037)
                    module.model.model.readout.classwise_readout.bias.add_(.21)
                    module.model.model.readout.pooled_context_readout.bias.sub_(.13)
                    norm = next(m for m in module.model.model.features.modules() if isinstance(m, torch.nn.BatchNorm2d))
                    norm.running_mean.add_(.17)
                    norm.num_batches_tracked.add_(9)
                module.eval()
                with torch.no_grad():
                    expected = module(image)
                config_path = Path(folder) / "config.json"
                config.save_to_file(config_path)
                config = ExpConfig.load_from_file(config_path)
                payload = _checkpoint(module, variant)
                for key in (CHECKPOINT_CONTRACT_KEY, BATCH_CHECKPOINT_KEY, ORDER_CHECKPOINT_KEY, TRAINING_CHECKPOINT_KEY):
                    self.assertIn(key, payload)
                self.assertEqual("hyper_parameters" in payload, variant == "full")
                path = Path(folder) / "model.ckpt"
                torch.save(payload, path)
                with patch.object(EfficientNet_V2_S_Weights, "get_state_dict", side_effect=AssertionError("network forbidden")), \
                        patch("torch.hub.download_url_to_file", side_effect=AssertionError("network forbidden")), \
                        patch.object(DICANetSExperiment, "_verify_original_artifact", side_effect=AssertionError("cache forbidden")):
                    restored = load_model_for_experiment(config, checkpoint_path=path)
                restored.startup_model(data)
                restored.eval()
                self.assertFalse(restored.model.initialize_pretrained)
                self.assertEqual(restored.model.to_config(), module.model.to_config())
                self.assertEqual(restored.output_class_order, list(data.label_encoder.classes))
                self.assertEqual(restored._ingredient_projection, data.projection_config)
                self.assertEqual(restored.exact_batch_plan, module.exact_batch_plan)
                self.assertEqual(restored.grad_accum, 16)
                self.assertEqual(set(restored.state_dict()), set(module.state_dict()))
                for name, value in module.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
                with torch.no_grad():
                    torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)
                restored.train()
                self.assertEqual(restored.model.model.features.training, not frozen)
                self.assertTrue(all(p.requires_grad for p in restored.model.model.readout.parameters()))

    def test_fresh_canonical_construction_requests_approved_artifact(self):
        module, config, _ = _fixture(selected=True)
        import torchvision.models
        stock = torchvision.models.efficientnet_v2_s(weights=None)
        with patch("src.models.experimental_dica_net.efficientnet_v2_s", return_value=stock) as factory, \
                patch.object(DICANetSExperiment, "_verify_original_artifact") as artifact:
            fresh = load_model_for_experiment(config)
        factory.assert_called_once_with(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        artifact.assert_called_once_with()
        self.assertTrue(fresh.model.initialize_pretrained)
        self.assertEqual(fresh.output_class_order, module.output_class_order)

    def test_all_head_state_and_original_feature_buffers_are_required(self):
        module, _, _ = _fixture(selected=True, frozen=True)
        original = _checkpoint(module, "light")
        state = original["state_dict"]
        keys = ["_model.model.readout.ingredient_queries", "_model.model.readout.f16_scale_embedding",
                "_model.model.readout.f32_projection.weight", "_model.model.readout.blocks.0.k_projection.weight",
                "_model.model.readout.final_query_norm.weight", "_model.model.readout.classwise_readout.bias",
                "_model.model.readout.pooled_context_readout.weight"]
        keys.append(next(key for key in state if key.endswith("running_mean")))
        for key in keys:
            for damage in ("missing", "shape"):
                with self.subTest(key=key, damage=damage):
                    payload = dict(original, state_dict=dict(state))
                    if damage == "missing":
                        del payload["state_dict"][key]
                    else:
                        payload["state_dict"][key] = state[key].reshape(-1)[:1]
                    with self.assertRaises(RuntimeError):
                        module.load_weights_from_checkpoint_data(payload)

    def test_saved_topology_provenance_output_order_and_projection_are_not_reinterpreted(self):
        module, config, _ = _fixture(selected=True, frozen=True)
        original = _checkpoint(module, "full")
        for field, value in (("width", 256), ("query_blocks", 2), ("feature_taps", [4,7]),
                             ("context_coefficient", 0), ("token_order", "reversed")):
            payload = copy.copy(original)
            payload[CHECKPOINT_CONTRACT_KEY] = copy.deepcopy(original[CHECKPOINT_CONTRACT_KEY])
            payload[CHECKPOINT_CONTRACT_KEY]["architecture"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.on_load_checkpoint(payload)
        for field, value in ((ORDER_CHECKPOINT_KEY, list(reversed(module.output_class_order))),
                             ("ingredient_projection", None)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.on_load_checkpoint(dict(original, **{field: value}))
        payload = copy.copy(original)
        payload["hyper_parameters"] = decode_config(copy.deepcopy(original["hyper_parameters"]))
        payload["hyper_parameters"]["torch_model"]["encoder_provenance"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            module.on_load_checkpoint(payload)
        for mutate in ("order", "projection", "width", "query_count"):
            changed = copy.deepcopy(config)
            if mutate == "order":
                changed.label_encoder["classes"] = list(reversed(module.output_class_order))
            elif mutate == "projection":
                changed.datamodule["ingredient_projection"]["artifact_hash"] = "0" * 64
            elif mutate == "width":
                changed.torch_model["num_classes"] = 165
            else:
                changed.torch_model["experimental_contract"]["num_classes"] = 165
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "original.ckpt"
                torch.save(original, path)
                with self.assertRaises(ValueError):
                    load_model_for_experiment(changed, checkpoint_path=path)

    def test_full_task_checkpoint_cannot_resume_as_new_selected_task(self):
        full, _, _ = _fixture(selected=False)
        selected, _, _ = _fixture(selected=True)
        with self.assertRaises(ValueError):
            selected.load_weights_from_checkpoint_data(_checkpoint(full, "light"))
        self.assertEqual(selected.model.model.readout.WIDTH, full.model.model.readout.WIDTH)
        self.assertEqual(selected.model.model.readout.HEADS, full.model.model.readout.HEADS)
        self.assertEqual(len(selected.model.model.readout.blocks), len(full.model.model.readout.blocks))


if __name__ == "__main__":
    unittest.main()

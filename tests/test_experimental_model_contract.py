import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.commons.config_enc_dec import decode_config, encode_config
from src.commons.exp_config import ExpConfig
from src.data_processing.common import BaseDataModule
from src.data_processing.experimental_transforms import (
    ExperimentalFitPad224, MEAN, STD, fit_pad_geometry,
)
from src.models.experimental_contract import (
    CHECKPOINT_CONTRACT_KEY, ExperimentalModelContract, construct_experimental_model,
    experimental_smoke_policy, make_experimental_linear, validate_checkpoint_contract,
)
from src.lightning.custom_callbacks import LightModelCheckpoint


class ExperimentalTransformTests(unittest.TestCase):
    def test_geometry_landscape_portrait_square_odd_and_half_up(self):
        for width, height, expected in (
                (400, 200, (224, 112, 0, 56, 0, 56)),
                (200, 400, (112, 224, 56, 0, 56, 0)),
                (31, 31, (224, 224, 0, 0, 0, 0)),
                (301, 200, (224, 149, 0, 37, 0, 38)),
                (448, 1, (224, 1, 0, 111, 0, 112)),
                (1, 100000, (1, 224, 111, 0, 112, 0))):
            with self.subTest(size=(width, height)):
                self.assertEqual(fit_pad_geometry(width, height), expected)
        for width, height in ((0, 1), (1, -1), (True, 1), (1, 1.0)):
            with self.assertRaises(ValueError):
                fit_pad_geometry(width, height)

    def test_float_conversion_rgb_and_zero_normalized_padding(self):
        transform = ExperimentalFitPad224()
        image = torch.full((3, 20, 40), 128, dtype=torch.uint8)
        actual = transform(image)
        self.assertEqual(tuple(actual.shape), (3, 224, 224))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, transform(image.float() / 255), rtol=0, atol=0)
        torch.testing.assert_close(actual[:, :56], torch.zeros(3, 56, 224), rtol=0, atol=0)
        torch.testing.assert_close(actual[:, 168:], torch.zeros(3, 56, 224), rtol=0, atol=0)
        expected = (torch.full((3,), 128 / 255) - torch.tensor(MEAN)) / torch.tensor(STD)
        torch.testing.assert_close(actual[:, 112, 112], expected)

    def test_grayscale_rgba_and_pil_modes_use_rgb(self):
        transform = ExperimentalFitPad224()
        gray = torch.full((1, 17, 31), 90, dtype=torch.uint8)
        rgb = gray.expand(3, 17, 31)
        rgba = torch.cat((rgb, torch.zeros_like(gray)), dim=0)
        torch.testing.assert_close(transform(gray), transform(rgb), rtol=0, atol=0)
        torch.testing.assert_close(transform(rgba), transform(rgb), rtol=0, atol=0)
        for mode in ("L", "RGBA", "CMYK"):
            image = Image.new(mode, (31, 17))
            torch.testing.assert_close(transform(image), transform(image.convert("RGB")), rtol=0, atol=0)

    def test_full_frame_borders_are_preserved_without_crop(self):
        image = torch.zeros(3, 10, 40)
        image[0, :, 0] = 1
        image[1, :, -1] = 1
        actual = ExperimentalFitPad224()(image)
        unnormalized = actual * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
        self.assertGreater(float(unnormalized[0, 112, 0]), 0.9)
        self.assertGreater(float(unnormalized[1, 112, -1]), 0.9)

    def test_validation_deterministic_and_flip_after_padding(self):
        image = torch.arange(3 * 13 * 31, dtype=torch.float32).reshape(3, 13, 31) / (3 * 13 * 31)
        state = torch.get_rng_state().clone()
        plain = ExperimentalFitPad224()(image)
        torch.testing.assert_close(plain, ExperimentalFitPad224()(image), rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(ExperimentalFitPad224(1)(image), plain.flip(-1), rtol=0, atol=0)

    def test_datamodule_does_not_append_dataset_normalization(self):
        transform = ExperimentalFitPad224()
        result = BaseDataModule._init_transform(transform, [.1, .2, .3], [.8, .7, .6])
        self.assertIs(result, transform)
        torch.testing.assert_close(result(Image.new("RGB", (40, 30))), transform(Image.new("RGB", (40, 30))),
                                   rtol=0, atol=0)

    def test_transform_primitive_roundtrip_and_corruption_rejection(self):
        transform = ExperimentalFitPad224(.5)
        config = transform.to_config()
        restored = ExperimentalFitPad224.from_config(decode_config(encode_config(config)))
        self.assertEqual(restored.to_config(), config)
        mutations = [dict(schema_version=2), dict(schema_version=True), dict(dtype="float16"),
                     dict(input_shape=[384, 384]), dict(extra="unknown"),
                     dict(padding={"placement": "center_odd_right_bottom", "fill": [0, 0, 0]}),
                     dict(order=list(reversed(config["order"])))]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                ExperimentalFitPad224.from_config(config | mutation)
        for config in (None, {}, {"augmentation": {"horizontal_flip_probability": float("nan")}}):
            with self.assertRaises(ValueError):
                ExperimentalFitPad224.from_config(config)

    def test_invalid_inputs_and_augmentation_are_rejected(self):
        for invalid in (True, -0.1, 1.1, float("nan"), float("inf"), "0"):
            with self.assertRaises(ValueError):
                ExperimentalFitPad224(invalid)
        for image in (torch.zeros(2, 4, 4), torch.zeros(2, 3, 4, 4), torch.zeros(3, 0, 4),
                      torch.zeros(3, 4, 4, dtype=torch.int16), torch.full((3, 4, 4), -1.),
                      torch.full((3, 4, 4), 256.), torch.full((3, 4, 4), float("nan"))):
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises(ValueError):
                ExperimentalFitPad224()(image)


class FixtureExperimentalModel(torch.nn.Module):
    """Tiny no-network fixture; not an implemented portfolio adapter."""
    def __init__(self, *, contract, initialize_pretrained):
        super().__init__()
        self.experimental_contract = contract
        self.initialize_pretrained = initialize_pretrained
        self.linear = make_experimental_linear(3, contract.num_classes, contract.head_seed)

    def forward(self, image):
        return self.linear(image)


class ExperimentalModelContractTests(unittest.TestCase):
    def test_portfolio_and_shape_identity_roundtrip(self):
        for model_id in ("efficientnet_v2_s", "maxvit_t", "p2_s"):
            for width in (1, 50, 59, 165):
                for adaptation in ("full", "frozen_encoder"):
                    with self.subTest(model=model_id, width=width, adaptation=adaptation):
                        contract = ExperimentalModelContract(model_id, width, adaptation, 7, .5)
                        self.assertEqual(ExperimentalModelContract.from_config(
                            json.loads(json.dumps(contract.to_config()))), contract)
                        self.assertEqual(contract.to_config()["transform"]["input_shape"], [224, 224])
        architecture = ExperimentalModelContract("p2_s", 165).to_config()["architecture"]
        self.assertEqual(architecture["memory_tokens"], [196, 49])
        self.assertEqual((architecture["width"], architecture["heads"], architecture["query_blocks"]), (128, 4, 1))
        self.assertFalse(architecture["query_self_attention"])

    def test_model_identity_config_expconfig_roundtrip(self):
        contract = ExperimentalModelContract("p2_s", 59, "frozen_encoder", 11)
        config = ExpConfig(tm_experimental_contract=contract.to_config())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            config.save_to_file(path)
            restored = ExpConfig.load_from_file(path)
        self.assertEqual(ExperimentalModelContract.from_config(restored.hp["torch_model"]["experimental_contract"]),
                         contract)
        self.assertIsNone(ExpConfig().datamodule["ingredient_projection"])
        self.assertNotIn("experimental_contract", ExpConfig().hp["torch_model"])

    def test_config_rejects_unknown_or_inconsistent_semantics(self):
        contract = ExperimentalModelContract("p2_s", 59).to_config()
        mutations = [dict(schema_version=True), dict(schema_version=2), dict(architecture_version=2),
                     dict(weights_enum="random"), dict(output="probabilities"), dict(extra="unsupported"),
                     dict(num_classes=True), dict(adaptation="head_only_unknown")]
        changed_architecture = copy.deepcopy(contract["architecture"])
        changed_architecture["width"] = 256
        mutations.append(dict(architecture=changed_architecture))
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                ExperimentalModelContract.from_config(contract | mutation)
        for invalid in (None, {}, contract | {"transform": None}):
            with self.assertRaises(ValueError):
                ExperimentalModelContract.from_config(invalid)
        for args in (("selector", 59), ("p2_s", 0), ("p2_s", 59, "full", True)):
            with self.assertRaises(ValueError):
                ExperimentalModelContract(*args)

    def test_shared_head_initialization_determinism_bias_and_private_rng(self):
        state = torch.get_rng_state().clone()
        head = make_experimental_linear(16, 5, 42)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(head.weight.device.type, "cpu")
        self.assertTrue(torch.count_nonzero(head.bias) == 0)
        self.assertLessEqual(float(head.weight.detach().abs().max()), math.sqrt(6 / 21))
        torch.testing.assert_close(head.weight, make_experimental_linear(16, 5, 42).weight, rtol=0, atol=0)
        self.assertFalse(torch.equal(head.weight, make_experimental_linear(16, 5, 43).weight))
        for args in ((True, 5, 42), (16, 0, 42), (16, 5, False), (16, 5, -1)):
            with self.assertRaises(ValueError):
                make_experimental_linear(*args)

    def test_factory_uses_offline_restore_and_strict_complete_state(self):
        contract = ExperimentalModelContract("efficientnet_v2_s", 3)
        fresh = construct_experimental_model(FixtureExperimentalModel, contract)
        self.assertTrue(fresh.initialize_pretrained)
        with torch.no_grad():
            fresh.linear.weight.add_(.25)
        restored = construct_experimental_model(FixtureExperimentalModel, contract, state_dict=fresh.state_dict())
        self.assertFalse(restored.initialize_pretrained)
        self.assertEqual(restored.experimental_contract.to_config()["weights_enum"],
                         "EfficientNet_V2_S_Weights.IMAGENET1K_V1")
        inputs = torch.tensor([[.1, .2, .3]])
        torch.testing.assert_close(fresh(inputs), restored(inputs), rtol=0, atol=0)
        for state in ({}, [], {"linear.weight": fresh.linear.weight}):
            with self.assertRaises((ValueError, RuntimeError)):
                construct_experimental_model(FixtureExperimentalModel, contract, state_dict=state)
        with self.assertRaises(ValueError):
            construct_experimental_model(FixtureExperimentalModel, contract.to_config())

    def test_factory_rejects_constructor_identity_mismatch(self):
        contract = ExperimentalModelContract("efficientnet_v2_s", 3)
        def mismatched_constructor(**kwargs):
            return FixtureExperimentalModel(contract=ExperimentalModelContract("maxvit_t", 3),
                                            initialize_pretrained=kwargs["initialize_pretrained"])
        with self.assertRaises(ValueError):
            construct_experimental_model(mismatched_constructor, contract)

    def test_checkpoint_rejects_same_shape_different_protocol(self):
        contract = ExperimentalModelContract("efficientnet_v2_s", 3)
        checkpoint = {CHECKPOINT_CONTRACT_KEY: contract.to_config()}
        validate_checkpoint_contract(checkpoint, contract)
        for changed in (ExperimentalModelContract("maxvit_t", 3),
                        ExperimentalModelContract("efficientnet_v2_s", 3, "frozen_encoder"),
                        ExperimentalModelContract("efficientnet_v2_s", 3, horizontal_flip_probability=.5)):
            with self.assertRaises(ValueError):
                validate_checkpoint_contract(checkpoint, changed)
        with self.assertRaises(ValueError):
            validate_checkpoint_contract({}, contract)

    def test_top_level_identity_survives_existing_light_checkpoint_pruning(self):
        contract = ExperimentalModelContract("p2_s", 59)
        checkpoint = {CHECKPOINT_CONTRACT_KEY: contract.to_config(), "state_dict": {"fixture": torch.ones(1)},
                      "hyper_parameters": {}, "datamodule_hyper_parameters": {}, "trainer_hyper_parameters": {}}
        LightModelCheckpoint().on_save_checkpoint(None, None, checkpoint)
        self.assertNotIn("hyper_parameters", checkpoint)
        self.assertIn("state_dict", checkpoint)
        validate_checkpoint_contract(checkpoint, contract)

    def test_head_dtype_does_not_follow_an_external_default_dtype(self):
        prior = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            self.assertEqual(make_experimental_linear(3, 2, 42).weight.dtype, torch.float32)
        finally:
            torch.set_default_dtype(prior)

    def test_smoke_settings_are_bounded_independent_not_resource_evidence(self):
        policy = experimental_smoke_policy()
        self.assertEqual(policy["requested_batch_size"], 128)
        self.assertEqual(policy["physical_probe_candidates"], [128, 64, 32, 16, 8])
        self.assertEqual(policy["precision"], "32-true")
        self.assertEqual((policy["probe_optimizer_updates"], policy["smoke_optimizer_updates"]), (2, 4))
        self.assertIsNone(policy["pos_weight"])
        self.assertFalse(policy["drop_last"])
        policy["physical_probe_candidates"].clear()
        self.assertEqual(len(experimental_smoke_policy()["physical_probe_candidates"]), 5)


if __name__ == "__main__":
    unittest.main()

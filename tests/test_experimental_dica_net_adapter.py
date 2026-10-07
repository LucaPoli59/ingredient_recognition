"""No-network DICA-Net adapter tests using real weights-none TorchVision features."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from src.commons.config_enc_dec import decode_config, encode_config
from src.models.experimental_contract import ExperimentalModelContract, construct_experimental_model
from src.models.experimental_dica_net import DICANetSExperiment
from src.models.dica_net import DICAReadoutS


class DICANetAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_exact_weight_route_loads_intact_features_before_new_modules(self):
        stock = efficientnet_v2_s(weights=None)
        original = {name: value.clone() for name, value in stock.features.state_dict().items()}
        with patch("src.models.experimental_dica_net.efficientnet_v2_s", return_value=stock) as factory, \
                patch.object(DICANetSExperiment, "_verify_original_artifact") as verify:
            model = DICANetSExperiment(num_classes=165)
        factory.assert_called_once_with(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        verify.assert_called_once_with()
        self.assertIs(model.model.features, stock.features)
        for name, value in model.model.features.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertIs(model.classifier_target_layer, model.model.readout)
        self.assertFalse(any("classifier" in key for key in model.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 20_814_874)
        self.assertIsNone(model.max_allowed_batch_size)
        self.assertFalse(model.support_layer_pretrain)
        self.assertNotIn("initialize_pretrained", model.to_config())

    def test_initializer_matches_independent_draw_order_and_preserves_cpu_rng(self):
        seed = 174
        rng = torch.get_rng_state().clone()
        readout = DICAReadoutS(59, head_seed=seed)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        generator = torch.Generator().manual_seed(seed)
        reference = {}
        def xavier(name, shape):
            reference[name] = torch.nn.init.xavier_uniform_(torch.empty(shape), gain=1, generator=generator)
        xavier("f16_projection.weight", (128, 160, 1, 1))
        xavier("f32_projection.weight", (128, 1280, 1, 1))
        for name, shape in (("f16_scale_embedding", (128,)), ("f32_scale_embedding", (128,)),
                            ("ingredient_queries", (59, 128))):
            reference[name] = torch.empty(shape).normal_(0, .02, generator=generator)
        for name, shape in (("q_projection", (128, 128)), ("k_projection", (128, 128)),
                            ("v_projection", (128, 128)), ("attention_output", (128, 128)),
                            ("ffn_input", (512, 128)), ("ffn_output", (128, 512))):
            xavier(f"blocks.0.{name}.weight", shape)
        xavier("classwise_readout.weight", (59, 128))
        xavier("pooled_context_readout.weight", (59, 1280))
        for name, parameter in readout.named_parameters():
            expected = reference.get(name)
            if expected is None:
                expected = torch.zeros_like(parameter) if name.endswith("bias") else torch.ones_like(parameter)
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertEqual(parameter.device.type, "cpu")
        same = DICAReadoutS(59, head_seed=seed)
        different = DICAReadoutS(59, head_seed=seed+1)
        for name, tensor in readout.state_dict().items():
            torch.testing.assert_close(tensor, same.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(readout.ingredient_queries, different.ingredient_queries))
        self.assertFalse(torch.equal(readout.blocks[0].q_projection.weight, readout.blocks[0].k_projection.weight))
        # Equal seeds do not pair random rows across vocabulary sizes.
        self.assertFalse(torch.equal(readout.classwise_readout.weight, DICAReadoutS(165, seed).classwise_readout.weight[:59]))

    def test_config_roundtrips_full_frozen_all_widths_and_shared_transforms(self):
        for width in (1, 50, 59, 165):
            for adaptation in ("full", "frozen_encoder"):
                with self.subTest(width=width, adaptation=adaptation):
                    contract = ExperimentalModelContract("p2_s", width, adaptation, 7, .5)
                    model = DICANetSExperiment(contract=contract, initialize_pretrained=False)
                    config = decode_config(encode_config(model.to_config()))
                    restored = DICANetSExperiment.load_from_config(config, initialize_pretrained=False)
                    self.assertEqual(restored.to_config(), model.to_config())
                    self.assertEqual(restored.experimental_contract, contract)
                    self.assertFalse(restored.initialize_pretrained)
                    self.assertEqual(restored.transform_aug.to_config()["augmentation"]["horizontal_flip_probability"], .5)
                    self.assertEqual(restored.transform_plain.to_config()["augmentation"]["horizontal_flip_probability"], 0.)
                    self.assertEqual(restored.to_config()["encoder_provenance"]["sha256"], DICANetSExperiment.WEIGHTS_SHA256)
        image = torch.arange(3 * 13 * 31, dtype=torch.float32).reshape(3, 13, 31) / (3 * 13 * 31)
        flip = DICANetSExperiment(num_classes=1, initialize_pretrained=False, horizontal_flip_probability=1)
        torch.testing.assert_close(flip.transform_aug(image), flip.transform_plain(image).flip(-1), rtol=0, atol=0)

    def test_full_and_frozen_modes_buffers_trainability_and_input_gradients(self):
        for mode in ("full", "frozen_encoder"):
            with self.subTest(mode=mode):
                model = DICANetSExperiment(num_classes=59, adaptation=mode, initialize_pretrained=False)
                for training in (True, False, True):
                    model.train(training)
                    self.assertEqual(model.model.readout.training, training)
                    self.assertEqual(model.model.features.training, training and mode == "full")
                    self.assertTrue(all(p.requires_grad for p in model.model.readout.parameters()))
                    self.assertTrue(all(p.requires_grad == (mode == "full") for p in model.model.features.parameters()))
                original = {name: value.clone() for name, value in model.model.features.named_buffers()}
                image = torch.randn(2, 3, 224, 224, requires_grad=True)
                model(image).square().mean().backward()
                self.assertGreater(image.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(image.grad).all())
                for parameter in model.model.readout.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                if mode == "frozen_encoder":
                    self.assertTrue(all(not module.training for module in model.model.features.modules()))
                    self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 383_616 + 1538 * 59)
                    self.assertTrue(all(p.grad is None for p in model.model.features.parameters()))
                    for name, value in model.model.features.named_buffers():
                        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
                    before = model.model.readout.ingredient_queries.detach().clone()
                    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
                    optimizer.step()
                    self.assertFalse(torch.equal(before, model.model.readout.ingredient_queries))
                    with torch.no_grad():
                        torch.testing.assert_close(model(image), model(image), rtol=0, atol=0)
                else:
                    self.assertTrue(all(p.grad is not None for p in model.model.features.parameters()))
                    counters = [(name, value) for name, value in model.model.features.named_buffers() if name.endswith("num_batches_tracked")]
                    self.assertTrue(all(torch.equal(value, original[name] + 1) for name, value in counters))

    def test_complete_standalone_state_restores_offline_and_rejects_missing_heads(self):
        contract = ExperimentalModelContract("p2_s", 59, "frozen_encoder")
        model = DICANetSExperiment(contract=contract, initialize_pretrained=False).eval()
        with patch.object(EfficientNet_V2_S_Weights, "get_state_dict", side_effect=AssertionError("network forbidden")), \
                patch.object(DICANetSExperiment, "_verify_original_artifact", side_effect=AssertionError("cache forbidden")):
            restored = construct_experimental_model(DICANetSExperiment, contract, state_dict=model.state_dict()).eval()
        image = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)
        state = dict(model.state_dict())
        del state["model.readout.ingredient_queries"]
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(state, strict=True)

    def test_constructor_and_saved_protocol_mismatches_fail_before_factory(self):
        invalid = [dict(num_classes=True), dict(num_classes=59.0), dict(num_classes=0), dict(input_shape=384),
                   dict(lp_phase=-1), dict(trns_aug="custom"), dict(initialize_pretrained=0), dict(head_seed=True),
                   dict(contract=ExperimentalModelContract("efficientnet_v2_s", 59)), dict(encoder_provenance={}),
                   dict(contract=ExperimentalModelContract("p2_s", 59), num_classes=165),
                   dict(contract=ExperimentalModelContract("p2_s", 59), head_seed=3)]
        with patch("src.models.experimental_dica_net.efficientnet_v2_s") as factory:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    DICANetSExperiment(**kwargs)
            factory.assert_not_called()
        config = DICANetSExperiment(num_classes=59, initialize_pretrained=False).to_config()
        variants = [dict(config, num_classes=165), dict(config, input_shape=[384,384]), dict(config, extra=True),
                    dict(config, initialize_pretrained=False), dict(config, encoder_provenance=None)]
        for key in config:
            damaged = copy.deepcopy(config)
            del damaged[key]
            variants.append(damaged)
        for path, value in ((["architecture", "feature_taps"], [4, 7]), (["architecture", "width"], 256),
                            (["architecture", "token_order"], "coarse_first"), (["architecture", "scale"], "M"),
                            (["architecture", "query_blocks"], 2), (["architecture", "added_dropout"], .1),
                            (["architecture", "query_coefficient"], .5), (["architecture_version"], 2),
                            (["head_initialization", "qkv_initialization"], "packed")):
            damaged = copy.deepcopy(config)
            target = damaged["experimental_contract"]
            for field in path[:-1]:
                target = target[field]
            target[path[-1]] = value
            variants.append(damaged)
        for field, value in (("sha256", "0" * 64), ("batch_norm_momentum", .01), ("size_bytes", 1),
                             ("weights_enum", "selector_checkpoint"), ("schema_version", True)):
            damaged = copy.deepcopy(config)
            damaged["encoder_provenance"][field] = value
            variants.append(damaged)
        with patch("src.models.experimental_dica_net.efficientnet_v2_s") as factory:
            for variant in variants:
                with self.subTest(variant=variant), self.assertRaises(ValueError):
                    DICANetSExperiment.load_from_config(variant, initialize_pretrained=False)
            factory.assert_not_called()

    def test_pinned_artifact_rejects_wrong_cache_and_offline_needs_no_cache(self):
        with tempfile.TemporaryDirectory() as folder, patch("torch.hub.get_dir", return_value=folder):
            with self.assertRaises(FileNotFoundError):
                DICANetSExperiment._verify_original_artifact()
            path = Path(folder) / "checkpoints" / "efficientnet_v2_s-dd5fe13b.pth"
            path.parent.mkdir()
            path.write_bytes(b"not the approved artifact")
            with self.assertRaises(ValueError):
                DICANetSExperiment._verify_original_artifact()
            model = DICANetSExperiment(num_classes=1, initialize_pretrained=False)
            self.assertEqual(model.num_classes, 1)

    def test_normalization_drift_is_rejected_without_resetting_statistics(self):
        stock = efficientnet_v2_s(weights=None)
        norm = next(m for m in stock.features.modules() if isinstance(m, torch.nn.BatchNorm2d))
        norm.momentum = .99
        norm.running_mean.fill_(.4)
        before = norm.running_mean.clone()
        with patch("src.models.experimental_dica_net.efficientnet_v2_s", return_value=stock):
            with self.assertRaises(RuntimeError):
                DICANetSExperiment(num_classes=1, initialize_pretrained=False)
        torch.testing.assert_close(norm.running_mean, before, rtol=0, atol=0)

    def test_context_only_head_is_not_exposed_as_complete_factorization_classifier(self):
        model = DICANetSExperiment(num_classes=3, initialize_pretrained=False)
        self.assertIs(model.classifier_target_layer, model.model.readout)
        with self.assertRaises(NotImplementedError):
            _ = model.factorization_classifier_layer


if __name__ == "__main__":
    unittest.main()

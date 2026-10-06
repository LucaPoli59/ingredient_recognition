"""Real TorchVision topology tests with weights=None; no benchmark evidence."""

import copy
import unittest
from unittest.mock import patch

import torch
from torchvision.models import MaxVit_T_Weights, maxvit_t
from torchvision.models.maxvit import PartitionAttentionLayer
from torchvision.ops import StochasticDepth

from src.commons.config_enc_dec import decode_config, encode_config
from src.models.experimental_contract import ExperimentalModelContract, make_experimental_linear
from src.models.experimental_maxvit import MaxViTTExperiment


class MaxViTTExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_fresh_exact_enum_whole_readout_replacement_and_intact_backbone_state(self):
        stock = maxvit_t(weights=None)
        stem, blocks, old_head = stock.stem, stock.blocks, stock.classifier
        batch_norm = next(module for module in stem.modules() if isinstance(module, torch.nn.BatchNorm2d))
        # Non-default buffer values prove that adaptation does not reset them.
        batch_norm.running_mean.fill_(.25)
        batch_norm.running_var.fill_(1.7)
        batch_norm.num_batches_tracked.fill_(19)
        before = {key: value.clone() for key, value in stock.state_dict().items()
                  if key.startswith(("stem.", "blocks."))}
        with patch("src.models.experimental_maxvit.maxvit_t", return_value=stock) as constructor:
            model = MaxViTTExperiment(num_classes=59, head_seed=7)
        constructor.assert_called_once_with(weights=MaxVit_T_Weights.IMAGENET1K_V1)
        self.assertIs(model.model.stem, stem)
        self.assertIs(model.model.blocks, blocks)
        self.assertIsNot(model.model.classifier, old_head)
        self.assertEqual([type(layer) for layer in model.model.classifier],
                         [torch.nn.AdaptiveAvgPool2d, torch.nn.Flatten, torch.nn.Linear])
        self.assertIs(model.classifier_target_layer, model.model.classifier[2])
        expected_head = make_experimental_linear(512, 59, 7)
        torch.testing.assert_close(model.classifier_target_layer.weight, expected_head.weight, rtol=0, atol=0)
        torch.testing.assert_close(model.classifier_target_layer.bias, expected_head.bias, rtol=0, atol=0)
        after = {key: value for key, value in model.model.state_dict().items()
                 if key.startswith(("stem.", "blocks."))}
        self.assertEqual(set(after), set(before))
        self.assertEqual(sum(key.endswith("relative_position_index") for key in before), 22)
        for key, value in after.items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(model.initialize_pretrained)
        self.assertNotIn("initialize_pretrained", model.to_config())
        self.assertIsNone(model.max_allowed_batch_size)
        self.assertFalse(model.support_layer_pretrain)

    def test_actual_shapes_and_parameter_counts_for_all_required_widths(self):
        for width in (1, 50, 59, 165):
            with self.subTest(width=width):
                model = MaxViTTExperiment(num_classes=width, initialize_pretrained=False).eval()
                self.assertEqual(sum(p.numel() for p in model.model.stem.parameters())
                                 + sum(p.numel() for p in model.model.blocks.parameters()), 30_143_944)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 30_143_944 + 513 * width)
                with torch.no_grad():
                    logits = model(torch.zeros(1, 3, 224, 224))
                self.assertEqual(tuple(logits.shape), (1, width))
                self.assertTrue(torch.isfinite(logits).all())
                if width == 165:
                    self.assertEqual(sum(p.numel() for p in model.parameters()), 30_228_589)

    def test_config_roundtrip_preserves_primitive_identity_and_fixed_normalization(self):
        for adaptation in ("full", "frozen_encoder"):
            with self.subTest(adaptation=adaptation):
                contract = ExperimentalModelContract("maxvit_t", 59, adaptation, 9, .5)
                model = MaxViTTExperiment(contract=contract, initialize_pretrained=False)
                config = decode_config(encode_config(model.to_config()))
                restored = MaxViTTExperiment.load_from_config(config, initialize_pretrained=False)
                self.assertEqual(restored.to_config(), model.to_config())
                self.assertEqual(restored.experimental_contract, contract)
                self.assertFalse(restored.initialize_pretrained)
                architecture = contract.to_config()["architecture"]
                self.assertEqual(architecture["input_size"], [224, 224])
                self.assertEqual(architecture["partition_size"], 7)
                self.assertEqual(architecture["normalization"], {
                    "batch_norm_eps": .001, "batch_norm_momentum": .01,
                    "pretrained_running_statistics": "preserve"})
                batch_norms = [module for module in restored.modules() if isinstance(module, torch.nn.BatchNorm2d)]
                self.assertEqual(len(batch_norms), 34)
                self.assertTrue(all(module.eps == .001 and module.momentum == .01 for module in batch_norms))
                self.assertEqual(restored.transform_aug.horizontal_flip_probability, .5)
                self.assertEqual(restored.transform_plain.horizontal_flip_probability, 0.)
        # Adding MaxViT's policy must not change another family's serialized identity.
        self.assertEqual(ExperimentalModelContract("efficientnet_v2_s", 165).to_config()["architecture"],
                         {"readout": "gap_biased_linear", "added_dropout": 0.0})
        self.assertNotIn("normalization", ExperimentalModelContract("p2_s", 165).to_config()["architecture"])

    def test_config_loader_fresh_route_keeps_approved_pretrained_default(self):
        model = MaxViTTExperiment(num_classes=59, initialize_pretrained=False)
        stock = maxvit_t(weights=None)
        with patch("src.models.experimental_maxvit.maxvit_t", return_value=stock) as constructor:
            restored = MaxViTTExperiment.load_from_config(model.to_config())
        constructor.assert_called_once_with(weights=MaxVit_T_Weights.IMAGENET1K_V1)
        self.assertTrue(restored.initialize_pretrained)

    def test_frozen_real_encoder_eval_state_buffers_and_trainable_head(self):
        model = MaxViTTExperiment(num_classes=59, adaptation="frozen_encoder", initialize_pretrained=False)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 513 * 59)
        for mode in (False, True, False, True):
            self.assertIs(model.train(mode), model)
            self.assertEqual(model.model.classifier.training, mode)
            for encoder in (model.model.stem, model.model.blocks):
                self.assertFalse(any(module.training for module in encoder.modules()))
        stochastic = [module for module in model.model.blocks.modules() if isinstance(module, StochasticDepth)]
        self.assertTrue(stochastic)
        self.assertTrue(any(module.p > 0 for module in stochastic))
        self.assertTrue(all(not module.training for module in stochastic))
        buffers = {key: value.clone() for key, value in model.named_buffers()}
        old_head = model.classifier_target_layer.weight.detach().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        image = torch.randn(2, 3, 224, 224)
        model(image).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(model.classifier_target_layer.weight, old_head))
        for encoder in (model.model.stem, model.model.blocks):
            self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        for key, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(model(image), model(image), rtol=0, atol=0)

    def test_full_real_encoder_trains_and_updates_normalization_buffers(self):
        model = MaxViTTExperiment(num_classes=3, initialize_pretrained=False).train()
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(module.training for module in model.modules()))
        batch_norm = next(module for module in model.model.stem.modules()
                          if isinstance(module, torch.nn.BatchNorm2d))
        mean = batch_norm.running_mean.clone()
        tracked = batch_norm.num_batches_tracked.clone()
        model(torch.randn(2, 3, 224, 224)).square().mean().backward()
        self.assertFalse(torch.equal(mean, batch_norm.running_mean))
        self.assertEqual(batch_norm.num_batches_tracked.item(), tracked.item() + 1)
        for parameter in (next(model.model.stem.parameters()), next(model.model.blocks.parameters()),
                          model.classifier_target_layer.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_real_forward_hooks_complete_pooled_head_and_input_gradients(self):
        for adaptation in ("full", "frozen_encoder"):
            with self.subTest(adaptation=adaptation):
                model = MaxViTTExperiment(num_classes=3, adaptation=adaptation,
                                         initialize_pretrained=False).eval()
                captured = {}
                def capture_features(module, inputs, output):
                    captured["features"] = output
                    output.retain_grad()
                def capture_classifier(module, inputs, output):
                    captured["pooled"] = inputs[0]
                    captured["logits"] = output
                feature_handle = model.conv_target_layer.register_forward_hook(capture_features)
                head_handle = model.classifier_target_layer.register_forward_hook(capture_classifier)
                try:
                    image = torch.randn(1, 3, 224, 224, requires_grad=True)
                    logits = model(image)
                    torch.testing.assert_close(logits, captured["logits"], rtol=0, atol=0)
                    torch.testing.assert_close(captured["pooled"], captured["features"].mean((2, 3)), rtol=0, atol=0)
                    self.assertEqual(tuple(captured["features"].shape), (1, 512, 7, 7))
                    logits[0, 0].backward()
                finally:
                    feature_handle.remove()
                    head_handle.remove()
                for gradient in (image.grad, captured["features"].grad):
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(torch.count_nonzero(gradient).item(), 0)
                self.assertIs(model.factorization_classifier_layer, model.model.classifier[-1])
                self.assertEqual(tuple(model.factorization_classifier_layer(torch.randn(4, 512)).shape), (4, 3))
                self.assertIsNone(model.gradcam_reshape_transform)

    def test_transform_parity_and_raw_logits(self):
        model = MaxViTTExperiment(num_classes=3, horizontal_flip_probability=1,
                                 initialize_pretrained=False).eval()
        image = torch.arange(3 * 13 * 31, dtype=torch.float32).reshape(3, 13, 31) / (3 * 13 * 31)
        torch.testing.assert_close(model.transform_aug(image), model.transform_plain(image).flip(-1), rtol=0, atol=0)
        with torch.no_grad():
            model.classifier_target_layer.weight.zero_()
            model.classifier_target_layer.bias.copy_(torch.tensor([-2., 0., 3.]))
            logits = model(model.transform_plain(image).unsqueeze(0))
        torch.testing.assert_close(logits, torch.tensor([[-2., 0., 3.]]), rtol=0, atol=0)

    def test_invalid_constructor_and_runtime_input_rejected(self):
        invalid = [dict(num_classes=True), dict(num_classes=59.0), dict(num_classes=0),
                   dict(input_shape=384), dict(input_shape=True), dict(input_shape=(224, 224.0)),
                   dict(input_shape=(224, 223)), dict(input_shape=(224,)), dict(lp_phase=-1),
                   dict(trns_aug=lambda: []), dict(trns_bld_plain=lambda x: []),
                   dict(initialize_pretrained=0), dict(adaptation="unknown"),
                   dict(head_seed=True), dict(horizontal_flip_probability=float("nan")),
                   dict(contract=ExperimentalModelContract("efficientnet_v2_s", 59)),
                   dict(contract=ExperimentalModelContract("maxvit_t", 59), num_classes=165),
                   dict(contract=ExperimentalModelContract("maxvit_t", 59), adaptation="frozen_encoder"),
                   dict(contract=ExperimentalModelContract("maxvit_t", 59), head_seed=43),
                   dict(contract=ExperimentalModelContract("maxvit_t", 59), horizontal_flip_probability=.5),
                   dict(contract={}), dict(experimental_contract={}),
                   dict(contract=ExperimentalModelContract("maxvit_t", 59), experimental_contract={})]
        with patch("src.models.experimental_maxvit.maxvit_t") as constructor:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    MaxViTTExperiment(**kwargs)
            constructor.assert_not_called()
        model = MaxViTTExperiment(num_classes=3, initialize_pretrained=False)
        for image in (torch.zeros(3, 224, 224), torch.zeros(0, 3, 224, 224),
                      torch.zeros(1, 1, 224, 224), torch.zeros(1, 3, 384, 384), "image"):
            with self.assertRaises(ValueError):
                model(image)

    def test_config_identity_normalization_and_geometry_tampering_rejected(self):
        model = MaxViTTExperiment(num_classes=59, initialize_pretrained=False)
        config = model.to_config()
        mutations = [dict(num_classes=165), dict(num_classes=59.0), dict(input_shape=[384, 384]),
                     dict(type=torch.nn.Linear), dict(lp_phase=-1), dict(trns_aug="lost_callable"),
                     dict(unknown=True), dict(initialize_pretrained=False),
                     dict(experimental_contract=ExperimentalModelContract("efficientnet_v2_s", 59).to_config())]
        for key, value in (("partition_size", 8), ("input_size", [448, 448]),
                           ("normalization", {"batch_norm_eps": .001, "batch_norm_momentum": .99,
                                              "pretrained_running_statistics": "preserve"})):
            contract = copy.deepcopy(config["experimental_contract"])
            contract["architecture"][key] = value
            mutations.append(dict(experimental_contract=contract))
        with patch("src.models.experimental_maxvit.maxvit_t") as constructor:
            for mutation in mutations:
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    MaxViTTExperiment.load_from_config(config | mutation, initialize_pretrained=False)
            for key in config:
                incomplete = copy.deepcopy(config)
                del incomplete[key]
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    MaxViTTExperiment.load_from_config(incomplete, initialize_pretrained=False)
            constructor.assert_not_called()

    def test_library_policy_drift_is_rejected_without_repairing_loaded_state(self):
        stock = maxvit_t(weights=None)
        batch_norm = next(module for module in stock.modules() if isinstance(module, torch.nn.BatchNorm2d))
        attention = next(module for module in stock.modules() if isinstance(module, PartitionAttentionLayer))
        mutations = [(stock, "partition_size", 8), (stock.blocks[-1], "grid_size", (14, 14)),
                     (attention, "p", 8), (batch_norm, "momentum", .99), (batch_norm, "eps", 1e-5),
                     (stock.classifier[5], "out_features", 999)]
        for module, attribute, altered in mutations:
            original = getattr(module, attribute)
            setattr(module, attribute, altered)
            try:
                with self.subTest(attribute=attribute), patch(
                        "src.models.experimental_maxvit.maxvit_t", return_value=stock), self.assertRaises(RuntimeError):
                    MaxViTTExperiment(num_classes=59, initialize_pretrained=False)
                self.assertEqual(getattr(module, attribute), altered)
            finally:
                setattr(module, attribute, original)


if __name__ == "__main__":
    unittest.main()

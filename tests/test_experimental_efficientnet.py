"""No-network adapter tests; random backbone fixtures are not benchmark models."""

import copy
import unittest
from unittest.mock import patch

import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from src.commons.config_enc_dec import decode_config, encode_config
from src.models.experimental_contract import ExperimentalModelContract, construct_experimental_model
from src.models.experimental_efficientnet import EfficientNetV2SExperiment


class _TinyLibraryFixture(torch.nn.Module):
    """Small factory stand-in, preserving the library's relevant public surface."""
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 1280, 3, stride=32, padding=1, bias=False),
            torch.nn.BatchNorm2d(1280), torch.nn.Dropout2d(.7),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(.2, inplace=True),
                                              torch.nn.Linear(1280, 1000))

    def forward(self, image):
        features = self.features(image)
        return self.classifier(torch.flatten(self.avgpool(features), 1))


class EfficientNetV2SExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def _tiny(**kwargs):
        return _TinyLibraryFixture()

    def test_fresh_default_uses_exact_original_enum_before_head_replacement(self):
        stock = _TinyLibraryFixture()
        features = stock.features
        original_head = stock.classifier
        original_state = {name: value.clone() for name, value in features.state_dict().items()}
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s", return_value=stock) as constructor:
            model = EfficientNetV2SExperiment(num_classes=59, head_seed=7)
        constructor.assert_called_once_with(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.assertIs(model.model.features, features)
        self.assertIsNot(model.model.classifier, original_head)
        self.assertIsInstance(model.classifier_target_layer, torch.nn.Linear)
        self.assertEqual((model.classifier_target_layer.in_features, model.classifier_target_layer.out_features),
                         (1280, 59))
        self.assertEqual(torch.count_nonzero(model.classifier_target_layer.bias).item(), 0)
        for name, tensor in features.state_dict().items():
            torch.testing.assert_close(tensor, original_state[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertIsNone(model.max_allowed_batch_size)
        self.assertFalse(model.support_layer_pretrain)
        self.assertTrue(model.initialize_pretrained)
        self.assertNotIn("initialize_pretrained", model.to_config())
        self.assertEqual(model.to_config()["experimental_contract"]["weights_enum"], model.WEIGHTS_ENUM)

    def test_actual_random_library_backbone_shape_width_count_and_no_classifier_dropout(self):
        # The real public constructor is used without weights/network; this
        # verifies topology, not an actually loaded ImageNet artifact.
        for width in (1, 50, 59, 165):
            with self.subTest(width=width):
                model = EfficientNetV2SExperiment(num_classes=width, initialize_pretrained=False).eval()
                self.assertIsInstance(model.model.classifier, torch.nn.Linear)
                self.assertFalse(any(isinstance(module, torch.nn.Dropout) for module in model.model.classifier.modules()))
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()),
                                 20_177_488 + 1281 * width)
                with torch.no_grad():
                    logits = model(torch.zeros(1, 3, 224, 224))
                self.assertEqual(tuple(logits.shape), (1, width))
                self.assertTrue(torch.isfinite(logits).all())
                if width == 165:
                    self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 20_388_853)

    def test_actual_feature_weights_and_buffers_survive_new_head_initialization(self):
        stock = efficientnet_v2_s(weights=None)
        features = stock.features
        original_state = {name: value.clone() for name, value in features.state_dict().items()}
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s", return_value=stock):
            model = EfficientNetV2SExperiment(num_classes=165, initialize_pretrained=False)
        self.assertIs(model.model.features, features)
        self.assertEqual(set(model.model.features.state_dict()), set(original_state))
        for name, tensor in model.model.features.state_dict().items():
            torch.testing.assert_close(tensor, original_state[name], rtol=0, atol=0)

    def test_full_and_frozen_config_roundtrip_all_widths_and_augmentation(self):
        for width in (1, 50, 59, 165):
            for adaptation in ("full", "frozen_encoder"):
                with self.subTest(width=width, adaptation=adaptation), patch(
                        "src.models.experimental_efficientnet.efficientnet_v2_s", side_effect=self._tiny) as constructor:
                    contract = ExperimentalModelContract("efficientnet_v2_s", width, adaptation, 7, .5)
                    model = EfficientNetV2SExperiment(contract=contract, initialize_pretrained=False)
                    config = decode_config(encode_config(model.to_config()))
                    restored = EfficientNetV2SExperiment.load_from_config(config, initialize_pretrained=False)
                    self.assertEqual(restored.to_config(), model.to_config())
                    self.assertEqual(restored.experimental_contract, contract)
                    self.assertFalse(restored.initialize_pretrained)
                    self.assertEqual(restored.transform_aug.to_config()["augmentation"]["horizontal_flip_probability"], .5)
                    self.assertEqual(restored.transform_plain.to_config()["augmentation"]["horizontal_flip_probability"], 0.)
                    self.assertTrue(all(call.kwargs == {"weights": None} for call in constructor.call_args_list))

    def test_config_loader_fresh_route_keeps_default_pretrained(self):
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s", side_effect=self._tiny) as constructor:
            model = EfficientNetV2SExperiment(num_classes=59, initialize_pretrained=False)
            EfficientNetV2SExperiment.load_from_config(model.to_config())
        self.assertEqual(constructor.call_args_list[-1].kwargs,
                         {"weights": EfficientNet_V2_S_Weights.IMAGENET1K_V1})

    def test_offline_complete_state_restore_exact_logits_and_missing_state_rejection(self):
        contract = ExperimentalModelContract("efficientnet_v2_s", 59)
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s", side_effect=self._tiny) as constructor:
            original = EfficientNetV2SExperiment(contract=contract, initialize_pretrained=False).eval()
            with torch.no_grad():
                original.model.classifier.weight.add_(.01)
            restored = construct_experimental_model(EfficientNetV2SExperiment, contract,
                                                   state_dict=original.state_dict()).eval()
            image = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                torch.testing.assert_close(original(image), restored(image), rtol=0, atol=0)
            self.assertEqual(restored.experimental_contract, contract)
            self.assertEqual(constructor.call_args_list[-1].kwargs, {"weights": None})
            state = dict(original.state_dict())
            del state["model.classifier.bias"]
            with self.assertRaises(RuntimeError):
                construct_experimental_model(EfficientNetV2SExperiment, contract, state_dict=state)

    def test_frozen_encoder_eval_buffers_and_trainable_head_persist_across_train(self):
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s", side_effect=self._tiny):
            model = EfficientNetV2SExperiment(num_classes=165, adaptation="frozen_encoder",
                                            initialize_pretrained=False)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                         1281 * 165)
        for mode in (False, True, False, True):
            self.assertIs(model.train(mode), model)
            self.assertEqual(model.model.classifier.training, mode)
            self.assertFalse(any(module.training for module in model.model.features.modules()))
        buffers = {name: value.clone() for name, value in model.model.features.named_buffers()}
        before = model.model.classifier.weight.detach().clone()
        optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.1)
        image = torch.randn(2, 3, 224, 224)
        logits = model(image)
        logits.square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.model.classifier.weight))
        self.assertTrue(all(parameter.grad is None for parameter in model.model.features.parameters()))
        for name, value in model.model.features.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(model(image), model(image), rtol=0, atol=0)

    def test_full_encoder_train_mode_and_gradients(self):
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s", side_effect=self._tiny):
            model = EfficientNetV2SExperiment(num_classes=3, initialize_pretrained=False)
        model.train()
        self.assertTrue(all(module.training for module in model.model.features.modules()))
        model(torch.randn(2, 3, 224, 224)).square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.model.features.parameters()))

    def test_actual_forward_hooks_and_input_gradients_in_full_and_frozen_modes(self):
        for adaptation in ("full", "frozen_encoder"):
            with self.subTest(adaptation=adaptation), patch(
                    "src.models.experimental_efficientnet.efficientnet_v2_s", side_effect=self._tiny):
                model = EfficientNetV2SExperiment(num_classes=3, adaptation=adaptation,
                                                initialize_pretrained=False).eval()
            captured = {}
            def hook(module, inputs, output):
                captured["activation"] = output
                output.retain_grad()
            handle = model.conv_target_layer.register_forward_hook(hook)
            image = torch.randn(1, 3, 224, 224, requires_grad=True)
            model(image)[0, 0].backward()
            handle.remove()
            self.assertTrue(torch.isfinite(image.grad).all())
            self.assertGreater(torch.count_nonzero(image.grad).item(), 0)
            self.assertEqual(tuple(captured["activation"].shape), (1, 1280, 7, 7))
            self.assertTrue(torch.isfinite(captured["activation"].grad).all())
            self.assertIs(model.factorization_classifier_layer, model.model.classifier)
            self.assertIsNone(model.gradcam_reshape_transform)

    def test_transform_plain_aug_shared_policy_and_raw_logits(self):
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s", side_effect=self._tiny):
            model = EfficientNetV2SExperiment(num_classes=3, horizontal_flip_probability=1,
                                            initialize_pretrained=False).eval()
        image = torch.arange(3 * 13 * 31, dtype=torch.float32).reshape(3, 13, 31) / (3 * 13 * 31)
        torch.testing.assert_close(model.transform_aug(image), model.transform_plain(image).flip(-1), rtol=0, atol=0)
        with torch.no_grad():
            model.model.classifier.weight.zero_()
            model.model.classifier.bias.copy_(torch.tensor([-2., 0., 3.]))
            logits = model(model.transform_plain(image).unsqueeze(0))
        torch.testing.assert_close(logits, torch.tensor([[-2., 0., 3.]]), rtol=0, atol=0)

    def test_invalid_constructor_and_input_overrides_rejected_before_factory(self):
        invalid = [dict(num_classes=True), dict(num_classes=59.0), dict(num_classes=0),
                   dict(input_shape=384), dict(input_shape=True), dict(input_shape=(224, 224.0)),
                   dict(input_shape=(224, 223)), dict(input_shape=(224,)), dict(lp_phase=-1),
                   dict(trns_aug=lambda: []), dict(trns_bld_plain=lambda x: []),
                   dict(initialize_pretrained=0), dict(adaptation="unknown"),
                   dict(head_seed=True), dict(horizontal_flip_probability=float("nan")),
                   dict(contract=ExperimentalModelContract("maxvit_t", 59)),
                   dict(contract=ExperimentalModelContract("efficientnet_v2_s", 59), num_classes=165),
                   dict(contract=ExperimentalModelContract("efficientnet_v2_s", 59), adaptation="frozen_encoder"),
                   dict(contract=ExperimentalModelContract("efficientnet_v2_s", 59), head_seed=43),
                   dict(contract=ExperimentalModelContract("efficientnet_v2_s", 59), horizontal_flip_probability=.5),
                   dict(contract={}), dict(experimental_contract={})]
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s") as constructor:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    EfficientNetV2SExperiment(**kwargs)
            constructor.assert_not_called()
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s", side_effect=self._tiny):
            model = EfficientNetV2SExperiment(num_classes=3, initialize_pretrained=False)
        for image in (torch.zeros(3, 224, 224), torch.zeros(0, 3, 224, 224),
                      torch.zeros(1, 1, 224, 224), torch.zeros(1, 3, 384, 384)):
            with self.assertRaises(ValueError):
                model(image)

    def test_inconsistent_config_rejected_before_construction(self):
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s", side_effect=self._tiny):
            model = EfficientNetV2SExperiment(num_classes=59, initialize_pretrained=False)
        config = model.to_config()
        mutations = [dict(num_classes=165), dict(num_classes=59.0), dict(input_shape=[384, 384]),
                     dict(type=torch.nn.Linear), dict(lp_phase=-1), dict(trns_aug="lost_callable"),
                     dict(unknown=True), dict(initialize_pretrained=False)]
        other_contract = ExperimentalModelContract("maxvit_t", 59).to_config()
        mutations.append(dict(experimental_contract=other_contract))
        with patch("src.models.experimental_efficientnet.efficientnet_v2_s") as constructor:
            for mutation in mutations:
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    EfficientNetV2SExperiment.load_from_config(config | mutation, initialize_pretrained=False)
            for key in config:
                incomplete = copy.deepcopy(config)
                del incomplete[key]
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    EfficientNetV2SExperiment.load_from_config(incomplete, initialize_pretrained=False)
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.data_processing.transformations import (
    selector_fit_pad_geometry,
    transform_plain_selector_efficientnet_v2_s,
)
from src.models.efficientnet import EfficientNetV2SSelector


class _FakeEfficientNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(4, 1000),
        )

    def forward(self, values):
        values = self.features(values).mean(dim=(-2, -1))
        return self.classifier(values)


class SelectorTransformAndModelTests(unittest.TestCase):
    def test_geometry_preserves_full_frame_and_assigns_odd_pixel_right_or_bottom(self):
        self.assertEqual(selector_fit_pad_geometry(7, 4), (384, 219, 0, 82, 0, 83))
        self.assertEqual(selector_fit_pad_geometry(4, 7), (219, 384, 82, 0, 83, 0))
        self.assertEqual(selector_fit_pad_geometry(5, 5), (384, 384, 0, 0, 0, 0))

    def test_plain_transform_converts_grayscale_and_mean_padding_normalizes_to_zero(self):
        image = Image.fromarray(np.full((4, 7), 128, dtype=np.uint8), mode="L")
        transform = transform_plain_selector_efficientnet_v2_s()
        first = transform(image)
        second = transform(image)

        self.assertEqual(tuple(first.shape), (3, 384, 384))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first[:, 0, 0], torch.zeros(3), atol=1e-6))

    def test_tensor_channel_conversion_supports_rgba(self):
        image = torch.zeros((4, 6, 3), dtype=torch.uint8)
        transformed = transform_plain_selector_efficientnet_v2_s()(image)
        self.assertEqual(tuple(transformed.shape), (3, 384, 384))

    @patch("src.models.efficientnet.efficientnet_v2_s", return_value=_FakeEfficientNet())
    def test_selector_replaces_only_linear_head_and_keeps_every_parameter_trainable(self, builder):
        model = EfficientNetV2SSelector(num_classes=165)

        builder.assert_called_once()
        self.assertEqual(model.classifier_target_layer.in_features, 4)
        self.assertEqual(model.classifier_target_layer.out_features, 165)
        self.assertTrue(model.classifier_target_layer.bias is not None)
        self.assertEqual(model.model.classifier[0].p, 0.2)
        self.assertTrue(model.model.classifier[0].inplace)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(tuple(model(torch.zeros(2, 3, 8, 8)).shape), (2, 165))

    def test_selector_rejects_non_frozen_shape_or_initialization(self):
        with self.assertRaises(ValueError):
            EfficientNetV2SSelector(input_shape=(224, 224))
        with self.assertRaises(ValueError):
            EfficientNetV2SSelector(pretrained=False)


if __name__ == "__main__":
    unittest.main()

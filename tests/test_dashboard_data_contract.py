import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from src.commons.exp_config import ExpConfig
from src.dashboards.runtime import load_visualization_datamodule
from src.data_processing.images_recipes import ImagesRecipesBaseDataModule
from src.data_processing.common import BaseDataModule
from src.data_processing.labels_encoders import MultiLabelBinarizerRobust
from src.lightning.lgn_models import BaseLGNM
from src.models.resnet import Resnet18


class FixtureDataModule(ImagesRecipesBaseDataModule):
    @classmethod
    def load_from_config(cls, config, **kwargs):
        return super().load_from_config(config, images_stats_path=Path(config["data_dir"]) / "train_images_stats.csv",
                                        **kwargs)


class DashboardDataContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "imgs" / "standard"
        self.images.mkdir(parents=True)
        Image.new("RGB", (40, 30), (30, 90, 150)).save(self.images / "example.png")
        rows = [{"image": "example.png", "ingredients_target": ["salt", "pepper"],
                 "ingredients_ok": ["salt", "pepper"]}]
        for split in ("train", "val", "test"):
            folder = self.root / split
            folder.mkdir()
            (folder / "ingredients_target_v5_metadata.json").write_text(json.dumps(rows))
        (self.root / "train_images_stats.csv").write_text(",red,green,blue\nmean,0.1,0.2,0.3\nstd,0.8,0.7,0.6\n")

    def model_and_config(self, **kwargs):
        config = ExpConfig(dm_type=FixtureDataModule, dm_data_dir=str(self.root), dm_num_workers=0, tm_type=Resnet18,
                           tm_pretrained=False, tm_input_shape=(32, 32), tm_num_classes=2,
                           metrics_={}, **kwargs)
        return BaseLGNM.load_from_config(config.hp), config

    def test_validation_uses_model_transform_and_shared_image_root(self):
        model, config = self.model_and_config()
        with patch.object(ImagesRecipesBaseDataModule, "test_dataloader", side_effect=AssertionError("test loader")):
            dm = load_visualization_datamodule(config, model)
            image, labels = dm.val_dataset[0]
        with Image.open(self.images / "example.png") as source:
            expected = BaseDataModule._init_transform(model.transform_plain, [.1, .2, .3], [.8, .7, .6])(source)
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
        self.assertEqual(tuple(image.shape), (3, 32, 32))
        self.assertEqual(dm.val_dataset.images_paths[0], self.images / "example.png")
        self.assertEqual(labels.tolist(), [1, 1])
        self.assertNotIn("<UNK>", dm.label_encoder.classes)
        self.assertFalse(hasattr(dm, "test_dataset") and dm.test_dataset is not None)

    def test_model_encoder_dimension_mismatch_is_rejected(self):
        model, config = self.model_and_config()
        model.num_classes = 3
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            load_visualization_datamodule(config, model)

    def test_saved_legacy_encoder_is_preserved(self):
        encoder = MultiLabelBinarizerRobust()
        encoder.fit(np.array([["salt", "pepper"]], dtype=object))
        config = ExpConfig(dm_type=FixtureDataModule, dm_data_dir=str(self.root), dm_num_workers=0,
                           dm_feature_label="ingredients_ok", dm_label_encoder=encoder.to_config(),
                           tm_type=Resnet18, tm_pretrained=False, tm_num_classes=3, metrics_={})
        dm = load_visualization_datamodule(config, BaseLGNM.load_from_config(config.hp))
        self.assertEqual(list(dm.label_encoder.classes), ["pepper", "salt", "<UNK>"])
        self.assertEqual(dm.val_dataset.label_data.shape, (1, 3))

    def test_model_owned_normalization_is_not_replaced_by_dataset_statistics(self):
        from torchvision.models import ResNet18_Weights
        model, config = self.model_and_config()
        # Exercise the pretrained preprocessing contract without downloading weights.
        model.model.tr_weights = ResNet18_Weights.IMAGENET1K_V1
        dm = load_visualization_datamodule(config, model)
        with Image.open(self.images / "example.png") as source:
            expected = model.transform_plain(source)
        torch.testing.assert_close(dm.val_dataset[0][0], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

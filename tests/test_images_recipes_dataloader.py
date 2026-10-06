import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.commons.exp_config import ExpConfig
from src.data_processing.images_recipes import ImagesRecipesBaseDataModule, _resolve_pin_memory


class ImageLoaderPolicyTests(unittest.TestCase):
    def test_automatic_policy_is_platform_aware(self):
        for platform, expected in (("nt", True), ("posix", False), ("unknown", False)):
            self.assertEqual(_resolve_pin_memory(None, platform), expected)

    def test_boolean_overrides_are_independent_of_platform(self):
        for platform in ("nt", "posix"):
            for value in (True, False):
                self.assertEqual(_resolve_pin_memory(value, platform), value)

    def test_non_boolean_overrides_are_rejected(self):
        for value in (0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                _resolve_pin_memory(value)

    def test_all_four_loaders_use_the_same_policy_and_worker_rules(self):
        # Isolate loader construction from metadata and pixel I/O.
        dm = object.__new__(ImagesRecipesBaseDataModule)
        dm.batch_size = 2
        for stage in ("train", "val", "test", "predict"):
            setattr(dm, f"{stage}_dataset", [0, 1])
        for policy in (None, True, False):
            for workers in (0, 2):
                dm.pin_memory, dm.num_workers = policy, workers
                for stage in ("train", "val", "test", "predict"):
                    loader = getattr(dm, f"{stage}_dataloader")()
                    self.assertEqual(loader.pin_memory, _resolve_pin_memory(policy))
                    self.assertEqual(loader.persistent_workers, workers > 0)
                    self.assertEqual(loader.num_workers, workers)

    def test_portable_policy_survives_configuration_roundtrip(self):
        for policy in (None, True, False):
            config = ExpConfig(dm_pin_memory=policy)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                config.save_to_file(path)
                restored = ExpConfig.load_from_file(path)
            with patch.object(ImagesRecipesBaseDataModule, "__init__", return_value=None) as init:
                ImagesRecipesBaseDataModule.load_from_config(restored.datamodule, batch_size=2)
            self.assertIs(init.call_args.kwargs["pin_memory"], policy)


if __name__ == "__main__":
    unittest.main()

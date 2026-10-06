import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightning as lgn
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from src.commons.config_enc_dec import encode_config
from src.commons.exp_config import ExpConfig, HTunerExpConfig
from src.data_processing.images_recipes import ImagesRecipesBaseDataModule
from src.data_processing.labels_encoders import MultiLabelBinarizer, MultiLabelBinarizerRobust
from src.ingredient_selection import runtime
from src.lightning.custom_callbacks import FullModelCheckpoint, LightModelCheckpoint
from src.lightning.lgn_models import BaseLGNM, BaseWithSchedulerLGNM
from src.models.commons import BaseModel
from src.training.commons import load_datamodule, model_training
from scripts.analise_exp.compare_experiments.normalization import label_contract
from scripts.analise_exp.compare_experiments.comparability import trial_signature, compare_signatures, comparison_cohort
from scripts.ingredient_selection.verify_runtime import verify_runtime


class TinyImageModel(BaseModel):
    PRETTY_NAME = "Projection smoke"

    def __init__(self, num_classes=59, input_shape=(4, 4), **kwargs):
        super().__init__(num_classes=num_classes, input_shape=input_shape, **kwargs)
        self.head = torch.nn.Linear(3, num_classes)

    def forward(self, images):
        return self.head(images.mean(dim=(-2, -1)))

    @property
    def conv_target_layer(self):
        return self.head

    @property
    def classifier_target_layer(self):
        return self.head


class FrozenRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.projection = runtime.resolve_projection(runtime.PROJECTION_ID)

    def test_default_does_not_open_a_resource(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("unexpected I/O")):
            self.assertIsNone(runtime.resolve_projection())
            self.assertIsNone(ExpConfig().datamodule["ingredient_projection"])

    def test_only_exact_approved_resource_is_accepted(self):
        payload = json.loads((runtime._RESOURCES / f"{runtime.PROJECTION_ID}.json").read_text())
        payload["class_order"].reverse()
        with patch.object(Path, "read_text", return_value=json.dumps(payload)):
            with self.assertRaisesRegex(ValueError, "artifact hash"):
                runtime.resolve_projection(runtime.PROJECTION_ID)
        for bad in ("../arbitrary", "", {}, False, 59):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runtime.resolve_projection(bad)

    def test_saved_contract_roundtrip_and_tamper_rejection(self):
        saved = self.projection.to_config()
        self.assertEqual(runtime.resolve_projection(saved), self.projection)
        for key, value in (("artifact_hash", "0" * 64), ("class_order", list(reversed(saved["class_order"]))),
                           ("base_class_indices", [0]), ("class_order_hash", "wrong")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.resolve_projection(saved | {key: value})

    def test_target_projection_keeps_empty_rows_and_rejects_unknown_base_labels(self):
        included = self.projection.class_order[0]
        excluded = next(x for x in self.projection.base_class_order if x not in self.projection.class_order)
        rows = [[excluded], [included, excluded], []]
        before = copy.deepcopy(rows)
        self.assertEqual(self.projection.project_targets(rows), [[], [included], []])
        self.assertEqual(rows, before)
        with self.assertRaisesRegex(ValueError, "outside the base"):
            self.projection.project_targets([["not-a-base-label"]])
        with self.assertRaises(ValueError):
            self.projection.project_targets([included])

    def test_column_projection_numpy_torch_and_empty_batch(self):
        values = np.arange(330).reshape(2, 165)
        expected = values[:, list(self.projection.base_class_indices)]
        np.testing.assert_array_equal(runtime.project_output_columns(values, self.projection.base_class_order), expected)
        tensor = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        selected = runtime.project_output_columns(tensor, self.projection.base_class_order)
        np.testing.assert_array_equal(selected.detach().numpy(), expected)
        selected.sum().backward()
        self.assertEqual(tensor.grad.sum().item(), 118)
        self.assertEqual(runtime.project_output_columns(values[:0], self.projection.base_class_order).shape, (0, 59))

    def test_column_projection_never_guesses_order_or_width(self):
        for values, names in ((np.zeros((2, 59)), self.projection.base_class_order),
                              (np.zeros((2, 165)), self.projection.base_class_order[::-1]),
                              (np.zeros(165), self.projection.base_class_order)):
            with self.assertRaises(ValueError):
                runtime.project_output_columns(values, names)

    def test_configuration_serializes_explicit_identity_and_order(self):
        for kind in (ExpConfig, HTunerExpConfig):
            configured = kind(dm_ingredient_projection=runtime.PROJECTION_ID)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                configured.save_to_file(path)
                restored = kind.load_from_file(path)
            self.assertEqual(restored.datamodule["ingredient_projection"], self.projection.to_config())
            self.assertEqual(restored.hp["ingredient_projection"], self.projection.to_config())

    def test_configuration_rejects_incompatible_field_category_encoder_and_head(self):
        for extra in ({"dm_feature_label": "ingredients_ok"}, {"dm_category": "italian"},
                      {"dm_metadata_filename": "metadata.json"}, {"tm_num_classes": 165},
                      {"dm_label_encoder": MultiLabelBinarizerRobust().to_config()},
                      {"dm_label_encoder": MultiLabelBinarizer(classes=["salt"]).to_config()}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                ExpConfig(dm_ingredient_projection=runtime.PROJECTION_ID, **extra)

    def test_checkpoint_configuration_requires_matching_top_level_identity(self):
        config = ExpConfig(dm_ingredient_projection=runtime.PROJECTION_ID, tm_num_classes=59)
        checkpoint = {key: encode_config(value) for key, value in config.config.items()}
        checkpoint["ingredient_projection"] = self.projection.to_config()
        restored = ExpConfig.load_from_ckpt_data(checkpoint)
        self.assertEqual(restored.datamodule["ingredient_projection"], self.projection.to_config())
        checkpoint.pop("ingredient_projection")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            ExpConfig.load_from_ckpt_data(checkpoint)

    def test_model_binding_and_light_checkpoint_identity(self):
        module = BaseLGNM(TinyImageModel(), .01, 2, torch.optim.SGD, torch.nn.BCEWithLogitsLoss, metrics={})
        module.bind_ingredient_projection(runtime.PROJECTION_ID)
        checkpoint = {"hyper_parameters": {}, "datamodule_hyper_parameters": {}}
        LightModelCheckpoint().on_save_checkpoint(None, module, checkpoint)
        module.on_save_checkpoint(checkpoint)
        self.assertEqual(checkpoint, {"ingredient_projection": self.projection.to_config()})
        module.on_load_checkpoint(checkpoint)
        with self.assertRaises(ValueError):
            module.on_load_checkpoint({})
        with self.assertRaises(ValueError):
            module.bind_ingredient_projection(None)
        unbound = BaseLGNM(TinyImageModel(), .01, 2, torch.optim.SGD, torch.nn.BCEWithLogitsLoss, metrics={})
        with self.assertRaises(ValueError):
            unbound.on_load_checkpoint(checkpoint)

    def test_scheduler_model_preserves_projection_binding(self):
        config = ExpConfig(dm_ingredient_projection=runtime.PROJECTION_ID, tm_type=TinyImageModel,
                           tm_num_classes=59, lgn_model_type=BaseWithSchedulerLGNM, metrics_={})
        module = BaseWithSchedulerLGNM.load_from_config(config.hp)
        checkpoint = {}
        module.on_save_checkpoint(checkpoint)
        self.assertEqual(checkpoint["ingredient_projection"], self.projection.to_config())

    def test_analysis_uses_saved_projection_when_hpo_drops_encoder(self):
        config = ExpConfig(dm_ingredient_projection=runtime.PROJECTION_ID)
        config.drop("lb")
        contract = label_contract(config.config)
        self.assertEqual(contract["classes"], list(self.projection.class_order))
        self.assertEqual(contract["count"], 59)
        self.assertEqual(contract["projection_artifact_hash"], self.projection.artifact_hash)
        selected = trial_signature(config.config)
        full_config = ExpConfig()
        full_config.datamodule["label_encoder"]["classes"] = list(self.projection.base_class_order)
        full = trial_signature(full_config.config)
        self.assertEqual(compare_signatures(selected, full)["status"], "incompatible")
        self.assertNotEqual(comparison_cohort(selected), comparison_cohort(full))

    def test_analysis_rejects_projection_and_encoder_conflicts(self):
        config = ExpConfig(dm_ingredient_projection=runtime.PROJECTION_ID)
        config.datamodule["label_encoder"]["classes"] = list(self.projection.class_order[::-1])
        with self.assertRaisesRegex(ValueError, "encoder order"):
            label_contract(config.config)
        config.drop("lb")
        config.hp.pop("ingredient_projection")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            label_contract(config.config)

    def test_checkpoint_rejects_disagreement_between_identity_sections(self):
        config = ExpConfig(dm_ingredient_projection=runtime.PROJECTION_ID, tm_type=TinyImageModel,
                           tm_num_classes=59, metrics_={})
        module = BaseLGNM.load_from_config(config.hp)
        checkpoint = {"ingredient_projection": self.projection.to_config(),
                      "datamodule_hyper_parameters": encode_config({"ingredient_projection": None})}
        with self.assertRaisesRegex(ValueError, "fields disagree"):
            module.on_load_checkpoint(checkpoint)

    def test_full_legacy_checkpoint_without_projection_still_loads(self):
        config = ExpConfig(tm_type=TinyImageModel, tm_num_classes=165, metrics_={})
        module = BaseLGNM.load_from_config(config.hp)
        checkpoint = {"hyper_parameters": encode_config(config.hp)}
        restored = ExpConfig.load_from_ckpt_data(checkpoint)
        module.on_load_checkpoint(checkpoint)
        self.assertEqual(restored.torch_model["num_classes"], 165)
        self.assertIsNone(restored.datamodule["ingredient_projection"])


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        payload = json.loads((runtime._RESOURCES / f"{runtime.PROJECTION_ID}.json").read_text())
        base, selected = payload["base_vocabulary"]["class_order"], payload["class_order"]
        excluded = next(name for name in base if name not in selected)
        self.targets = [base, [excluded], [selected[0], excluded], []]
        self.images = self.root / "imgs" / "standard"
        self.images.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (4, 4), (20 * i, 40, 80)).save(self.images / f"{i}.png")
        metadata = payload["base_vocabulary"]["metadata_filename"]
        self.metadata = metadata
        for split in ("train", "val", "test"):
            directory = self.root / split
            directory.mkdir()
            rows = [{"id": f"{split}-{i}", "image": f"{i}.png", "cuisine": "Italian",
                     "ingredients_target": labels, "ingredients_ok": ["salt"]} for i, labels in enumerate(self.targets)]
            path = directory / metadata
            path.write_text(json.dumps(rows))
            if split != "test":
                payload["base_vocabulary"]["metadata_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.stats = self.root / "stats.csv"
        self.stats.write_text(",red,green,blue\nmean,0.5,0.5,0.5\nstd,0.2,0.2,0.2\n")
        self.original_bytes = {(split, metadata): (self.root / split / metadata).read_bytes()
                               for split in ("train", "val", "test")}
        # Register only a test-local copy, preserving real membership but binding synthetic metadata.
        payload.pop("artifact_hash")
        payload["artifact_hash"] = runtime._hash(payload)
        resources = self.root / "resources"
        resources.mkdir()
        (resources / f"{runtime.PROJECTION_ID}.json").write_text(json.dumps(payload))
        for patcher in (patch.object(runtime, "_RESOURCES", resources),
                        patch.dict(runtime._APPROVED, {runtime.PROJECTION_ID: payload["artifact_hash"]})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.projection = runtime.resolve_projection(runtime.PROJECTION_ID)

    def dm(self, selected=True, **kwargs):
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        return ImagesRecipesBaseDataModule(data_dir=self.root, images_stats_path=self.stats,
                                          batch_size=2, num_workers=0, transform_plain=transform,
                                          transform_aug=transform,
                                          ingredient_projection=runtime.PROJECTION_ID if selected else None, **kwargs)

    def config(self):
        return ExpConfig(dm_ingredient_projection=runtime.PROJECTION_ID, dm_data_dir=str(self.root),
                         dm_num_workers=0, tm_type=TinyImageModel, tm_input_shape=(4, 4),
                         optimizer=torch.optim.SGD, weighted_loss=False, metrics_={}, batch_size=2)

    def test_datamodule_keeps_full_record_population_and_default_parity(self):
        selected, full = self.dm(), self.dm(False)
        selected.prepare_data()
        full.prepare_data()
        selected.setup()
        self.assertEqual(full.get_num_classes(), 165)
        self.assertEqual(selected.get_num_classes(), 59)
        self.assertEqual(selected.hparams["ingredient_projection"], self.projection.to_config())
        for split in ("train", "val", "test", "predict"):
            self.assertEqual(selected._images_paths[split], full._images_paths[split])
            expected = self.projection.project_columns(full._label_data[split], full.label_encoder.classes)
            np.testing.assert_array_equal(selected._label_data[split], expected)
            self.assertEqual(expected.shape, (4, 59))
            self.assertEqual(expected[1].sum(), 0)
            self.assertEqual(expected[3].sum(), 0)
        images, labels = next(iter(selected.val_dataloader()))
        self.assertEqual(tuple(labels.shape), (2, 59))
        self.assertEqual(tuple(images.shape), (2, 3, 4, 4))
        for (split, name), content in self.original_bytes.items():
            self.assertEqual((self.root / split / name).read_bytes(), content)

    def test_prepare_data_is_idempotent_without_refitting_selected_order(self):
        dm = self.dm()
        with patch.object(dm.label_encoder, "fit", side_effect=AssertionError("unexpected refit")):
            dm.prepare_data()
            before = dm._label_data["train"].copy()
            dm.prepare_data()
        np.testing.assert_array_equal(before, dm._label_data["train"])

    def test_rejects_changed_training_metadata_before_loading_splits(self):
        path = self.root / "train" / self.metadata
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "train metadata"):
            self.dm().prepare_data()

    def test_unknown_test_target_fails_even_when_projection_would_discard_it(self):
        path = self.root / "test" / self.metadata
        rows = json.loads(path.read_text())
        rows[0]["ingredients_target"].append("outside-base")
        path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "outside the base"):
            self.dm().prepare_data()

    def test_rejects_reordered_or_robust_encoder(self):
        wrong = MultiLabelBinarizer(classes=list(reversed(self.projection.class_order)))
        wrong.fit()
        for encoder in (wrong, MultiLabelBinarizerRobust()):
            with self.assertRaises(ValueError):
                self.dm(label_encoder=encoder)

    def test_fitted_encoder_mapping_is_validated_even_with_correct_names(self):
        encoder = MultiLabelBinarizer(classes=list(self.projection.class_order))
        encoder.fit()
        encoder.encode_map[encoder.classes[0]] = 1
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.dm(label_encoder=encoder)

    def test_datamodule_config_roundtrip(self):
        dm = self.dm()
        dm.prepare_data()
        restored = ImagesRecipesBaseDataModule.load_from_config(dict(dm.hparams), batch_size=2,
                                                               images_stats_path=self.stats)
        restored.prepare_data()
        np.testing.assert_array_equal(dm._label_data["val"], restored._label_data["val"])
        self.assertEqual(restored.projection_config, dm.projection_config)

    def test_read_only_runtime_verifier_never_opens_test_metadata(self):
        original_open = Path.open

        def guarded(path, *args, **kwargs):
            if path.name == self.metadata and path.parent.name not in ("train", "val"):
                raise AssertionError("test metadata opened")
            return original_open(path, *args, **kwargs)

        import builtins
        real_open = builtins.open

        def guarded_builtin(path, mode="r", *args, **kwargs):
            if Path(path).name == self.metadata and Path(path).parent.name not in ("train", "val"):
                raise AssertionError("test metadata opened")
            if mode != "r":
                raise AssertionError("unexpected write")
            return real_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", guarded), patch("builtins.open", guarded_builtin):
            result = verify_runtime(self.root)
        self.assertEqual(result["writes_performed"], [])
        self.assertFalse(result["test_split_accessed"])
        self.assertEqual(result["splits"]["train"]["empty_projected_targets_retained"], 2)

    def test_canonical_training_checks_projection_before_model_construction(self):
        with patch("src.training.commons.BaseLGNM.load_from_config", side_effect=AssertionError("model constructed")):
            with self.assertRaisesRegex(ValueError, "DataModule"):
                model_training(self.config(), self.dm(False))

    def test_canonical_training_binds_config_before_delegating(self):
        dm, config = self.dm(), self.config()
        dm.prepare_data()
        trainer = unittest.mock.Mock()
        trainer.debug = False
        with patch("src.training.commons.BaseTrainer.load_from_config", return_value=trainer):
            model_training(config, dm)
        module = trainer.fit.call_args.kwargs["model"]
        self.assertEqual(module.num_classes, 59)
        self.assertEqual(module.hparams["ingredient_projection"], self.projection.to_config())
        self.assertEqual(config.label_encoder["classes"], list(self.projection.class_order))

    def test_canonical_training_loads_selected_datamodule_when_not_supplied(self):
        dm, config = self.dm(), self.config()
        dm.prepare_data()
        trainer = unittest.mock.Mock(debug=False)
        with patch("src.training.commons.load_datamodule", return_value=dm) as load, \
                patch("src.training.commons.BaseTrainer.load_from_config", return_value=trainer):
            model_training(config)
        load.assert_called_once_with(config)
        self.assertEqual(trainer.fit.call_args.kwargs["model"].num_classes, 59)

    def test_cpu_fit_full_and_light_checkpoint_reload(self):
        for callback_type in (FullModelCheckpoint, LightModelCheckpoint):
            with self.subTest(callback=callback_type.__name__):
                dm, config = self.dm(), self.config()
                dm.prepare_data()
                config.update_config(dm_label_encoder=dm.label_encoder.to_config(), tm_num_classes=59)
                model = BaseLGNM.load_from_config(config.hp)
                model.startup_model(dm)
                callback = callback_type(dirpath=self.root / callback_type.__name__, save_last=True, save_top_k=0)
                trainer = lgn.Trainer(accelerator="cpu", max_epochs=1, limit_train_batches=1,
                                      limit_val_batches=1, num_sanity_val_steps=0, logger=False,
                                      enable_progress_bar=False, enable_model_summary=False, callbacks=[callback])
                trainer.hparams = dict(config.trainer)
                trainer.fit(model, datamodule=dm)
                self.assertEqual(trainer.global_step, 1)
                checkpoint = torch.load(callback.last_model_path, weights_only=False)
                self.assertEqual(checkpoint["ingredient_projection"], self.projection.to_config())
                if callback_type is FullModelCheckpoint:
                    restored_config = ExpConfig.load_from_ckpt_data(checkpoint)
                else:
                    self.assertNotIn("hyper_parameters", checkpoint)
                    saved = self.root / "trial_config.json"
                    config.save_to_file(saved)
                    restored_config = ExpConfig.load_from_file(saved)
                restored = BaseLGNM.load_from_config(restored_config.hp)
                restored.load_weights_from_checkpoint(callback.last_model_path, weights_only=False)
                values = torch.rand(2, 3, 4, 4)
                torch.testing.assert_close(model(values), restored(values), rtol=0, atol=0)
                if callback_type is FullModelCheckpoint:
                    resumed_dm = self.dm()
                    resumed_dm.prepare_data()
                    restored.startup_model(resumed_dm)
                    resumed = lgn.Trainer(accelerator="cpu", max_epochs=2, limit_train_batches=1,
                                          limit_val_batches=1, num_sanity_val_steps=0, logger=False,
                                          enable_progress_bar=False, enable_model_summary=False,
                                          enable_checkpointing=False)
                    resumed.fit(restored, datamodule=resumed_dm, ckpt_path=callback.last_model_path)
                    self.assertEqual(resumed.global_step, 2)
                checkpoint["ingredient_projection"]["class_order"].reverse()
                with self.assertRaises(ValueError):
                    restored.on_load_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()

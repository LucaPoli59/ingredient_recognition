"""Synthetic CPU contract tests, not pretrained qualification or food evaluation."""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import lightning as lgn
import torch
from torch.utils.data import DataLoader, IterableDataset, RandomSampler, TensorDataset, WeightedRandomSampler

from src.commons.config_enc_dec import decode_config, encode_config
from src.commons.exp_config import ExpConfig
from src.data_processing.labels_encoders import MultiLabelBinarizer
from src.ingredient_selection.runtime import PROJECTION_ID, resolve_projection
from src.lightning.custom_callbacks import FullModelCheckpoint, LightModelCheckpoint
from src.lightning.experimental_lgn import (
    ExperimentalLGNM, BATCH_CHECKPOINT_KEY, ORDER_CHECKPOINT_KEY, TRAINING_CHECKPOINT_KEY,
)
from src.lightning.lgn_models import BaseLGNM
from src.models.commons import BaseModel
from src.models.experimental_contract import CHECKPOINT_CONTRACT_KEY, ExperimentalModelContract
from src.training.batching import ExactBatchPlan, finite_loader_consumed_records
from src.training.experimental_runtime import (
    is_experimental_config, load_model_for_experiment, prepare_experimental_config,
)


class TinyExperimentalAdapter(BaseModel):
    """Factory-observable analytical fixture; no TorchVision weights or image reads."""

    initialization_requests = []
    PRETTY_NAME = "synthetic experimental adapter"

    def __init__(self, contract=None, initialize_pretrained=True):
        contract = contract or ExperimentalModelContract("efficientnet_v2_s", 2)
        super().__init__(num_classes=contract.num_classes, input_shape=224)
        self.experimental_contract = contract
        type(self).initialization_requests.append(initialize_pretrained)
        self.linear = torch.nn.Linear(1, contract.num_classes)
        with torch.no_grad():
            self.linear.weight.fill_(.3)
            self.linear.bias.fill_(-.1)

    def forward(self, inputs):
        return self.linear(inputs)

    @property
    def conv_target_layer(self):
        return None

    @property
    def classifier_target_layer(self):
        return self.linear

    def to_config(self):
        return super().to_config() | {"experimental_contract": self.experimental_contract.to_config()}

    @classmethod
    def load_from_config(cls, config, *, initialize_pretrained=True):
        if config["type"] is not cls:
            raise ValueError("fixture model type disagrees")
        contract = ExperimentalModelContract.from_config(config["experimental_contract"])
        if config["num_classes"] != contract.num_classes:
            raise ValueError("fixture model width disagrees")
        return cls(contract, initialize_pretrained)


class ObservedExperimentalLGNM(ExperimentalLGNM):
    """Capture the value passed to actual Lightning logging, before loss scaling."""

    def __init__(self, *args, **kwargs):
        self.observed_losses = []
        super().__init__(*args, **kwargs)

    def log(self, name, value, *args, **kwargs):
        if name == "train_loss":
            self.observed_losses.append((float(value.detach()), kwargs["batch_size"]))
        return super().log(name, value, *args, **kwargs)


class SyntheticIterable(IterableDataset):
    def __iter__(self):
        yield torch.zeros(1), torch.zeros(2)


def synthetic_data(records=221):
    inputs = torch.arange(records, dtype=torch.float32).reshape(-1, 1) / max(records, 1)
    targets = torch.stack((torch.arange(records) % 3 == 0, torch.arange(records) % 5 == 0), dim=1).float()
    return TensorDataset(inputs, targets)


def synthetic_datamodule(order=("zucchini", "apple"), weights=None):
    encoder = MultiLabelBinarizer(classes=list(order))
    encoder.fit()
    return SimpleNamespace(label_encoder=encoder, prepare_data=lambda: None,
                           classes_weights=torch.tensor([2., 3.]) if weights is None else weights,
                           get_num_classes=lambda: len(order), projection_config=None)


def make_module(*, weighted=False, module_type=ExperimentalLGNM, output_order=("zucchini", "apple"), **kwargs):
    return module_type(TinyExperimentalAdapter(), lr=.1, batch_size=128,
                       optimizer=torch.optim.SGD, loss_fn=torch.nn.BCEWithLogitsLoss,
                       weighted_loss=weighted, physical_batch_size=8,
                       output_class_order=list(output_order) if output_order is not None else None,
                       **kwargs)


def experiment_config(module):
    order = list(module.output_class_order)
    return ExpConfig(hp_=copy.deepcopy(dict(module.hparams)), lb_classes=order,
                     lb_encode_map={name: index for index, name in enumerate(order)}, lb_fitted=True)


def checkpoint_for(module, variant="full"):
    checkpoint = {"state_dict": copy.deepcopy(module.state_dict()),
                  "hyper_parameters": copy.deepcopy(dict(module.hparams)),
                  "datamodule_hyper_parameters": {"ingredient_projection": None}}
    module.on_save_checkpoint(checkpoint)
    if variant == "full":
        FullModelCheckpoint().on_save_checkpoint(SimpleNamespace(hparams={}), module, checkpoint)
    elif variant == "light":
        LightModelCheckpoint().on_save_checkpoint(None, module, checkpoint)
    return checkpoint


class SyntheticLightningDataModule(lgn.LightningDataModule):
    def __init__(self):
        super().__init__()
        fixture = synthetic_datamodule()
        self.label_encoder = fixture.label_encoder
        self.classes_weights = fixture.classes_weights
        self.projection_config = None
        self.save_hyperparameters({"ingredient_projection": None})

    def get_num_classes(self):
        return len(self.label_encoder.classes)

    def train_dataloader(self):
        return DataLoader(synthetic_data(), batch_size=8, shuffle=False)


class ExperimentalLightningTests(unittest.TestCase):
    def test_real_lightning_partial_and_truncated_groups_match_direct_sgd(self):
        dataset = synthetic_data()
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for limit in (1.0, 22):
                with self.subTest(limit=limit), tempfile.TemporaryDirectory() as folder:
                    module = make_module(module_type=ObservedExperimentalLGNM)
                    module.startup_model(synthetic_datamodule())
                    reference = TinyExperimentalAdapter()
                    reference.load_state_dict(module.model.state_dict())
                    records = finite_loader_consumed_records(len(dataset), module.exact_batch_plan, limit)
                    inputs, targets = dataset.tensors
                    optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
                    expected_losses = []
                    for start in range(0, records, 128):
                        stop = min(start + 128, records)
                        for offset in range(start, stop, 8):
                            end = min(offset + 8, stop)
                            value = torch.nn.functional.binary_cross_entropy_with_logits(
                                reference(inputs[offset:end]), targets[offset:end])
                            expected_losses.append((float(value.detach()), end - offset))
                        optimizer.zero_grad()
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            reference(inputs[start:stop]), targets[start:stop])
                        loss.backward()
                        optimizer.step()
                    trainer = lgn.Trainer(accelerator="cpu", devices=1, max_epochs=1,
                                          accumulate_grad_batches=16, limit_train_batches=limit,
                                          logger=False, enable_checkpointing=False, enable_progress_bar=False,
                                          enable_model_summary=False, num_sanity_val_steps=0,
                                          default_root_dir=folder)
                    trainer.fit(module, train_dataloaders=DataLoader(dataset, batch_size=8, shuffle=False))
                    self.assertEqual(trainer.global_step, 2)
                    self.assertEqual(module._expected_epoch_updates, 2)
                    self.assertIsNone(module._consumed_records)
                    self.assertEqual(sum(size for _, size in module.observed_losses), records)
                    self.assertEqual(len(module.observed_losses), math.ceil(records / 8))
                    for actual, expected in zip(module.observed_losses, expected_losses):
                        self.assertEqual(actual[1], expected[1])
                        self.assertAlmostEqual(actual[0], expected[0], places=6)
                    logged_mean = sum(loss * size for loss, size in expected_losses) / records
                    self.assertAlmostEqual(float(trainer.callback_metrics["train_loss_epoch"]), logged_mean, places=6)
                    for actual, expected in zip(module.model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        finally:
            torch.set_num_threads(old_threads)

    def test_optimizer_and_scheduler_classes_remain_reconstructible(self):
        module = make_module(lr_scheduler=torch.optim.lr_scheduler.StepLR,
                             lr_scheduler_params={"step_size": 1, "gamma": .5})
        first = module.configure_optimizers()
        second = module.configure_optimizers()
        self.assertIs(module.optimizer, torch.optim.SGD)
        self.assertIs(module.lr_scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertIsInstance(first["optimizer"], torch.optim.SGD)
        self.assertIsInstance(second["lr_scheduler"], torch.optim.lr_scheduler.StepLR)
        self.assertIsNot(first["optimizer"], second["optimizer"])

    def test_hpo_variable_filter_cannot_discard_reconstruction_identity(self):
        module = make_module(hparams_to_register=["lr"])
        required = {"lgn_model_type", "torch_model", "batch_size", "lr", "optimizer", "loss_fn",
                    "weighted_loss", "exact_batch_plan", "output_class_order"}
        self.assertTrue(required <= set(module.hparams))
        module.startup_model(synthetic_datamodule())
        module.on_load_checkpoint(checkpoint_for(module))
        restored = ExperimentalLGNM.load_from_config(dict(module.hparams), initialize_pretrained=False)
        self.assertEqual(restored.exact_batch_plan, module.exact_batch_plan)
        self.assertEqual(restored.output_class_order, module.output_class_order)

    def test_projection_and_output_order_are_crossvalidated_without_an_encoder(self):
        projection = resolve_projection(PROJECTION_ID)
        order = list(projection.class_order)
        for saved_order in (order, list(reversed(order))):
            module = ExperimentalLGNM(TinyExperimentalAdapter(ExperimentalModelContract("efficientnet_v2_s", 59)),
                                      lr=.1, batch_size=128, optimizer=torch.optim.SGD,
                                      loss_fn=torch.nn.BCEWithLogitsLoss, physical_batch_size=8,
                                      output_class_order=saved_order)
            config = dict(module.hparams) | {"ingredient_projection": projection.to_config()}
            if saved_order == order:
                module.bind_ingredient_projection(projection.to_config())
                self.assertEqual(module.output_class_order, order)
                restored = ExperimentalLGNM.load_from_config(config, initialize_pretrained=False)
                self.assertEqual(restored.output_class_order, order)
                self.assertEqual(restored._ingredient_projection, projection.to_config())
            else:
                with self.assertRaises(ValueError):
                    module.bind_ingredient_projection(projection.to_config())
                with self.assertRaises(ValueError):
                    ExperimentalLGNM.load_from_config(config, initialize_pretrained=False)

    def test_native_full_and_light_checkpoints_restore_offline_complete_state(self):
        for weighted in (False, True):
            for variant in ("native", "full", "light"):
                with self.subTest(weighted=weighted, variant=variant), tempfile.TemporaryDirectory() as folder:
                    module = make_module(weighted=weighted)
                    module.startup_model(synthetic_datamodule())
                    with torch.no_grad():
                        module.model.linear.bias.add_(.27)
                    config = experiment_config(module)
                    config_path = Path(folder) / "config.json"
                    config.save_to_file(config_path)
                    config = ExpConfig.load_from_file(config_path)
                    TinyExperimentalAdapter.initialization_requests.clear()
                    fresh = load_model_for_experiment(config)
                    self.assertEqual(TinyExperimentalAdapter.initialization_requests, [True])
                    self.assertIs(fresh.loss_fn, torch.nn.BCEWithLogitsLoss)
                    checkpoint = checkpoint_for(module, variant)
                    path = Path(folder) / "fixture.ckpt"
                    torch.save(checkpoint, path)
                    restored = load_model_for_experiment(config, checkpoint_path=path)
                    self.assertEqual(TinyExperimentalAdapter.initialization_requests, [True, False])
                    self.assertEqual(restored.exact_batch_plan, module.exact_batch_plan)
                    self.assertEqual(restored.output_class_order, ["zucchini", "apple"])
                    self.assertEqual(restored.model.experimental_contract, module.model.experimental_contract)
                    self.assertEqual(restored.model.to_config()["experimental_contract"]["weights_enum"],
                                     "EfficientNet_V2_S_Weights.IMAGENET1K_V1")
                    restored.startup_model(synthetic_datamodule())
                    self.assertTrue(restored.prepared)
                    self.assertEqual(set(restored.state_dict()), set(module.state_dict()))
                    self.assertEqual("loss_fn.pos_weight" in restored.state_dict(), weighted)
                    for key, value in module.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                    inputs = torch.tensor([[.2], [.8]])
                    torch.testing.assert_close(restored(inputs), module(inputs), rtol=0, atol=0)

    def test_actual_trainer_saved_full_and_light_checkpoints_restore_offline(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for weighted, callback_type in ((False, FullModelCheckpoint), (True, LightModelCheckpoint)):
                with self.subTest(weighted=weighted, callback=callback_type), tempfile.TemporaryDirectory() as folder:
                    module = make_module(weighted=weighted)
                    data_module = SyntheticLightningDataModule()
                    module.startup_model(data_module)
                    config = experiment_config(module)
                    callback = callback_type(save_top_k=0)
                    trainer = lgn.Trainer(accelerator="cpu", devices=1, max_epochs=1,
                                          accumulate_grad_batches=16, logger=False,
                                          callbacks=[callback], enable_progress_bar=False,
                                          enable_model_summary=False, num_sanity_val_steps=0,
                                          default_root_dir=folder)
                    # FullModelCheckpoint normally obtains these from BaseTrainer.
                    trainer.hparams = {}
                    trainer.fit(module, datamodule=data_module)
                    path = Path(folder) / "actual.ckpt"
                    trainer.save_checkpoint(path)
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    self.assertEqual(payload["global_step"], 2)
                    for key in (CHECKPOINT_CONTRACT_KEY, BATCH_CHECKPOINT_KEY,
                                ORDER_CHECKPOINT_KEY, TRAINING_CHECKPOINT_KEY):
                        self.assertIn(key, payload)
                    self.assertEqual("hyper_parameters" in payload, callback_type is FullModelCheckpoint)
                    TinyExperimentalAdapter.initialization_requests.clear()
                    restored = load_model_for_experiment(config, checkpoint_path=path)
                    self.assertEqual(TinyExperimentalAdapter.initialization_requests, [False])
                    restored.startup_model(data_module)
                    self.assertEqual(restored.exact_batch_plan, module.exact_batch_plan)
                    for key, value in module.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                    resume_trainer = lgn.Trainer(accelerator="cpu", devices=1, max_epochs=2,
                                                 accumulate_grad_batches=16, logger=False,
                                                 callbacks=[callback_type(save_top_k=0)], enable_progress_bar=False,
                                                 enable_model_summary=False, num_sanity_val_steps=0,
                                                 default_root_dir=folder)
                    resume_trainer.hparams = {}
                    resume_trainer.fit(restored, datamodule=data_module, ckpt_path=path)
                    self.assertEqual(resume_trainer.global_step, 4)
                    self.assertEqual(restored._processed_records, 221)
        finally:
            torch.set_num_threads(old_threads)

    def test_missing_checkpoint_identity_and_inconsistent_fields_are_rejected(self):
        module = make_module()
        module.startup_model(synthetic_datamodule())
        original = checkpoint_for(module)
        corruptions = []
        for key in (CHECKPOINT_CONTRACT_KEY, BATCH_CHECKPOINT_KEY, ORDER_CHECKPOINT_KEY, TRAINING_CHECKPOINT_KEY):
            damaged = copy.deepcopy(original)
            del damaged[key]
            corruptions.append(damaged)
        damaged = copy.deepcopy(original)
        damaged[ORDER_CHECKPOINT_KEY] = ["apple", "zucchini"]
        corruptions.append(damaged)
        damaged = copy.deepcopy(original)
        damaged[BATCH_CHECKPOINT_KEY] = ExactBatchPlan.resolve(128, physical_batch_size=16).to_config()
        corruptions.append(damaged)
        damaged = copy.deepcopy(original)
        damaged[CHECKPOINT_CONTRACT_KEY] = ExperimentalModelContract("maxvit_t", 2).to_config()
        corruptions.append(damaged)
        damaged = checkpoint_for(module, "light")
        params = decode_config(damaged[TRAINING_CHECKPOINT_KEY])
        params["lr"] = .2
        damaged[TRAINING_CHECKPOINT_KEY] = encode_config(params)
        corruptions.append(damaged)
        for field, value in (("output_class_order", ["apple", "zucchini"]), ("weighted_loss", True),
                             ("batch_size", 64), ("lgn_model_type", BaseLGNM)):
            damaged = copy.deepcopy(original)
            params = decode_config(damaged["hyper_parameters"])
            params[field] = value
            damaged["hyper_parameters"] = encode_config(params)
            corruptions.append(damaged)
        for index, checkpoint in enumerate(corruptions):
            with self.subTest(corruption=index), self.assertRaises(ValueError):
                module.on_load_checkpoint(checkpoint)

    def test_incomplete_model_state_and_illegal_loss_buffers_are_rejected(self):
        for weighted, mutation in (
                (False, lambda state: state.pop("_model.linear.weight")),
                (False, lambda state: state.update({"loss_fn.pos_weight": torch.ones(2)})),
                (True, lambda state: state.pop("loss_fn.pos_weight")),
                (True, lambda state: state.update({"loss_fn.pos_weight": torch.ones(1)})),
                (True, lambda state: state.update({"loss_fn.pos_weight": torch.tensor([-1., 2.])})),
                (True, lambda state: state.update({"loss_fn.pos_weight": torch.tensor([float("nan"), 2.])}))):
            with self.subTest(weighted=weighted, mutation=mutation), tempfile.TemporaryDirectory() as folder:
                module = make_module(weighted=weighted)
                module.startup_model(synthetic_datamodule())
                checkpoint = checkpoint_for(module)
                mutation(checkpoint["state_dict"])
                path = Path(folder) / "damaged.ckpt"
                torch.save(checkpoint, path)
                with self.assertRaises((ValueError, RuntimeError)):
                    load_model_for_experiment(experiment_config(module), checkpoint_path=path)

    def test_startup_rejects_encoder_width_order_and_restored_positive_weights(self):
        module = make_module(weighted=True)
        module.startup_model(synthetic_datamodule())
        for datamodule in (synthetic_datamodule(("apple", "zucchini")),
                           synthetic_datamodule(("zucchini",)),
                           synthetic_datamodule(weights=torch.tensor([1., 3.]))):
            with self.subTest(order=datamodule.label_encoder.classes), self.assertRaises(ValueError):
                module.startup_model(datamodule)

    def test_class_order_and_batch_configuration_are_explicit_and_immutable(self):
        for order in (("same", "same"), ("", "apple"), ("one",), ("one", 2)):
            with self.subTest(order=order), self.assertRaises(ValueError):
                make_module(output_order=order)
        module = make_module(output_order=None)
        with self.assertRaises(ValueError):
            module.on_save_checkpoint({})
        module.bind_output_class_order(["zucchini", "apple"])
        module.batch_size = 128
        self.assertEqual(module.batch_size, 8)
        with self.assertRaises(ValueError):
            module.batch_size = 16
        with self.assertRaises(ValueError):
            module.bind_output_class_order(["apple", "zucchini"])
        for override in ({"physical_batch_size": 16}, {"exact_batch_plan": ExactBatchPlan.resolve(64).to_config()}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                ExperimentalLGNM.load_from_config(dict(module.hparams), lgn_model_kwargs=override)

    def test_epoch_start_rejects_loaders_outside_the_finite_contract(self):
        module = make_module()
        dataset = synthetic_data()
        standard = DataLoader(dataset, batch_size=8)
        trainer_fields = dict(train_dataloader=standard, world_size=1, accumulate_grad_batches=16,
                              fast_dev_run=False, limit_train_batches=1.0, num_training_batches=28,
                              global_step=0, max_steps=-1)
        illegal_loaders = [None, DataLoader(dataset, batch_size=8, drop_last=True),
                           DataLoader(dataset, batch_size=16),
                           DataLoader(SyntheticIterable(), batch_size=8),
                           DataLoader(dataset, batch_size=8, sampler=RandomSampler(dataset, replacement=True)),
                           DataLoader(dataset, batch_size=8,
                                      sampler=WeightedRandomSampler(torch.ones(len(dataset)), len(dataset)))]
        invalid = [{"train_dataloader": value} for value in illegal_loaders]
        invalid += [{"world_size": 2}, {"accumulate_grad_batches": 8}, {"fast_dev_run": True},
                    {"num_training_batches": 27}, {"limit_train_batches": .001}]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                module._trainer = SimpleNamespace(**(trainer_fields | overrides))
                module.on_train_epoch_start()
        module._trainer = SimpleNamespace(**trainer_fields)
        module.on_train_epoch_start()
        self.assertEqual(module._consumed_records, 221)
        self.assertEqual(module._expected_epoch_updates, 2)
        module._trainer.global_step = 1
        with self.assertRaises(ValueError):
            module.on_train_epoch_end()
        module._trainer.global_step = 2
        module._processed_records = 220
        with self.assertRaises(ValueError):
            module.on_train_epoch_end()
        module._processed_records = 221
        module.on_train_epoch_end()
        self.assertIsNone(module._consumed_records)
        module._trainer = None

    def test_training_resume_rejects_partial_or_malformed_saved_batch_progress(self):
        module = make_module()
        module.startup_model(synthetic_datamodule())
        module._trainer = SimpleNamespace(num_training_batches=28)
        module.on_train_start()
        for progress in ({"current": {"ready": 0}, "is_last_batch": False},
                         {"current": {"ready": 28}, "is_last_batch": True}):
            checkpoint = checkpoint_for(module)
            checkpoint["loops"] = {"fit_loop": {"epoch_loop.batch_progress": progress}}
            module.on_load_checkpoint(checkpoint)
            module.on_train_start()
        for progress in ({}, [], {"current": {}}, {"current": {"ready": True}},
                         {"current": {"ready": -1}}, {"current": {"ready": 0.}},
                         {"current": {"ready": 2}, "is_last_batch": False},
                         {"current": {"ready": 16}, "is_last_batch": False},
                         {"current": {"ready": 28}, "is_last_batch": False},
                         {"current": {"ready": 29}, "is_last_batch": True}):
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                checkpoint = checkpoint_for(module)
                checkpoint["loops"] = {"fit_loop": {"epoch_loop.batch_progress": progress}}
                module.on_load_checkpoint(checkpoint)
                module.on_train_start()
        module._trainer = None

    def test_training_requires_verified_mean_reduced_loss(self):
        module = make_module()
        module.startup_model(synthetic_datamodule())
        batch = tuple(tensor[:8] for tensor in synthetic_data().tensors)
        with self.assertRaises(ValueError):
            module.training_step(batch, 0)
        module._consumed_records = 221
        module.loss_fn.reduction = "sum"
        with self.assertRaises(ValueError):
            module.training_step(batch, 0)

    def test_runtime_rejects_disagreeing_saved_encoder_contract_and_restore_intent(self):
        module = make_module()
        config = experiment_config(module)
        config.label_encoder["classes"] = ["apple", "zucchini"]
        with self.assertRaises(ValueError):
            load_model_for_experiment(config)
        config = experiment_config(module)
        config.label_encoder["fitted"] = False
        config.hp.pop("output_class_order")
        with self.assertRaises(ValueError):
            load_model_for_experiment(config, checkpoint_path="must-not-be-opened.ckpt")
        config = experiment_config(module)
        config.torch_model.pop("experimental_contract")
        with self.assertRaises(ValueError):
            is_experimental_config(config)
        config = experiment_config(module)
        config.hp["lgn_model_type"] = BaseLGNM
        with self.assertRaises(ValueError):
            is_experimental_config(config)
        with self.assertRaises(ValueError):
            BaseLGNM(TinyExperimentalAdapter(), lr=.01, batch_size=128,
                     optimizer=torch.optim.SGD, loss_fn=torch.nn.BCEWithLogitsLoss)
        with self.assertRaises(ValueError):
            ExperimentalLGNM.load_from_checkpoint("must-not-be-opened.ckpt")

    def test_prepare_config_persists_exact_plan_fitted_encoder_and_order_before_save(self):
        module = make_module()
        config = experiment_config(module)
        config.hp.pop("exact_batch_plan")
        config.hp.pop("output_class_order")
        config.datamodule["label_encoder"] = MultiLabelBinarizer().to_config()
        data_module = synthetic_datamodule()
        prepare_experimental_config(config, data_module, {"physical_batch_size": 8})
        self.assertEqual(config.hp["exact_batch_plan"], module.exact_batch_plan.to_config())
        self.assertEqual(config.hp["output_class_order"], module.output_class_order)
        self.assertEqual(config.label_encoder, data_module.label_encoder.to_config())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prepared.json"
            config.save_to_file(path)
            restored = ExpConfig.load_from_file(path)
        reconstructed = load_model_for_experiment(restored)
        self.assertEqual(reconstructed.exact_batch_plan, module.exact_batch_plan)
        self.assertEqual(reconstructed.output_class_order, module.output_class_order)
        reversed_module = synthetic_datamodule(("apple", "zucchini"))
        with self.assertRaises(ValueError):
            prepare_experimental_config(config, reversed_module)

    def test_encoder_column_mapping_is_validated_not_only_the_class_names(self):
        module = make_module()
        config = experiment_config(module)
        config.label_encoder["encode_map"] = {"zucchini": 1, "apple": 0}
        with self.assertRaises(ValueError):
            load_model_for_experiment(config)
        data_module = synthetic_datamodule()
        data_module.label_encoder.encode_map = {"zucchini": 1, "apple": 0}
        with self.assertRaises(ValueError):
            module.startup_model(data_module)
        config = experiment_config(module)
        with self.assertRaises(ValueError):
            prepare_experimental_config(config, data_module)

    def test_full_default_and_legacy_load_path_remain_unchanged(self):
        config = ExpConfig()
        self.assertFalse(is_experimental_config(config))
        self.assertIsNone(config.datamodule["ingredient_projection"])
        self.assertEqual(config.datamodule["feature_label"], "ingredients_target")
        self.assertNotIn("experimental_contract", config.torch_model)
        legacy = SimpleNamespace(load_weights_from_checkpoint=mock.Mock())
        with mock.patch.object(BaseLGNM, "load_from_config", return_value=legacy) as factory:
            self.assertIs(load_model_for_experiment(config, checkpoint_path="legacy.ckpt"), legacy)
        factory.assert_called_once()
        legacy.load_weights_from_checkpoint.assert_called_once_with(
            "legacy.ckpt", weights_only=False, drop_fields=None)
        legacy.load_weights_from_checkpoint.reset_mock()
        with mock.patch.object(BaseLGNM, "load_from_config", return_value=legacy):
            load_model_for_experiment(config, checkpoint_path="legacy.ckpt", legacy_drop_fields=["loss_fn.pos_weight"])
        legacy.load_weights_from_checkpoint.assert_called_once_with(
            "legacy.ckpt", weights_only=False, drop_fields=["loss_fn.pos_weight"])


if __name__ == "__main__":
    unittest.main()

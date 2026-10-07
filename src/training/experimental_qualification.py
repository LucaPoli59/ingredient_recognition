"""Bounded portfolio qualification through canonical training and consumers.

Disposable capacity probes and fresh train/validation smokes are engineering
evidence. They never choose architectures, hyperparameters or ingredients.
"""

from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from unittest.mock import patch

import lightning as lgn
import numpy as np
import torch
import torchvision
from PIL import Image

from settings.config import YUMMLY_PATH, YUMMLY_TARGET_METADATA_FILENAME
from src.commons.exp_config import ExpConfig
from src.data_processing.images_recipes import ImagesRecipesBaseDataModule
from src.ingredient_selection.runtime import PROJECTION_ID, resolve_projection, project_output_columns
from src.lightning.custom_callbacks import FullModelCheckpoint, LightModelCheckpoint
from src.lightning.experimental_lgn import ExperimentalLGNM
from src.lightning.lgn_trainers import BaseTrainer
from src.models.experimental_contract import ExperimentalModelContract, experimental_smoke_policy
from src.models.experimental_dica_net import DICANetSExperiment
from src.models.experimental_efficientnet import EfficientNetV2SExperiment
from src.models.experimental_maxvit import MaxViTTExperiment
from src.training.commons import load_datamodule, model_training, set_torch_constants
from src.training.experimental_runtime import load_model_for_experiment

ROOT = Path(__file__).resolve().parents[2]
MODEL_TYPES = {"efficientnet_v2_s": EfficientNetV2SExperiment,
               "maxvit_t": MaxViTTExperiment, "p2_s": DICANetSExperiment}
_ARTIFACTS = {
    "efficientnet_v2_s": ("efficientnet_v2_s-dd5fe13b.pth", 86721253,
        "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"),
    "maxvit_t": ("maxvit_t-bc5ab103.pth", 124538661,
        "bc5ab103d47a7c6c02dc35bf65796b3a6cccf3d51bce330326dbdc634a3ac0e1"),
}
RESTORE_TOLERANCE = {"rtol": 0.0, "atol": 0.0}  # Same CPU eval backend and saved state.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_record(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_directory(name):
    if not name or Path(name).name != name or name in (".", "..") or "\\" in name:
        raise ValueError("run-name must be a single directory name")
    return ROOT / "experiments" / "runtime_smoke" / name


def metadata_hashes():
    return {split: sha256(Path(YUMMLY_PATH) / split / YUMMLY_TARGET_METADATA_FILENAME)
            for split in ("train", "val", "test")}


def source_inventory():
    paths = set((ROOT / "src").rglob("*.py")) | set((ROOT / "settings").rglob("*.py"))
    paths.add(ROOT / "scripts/validation/experimental_model_smoke.py")
    paths.add(ROOT / "src/ingredient_selection/resources" / f"{PROJECTION_ID}.json")
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def verify_artifact(model_id):
    family = "maxvit_t" if model_id == "maxvit_t" else "efficientnet_v2_s"
    filename, size, expected = _ARTIFACTS[family]
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file() or path.stat().st_size != size or sha256(path) != expected:
        raise ValueError(f"approved original artifact unavailable or mismatched: {filename}")
    return {"filename": filename, "size_bytes": size, "sha256": expected,
            "url": f"https://download.pytorch.org/models/{filename}",
            "weights_enum": MODEL_TYPES[model_id].WEIGHTS_ENUM}


def original_encoder_matches(adapter, artifact):
    state = torch.load(Path(torch.hub.get_dir()) / "checkpoints" / artifact["filename"],
                       map_location="cpu", weights_only=True)
    retained = {name: value for name, value in adapter.model.state_dict().items()
                if name.startswith(("features.", "stem.", "blocks."))}
    expected = {name: value for name, value in state.items()
                if name.startswith(("features.", "stem.", "blocks."))}
    if retained.keys() != expected.keys() or not retained:
        raise ValueError("original encoder state keys differ")
    for name, value in retained.items():
        if not torch.equal(value.detach().cpu(), expected[name]):
            raise ValueError(f"original encoder state changed before training: {name}")
    return len(retained)


def make_smoke_config(model_id, adaptation, physical_batch, selected, trial, updates):
    policy = experimental_smoke_policy()
    if model_id not in MODEL_TYPES or adaptation not in ("full", "frozen_encoder"):
        raise ValueError("unknown qualification model/adaptation")
    if physical_batch not in policy["physical_probe_candidates"]:
        raise ValueError("physical batch must be a declared probe candidate")
    contract = ExperimentalModelContract(model_id, 59 if selected else 165, adaptation, policy["seed"])
    # This nontraining template supplies every adapter-specific serialization
    # field. Canonical model_training still constructs a fresh approved model.
    with torch.random.fork_rng(devices=[]):
        template = MODEL_TYPES[model_id](contract=contract, initialize_pretrained=False)
        model_config = template.to_config()
        del template
    return ExpConfig(hp_lgn_model_type=ExperimentalLGNM, hp_torch_model=model_config,
                     hp_batch_size=policy["requested_batch_size"], hp_physical_batch_size=physical_batch,
                     hp_max_physical_batch_size=physical_batch, hp_optimizer=torch.optim.AdamW,
                     hp_lr=policy["lr"], hp_weight_decay=policy["weight_decay"], hp_weighted_loss=False,
                     hp_loss_fn=torch.nn.BCEWithLogitsLoss, hp_lr_scheduler=None,
                     dm_ingredient_projection=PROJECTION_ID if selected else None,
                     dm_num_workers=policy["num_workers"], dm_pin_memory=policy["pin_memory"],
                     tr_type=BaseTrainer, tr_save_dir=str(trial), tr_max_epochs=1,
                     tr_limit_train_batches=updates * (policy["requested_batch_size"] // physical_batch),
                     tr_log_every_n_steps=1,
                     lgg_wandb_notes="Phase 5.5 bounded engineering qualification; not benchmark evidence")


class QualificationEvidence(lgn.Callback):
    def __init__(self, artifact, width, updates):
        self.artifact, self.width, self.updates = artifact, width, updates
        self.record = {"train_batches": 0, "train_records": 0, "zero_target_train_records": 0,
                       "cuda_validation_batches": 0, "optimizer_steps": 0,
                       "completed_optimizer_steps": 0, "step_seconds": []}

    def on_fit_start(self, trainer, module):
        self.record["original_encoder_entries_exact"] = original_encoder_matches(module.model, self.artifact)
        normalization = [m for m in module.model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        momentum = .01 if module.model.experimental_contract.model_id == "maxvit_t" else .1
        assert normalization and all(m.eps == .001 and m.momentum == momentum for m in normalization)
        self.record["batch_norm"] = {"modules": len(normalization), "eps": .001, "momentum": momentum,
                                     "physical_batch_statistics": module.model.adaptation == "full"}
        trainer.log_hparams({"qualification_policy": experimental_smoke_policy()})

    def on_train_start(self, trainer, module):
        assert module.device.type == "cuda"
        assert all(p.dtype == torch.float32 for p in module.model.parameters())
        self.name, self.parameter = next((name, p) for name, p in module.model.named_parameters() if p.requires_grad)
        self.before = self.parameter.detach().cpu().clone()
        torch.cuda.synchronize()
        self.started = self.last_step = time.perf_counter()
        self.record["total_parameters"] = sum(p.numel() for p in module.model.parameters())
        self.record["trainable_parameters"] = sum(p.numel() for p in module.model.parameters() if p.requires_grad)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        assert batch[0].is_cuda and batch[1].is_cuda and batch[1].shape[1] == self.width
        assert torch.isfinite(outputs["loss"]).all()
        self.record["train_batches"] += 1
        self.record["train_records"] += len(batch[0])
        self.record["zero_target_train_records"] += int((batch[1].sum(1) == 0).sum())
        if trainer.global_step > self.record["completed_optimizer_steps"]:
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.record["step_seconds"].append(now - self.last_step)
            self.last_step = now
            self.record["completed_optimizer_steps"] = trainer.global_step

    def on_before_optimizer_step(self, trainer, module, optimizer):
        gradients = [p.grad for p in module.model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert len(gradients) == sum(p.requires_grad for p in module.model.parameters())
        assert any(torch.count_nonzero(g) for g in gradients)
        self.record["optimizer_steps"] += 1

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        # Called before teardown. Post-fit module.device is not CUDA evidence.
        assert module.device.type == "cuda" and batch[0].is_cuda and batch[1].is_cuda
        assert batch[1].shape[1] == self.width
        assert outputs.is_cuda and torch.isfinite(outputs).all()
        self.record["cuda_validation_batches"] += 1

    def on_train_end(self, trainer, module):
        torch.cuda.synchronize()
        assert not torch.equal(self.before, self.parameter.detach().cpu())
        assert trainer.global_step == self.updates == self.record["optimizer_steps"] == self.record["completed_optimizer_steps"]
        assert self.record["train_records"] == self.updates * experimental_smoke_policy()["requested_batch_size"]
        assert self.record["cuda_validation_batches"] == 4  # Two sanity + two final.
        assert all(torch.isfinite(value).all() for value in trainer.callback_metrics.values())
        optimizer = trainer.optimizers[0]
        state_tensors = [v for state in optimizer.state.values() for key, v in state.items()
                         if key in ("exp_avg", "exp_avg_sq")]
        assert state_tensors and all(v.is_cuda for v in state_tensors)
        self.record.update(finite_loss_and_gradients=True, trainable_parameter_changed=self.name,
                           optimizer_state_allocated=True, optimizer_state_tensors=len(state_tensors),
                           training_seconds=time.perf_counter() - self.started,
                           cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)


def verify_analysis_identity(outputs, names, selected):
    projection = resolve_projection(PROJECTION_ID)
    if selected:
        if tuple(names) != projection.class_order or outputs.ndim != 2 or outputs.shape[1] != len(names):
            raise ValueError("selected analysis output identity differs")
    else:
        projected = project_output_columns(outputs, source_class_order=names, projection=projection.to_config())
        torch.testing.assert_close(projected, outputs[:, list(projection.base_class_indices)], rtol=0, atol=0)
        try:
            project_output_columns(outputs, source_class_order=list(reversed(names)), projection=projection.to_config())
        except ValueError:
            pass
        else:
            raise AssertionError("analysis accepted a reordered source vocabulary")


def selected_row_parity(dm):
    base = load_datamodule(ExpConfig(dm_num_workers=2))
    projection = resolve_projection(PROJECTION_ID)
    for split in ("train", "val"):
        full, selected = getattr(base, f"{split}_dataset"), getattr(dm, f"{split}_dataset")
        assert list(full.images_paths) == list(selected.images_paths)
        expected = projection.project_columns(full.label_data, base.label_encoder.classes)
        np.testing.assert_array_equal(expected, selected.label_data)
    return {"train_validation_record_order_exact": True, "every_target_row_exact_projection": True}


def restore_and_analyze(config, module, dm, checkpoint, job):
    checkpoint_hash = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = ExpConfig.load_from_ckpt_data(payload)
    assert saved.label_encoder == config.label_encoder
    assert saved.datamodule["ingredient_projection"] == config.datamodule["ingredient_projection"]
    module.cpu().eval()
    image, _ = dm.val_dataset[0]
    with torch.no_grad():
        expected = module(image[None])
    light = copy.copy(payload)
    LightModelCheckpoint().on_save_checkpoint(None, module, light)
    for key in ("optimizer_states", "lr_schedulers", "loops"):
        light.pop(key, None)
    light_path = job / "light_model.ckpt"
    torch.save(light, light_path)
    with patch("torch.hub.download_url_to_file", side_effect=AssertionError("offline restoration required")), \
            patch("torchvision.models.EfficientNet_V2_S_Weights.get_state_dict", side_effect=AssertionError("offline required")), \
            patch("torchvision.models.MaxVit_T_Weights.get_state_dict", side_effect=AssertionError("offline required")):
        for path, external in ((checkpoint, saved), (light_path, config)):
            restored = load_model_for_experiment(external, checkpoint_path=path).cpu().eval()
            with torch.no_grad():
                actual = restored(image[None])
            torch.testing.assert_close(actual, expected, **RESTORE_TOLERANCE)
            assert restored.output_class_order == list(dm.label_encoder.classes)
            assert restored._ingredient_projection == dm.projection_config
            del restored
    verify_analysis_identity(expected, module.output_class_order, dm.projection_config is not None)
    assert checkpoint_hash == sha256(checkpoint)
    return {"full_and_light_offline_restore": True, "cpu_logits_tolerance": RESTORE_TOLERANCE,
            "saved_order_projection_exact": True, "analysis_identity_verified": True,
            "checkpoint_sha256": checkpoint_hash, "light_sha256": sha256(light_path),
            "light_provenance": "same full payload through LightModelCheckpoint save hook; optimizer/loops removed"}


def load_dashboard(job):
    from src.dashboards import _commons
    cache = job / "dashboard_cache"
    cache.mkdir(exist_ok=False)
    _commons.DASH_CACHE = str(cache)
    from src.dashboards.dash.app import app
    import dash
    name = next(name for name, page in dash.page_registry.items() if page["path"] == "/model_visualization")
    return app, importlib.import_module(name)


def verify_dashboard_preprocessing(config, checkpoint, image):
    from src.dashboards.runtime import load_visualization_datamodule
    # The cache contains a BaseModel adapter; DataModule reconstruction requires
    # the Lightning wrapper which owns projection/encoder binding.
    reference = load_model_for_experiment(config, checkpoint_path=checkpoint).cpu().eval()
    reference_dm = load_visualization_datamodule(config, reference)
    torch.testing.assert_close(image, reference_dm.val_dataset[0][0], rtol=0, atol=0)


def dashboard_check(job, checkpoint, config):
    import codecs
    import pickle
    import jsonpickle
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    from src.dashboards.runtime import prepare_visualization_image
    app, page = load_dashboard(job)
    client = app.server.test_client()
    statuses = {url: client.get(url).status_code for url in
                ("/", "/model_visualization", "/_dash-layout", "/_dash-dependencies")}
    assert all(status == 200 for status in statuses.values())
    before_hash = sha256(checkpoint)
    token = context_value.set(AttributeDict(triggered_inputs=[{"prop_id": "load_exp_button.n_clicks", "value": 1}]))
    try:
        loaded = page.load_experiment(1, str(checkpoint.parent), None, None, None)
    finally:
        context_value.reset(token)
    assert loaded[-1] == "success", loaded[-2]
    data, index, image_url = page.load_images(loaded[0])[:3]
    response = client.get(image_url)
    assert response.status_code == 200 and response.mimetype.startswith("image/")
    assert len(data) == 5996
    encoder = jsonpickle.decode(loaded[2])
    assert list(encoder.classes) == list(config.label_encoder["classes"])
    transform = pickle.loads(codecs.decode(loaded[1].encode(), "base64"))
    cached = page._load_model().cuda().eval()
    with Image.open(data[0]["img"]) as source:
        image, _ = prepare_visualization_image(transform, source)
    verify_dashboard_preprocessing(config, checkpoint, image)
    with torch.no_grad():
        expected = cached(image[None].cuda())[0]
    assert expected.is_cuda and torch.isfinite(expected).all()
    table = page._create_preds_table(torch.sigmoid(expected).cpu().numpy(), encoder)
    observed = []
    original_load = page._load_model
    handles = []
    def checked_load():
        adapter = original_load()
        def check_input(_module, inputs):
            assert inputs[0].is_cuda
            observed.append(inputs[0].requires_grad)
        handles.append(adapter.register_forward_pre_hook(check_input))
        return adapter
    try:
        with patch.object(page, "_load_model", side_effect=checked_load):
            inferred = page.make_inference(1, None, index, data, .5, loaded[1], loaded[2], device="cuda")
    finally:
        for handle in handles:
            handle.remove()
    supports = cached.supports_feature_factorization
    assert inferred[-1] == ("success" if supports else "info"), inferred[-2]
    assert any(observed) and len(inferred[0].data) == 1
    assert len(inferred[1].data) == (1 if supports else 0)
    if not supports:
        assert "unavailable" in inferred[-2] and inferred[1].layout.annotations
    assert [row["Ingredients"] for row in inferred[3]] == table["Ingredients"].tolist()
    np.testing.assert_allclose([row["Confidence"] for row in inferred[3]], table["Confidence"], rtol=2e-5, atol=2e-6)
    assert sha256(checkpoint) == before_hash
    return {"http": statuses, "display_image_http": 200, "validation_records": len(data),
            "preprocessing_exact": True, "ordered_predictions_match_direct_cuda": True,
            "cuda_gradcam_input_gradients": True, "feature_factorization": "passed" if supports else "explicitly_unsupported",
            "checkpoint_unchanged": True, "cache_isolated": True,
            "validation_image": str(Path(data[0]["img"]).relative_to(ROOT)),
            "browser_visual_acceptance": "not_performed"}


def execute_job(job, model_id, adaptation, physical_batch, selected, kind):
    """One fresh subprocess per probe/smoke; exceptions retained in result.json."""
    job = Path(job)
    if job.resolve().parent.parent.parent != (ROOT / "experiments/runtime_smoke").resolve():
        raise ValueError("qualification job must stay in its isolated runtime-smoke run")
    job.mkdir(parents=True, exist_ok=False)
    result = {"purpose": "engineering_only", "model_id": model_id, "adaptation": adaptation,
              "selected": selected, "kind": kind, "status": "running", "policy": experimental_smoke_policy()}
    result_path = job / "result.json"
    write_record(result_path, result)
    try:
        if kind not in ("probe", "smoke") or (kind == "probe" and selected):
            raise ValueError("unsupported qualification job")
        if not torch.cuda.is_available():
            raise RuntimeError("qualification requires CUDA")
        policy = experimental_smoke_policy()
        set_torch_constants()
        torch.set_float32_matmul_precision(policy["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = policy["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = policy["cudnn_allow_tf32"]
        torch.set_num_threads(4)
        lgn.seed_everything(policy["seed"], workers=True)
        free, total = torch.cuda.mem_get_info()
        allowance = min(int(total * policy["memory_allowance_fraction"]), free - 256 * 2**20)
        if allowance <= 0:
            raise RuntimeError("no useful free CUDA allowance")
        torch.cuda.set_per_process_memory_fraction(allowance / total)
        result["resource"] = {"device": torch.cuda.get_device_name(), "total_mib": total / 2**20,
                              "free_before_mib": free / 2**20, "other_use_mib": (total - free) / 2**20,
                              "policy_allowance_mib": total * policy["memory_allowance_fraction"] / 2**20,
                              "operational_allowance_mib": allowance / 2**20,
                              "precision": policy["precision"], "matmul_precision": torch.get_float32_matmul_precision(),
                              "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn_tf32": torch.backends.cudnn.allow_tf32}
        result["artifact"] = verify_artifact(model_id)
        before_metadata = metadata_hashes()
        updates = policy["probe_optimizer_updates" if kind == "probe" else "smoke_optimizer_updates"]
        trial = job / "trial_0"
        trial.mkdir()
        scratch = job / "trainer_scratch"
        scratch.mkdir()
        config = make_smoke_config(model_id, adaptation, physical_batch, selected, trial, updates)
        with ExitStack() as guards:
            for method in ("test_dataloader", "predict_dataloader"):
                guards.enter_context(patch.object(ImagesRecipesBaseDataModule, method,
                    side_effect=AssertionError("test/predict image inference forbidden")))
            guards.enter_context(patch("src.lightning.lgn_trainers.EXPERIMENTS_TRASH_PATH", str(scratch)))
            guards.enter_context(patch("src.lightning.lgn_trainers.EXPERIMENTS_WANDB_PATH", str(job / "wandb")))
            dm = load_datamodule(config)
            result["data"] = {"metadata_sha256": before_metadata, "metadata_only_test_access": True,
                              "predictive_test_evaluation": False,
                              "train_records": len(dm.train_dataset), "validation_records": len(dm.val_dataset),
                              "zero_train_targets_retained": int((np.asarray(dm.train_dataset.label_data).sum(1) == 0).sum()),
                              "zero_validation_targets_retained": int((np.asarray(dm.val_dataset.label_data).sum(1) == 0).sum())}
            assert len(dm.train_dataset) == 47965 and len(dm.val_dataset) == 5996
            assert list(dm.label_encoder.classes) == list(resolve_projection(PROJECTION_ID).class_order if selected
                                                          else resolve_projection(PROJECTION_ID).base_class_order)
            if selected:
                result["data"]["projection_parity"] = selected_row_parity(dm)
            checkpoint = FullModelCheckpoint(dirpath=trial / "checkpoints", monitor="val_loss", save_top_k=1, save_last=True)
            evidence = QualificationEvidence(result["artifact"], dm.get_num_classes(), updates)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            trainer, module = model_training(config, dm, trainer_kwargs={
                "limit_val_batches": policy["validation_batches"], "precision": policy["precision"],
                "profiler": None, "callbacks": [checkpoint, evidence], "model_checkpoint_callback": checkpoint,
                "enable_progress_bar": False, "devices": 1, "gradient_clip_val": 0.0})
            result["training"] = evidence.record
            result["batch_plan"] = module.exact_batch_plan.to_config()
            result["model_contract"] = module.model.experimental_contract.to_config()
            result["class_order"] = module.output_class_order
            result["ingredient_projection"] = dm.projection_config
            result["pin_memory_resolved"] = dm._pin_memory_enabled
            assert result["training"]["cuda_peak_reserved_mib"] <= allowance / 2**20
            config.save_to_file(trial / "smoke_config.json")
            if kind == "smoke":
                path = trial / "best_model.ckpt"
                result["restoration"] = restore_and_analyze(config, module, dm, path, job)
                del module, trainer, evidence, dm
                torch.cuda.empty_cache()
                result["dashboard"] = dashboard_check(job, path, config)
        assert before_metadata == metadata_hashes()
        result.update(status="passed", metadata_unchanged=True)
        write_record(result_path, result)
        return 0
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=str(error),
                      cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        write_record(result_path, result)
        return 70
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_record(result_path, result)
        raise


def source_provenance(run):
    inventory = source_inventory()
    archive = run / "source_snapshot.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zipped:
        for relative in inventory:
            zipped.write(ROOT / relative, relative)
    installed = Path(torchvision.__file__).parent
    import importlib.metadata
    distribution = importlib.metadata.distribution("torchvision")
    license_path = next(distribution.locate_file(item) for item in distribution.files if str(item).endswith("/LICENSE"))
    return {"base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_inventory": inventory, "source_snapshot_sha256": sha256(archive),
            "torch": torch.__version__, "torch_revision": torch.version.git_version,
            "torchvision": torchvision.__version__, "torchvision_revision": torchvision.version.git_version,
            "lightning": lgn.__version__,
            "library_source_sha256": {name: sha256(installed / "models" / name) for name in ("efficientnet.py", "maxvit.py")},
            "torchvision_license_sha256": sha256(license_path),
            "notice": "Installed TorchVision BSD-3-Clause notice retained; weight terms remain separate."}


def run_portfolio(run_name, models=None):
    run = run_directory(run_name)
    run.mkdir(parents=True, exist_ok=False)
    models = list(MODEL_TYPES) if models is None else list(models)
    if not models or len(set(models)) != len(models) or any(name not in MODEL_TYPES for name in models):
        raise ValueError("models must be unique supported identities")
    manifest = {"purpose": "phase5_engineering_only", "status": "running", "models": {},
                "policy": experimental_smoke_policy(), "provenance": source_provenance(run)}
    write_record(run / "qualification.json", manifest)
    cli = ROOT / "scripts/validation/experimental_model_smoke.py"
    def dispatch(model_id, adaptation, physical, selected, kind):
        name = f"{kind}-{adaptation}-{physical}" + ("-selected" if selected else "")
        job = run / model_id / name
        job.parent.mkdir(exist_ok=True)
        arguments = [sys.executable, str(cli), "--worker", kind, "--job", str(job), "--model", model_id,
                     "--adaptation", adaptation, "--physical-batch", str(physical)]
        if selected:
            arguments.append("--selected")
        with (job.parent / f"{name}.log").open("x") as log:
            completed = subprocess.run(arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result_path = job / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"worker produced no evidence: {job.relative_to(ROOT)}")
        result = json.loads(result_path.read_text())
        if (completed.returncode, result["status"]) not in ((0, "passed"), (70, "oom")):
            raise RuntimeError(f"qualification worker failed: {job.relative_to(ROOT)}: {result.get('error')}")
        return result
    try:
        for model_id in models:
            entry = {"probes": []}
            manifest["models"][model_id] = entry
            cap = None
            for adaptation in ("full", "frozen_encoder"):
                for physical in experimental_smoke_policy()["physical_probe_candidates"]:
                    print(f"{model_id}: {adaptation} probe physical={physical}", flush=True)
                    result = dispatch(model_id, adaptation, physical, False, "probe")
                    entry["probes"].append({"adaptation": adaptation, "physical_batch": physical, "status": result["status"]})
                    write_record(run / "qualification.json", manifest)
                    if result["status"] == "passed":
                        cap = physical
                        break
                if cap is not None:
                    break
            if cap is None:
                raise RuntimeError(f"no useful permitted full/frozen execution for {model_id}")
            entry.update(adaptation=adaptation, measured_physical_cap=cap)
            for selected in (False, True):
                print(f"{model_id}: fresh {adaptation} smoke physical={cap}, selected={selected}", flush=True)
                entry["selected_smoke" if selected else "full_smoke"] = dispatch(model_id, adaptation, cap, selected, "smoke")
                write_record(run / "qualification.json", manifest)
            if source_inventory() != manifest["provenance"]["source_inventory"]:
                raise RuntimeError("source inventory changed during qualification")
        manifest["status"] = "passed"
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_record(run / "qualification.json", manifest)
    return run


def serve_dashboard(run_name, model_id, selected=False, port=8068):
    """Serve a completed smoke for browser acceptance with a fresh isolated cache."""
    run = run_directory(run_name)
    manifest = json.loads((run / "qualification.json").read_text())
    if source_inventory() != manifest["provenance"]["source_inventory"]:
        raise ValueError("browser acceptance source inventory differs from the recorded run")
    entry = manifest["models"][model_id]
    job = run / model_id / (f"smoke-{entry['adaptation']}-{entry['measured_physical_cap']}" +
                            ("-selected" if selected else ""))
    result = json.loads((job / "result.json").read_text())
    checkpoint = job / "trial_0/best_model.ckpt"
    if result["status"] != "passed" or sha256(checkpoint) != result["restoration"]["checkpoint_sha256"]:
        raise ValueError("browser acceptance requires an unchanged passed smoke checkpoint")
    policy = experimental_smoke_policy()
    set_torch_constants()
    torch.set_float32_matmul_precision(policy["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(total * .85, free - 256 * 2**20) / total)
    browser = job / f"browser-{time.time_ns()}"
    browser.mkdir()
    app, _ = load_dashboard(browser)
    with ExitStack() as guards:
        for method in ("test_dataloader", "predict_dataloader"):
            guards.enter_context(patch.object(ImagesRecipesBaseDataModule, method,
                side_effect=AssertionError("test/predict inference forbidden")))
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

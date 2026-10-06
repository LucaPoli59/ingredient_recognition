"""Bounded CUDA/data/checkpoint/Dash engineering smoke; never a benchmark run."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import importlib
import json
import pickle
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jsonpickle
import lightning as lgn
import numpy as np
import torch
from PIL import Image

from settings.config import YUMMLY_PATH, YUMMLY_TARGET_METADATA_FILENAME
from src.commons.exp_config import ExpConfig
from src.data_processing.images_recipes import ImagesRecipesBaseDataModule
from src.lightning.custom_callbacks import FullModelCheckpoint
from src.lightning.lgn_trainers import BaseTrainer
from src.models.resnet import Resnet18
from src.training.commons import load_datamodule, model_training, set_torch_constants


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_hashes():
    return {split: sha256(Path(YUMMLY_PATH) / split / YUMMLY_TARGET_METADATA_FILENAME)
            for split in ("train", "val", "test")}


class TrainingEvidence(lgn.Callback):
    def __init__(self):
        self.batches = 0
        self.before = None
        self.changed = False

    def on_train_start(self, trainer, module):
        self.before = next(module.model.parameters()).detach().cpu().clone()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        assert batch[0].is_cuda and batch[1].shape[1] == 165
        assert torch.isfinite(outputs["loss"]).all()
        self.batches += 1

    def on_train_end(self, trainer, module):
        self.changed = not torch.equal(self.before, next(module.model.parameters()).detach().cpu())


def train_smoke(run):
    trial = run / "trial_0"
    trial.mkdir()
    scratch = run / "trainer_scratch"
    scratch.mkdir()
    config = ExpConfig(tm_type=Resnet18, tm_pretrained=True, batch_size=16,
                       dm_num_workers=2, tr_type=BaseTrainer, tr_max_epochs=1,
                       tr_limit_train_batches=4, tr_log_every_n_steps=1, tr_save_dir=str(trial),
                       lgg_wandb_notes="Data 2.4 engineering smoke; not benchmark evidence")
    set_torch_constants()
    lgn.seed_everything(42, workers=True)
    torch.set_num_threads(4)
    dm = load_datamodule(config)
    assert dm.get_num_classes() == 165 and "<UNK>" not in dm.label_encoder.classes
    assert dm.projection_config is None and not dm._pin_memory_enabled
    config.update_config(dm_label_encoder=dm.label_encoder.to_config(), tm_num_classes=165)
    config.save_to_file(trial / "smoke_config.json")
    checkpoint = FullModelCheckpoint(dirpath=trial / "checkpoints", monitor="val_loss",
                                     save_top_k=1, save_last=True)
    evidence = TrainingEvidence()
    torch.cuda.reset_peak_memory_stats()
    # The existing trainer clears its scratch root after fit: isolate that side effect.
    with patch("src.lightning.lgn_trainers.EXPERIMENTS_TRASH_PATH", str(scratch)):
        trainer, model = model_training(config, dm, trainer_kwargs={
            "limit_val_batches": 2, "precision": "32-true", "profiler": None,
            "callbacks": [checkpoint, evidence], "model_checkpoint_callback": checkpoint,
            "enable_progress_bar": False,
        })
    assert trainer.global_step == 4 and evidence.batches == 4 and evidence.changed
    assert all(torch.isfinite(value).all() for value in trainer.callback_metrics.values())
    path = trial / "best_model.ckpt"
    assert path.is_file()
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
    restored_config = ExpConfig.load_from_ckpt_data(checkpoint_data)
    restored = restored_config.hp["lgn_model_type"].load_from_config(restored_config.hp)
    restored.load_weights_from_checkpoint(path, drop_fields=["loss_fn.pos_weight"])
    assert list(restored_config.label_encoder["classes"]) == list(config.label_encoder["classes"])
    model, restored = model.cpu().eval(), restored.cpu().eval()
    image, _ = dm.val_dataset[0]
    with torch.no_grad():
        expected = model(image.unsqueeze(0))
        actual = restored(image.unsqueeze(0))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return path, {"model": "Resnet18", "pretrained": True, "seed": 42, "epochs": 1,
                  "batch_size": 16, "workers": 2, "pin_memory_resolved": False,
                  "train_batches": evidence.batches, "validation_batches": 2,
                  "sanity_validation_batches": 2, "optimizer_updates": trainer.global_step,
                  "finite_losses": True, "parameters_changed": evidence.changed,
                  "outputs": 165, "unknown_output": False, "reload_logits_exact": True,
                  "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "checkpoint_sha256": sha256(path)}


def load_isolated_dashboard(run):
    from src.dashboards import _commons
    cache = run / "dashboard_cache"
    cache.mkdir(exist_ok=True)
    _commons.DASH_CACHE = str(cache)
    from src.dashboards.dash.app import app
    # Dash imports pages using its own registered module name.
    import dash
    page_name = next(name for name, page in dash.page_registry.items()
                     if page["path"] == "/model_visualization")
    return app, importlib.import_module(page_name)


def dashboard_smoke(run, checkpoint):
    from dash._callback_context import context_value
    from dash._utils import AttributeDict
    app, page = load_isolated_dashboard(run)
    client = app.server.test_client()
    statuses = {url: client.get(url).status_code for url in
                ("/", "/model_visualization", "/_dash-layout", "/_dash-dependencies")}
    assert set(statuses.values()) == {200}
    original_hash = sha256(checkpoint)
    config, model, _ = page._load_exp_from_select(str(checkpoint.parent), None)
    token = context_value.set(AttributeDict(triggered_inputs=[
        {"prop_id": "load_exp_button.n_clicks", "value": 1}]))
    try:
        loaded = page.load_experiment(1, str(checkpoint.parent), None, None, None)
    finally:
        context_value.reset(token)
    assert loaded[-1] == "success", str(loaded[-2])
    images = page.load_images(loaded[0])
    data, index, image_url = images[:3]
    response = client.get(image_url)
    assert response.status_code == 200 and response.mimetype.startswith("image/")
    assert len(data) == 5996
    assert Path(data[0]["img"]).parent == Path(YUMMLY_PATH) / "imgs" / "standard"
    encoder = jsonpickle.decode(loaded[2])
    assert list(encoder.classes) == list(config.label_encoder["classes"])
    assert "<UNK>" not in encoder.classes and encoder.num_classes == 165
    transform = pickle.loads(codecs.decode(loaded[1].encode(), "base64"))
    from src.dashboards.runtime import load_visualization_datamodule
    reference_dm = load_visualization_datamodule(config, model)
    with Image.open(data[0]["img"]) as source:
        image = transform(source)
    torch.testing.assert_close(image, reference_dm.val_dataset[0][0], rtol=0, atol=0)
    with torch.no_grad():
        logits = model.eval()(image.unsqueeze(0).to(model.device))[0]
    assert logits.shape == (165,) and torch.isfinite(logits).all()
    expected = page._create_preds_table(torch.sigmoid(logits).cpu().numpy(), encoder)
    inferred = page.make_inference(1, None, index, data, .5, loaded[1], loaded[2])
    assert inferred[-1] == "success", str(inferred[-2])
    actual = inferred[3]
    assert [row["Ingredients"] for row in actual] == expected["Ingredients"].tolist()
    np.testing.assert_allclose([row["Confidence"] for row in actual], expected["Confidence"], atol=1e-6)
    assert all(row["Ingredients"] in encoder.classes for row in actual)
    assert len(inferred[0].data) and len(inferred[1].data)
    assert original_hash == sha256(checkpoint)
    return {"http": statuses, "validation_records": len(data), "display_image_http": response.status_code,
            "saved_label_order_exact": True, "model_preprocessing_exact": True,
            "prediction_table_matches_direct_forward": True, "gradcam_and_factorization": True,
            "checkpoint_unchanged": True, "cache_isolated": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dashboard-only", action="store_true")
    parser.add_argument("--serve-dashboard", action="store_true")
    parser.add_argument("--port", type=int, default=8064)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be a single directory name")
    run = ROOT / "experiments" / "runtime_smoke" / args.run_name
    if args.serve_dashboard:
        if not (run / "trial_0" / "best_model.ckpt").is_file():
            parser.error("the smoke checkpoint must exist before serving the dashboard")
        app, _ = load_isolated_dashboard(run)
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("this bounded runtime smoke requires CUDA")
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(total - 512 * 2**20, free - 256 * 2**20) / total)
    hashes = metadata_hashes()
    record_path = run / "smoke_record.json"
    with ExitStack() as guards:
        for name in ("test_dataloader", "predict_dataloader"):
            guards.enter_context(patch.object(ImagesRecipesBaseDataModule, name,
                side_effect=AssertionError("predictive test access is forbidden in this smoke")))
        if args.dashboard_only:
            record = json.loads(record_path.read_text())
            checkpoint = run / "trial_0" / "best_model.ckpt"
            if hashes != record["metadata_sha256"]:
                raise ValueError("metadata no longer matches the saved smoke record")
            if sha256(checkpoint) != record["training"]["checkpoint_sha256"]:
                raise ValueError("checkpoint no longer matches the saved smoke record")
        else:
            run.mkdir(parents=True, exist_ok=False)
            record = {"purpose": "engineering_only", "base_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "metadata_sha256": hashes, "predictive_test_evaluation": False}
            checkpoint, record["training"] = train_smoke(run)
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        record["dashboard"] = dashboard_smoke(run, checkpoint)
    assert hashes == metadata_hashes()
    record["metadata_unchanged"] = True
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

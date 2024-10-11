from typing import Any, Tuple, List
import torch
from sklearn.metrics import label_ranking_loss
from torch import nn
import random
import tqdm
import lightning as lgn
from lightning import Callback
import numpy as np

import os

from torch.utils.data import DataLoader
from torchinfo import summary

from settings.config import *
from src.commons.utils import multi_label_accuracy, pred_digits_to_values
from src.data_processing.images_recipes import RecipesDataset
from src.data_processing.labels_encoders import MultiLabelBinarizer
from src.models.custom_schedulers import ConstantStartReduceOnPlateau


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    true_classes = torch.argmax(y_true, dim=1)
    pred_classes = torch.argmax(torch.sigmoid(y_pred), dim=1)
    # num of hits / num of classes mean over the batch
    return (true_classes == pred_classes).sum() / y_pred.size(0)

def multi_label_accuracy_nonzero(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    return multi_label_accuracy(y_pred, torch.round(y_true))

class RecipeAutoencoder(nn.Module):
    def __init__(self, input_dim, layers_dims, bottleneck_dim, relu_type=torch.nn.ReLU):
        super().__init__()
        self.layers_dims = layers_dims
        self.bottleneck_dim = bottleneck_dim
        self.input_dim = input_dim
        relu = relu_type

        if len(layers_dims) == 0:
            encoder = [
                nn.Linear(input_dim, bottleneck_dim)
            ]

            decoder = [
                nn.Linear(bottleneck_dim, input_dim)
            ]

        else:
            encoder = ([nn.Linear(input_dim, layers_dims[0]), relu()]
                       + [[nn.Linear(layers_dims[i], layers_dims[i+1]), relu()] for i in range(len(layers_dims)-1)]
                       + [nn.Linear(layers_dims[-1], bottleneck_dim)])

            decoder = ([nn.Linear(bottleneck_dim, layers_dims[-1]), relu()]
                       + [[nn.Linear(layers_dims[i], layers_dims[i-1]), relu()] for i in range(len(layers_dims)-1, 0, -1)]
                       + [nn.Linear(layers_dims[0], input_dim)])

        encoder = [layer for layers in ([layer if isinstance(layer, list) else [layer] for layer in encoder]) for layer
                   in layers]
        decoder = [layer for layers in ([layer if isinstance(layer, list) else [layer] for layer in decoder]) for layer
                     in layers]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class BaseRecipeAutoencoder(RecipeAutoencoder):
    def __init__(self, input_dim=182, layers_dims=None, bottleneck_dim=10, relu_type=torch.nn.ReLU):
        if layers_dims is None:
            layers_dims = [90, 45, 22]

        super().__init__(input_dim=input_dim, layers_dims=layers_dims, bottleneck_dim=bottleneck_dim,
                         relu_type=relu_type)


class AutoencoderLightningModel(lgn.LightningModule):
    def __init__(self, autoencoder, lr, optimizer, loss_fn, batch_size, momentum, weight_decay, non_zero_labels=False):
        super().__init__()
        self.autoencoder = autoencoder
        self.lr = lr
        self.loss_fn = loss_fn()
        self.accuracy_fn = multi_label_accuracy if not non_zero_labels else multi_label_accuracy_nonzero
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["autoencoder", "loss_fn"])
        self.lr_scheduler = None

    def forward(self, x):
        return self.autoencoder(x)

    def configure_optimizers(self):
        if self.optimizer == torch.optim.SGD:
            self.optimizer = self.optimizer(self.autoencoder.parameters(), lr=self.lr, momentum=self.momentum,
                                            weight_decay=self.weight_decay)
        else:
            self.optimizer = self.optimizer(self.autoencoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.lr_scheduler = ConstantStartReduceOnPlateau(self.optimizer, initial_lr=self.lr, warm_duration=15,
                                                      mode="min", patience=5, cooldown=1, min_lr=1e-6, factor=0.05)

        return self.optimizer

    def _base_step(self, batch) -> Tuple[Any, Any]:
        y = batch
        y_pred, latent_vector = self.autoencoder(y)
        loss = self.loss_fn(y_pred, y)
        with torch.no_grad():
            acc = self.accuracy_fn(y_pred, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._base_step(batch)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._base_step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._base_step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc})


class MultiLabelBinarizerNonZero(MultiLabelBinarizer):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def digits_int2float(self, digts: List[int]) -> List[float]:
        return (np.array(digts) + self.eps).tolist()

    def digits_float2int(self, digts: List[float]) -> List[int]:
        return np.round(np.array(digts) - self.eps).astype(int).tolist()

    def _encode(self, labels: List[str]):
        return self.digits_int2float(super()._encode(labels))

    def _inverse(self, encoded_labels: List[float]):
        return super()._inverse(self.digits_float2int(encoded_labels))

    def _decode_labels(self, encoded_labels: List[float]):
        return super()._decode_labels(self.digits_float2int(encoded_labels))


if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')  # For better performance with cuda

    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")
    CATEGORY = "all"
    EPOCHS = 100
    BATCH_SIZE = 256
    metadata_filename = "topk_" + METADATA_FILENAME
    INPUT_DIM = 100
    # INPUT_DIM = 182
    non_zero_labels = True

    LR = 5e-2
    OPTIMIZER = torch.optim.SGD
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0
    # WEIGHT_DECAY = 1e-4
    LOSS_FN = nn.BCEWithLogitsLoss

    if not os.path.exists(model_path):
        os.makedirs(os.path.join(model_path, "lightning_logs"))

    src = YUMMLY_PATH
    label_encoder = MultiLabelBinarizer() if not non_zero_labels else MultiLabelBinarizerNonZero()
    train_dataset = RecipesDataset(os.path.join(src, "train"), label_encoder=label_encoder, category=CATEGORY,
                                         feature_label="ingredients_ok", metadata_filename=metadata_filename)

    val_dataset = RecipesDataset(os.path.join(src, "val"), label_encoder=label_encoder, category=CATEGORY,
                                       feature_label="ingredients_ok", metadata_filename=metadata_filename)

    NUM_WORKERS = os.cpu_count() if EPOCHS > 3 else 2

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True, persistent_workers=True)

    layers_dims = [50, 25]
    bottleneck_dim = 15
    # torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Elu, torch.nn.PReLU
    relu_type = torch.nn.ELU

    autoencoder_model = BaseRecipeAutoencoder(layers_dims=layers_dims, input_dim=INPUT_DIM, bottleneck_dim=bottleneck_dim,
                                              relu_type=relu_type)
    lightning_model = AutoencoderLightningModel(autoencoder_model, LR, OPTIMIZER, LOSS_FN, BATCH_SIZE, MOMENTUM, WEIGHT_DECAY,
                                                non_zero_labels=non_zero_labels)

    bar_callback = lgn.pytorch.callbacks.RichProgressBar(leave=True)

    print(summary(autoencoder_model, (BATCH_SIZE, INPUT_DIM)))

    trainer = lgn.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        default_root_dir=model_path,
        precision="16-mixed",
        log_every_n_steps=len(train_dataloader),
        callbacks=[
            bar_callback,
            # device_stats_callback,
        ],

        accumulate_grad_batches=5,
        gradient_clip_val=0.01,
        gradient_clip_algorithm="value",

        # fast_dev_run=True,
        # profiler=profiler,
        enable_model_summary=False,
    )
    print("Training")



    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    print("Testing")
    labels = next(iter(val_dataloader))

    lightning_model.eval()
    lightning_model.to("cpu")
    labels.to("cpu")

    with torch.no_grad():
        y_pred, latent_vector = lightning_model(labels)
        print("true:\n", label_encoder.inverse_transform(labels[:10].numpy()), "\n")
        print("preds:\n", label_encoder.inverse_transform(pred_digits_to_values(y_pred[:10]).numpy()), "\n\n\n\n")

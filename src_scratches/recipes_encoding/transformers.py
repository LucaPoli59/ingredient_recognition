import os
from typing import Any, Tuple, List, Optional, Dict
from typing_extensions import Self

import lightning as lgn
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from set_transformer.modules import ISAB, PMA, SAB
from settings.config import *
from settings.config import DEF_UNKNOWN_TOKEN
from src.commons.utils import multi_label_accuracy, pred_digits_to_values, accuracy
from src.data_processing.images_recipes import RecipesDataset, RecipesIntMaskingDataset, RecipesIntFlavorDataset
from src.data_processing.labels_encoders import MultiLabelBinarizer, LabelEncoderInterface, TextIntEncoder
from src.models.custom_schedulers import ConstantStartReduceOnPlateau


class RecipeTransformer(torch.nn.Module):
    def __init__(self, num_classes, dim_input_emb, dim_output, dim_bottleneck, num_heads=4, num_ids=32, ln=False):
        """
        :param num_classes: Number of classes (used for the vocab embedding)
        :param dim_input_emb: Dimension of the embedding of each element of the input sequence
        :param dim_output: Dimension of the output for the self supervised task
        :param dim_bottleneck: Bottleneck dimension for the transformer
        :param num_heads: Number of heads for the transformer
        :param num_ids: Number of induced points for the transformer (learnable parameter of the ISAB)
        :param ln:  Whether to use layer normalization
        """
        super().__init__()
        self.vocab_size = num_classes
        self.dim_input_emb = dim_input_emb
        self.dim_output = dim_output
        self.dim_bottleneck = dim_bottleneck

        self.embedding = torch.nn.Embedding(num_classes, dim_input_emb, padding_idx=0)
        self.encoder = torch.nn.Sequential(
            ISAB(dim_input_emb, dim_bottleneck, num_heads, num_ids, ln=ln),
            ISAB(dim_bottleneck, dim_bottleneck, num_heads, num_ids, ln=ln),
            ISAB(dim_bottleneck, dim_bottleneck, num_heads, num_ids, ln=ln),

        )
        self.decoder = torch.nn.Sequential(
            PMA(dim_bottleneck, num_heads, 1, ln=ln),
            SAB(dim_bottleneck, dim_bottleneck, num_heads, ln=ln),
            SAB(dim_bottleneck, dim_bottleneck, num_heads, ln=ln),
            SAB(dim_bottleneck, dim_bottleneck, num_heads, ln=ln),
            torch.nn.Linear(dim_bottleneck, dim_output)
        )

    def forward(self, x):
        embedding = self.embedding(x)
        encoded = self.encoder(embedding)
        ris = self.decoder(encoded).squeeze(1)
        return ris, encoded


class BaseRecipeTransformer(RecipeTransformer):
    def __init__(self, num_classes, dim_output, dim_input_emb=50, dim_bottleneck=10, num_heads=4, num_ids=32, ln=False):
        super().__init__(num_classes, dim_input_emb, dim_output, dim_bottleneck, num_heads, num_ids, ln)

class SetTransformerLightningModel(lgn.LightningModule):
    def __init__(self, transformer, lr, optimizer, loss_fn, accuracy_fn, batch_size, momentum, weight_decay):
        super().__init__()
        self.transformer = transformer
        self.lr = lr
        self.loss_fn = loss_fn()
        self.accuracy_fn = accuracy_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["transformer", "loss_fn"])
        self.lr_scheduler = None

    def forward(self, x):
        return self.transformer(x)

    def configure_optimizers(self):
        if self.optimizer == torch.optim.SGD:
            self.optimizer = self.optimizer(self.transformer.parameters(), lr=self.lr, momentum=self.momentum,
                                            weight_decay=self.weight_decay)
        else:
            self.optimizer = self.optimizer(self.transformer.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.lr_scheduler = ConstantStartReduceOnPlateau(self.optimizer, initial_lr=self.lr, warm_duration=15,
                                                      mode="min", patience=5, cooldown=1, min_lr=1e-6, factor=0.05)

        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler, 'monitor': 'train_loss'}

    def _base_step(self, batch) -> Tuple[Any, Any]:
        labels, masked_label = batch

        masked_label_pred, labels_embedding = self.transformer(labels)
        loss = self.loss_fn(masked_label_pred, masked_label)
        with torch.no_grad():
            acc = self.accuracy_fn(masked_label_pred, masked_label)
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



if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')  # For better performance with cuda

    model_path = os.path.join(os.path.dirname(os.getcwd()), "model")
    CATEGORY = "all"
    EPOCHS = 50
    BATCH_SIZE = 256

    # INPUT_DIM = 182

    LR = 1e-3
    OPTIMIZER = torch.optim.SGD
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    accuracy_fn = accuracy
    # accuracy_fn = accuracy
    # WEIGHT_DECAY = 1e-4
    # LOSS_FN = nn.CrossEntropyLoss
    LOSS_FN = nn.MSELoss


    if not os.path.exists(model_path):
        os.makedirs(os.path.join(model_path, "lightning_logs"))

    src = YUMMLY_PATH
    label_encoder = TextIntEncoder()

    # feature_label = "ingredients_ok"
    # metadata_filename = "topk_" + METADATA_FILENAME
    p_masking = 0.85

    feature_label = "flavors"
    metadata_filename = METADATA_FILENAME
    train_dataset = RecipesIntFlavorDataset(os.path.join(src, "train"), category=CATEGORY, label_encoder=label_encoder,
                                          feature_label=feature_label, metadata_filename=metadata_filename)
                                          # p_mask=p_masking)

    val_dataset = RecipesIntFlavorDataset(os.path.join(src, "val"), category=CATEGORY, label_encoder=label_encoder,
                                        feature_label=feature_label, metadata_filename=metadata_filename)
                                        # p_mask=p_masking)

    input_dim = train_dataset.dim_vocab
    output_dim = train_dataset.dim_target
    dm_embedding = 50
    dm_bottleneck = 10
    n_heads = 6

    collate_fn = train_dataset.collate_fn

    NUM_WORKERS = os.cpu_count() if EPOCHS > 3 else 2

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True, persistent_workers=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True, persistent_workers=True, collate_fn=collate_fn)

    model = BaseRecipeTransformer(input_dim, output_dim, dim_input_emb=dm_embedding, dim_bottleneck=dm_bottleneck, num_heads=n_heads)
    lightning_model = SetTransformerLightningModel(model, LR, OPTIMIZER, LOSS_FN, accuracy_fn, BATCH_SIZE, MOMENTUM, WEIGHT_DECAY)

    print("Model: [Assuming 18 ingredients]:\n")
    summary(lightning_model, input_size=(BATCH_SIZE, 18), dtypes=[torch.int64])

    bar_callback = lgn.pytorch.callbacks.RichProgressBar(leave=True)

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
    batch = next(iter(val_dataloader))
    labels, masked_label = batch

    lightning_model.eval()
    lightning_model.to("cpu")
    labels.to("cpu")
    masked_label.to("cpu")

    with torch.no_grad():
        masked_label_pred, latent_vector = lightning_model(labels)

        print("ingredients labels:\n", label_encoder.inverse_transform(labels[:10].numpy()), "\n")
        print("true label:\n", masked_label[:10].numpy(), "\n")
        # print("true masked label:\n", label_encoder.inverse_transform([[train_dataset.ingr_hot2int(elem)] for elem in masked_label[:10].numpy()]), "\n")

        print("preds:\n", masked_label_pred[:10].numpy(), "\n")
        # print("preds converted:\n", label_encoder.inverse_transform([[train_dataset.ingr_hot2int(elem)] for elem in torch.sigmoid(masked_label_pred[:10]).numpy()]), "\n")
        # print("preds:\n", torch.sigmoid(masked_label_pred[:10]).numpy(), "\n")



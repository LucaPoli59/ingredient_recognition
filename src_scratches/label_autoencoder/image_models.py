"""Shared image models; importing this module does not change CUDA visibility."""
import copy

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

ARMS = ("direct", "ae32_aligned", "ae32_mse")


def normalization(values):
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6)


class ImageHead(nn.Module):
    """All forward/inference inputs are image features; no ingredient input exists."""
    def __init__(self, kind, feature_mean, feature_std, label_count, decoder=None,
                 latent_mean=None, latent_std=None):
        super().__init__()
        if kind not in ARMS:
            raise ValueError(f"Unknown head {kind}")
        self.kind = kind
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))
        if kind == "direct":
            self.projection = nn.Linear(len(feature_mean), label_count)
        else:
            if decoder is None or latent_mean is None or latent_std is None:
                raise ValueError("AE head requires decoder and train-fitted latent normalization")
            self.register_buffer("latent_mean", torch.as_tensor(latent_mean, dtype=torch.float32))
            self.register_buffer("latent_std", torch.as_tensor(latent_std, dtype=torch.float32))
            self.projection = nn.Linear(len(feature_mean), len(latent_mean))
            self.decoder = copy.deepcopy(decoder).requires_grad_(False).eval()
            # Start near the train-average code; both AE objectives share this initialization.
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.projection.bias)

    def forward(self, features):
        projected = self.projection((features - self.feature_mean) / self.feature_std)
        if self.kind == "direct":
            return projected, None
        logits = self.decoder(projected * self.latent_std + self.latent_mean)
        return logits, projected


def teacher_codes(autoencoder, labels, batch_size=1024):
    """Used for training targets/diagnostics, never to construct image predictions."""
    autoencoder.eval()
    with torch.inference_mode():
        return np.concatenate([autoencoder.encoder(torch.from_numpy(labels[start:start + batch_size])).numpy()
                               for start in range(0, len(labels), batch_size)])


def load_backbone(weights):
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.fc = nn.Identity()
    model.requires_grad_(False).eval().to(memory_format=torch.channels_last)
    return model


def load_head(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state_dict"]
    decoder = None
    if saved["kind"] != "direct":
        decoder = nn.Sequential(nn.Linear(saved["latent_dim"], saved["decoder_hidden_dim"]), nn.ReLU(),
                                nn.Linear(saved["decoder_hidden_dim"], saved["label_count"]))
    model = ImageHead(saved["kind"], state["feature_mean"], state["feature_std"], saved["label_count"],
                      decoder, state.get("latent_mean"), state.get("latent_std"))
    model.load_state_dict(state, strict=True)
    return model.eval(), saved

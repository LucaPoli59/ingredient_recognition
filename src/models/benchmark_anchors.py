"""Opt-in matched-input anchors; serialized historical adapters are unchanged."""

import hashlib
import json
import math
from pathlib import Path
import sys
from types import MethodType

import torch
from torchvision.models import resnet18

from src.models.commons import BaseModel
from src.models.experimental_contract import ExperimentalModelContract, make_experimental_linear
from src.data_processing.experimental_transforms import ExperimentalFitPad224


ARTIFACTS = {
    "resnet18_anchor": ("resnet18-f37072fd.pth", "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
    "dinov2_b14_anchor": ("dinov2_vitb14_reg4_pretrain.pth", "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71"),
}
DINO_SOURCE_HASH = "ebde9a04d8b7c1d09ad007f8bb6a189a639a223d769fb4365d97fb96b61b4e85"
DINO_HUB_HASH = "e8926c8d01f93e27a6a687a39a1d78716690037fa79153c6483141526ef2a7c2"
DINO_LICENSE_HASH = "600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_dino_source():
    root = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    # Matches C-locale sorted `sha256sum` lines, not a Git commit/tag claim.
    inventory = "".join(f"{sha256(p)}  {p.relative_to(root).as_posix()}\n"
                        for p in sorted((root / "dinov2").rglob("*.py")))
    if (hashlib.sha256(inventory.encode()).hexdigest() != DINO_SOURCE_HASH
            or sha256(root / "hubconf.py") != DINO_HUB_HASH
            or sha256(root / "LICENSE") != DINO_LICENSE_HASH):
        raise ValueError("DINO source cache differs from the pinned benchmark snapshot")
    for name, module in tuple(sys.modules.items()):
        if name == "dinov2" or name.startswith("dinov2."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(root.resolve()):
                raise ValueError("a different DINO implementation is already imported")
    return root


def _portable_attention_forward(attention, x, attn_bias=None):
    """Official dense CPU fallback, without changing global xFormers flags."""
    from dinov2.layers.attention import Attention, MemEffAttention
    if x.device.type == "cpu":
        if attn_bias is not None:
            raise ValueError("the benchmark supports dense image batches, not nested attention")
        return Attention.forward(attention, x)
    return MemEffAttention.forward(attention, x, attn_bias=attn_bias)


class BenchmarkAnchor(BaseModel):
    """Original ResNet18 or DINO register-B/14 encoder, new seeded head.

    Offline construction never contacts Hub. Fresh initialization requires the
    verified official artifacts already in the cache; it does not download them.
    """

    PRETTY_NAME = "Phase 6 comparison anchor"

    def __init__(self, contract, *, initialize_pretrained=True):
        if not isinstance(contract, ExperimentalModelContract) or contract.model_id not in ARTIFACTS:
            raise ValueError("a validated anchor contract is required")
        if type(initialize_pretrained) is not bool:
            raise ValueError("initialize_pretrained must be boolean")
        super().__init__(contract.num_classes, (224, 224))
        self.experimental_contract = contract
        self.is_dino = contract.model_id == "dinov2_b14_anchor"
        if self.is_dino:
            self.model = torch.hub.load(str(verified_dino_source()), "dinov2_vitb14_reg_lc",
                                        source="local", pretrained=False)
            encoder = self.model.backbone
            for block in encoder.blocks:
                block.attn.forward = MethodType(_portable_attention_forward, block.attn)
        else:
            self.model = resnet18(weights=None)
            encoder = self.model
        if initialize_pretrained:
            filename, expected = ARTIFACTS[contract.model_id]
            path = Path(torch.hub.get_dir()) / "checkpoints" / filename
            if sha256(path) != expected:
                raise ValueError("anchor original artifact hash differs")
            encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        if self.is_dino:
            if self.model.linear_head.in_features != 3840 or self.model.backbone.num_register_tokens != 4:
                raise ValueError("unexpected DINO register/linear-probe topology")
            self.model.linear_head = make_experimental_linear(3840, contract.num_classes, contract.head_seed)
            self.model.backbone.requires_grad_(False)
        else:
            self.model.fc = make_experimental_linear(512, contract.num_classes, contract.head_seed)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "is_dino", False) and hasattr(self, "model"):
            self.model.backbone.eval()
        return self

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 224, 224):
            raise ValueError("benchmark input must be [B,3,224,224]")
        return self.model(x)

    @property
    def transform_aug(self):
        return ExperimentalFitPad224(self.experimental_contract.horizontal_flip_probability)

    @property
    def transform_plain(self):
        return ExperimentalFitPad224(0.)

    @property
    def conv_target_layer(self):
        return self.model.backbone.blocks[-1].norm1 if self.is_dino else self.model.layer4[-1]

    @property
    def classifier_target_layer(self):
        return self.model.linear_head if self.is_dino else self.model.fc

    @property
    def gradcam_reshape_transform(self):
        if not self.is_dino:
            return None
        def reshape(tokens):
            patches = tokens[:, 5:, :]
            b, n, c = patches.shape
            side = math.isqrt(n)
            if side * side != n:
                raise ValueError("DINO patch tokens must form a square grid")
            return patches.reshape(b, side, side, c).permute(0, 3, 1, 2).contiguous()
        return reshape

    @property
    def factorization_classifier_layer(self):
        if self.is_dino:
            from src.models.dinov2 import _DinoV2PatchClassifier
            return _DinoV2PatchClassifier(self.model.linear_head, 768)
        return self.model.fc

    @classmethod
    def provenance(cls, model_id):
        filename, digest = ARTIFACTS[model_id]
        result = {"filename": filename, "sha256": digest}
        if model_id == "dinov2_b14_anchor":
            result.update(source_hash=DINO_SOURCE_HASH, hub_hash=DINO_HUB_HASH, license_hash=DINO_LICENSE_HASH)
        return result

    def to_config(self):
        return super().to_config() | {
            "experimental_contract": self.experimental_contract.to_config(),
            "anchor_provenance": self.provenance(self.experimental_contract.model_id)}

    @classmethod
    def load_from_config(cls, config, *, initialize_pretrained=True):
        expected = {"type", "num_classes", "input_shape", "trns_aug", "trns_bld_aug",
                    "trns_bld_plain", "lp_phase", "experimental_contract", "anchor_provenance"}
        if set(config) != expected or config["type"] is not cls:
            raise ValueError("invalid benchmark anchor configuration")
        contract = ExperimentalModelContract.from_config(config["experimental_contract"])
        if (contract.model_id not in ARTIFACTS or type(config["num_classes"]) is not int
                or config["num_classes"] != contract.num_classes or tuple(config["input_shape"]) != (224, 224)
                or any(config[k] is not None for k in ("trns_aug", "trns_bld_aug", "trns_bld_plain", "lp_phase"))
                or config["anchor_provenance"] != cls.provenance(contract.model_id)):
            raise ValueError("anchor configuration/provenance disagrees")
        return cls(contract, initialize_pretrained=initialize_pretrained)

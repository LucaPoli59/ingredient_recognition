"""DICA-Net-S tensor graph, the 4A-D2 design formerly called P2-S.

This core consumes an intact EfficientNetV2-S feature stack. The experiment
adapter, approved weight loading and persistence are separate Phase 5.4.2 work.
Equations: docs/research/topics/custom_attention_model_design/
architecture_compatibility_synthesis.md (route Q, Query2Label-inspired).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class _QueryBlock(nn.Module):
    """One pre-norm cross-attention/FFN block; labels never attend to labels."""

    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(128, eps=1e-5)
        self.q_projection = nn.Linear(128, 128)
        self.k_projection = nn.Linear(128, 128)
        self.v_projection = nn.Linear(128, 128)
        self.attention_output = nn.Linear(128, 128)
        self.ffn_norm = nn.LayerNorm(128, eps=1e-5)
        self.ffn_input = nn.Linear(128, 512)
        self.ffn_activation = nn.GELU(approximate="none")
        self.ffn_output = nn.Linear(512, 128)

    @staticmethod
    def _split_heads(tokens):
        return tokens.reshape(tokens.shape[0], tokens.shape[1], 4, 32).transpose(1, 2)

    def forward(self, queries, memory):
        q = self._split_heads(self.q_projection(self.query_norm(queries)))
        k = self._split_heads(self.k_projection(memory))
        v = self._split_heads(self.v_projection(memory))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(queries.shape[0], queries.shape[1], 128)
        residual = queries + self.attention_output(attended)
        return residual + self.ffn_output(self.ffn_activation(self.ffn_input(self.ffn_norm(residual))))


class _ClasswiseLinear(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, 128))
        self.bias = nn.Parameter(torch.empty(num_classes))

    def forward(self, queries):
        return (queries * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias


class DICAReadoutS(nn.Module):
    """Dual-scale Ingredient-query and Context Attention, S readout.

    Label-indexed tensors use caller-supplied column order. Only the class count
    varies; feature taps, width, heads and depth are fixed by 4A-D2.
    """

    WIDTH = 128
    HEADS = 4
    QUERY_BLOCKS = 1
    TOKEN_RANGES = ((0, 196), (196, 245))

    def __init__(self, num_classes: int, head_seed: int = 42):
        super().__init__()
        _positive_int(num_classes, "num_classes")
        if type(head_seed) is not int or not 0 <= head_seed < 2**63:
            raise ValueError("head_seed must be an integer in [0,2**63)")
        self.num_classes = num_classes
        self.head_seed = head_seed
        # Constructor defaults are overwritten in the declared order, without
        # consuming the caller's CPU RNG or ever visiting the supplied encoder.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.f16_projection = nn.Conv2d(160, 128, 1, bias=False, dtype=torch.float32)
            self.f32_projection = nn.Conv2d(1280, 128, 1, bias=False, dtype=torch.float32)
            self.f16_token_norm = nn.LayerNorm(128, eps=1e-5, dtype=torch.float32)
            self.f32_token_norm = nn.LayerNorm(128, eps=1e-5, dtype=torch.float32)
            self.f16_scale_embedding = nn.Parameter(torch.empty(128, dtype=torch.float32))
            self.f32_scale_embedding = nn.Parameter(torch.empty(128, dtype=torch.float32))
            self.ingredient_queries = nn.Parameter(torch.empty(num_classes, 128, dtype=torch.float32))
            self.blocks = nn.ModuleList([_QueryBlock().float()])
            self.final_query_norm = nn.LayerNorm(128, eps=1e-5, dtype=torch.float32)
            self.classwise_readout = _ClasswiseLinear(num_classes).float()
            self.pooled_context_readout = nn.Linear(1280, num_classes, dtype=torch.float32)
        self._initialize_new_modules()

    def _initialize_new_modules(self):
        generator = torch.Generator(device="cpu").manual_seed(self.head_seed)

        def initialize(module):
            if isinstance(module, (nn.Conv2d, nn.Linear, _ClasswiseLinear)):
                nn.init.xavier_uniform_(module.weight, gain=1.0, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for module in (self.f16_projection, self.f32_projection,
                       self.f16_token_norm, self.f32_token_norm):
            initialize(module)
        for embedding in (self.f16_scale_embedding, self.f32_scale_embedding, self.ingredient_queries):
            nn.init.normal_(embedding, mean=0.0, std=0.02, generator=generator)
        for block in self.blocks:
            for module in (block.query_norm, block.q_projection, block.k_projection,
                           block.v_projection, block.attention_output, block.ffn_norm,
                           block.ffn_input, block.ffn_output):
                initialize(module)
        for module in (self.final_query_norm, self.classwise_readout, self.pooled_context_readout):
            initialize(module)

    @staticmethod
    def _validate_features(f16, f32):
        for name, feature, shape in (("F16", f16, (160, 14, 14)), ("F32", f32, (1280, 7, 7))):
            if (not isinstance(feature, torch.Tensor) or feature.ndim != 4
                    or feature.shape[0] <= 0 or tuple(feature.shape[1:]) != shape
                    or not feature.is_floating_point()):
                raise ValueError(f"{name} must be floating [B,{shape[0]},{shape[1]},{shape[2]}], B>0")
        if f16.shape[0] != f32.shape[0] or f16.device != f32.device or f16.dtype != f32.dtype:
            raise ValueError("F16 and F32 must share batch size, device and dtype")

    def build_memory(self, f16, f32):
        self._validate_features(f16, f32)
        fine = self.f16_projection(f16).flatten(2).transpose(1, 2)
        coarse = self.f32_projection(f32).flatten(2).transpose(1, 2)
        fine = self.f16_token_norm(fine) + self.f16_scale_embedding
        coarse = self.f32_token_norm(coarse) + self.f32_scale_embedding
        return torch.cat((fine, coarse), dim=1)

    def forward_branches(self, f16, f32):
        """Raw query/context logits for engineering checks, not visibility scores."""
        memory = self.build_memory(f16, f32)
        queries = self.ingredient_queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        query_logits = self.classwise_readout(self.final_query_norm(queries))
        context_logits = self.pooled_context_readout(f32.mean(dim=(-2, -1)))
        return query_logits, context_logits

    def forward(self, f16, f32):
        query_logits, context_logits = self.forward_branches(f16, f32)
        return query_logits + context_logits


class DICANetSCore(nn.Module):
    """Once-only encoder traversal plus DICA-Net-S readout.

    Pass the intact ``features`` from TorchVision EfficientNetV2-S. This is a
    tensor core, not yet a BaseModel adapter or a qualified training entry point.
    Feature provenance, full/frozen policy and canonical saved configuration
    belong to the experiment adapter; this core neither loads nor freezes it.
    """

    PRETTY_NAME = "DICA-Net-S"

    def __init__(self, features: nn.Sequential, num_classes: int, head_seed: int = 42):
        super().__init__()
        if not isinstance(features, nn.Sequential) or len(features) != 8:
            raise ValueError("DICA-Net-S requires the intact eight-stage EfficientNetV2-S features")
        self.features = features
        self.readout = DICAReadoutS(num_classes, head_seed)
        self.num_classes = num_classes

    def forward_features(self, image):
        if (not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] <= 0
                or tuple(image.shape[1:]) != (3, 224, 224) or not image.is_floating_point()):
            raise ValueError("DICA-Net-S input must be floating [B,3,224,224], B>0")
        f16 = None
        for index, stage in enumerate(self.features):
            image = stage(image)
            if index == 5:
                f16 = image
        return f16, image

    def forward(self, image):
        return self.readout(*self.forward_features(image))

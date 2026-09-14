"""Read bounded checkpoint metadata without loading model tensors into the report."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def inspect_checkpoint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "unavailable", "reason": "checkpoint path not provided"}
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as error:
        return {"status": "unavailable", "reason": str(error), "source": str(path)}
    state = checkpoint.get("state_dict", {})
    return {
        "status": "available",
        "source": str(path),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "lightning_version": checkpoint.get("pytorch-lightning_version"),
        "state_dict_tensors": len(state),
        "state_dict_elements": sum(value.numel() for value in state.values()),
        "amp_scaler_snapshot": checkpoint.get(
            "MixedPrecision", checkpoint.get("native_amp_scaling_state")
        ),
    }

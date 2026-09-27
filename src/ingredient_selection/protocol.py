"""Frozen Phase 3 protocol values, hashes, and isolation checks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROTOCOL_ID = "phase3-d1-v3"
PILOT_GENERATION = "phase3-pilot-v1"


@dataclass(frozen=True)
class SelectorProtocol:
    protocol_id: str = PROTOCOL_ID
    seed: int = 42
    num_classes: int = 165
    image_size: int = 384
    batch_size: int = 128
    max_epochs: int = 40
    audit_epochs: tuple[int, ...] = tuple(range(0, 41, 2))
    early_window: tuple[int, ...] = (2, 4, 6)
    near_window: tuple[int, ...] = (30, 32, 34, 36, 38)
    late_window: tuple[int, ...] = (32, 34, 36, 38, 40)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    warmup_epochs: int = 2
    cosine_epochs: int = 38
    minimum_learning_rate: float = 1e-6
    f1_threshold: float = 0.5
    bootstrap_samples: int = 1000
    bootstrap_max_draws: int = 10_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_values_hash(values: Sequence[Any]) -> str:
    return sha256_json(list(values))


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(value.shape)))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def compute_positive_counts(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets)
    if values.ndim != 2:
        raise ValueError("targets must be a two-dimensional [records, labels] array")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("targets must be binary")
    counts = values.sum(axis=0, dtype=np.int64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"train targets contain labels without positives: {missing}")
    return counts


def compute_pos_weight(targets: np.ndarray) -> torch.Tensor:
    counts = compute_positive_counts(targets)
    total = int(np.asarray(targets).shape[0])
    return torch.as_tensor((total - counts) / counts, dtype=torch.float32)


def build_pilot_cohort(class_names: Sequence[str], positive_counts: Sequence[int]) -> dict[str, Any]:
    """Build the sealed 8-per-stratum Phase 3 pilot cohort."""
    names = [str(name) for name in class_names]
    supports = [int(value) for value in positive_counts]
    if len(names) != 165 or len(supports) != 165:
        raise ValueError("the frozen pilot rule requires exactly 165 labels and supports")
    if len(set(names)) != len(names):
        raise ValueError("class names must be unique")
    if any(value <= 0 for value in supports):
        raise ValueError("pilot supports must be positive")

    source = [
        {"class_name": name, "class_index": index, "train_positive_count": supports[index]}
        for index, name in enumerate(names)
    ]
    ordered = sorted(source, key=lambda item: (item["train_positive_count"], item["class_name"]))
    selected: list[dict[str, Any]] = []
    for stratum in range(3):
        members = ordered[stratum * 55:(stratum + 1) * 55]
        ranked = sorted(
            members,
            key=lambda item: (
                hashlib.sha256(
                    PILOT_GENERATION.encode("utf-8") + b"\0" + item["class_name"].encode("utf-8")
                ).hexdigest(),
                item["class_name"],
            ),
        )
        for rank, item in enumerate(ranked[:8]):
            selected.append(item | {"support_stratum": stratum, "hash_rank": rank})

    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generation": PILOT_GENERATION,
        "class_order_hash": ordered_values_hash(names),
        "support_hash": ordered_values_hash(supports),
        "strata": 3,
        "stratum_size": 55,
        "selected_per_stratum": 8,
        "labels": selected,
    }
    payload["artifact_hash"] = sha256_json(payload)
    return payload


def validate_pilot_cohort(
        artifact: dict[str, Any], class_names: Sequence[str], positive_counts: Sequence[int]) -> None:
    expected = build_pilot_cohort(class_names, positive_counts)
    if artifact != expected:
        raise ValueError("pilot cohort does not match the frozen support-stratified SHA-256 rule")


def configure_deterministic_runtime(seed: int = 42) -> dict[str, Any]:
    """Apply the Phase 3 deterministic-runtime contract and return resolved state."""
    if seed != 42:
        raise ValueError("Phase 3-D1 fixes the global seed to 42")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {
        "seed": seed,
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def git_revision_and_cleanliness(repository_root: str | os.PathLike) -> tuple[str, bool, list[str]]:
    root = Path(repository_root)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return revision, not status, status


def require_clean_tracked_worktree(repository_root: str | os.PathLike) -> str:
    revision, clean, status = git_revision_and_cleanliness(repository_root)
    if not clean:
        raise RuntimeError(
            "the selector campaign requires a clean tracked worktree; changed paths: "
            + ", ".join(line[3:] for line in status)
        )
    return revision

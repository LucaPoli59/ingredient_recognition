"""Train/validation-only data boundary for the Phase 3 selector."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import lightning as lgn
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data_processing.transformations import (
    transform_aug_selector_efficientnet_v2_s,
    transform_plain_selector_efficientnet_v2_s,
)
from src.ingredient_selection.protocol import compute_positive_counts, sha256_file


def _normalise_cuisine(value: Any) -> str:
    if isinstance(value, list):
        values = sorted(str(item).strip().casefold() for item in value if str(item).strip())
        return "|".join(values) if values else "<unknown>"
    text = str(value or "").strip().casefold()
    return text if text else "<unknown>"


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError(f"metadata must contain a non-empty JSON list: {path}")
    return records


def _class_order(records: Sequence[dict[str, Any]], feature_label: str) -> list[str]:
    return sorted({str(label) for record in records for label in record[feature_label]})


def _encode_targets(
        records: Sequence[dict[str, Any]], class_names: Sequence[str], feature_label: str) -> np.ndarray:
    index = {name: position for position, name in enumerate(class_names)}
    targets = np.zeros((len(records), len(class_names)), dtype=np.uint8)
    for row, record in enumerate(records):
        labels = record.get(feature_label)
        if not isinstance(labels, list):
            raise ValueError(f"record {record.get('id', row)!r} has no list field {feature_label!r}")
        unknown = sorted(set(map(str, labels)) - index.keys())
        if unknown:
            raise ValueError(f"metadata contains labels outside the train class order: {unknown}")
        for label in labels:
            targets[row, index[str(label)]] = 1
    return targets


@dataclass(frozen=True)
class SelectorSplit:
    name: str
    metadata_path: Path
    image_paths: tuple[Path, ...]
    record_ids: tuple[str, ...]
    cuisines: tuple[str, ...]
    targets: np.ndarray
    metadata_sha256: str

    @property
    def supports(self) -> np.ndarray:
        return self.targets.sum(axis=0, dtype=np.int64)


@dataclass(frozen=True)
class SelectorDataBundle:
    root: Path
    image_root: Path
    metadata_filename: str
    feature_label: str
    class_names: tuple[str, ...]
    train: SelectorSplit
    val: SelectorSplit

    @classmethod
    def load(
            cls,
            root: str | Path,
            metadata_filename: str = "ingredients_target_v5_metadata.json",
            feature_label: str = "ingredients_target",
            images_subdir: str | Path = Path("imgs") / "standard",
            expected_classes: int = 165,
    ) -> "SelectorDataBundle":
        data_root = Path(root).resolve()
        image_root = (data_root / images_subdir).resolve()
        if not image_root.is_dir():
            raise FileNotFoundError(f"selector image directory not found: {image_root}")

        paths = {
            split: data_root / split / metadata_filename
            for split in ("train", "val")
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"selector metadata not found: {path}")
        train_records = _load_json_records(paths["train"])
        val_records = _load_json_records(paths["val"])
        class_names = _class_order(train_records, feature_label)
        if len(class_names) != expected_classes:
            raise ValueError(f"train metadata defines {len(class_names)} classes, expected {expected_classes}")

        def build_split(name: str, records: list[dict[str, Any]]) -> SelectorSplit:
            ids = tuple(str(record["id"]) for record in records)
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} metadata contains duplicate record IDs")
            images = tuple(image_root / str(record["image"]) for record in records)
            missing = next((path for path in images if not path.is_file()), None)
            if missing is not None:
                raise FileNotFoundError(f"{name} image not found: {missing}")
            return SelectorSplit(
                name=name,
                metadata_path=paths[name],
                image_paths=images,
                record_ids=ids,
                cuisines=tuple(_normalise_cuisine(record.get("cuisine")) for record in records),
                targets=_encode_targets(records, class_names, feature_label),
                metadata_sha256=sha256_file(paths[name]),
            )

        train = build_split("train", train_records)
        val = build_split("val", val_records)
        compute_positive_counts(train.targets)
        return cls(
            root=data_root,
            image_root=image_root,
            metadata_filename=metadata_filename,
            feature_label=feature_label,
            class_names=tuple(class_names),
            train=train,
            val=val,
        )


class SelectorDataset(Dataset):
    def __init__(self, split: SelectorSplit, transform):
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split.record_ids)

    def __getitem__(self, index: int):
        with Image.open(self.split.image_paths[index]) as image:
            transformed = self.transform(image)
        target = torch.as_tensor(self.split.targets[index], dtype=torch.float32)
        return transformed, target, self.split.record_ids[index]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SelectorDataModule(lgn.LightningDataModule):
    """Physical microbatch loader set with separate deterministic audit datasets."""

    def __init__(
            self,
            bundle: SelectorDataBundle,
            batch_size: int = 8,
            num_workers: int = 0,
            seed: int = 42,
            pin_memory: bool = False,
    ):
        super().__init__()
        if batch_size < 1 or 128 % batch_size:
            raise ValueError("physical batch size must be a positive divisor of 128")
        if seed != 42:
            raise ValueError("Phase 3-D1 requires seed 42")
        self.bundle = bundle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self._train_generator = torch.Generator().manual_seed(seed)
        train_transform = transform_aug_selector_efficientnet_v2_s()
        audit_transform = transform_plain_selector_efficientnet_v2_s()
        self.train_dataset = SelectorDataset(bundle.train, train_transform)
        self.audit_train_dataset = SelectorDataset(bundle.train, audit_transform)
        self.val_dataset = SelectorDataset(bundle.val, audit_transform)

    def _loader(self, dataset: Dataset, shuffle: bool, generator=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True, generator=self._train_generator)

    def audit_train_dataloader(self) -> DataLoader:
        return self._loader(self.audit_train_dataset, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

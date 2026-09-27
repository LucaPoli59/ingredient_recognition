"""Validated, deterministic artifact persistence for selector campaigns."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.ingredient_selection.protocol import canonical_json_bytes, sha256_json


METRICS_KEY = ["run_id", "audit_epoch", "split", "class_index"]


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str | Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    _atomic_bytes(Path(path), encoded)


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


class SelectorArtifactStore:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).resolve()
        self.scores_dir = self.output_dir / "audit_scores"

    def initialize(self, *, allow_existing: bool = False) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()) and not allow_existing:
            raise FileExistsError(f"selector output directory is not empty: {self.output_dir}")
        self.scores_dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "campaign_manifest.json"

    @property
    def pilot_path(self) -> Path:
        return self.output_dir / "pilot_cohort.json"

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "metrics_per_label_epoch.csv"

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        identity = manifest.get("campaign_identity")
        if identity is None:
            raise ValueError("campaign manifest must contain campaign_identity")
        expected = sha256_json(identity)
        if manifest.get("campaign_identity_hash") not in (None, expected):
            raise ValueError("campaign identity hash is inconsistent")
        payload = dict(manifest)
        payload["campaign_identity_hash"] = expected
        write_json(self.manifest_path, payload)

    def write_pilot(self, pilot: dict[str, Any]) -> None:
        write_json(self.pilot_path, pilot)

    def existing_audit_epochs(self) -> set[int]:
        if not self.metrics_path.is_file():
            return set()
        metrics = pd.read_csv(self.metrics_path)
        return set(int(value) for value in metrics["audit_epoch"].unique())

    def append_metric_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        incoming = pd.DataFrame(list(rows))
        if incoming.empty:
            raise ValueError("cannot append an empty audit metric set")
        if incoming.duplicated(METRICS_KEY).any():
            raise ValueError("incoming audit metrics contain duplicate keys")
        if self.metrics_path.is_file():
            current = pd.read_csv(self.metrics_path)
            overlap = current.merge(incoming, how="inner", on=METRICS_KEY)
            if not overlap.empty:
                raise ValueError("audit metrics already exist for this run/epoch/split/class")
            incoming = pd.concat([current, incoming], ignore_index=True)
        incoming = incoming.sort_values(METRICS_KEY).reset_index(drop=True)
        _atomic_bytes(self.metrics_path, incoming.to_csv(index=False, lineterminator="\n").encode("utf-8"))

    def write_validation_scores(
            self,
            epoch: int,
            record_ids: list[str],
            targets: np.ndarray,
            logits: np.ndarray,
    ) -> Path:
        path = self.scores_dir / f"validation_epoch_{epoch:02d}.npz"
        if path.exists():
            raise FileExistsError(f"validation scores already exist: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
        os.close(descriptor)
        try:
            np.savez_compressed(
                temporary,
                record_ids=np.asarray(record_ids, dtype=str),
                targets=np.asarray(targets, dtype=np.uint8),
                logits=np.asarray(logits, dtype=np.float32),
            )
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return path

    def write_bootstrap(self, rows: Iterable[dict[str, Any]]) -> None:
        frame = pd.DataFrame(list(rows)).sort_values("class_index")
        _atomic_bytes(
            self.output_dir / "final_validation_bootstrap.csv",
            frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        )

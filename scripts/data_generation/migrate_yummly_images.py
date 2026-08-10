"""Stage Yummly split-local images in the shared ``imgs/standard`` directory.

The command is intentionally additive: it never removes or rewrites an image
from a split directory. Run without ``--apply`` first to inspect the planned
migration, then run with ``--apply`` to build and verify the common store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Iterable


SPLITS = ("train", "val", "test")
DEFAULT_METADATA_FILENAMES = ("metadata.json", "sel_ing_2410_metadata.json")


@dataclass(frozen=True)
class ImageSource:
    split: str
    metadata_filename: str
    source: Path
    relative_path: Path


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(image_ref: object) -> Path:
    if not isinstance(image_ref, str) or not image_ref:
        raise ValueError(f"invalid image reference: {image_ref!r}")
    relative_path = PurePath(image_ref)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe image reference: {image_ref!r}")
    return Path(relative_path)


def existing_metadata_filenames(dataset_root: Path, requested: Iterable[str]) -> list[str]:
    filenames = []
    for filename in requested:
        present = [dataset_root / split / filename for split in SPLITS]
        if all(path.is_file() for path in present):
            filenames.append(filename)
        elif any(path.exists() for path in present):
            missing = [str(path) for path in present if not path.is_file()]
            raise FileNotFoundError(f"metadata generation {filename!r} is incomplete: {missing}")
    if not filenames:
        raise FileNotFoundError("no complete metadata generation was found")
    return filenames


def collect_image_sources(dataset_root: Path, metadata_filenames: Iterable[str]) -> list[ImageSource]:
    sources: list[ImageSource] = []
    errors: list[str] = []
    for split in SPLITS:
        split_root = dataset_root / split
        for metadata_filename in metadata_filenames:
            with (split_root / metadata_filename).open(encoding="utf-8") as metadata_file:
                recipes = json.load(metadata_file)
            if not isinstance(recipes, list):
                errors.append(f"{split}/{metadata_filename} does not contain a list")
                continue
            for record_index, recipe in enumerate(recipes):
                try:
                    relative_path = safe_relative_path(recipe.get("image"))
                except ValueError as error:
                    errors.append(f"{split}/{metadata_filename} record {record_index}: {error}")
                    continue
                source = split_root / relative_path
                if not source.is_file():
                    errors.append(f"{split}/{metadata_filename} record {record_index}: missing {source}")
                    continue
                sources.append(ImageSource(split, metadata_filename, source, relative_path))
    if errors:
        preview = "\n".join(errors[:10])
        more = "" if len(errors) <= 10 else f"\n... and {len(errors) - 10} more errors"
        raise FileNotFoundError(f"cannot stage common images:\n{preview}{more}")
    return sources


def deduplicate_sources(sources: Iterable[ImageSource]) -> dict[Path, list[ImageSource]]:
    by_destination: dict[Path, list[ImageSource]] = defaultdict(list)
    checksum_by_source: dict[Path, str] = {}
    for source in sources:
        by_destination[source.relative_path].append(source)
    for relative_path, candidates in by_destination.items():
        checksums = set()
        for candidate in candidates:
            if candidate.source not in checksum_by_source:
                checksum_by_source[candidate.source] = sha256(candidate.source)
            checksums.add(checksum_by_source[candidate.source])
        if len(checksums) != 1:
            locations = ", ".join(str(candidate.source) for candidate in candidates)
            raise ValueError(f"conflicting image bytes for {relative_path}: {locations}")
    return by_destination


def stage_images(destination_root: Path, sources_by_destination: dict[Path, list[ImageSource]]) -> None:
    staging_root = destination_root.with_name(f".{destination_root.name}.staging")
    if destination_root.exists():
        raise FileExistsError(f"destination already exists: {destination_root}")
    if staging_root.exists():
        raise FileExistsError(f"staging directory already exists: {staging_root}")

    staging_root.mkdir(parents=True)
    try:
        for relative_path, candidates in sorted(sources_by_destination.items()):
            source = candidates[0].source
            destination = staging_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256(source) != sha256(destination):
                raise RuntimeError(f"checksum mismatch while copying {source}")
        os.replace(staging_root, destination_root)
    except Exception:
        # Keep the staging directory for inspection; it is never removed automatically.
        raise


def verify_store(destination_root: Path, sources_by_destination: dict[Path, list[ImageSource]]) -> None:
    errors = []
    for relative_path, candidates in sources_by_destination.items():
        destination = destination_root / relative_path
        if not destination.is_file():
            errors.append(f"missing destination {destination}")
            continue
        source_checksum = sha256(candidates[0].source)
        if source_checksum != sha256(destination):
            errors.append(f"checksum mismatch for {relative_path}")
    if errors:
        preview = "\n".join(errors[:10])
        more = "" if len(errors) <= 10 else f"\n... and {len(errors) - 10} more errors"
        raise RuntimeError(f"common image store verification failed:\n{preview}{more}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/input/yummly"))
    parser.add_argument("--metadata-filename", action="append", dest="metadata_filenames")
    parser.add_argument("--apply", action="store_true", help="copy images into a staged common store")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    requested = args.metadata_filenames or DEFAULT_METADATA_FILENAMES
    metadata_filenames = existing_metadata_filenames(dataset_root, requested)
    sources = collect_image_sources(dataset_root, metadata_filenames)
    sources_by_destination = deduplicate_sources(sources)
    destination_root = dataset_root / "imgs" / "standard"

    print(f"metadata generations: {', '.join(metadata_filenames)}")
    print(f"metadata image references: {len(sources)}")
    print(f"unique image paths: {len(sources_by_destination)}")
    print(f"destination: {destination_root}")

    if not args.apply:
        print("dry run complete; no files were written")
        return

    stage_images(destination_root, sources_by_destination)
    verify_store(destination_root, sources_by_destination)
    print("common image store staged and checksum-verified; split-local images were not removed")


if __name__ == "__main__":
    main()

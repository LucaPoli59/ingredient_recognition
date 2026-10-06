"""Read-only full/selected encoding parity on frozen v5 train and validation metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from settings.config import YUMMLY_PATH
from src.data_processing.images_recipes import images_recipes_processing
from src.data_processing.labels_encoders import MultiLabelBinarizer
from src.ingredient_selection.runtime import PROJECTION_ID, resolve_projection


def verify_runtime(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    projection = resolve_projection(PROJECTION_ID)
    projection.verify_metadata(root)
    full_encoder = MultiLabelBinarizer()
    selected_encoder = MultiLabelBinarizer(classes=list(projection.class_order))
    selected_encoder.fit()
    splits = {}
    for split in ("train", "val"):
        kwargs = dict(data_dir=root / split, metadata_filename=projection.metadata_filename,
                      recipe_feature_label=projection.target_field, images_dir=root / "imgs" / "standard")
        full_paths, full, full_encoder = images_recipes_processing(**kwargs, label_encoder=full_encoder)
        paths, selected, selected_encoder = images_recipes_processing(
            **kwargs, label_encoder=selected_encoder, ingredient_projection=projection)
        if paths != full_paths:
            raise AssertionError(f"{split} record order/population changed")
        np.testing.assert_array_equal(selected, projection.project_columns(full, full_encoder.classes))
        splits[split] = {"records": len(paths), "selected_outputs": selected.shape[1],
                         "empty_projected_targets_retained": int(np.count_nonzero(selected.sum(axis=1) == 0)),
                         "full_column_parity": True}
    projection.verify_metadata(root)
    return {"projection": projection.to_config(), "splits": splits,
            "test_split_accessed": False, "inference_performed": False, "writes_performed": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(YUMMLY_PATH))
    args = parser.parse_args()
    print(json.dumps(verify_runtime(args.data_dir), indent=2))


if __name__ == "__main__":
    main()

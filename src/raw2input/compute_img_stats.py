"""Compute channel statistics from the selected training metadata generation."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torchvision.transforms import v2

from settings.config import DEF_IMAGE_SHAPE, IMG_STATS_FILENAME, YUMMLY_PATH, YUMMLY_TARGET_METADATA_FILENAME
from src.data_processing.images_recipes import ImagesRecipesDataset
from src.data_processing.labels_encoders import MultiLabelBinarizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(YUMMLY_PATH))
    parser.add_argument("--metadata-filename", default=YUMMLY_TARGET_METADATA_FILENAME)
    parser.add_argument("--feature-label", default="ingredients_target")
    parser.add_argument("--images-subdir", type=Path, default=Path("imgs") / "standard")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(DEF_IMAGE_SHAPE),
        v2.ToDtype(torch.float32, scale=True),
    ])

    train_dataset = ImagesRecipesDataset(
        args.dataset_root / "train",
        metadata_filename=args.metadata_filename,
        feature_label=args.feature_label,
        images_dir=args.dataset_root / args.images_subdir,
        transform=transform,
        label_encoder=MultiLabelBinarizer(),
        category="all",
    )

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    pixel_sum = torch.zeros(3, dtype=torch.float32).to(device)
    pixel_squared_sum = torch.zeros(3, dtype=torch.float32).to(device)

    for img_batch, _ in loader:
        img_batch = img_batch.to(device)
        pixel_sum += img_batch.sum(dim=(0, 2, 3))
        pixel_squared_sum += (img_batch ** 2).sum(dim=(0, 2, 3))

    n_pixel = len(train_dataset) * DEF_IMAGE_SHAPE[0] * DEF_IMAGE_SHAPE[1]
    mean = pixel_sum / n_pixel
    std = torch.sqrt(pixel_squared_sum / n_pixel - (mean ** 2))

    mean = mean.cpu().numpy()
    std = std.cpu().numpy()

    print(f"Mean: {mean}, Std: {std}")

    stats_df = pd.DataFrame({"mean": mean, "std": std}).T
    stats_df.columns.name = "channels"
    stats_df.index.name = "stat"
    output_path = args.output or args.dataset_root / IMG_STATS_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path)

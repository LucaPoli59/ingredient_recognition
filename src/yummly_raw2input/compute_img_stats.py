"""File used to compute the mean and std of the training images, to be used in the normalization of the images during training."""

import os
import pandas as pd
import torch
from torchvision.transforms import v2

from settings.config import YUMMLY_PATH, YUMMLY_RECIPES_PATH, DEF_IMAGE_SHAPE, YUMMLY_IMG_STATS_PATH, INPUT_PATH
from src.data_processing.data_handling import ImagesRecipesDataset
from src.data_processing.labels_encoders import MultiLabelBinarizerRobust


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(DEF_IMAGE_SHAPE),
        v2.ToDtype(torch.float32, scale=True),
    ])

    label_encoder = MultiLabelBinarizerRobust()
    # CLEANLAB_OUT_DATA = os.path.join(INPUT_PATH, 'yummly_corrected')
    # train_dataset = ImagesRecipesDataset(os.path.join(CLEANLAB_OUT_DATA, "train"),
    #                                      transform=transform, label_encoder=label_encoder, category="all")

    train_dataset = ImagesRecipesDataset(os.path.join(YUMMLY_PATH, "train"),
                                         transform=transform, label_encoder=label_encoder, category="all")

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, num_workers=4, shuffle=False)

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
    stats_df.to_csv(YUMMLY_IMG_STATS_PATH)

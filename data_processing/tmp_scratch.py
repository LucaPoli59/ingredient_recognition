from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from external_code.helpers import plot

from data_processing.ImagesRecipesDataset import ImagesRecipesDataset
from data_processing.MultiLabelBinarizerRobust import MultiLabelBinarizerRobust
from config import *

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.TrivialAugmentWide(num_magnitude_bins=31),
    v2.ToDtype(torch.float32, scale=True)
])

encoder = MultiLabelBinarizerRobust()

train_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "train"), os.path.join(RECIPES_PATH, 'train.json'),
                                     transform=transforms, label_encoder=encoder)


# image = train_dataset.load_image(0)
# plot([image] + [transforms(image) for _ in range(3)])
# plt.show()


val_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "val"), os.path.join(RECIPES_PATH, 'val.json'),
                                   label_encoder=encoder)
test_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "test"), os.path.join(RECIPES_PATH, 'test.json'),
                                    label_encoder=encoder)

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

img, y = next(iter(train_dataloader))
plot(img[:4])
plt.show()

print(encoder.inverse_transform([y[0]]))

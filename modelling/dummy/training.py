import torch
import os
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from typing import Callable

from data_processing.ImagesRecipesDataset import ImagesRecipesDataset
from data_processing.MultiLabelBinarizerRobust import MultiLabelBinarizerRobust
from modelling.dummy.model import DummyModel
from config import *


def train_step(model: torch.nn.Module,
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
               device: torch.device):
    # Set model in training mode
    model.train()

    loss, accuracy = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward
        y_pred = model(X)

        # Loss and accuracy computation
        batch_loss = loss_fn(y_pred, y)
        loss += batch_loss.item()
        accuracy += accuracy_fn(y_pred, y)

        # Zero grad optimizer
        optimizer.zero_grad()

        # Backward
        batch_loss.backward()

        # Step of optimization
        optimizer.step()

    avg_loss, avg_acc = loss / len(dataloader), accuracy / len(dataloader)

    return avg_loss, avg_acc


def eval_step(model: torch.nn.Module,
              dataloader: DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
              device: torch.device):
    model.eval()

    loss, accuracy = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            accuracy += accuracy_fn(y_pred, y)

    avg_loss, avg_acc = loss / len(dataloader), accuracy / len(dataloader)

    return avg_loss, avg_acc


def multi_label_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = torch.round(torch.sigmoid(y_pred)) # get the binary predictions
    # num of hits / num of classes mean over the batch
    return torch.mean((y_pred == y_true).sum(dim=1) / y_pred.size(1)).item()


def train(model: torch.nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          loss_fn: torch.nn.Module,
          accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
          optimizer: torch.optim.Optimizer,
          epochs: int = 10,
          device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device)

    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in tqdm(range(epochs), desc=f"Training model"):
        train_loss, train_acc = train_step(model, train_dataloader, optimizer, loss_fn, accuracy_fn, device)

        val_loss, val_acc = eval_step(model, val_dataloader, loss_fn, accuracy_fn, device)

        print(f"\nEpoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    return results


if __name__ == "__main__":
    # Load the dataset
    INPUT_SHAPE = 224
    CATEGORY = "mexican"

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((INPUT_SHAPE, INPUT_SHAPE)),
        v2.TrivialAugmentWide(num_magnitude_bins=31),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # Load the dataset
    mlb = MultiLabelBinarizerRobust()
    train_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "train"), os.path.join(RECIPES_PATH, "train.json"),
                                         transform=transform, label_encoder=mlb, category=CATEGORY)
    val_dataset = ImagesRecipesDataset(os.path.join(IMAGES_PATH, "val"), os.path.join(RECIPES_PATH, "val.json"),
                                       transform=transform, label_encoder=mlb, category=CATEGORY)

    BATCH_SIZE = 128
    NUM_WORKERS = 1
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Creation of the model
    model = DummyModel(INPUT_SHAPE, len(mlb.classes)+1)

    # Training hyperparameters
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    accuracy_fn = multi_label_accuracy

    # Training
    results = train(model, train_dataloader, val_dataloader, loss_fn, accuracy_fn, optimizer, epochs=10)
    print("Training Done!")
    print(results)

    # Save the model
    torch.save(model.state_dict(), "model.pth")

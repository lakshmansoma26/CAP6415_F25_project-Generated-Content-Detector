import os
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(
    data_root: str = "data/raw",
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:

    # Standard ImageNet-like preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    print(f"[INFO] Train samples: {len(train_dataset)}")
    print(f"[INFO] Test samples:  {len(test_dataset)}")
    print(f"[INFO] Classes: {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


def build_model(num_classes: int = 2) -> nn.Module:
    # For older torchvision versions:
    model = models.resnet18(pretrained=True)

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 10 == 0:
            print(f"  [Train] Epoch {epoch} | Step {i+1}/{len(loader)}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    epoch: int,
    mode: str = "Val",
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"  [{mode}] Epoch {epoch} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def train_resnet(
    epochs: int = 15,
    lr: float = 1e-4,
    batch_size: int = 32,
):
    os.makedirs("results/models", exist_ok=True)

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = build_model(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    best_acc = 0.0
    best_path = os.path.join("results", "models", "resnet18_best.pth")

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, epoch, mode="Test"
        )

        elapsed = time.time() - start_time
        print(
            f"[EPOCH {epoch}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}, "
            f"Time: {elapsed:.1f}s"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(val_loss)
        history["test_acc"].append(val_acc)

        # Save best model by test accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": val_acc,
                },
                best_path,
            )
            print(f"[INFO] Saved new best model with Test Acc = {val_acc:.4f}")

    # Save metrics to CSV
    df = pd.DataFrame(history)
    metrics_path = os.path.join("results", "week5_cnn_resnet18_metrics.csv")
    df.to_csv(metrics_path, index=False)
    print(f"[INFO] Saved training history to {metrics_path}")
    print(f"[INFO] Best Test Acc: {best_acc:.4f} | Model: {best_path}")


if __name__ == "__main__":
    train_resnet(
        epochs=15,
        lr=1e-4,
        batch_size=32,
    )

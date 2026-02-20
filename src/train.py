from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Tuple
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import models
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    EPOCHS,
    IMG_SIZE,
    LR,
    MODEL_DIR,
    LOG_DIR,
    NUM_WORKERS,
    RAW_DATA_DIR,
    SEED,
    SPLIT_FILE,
    UNFREEZE_LAST_N_BLOCKS,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from dataset import GameGenreDataset
from mapping import GENRES
from utils import set_seed


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    if UNFREEZE_LAST_N_BLOCKS >= 1:
        for p in model.layer4.parameters():
            p.requires_grad = True
    if UNFREEZE_LAST_N_BLOCKS >= 2:
        for p in model.layer3.parameters():
            p.requires_grad = True

    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    set_seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "training_log.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_train = GameGenreDataset(split="train", augment=True)

    val_size = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size

    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    full_train_noaug = GameGenreDataset(split="train", augment=False)
    val_ds = Subset(full_train_noaug, val_ds.indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_model(num_classes=len(GENRES)).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = MODEL_DIR / "best_resnet50.pt"

    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")
    print(f"Trainable params: {sum(p.numel() for p in params):,}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.time()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y).sum().item()
            running_total += x.size(0)

            pbar.set_postfix(
                loss=running_loss / max(running_total, 1),
                acc=running_correct / max(running_total, 1),
            )

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "genres": GENRES,
                    "img_size": IMG_SIZE,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "config": {
                        "epochs": EPOCHS,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                        "unfreeze_last_n_blocks": UNFREEZE_LAST_N_BLOCKS,
                        "val_split": VAL_SPLIT,
                        "seed": SEED,
                    },
                },
                best_path,
            )
            print(f"✅ Saved best model to: {best_path} (val_acc={val_acc:.4f})")

    print(f"Best val acc: {best_val_acc:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()

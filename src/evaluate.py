import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset import GameGenreDataset
from mapping import GENRES
from config import BATCH_SIZE, NUM_WORKERS

MODEL_PATH = Path("outputs/models/best_resnet50.pt")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    import torchvision.models as models
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(GENRES))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = GameGenreDataset(split="test", augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = load_model(device)

    all_preds = []
    all_labels = []

    for x, y in tqdm(loader):
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    print("\nClassification report:\n")
    print(classification_report(all_labels, all_preds, target_names=GENRES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=GENRES, yticklabels=GENRES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    out_path = FIG_DIR / "confusion_matrix.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved confusion matrix to {out_path}")


if __name__ == "__main__":
    evaluate()

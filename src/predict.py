import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

from mapping import GENRES

MODEL_PATH = Path("outputs/models/best_resnet50.pt")


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(GENRES))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top3 = torch.topk(probs, 3)

    print("\nPrediction:")
    for i in range(3):
        cls = GENRES[top3.indices[i]]
        conf = top3.values[i].item() * 100
        print(f"{cls:20s} {conf:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    predict(args.image)

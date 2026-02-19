import json
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from config import RAW_DATA_DIR, SPLIT_FILE, IMG_SIZE
from mapping import genre_for_game, GENRES
from utils import list_images


class GameGenreDataset(Dataset):
    def __init__(self, split: str = "train", augment: bool = False):
        assert split in ["train", "test"]

        with open(SPLIT_FILE, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        self.games = split_data["train_games"] if split == "train" else split_data["test_games"]

        self.label_to_idx = {g: i for i, g in enumerate(GENRES)}

        self.samples = []
        for root in ["train", "test"]:  
            base = RAW_DATA_DIR / root
            if not base.exists():
                continue

            for game_folder in base.iterdir():
                if not game_folder.is_dir():
                    continue

                game_name = game_folder.name
                if game_name not in self.games:
                    continue

                genre = genre_for_game(game_name)
                label = self.label_to_idx[genre]

                for img_path in list_images(game_folder):
                    self.samples.append((img_path, label))

        if augment and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        for split in ["train", "test"]:
            ds = GameGenreDataset(split=split, augment=False)
            print(f"\n{split.upper()} samples: {len(ds)}")

            counts = Counter()
            for _, label in ds.samples:
                counts[label] += 1

            print("Per-class distribution:")
            for genre, idx in {g: i for i, g in enumerate(GENRES)}.items():
                print(f"{genre:20s}: {counts[idx]}")

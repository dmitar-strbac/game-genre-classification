import random
import numpy as np
import torch
from pathlib import Path
from config import IMG_EXTENSIONS

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_images(folder: Path):
    return [
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMG_EXTENSIONS
    ]

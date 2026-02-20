from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "split_v2.json"

IMG_SIZE = 224

BATCH_SIZE = 32
NUM_WORKERS = 4

SEED = 42

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

EPOCHS = 20
LR = 5e-5
WEIGHT_DECAY = 1e-4

UNFREEZE_LAST_N_BLOCKS = 2  

VAL_SPLIT = 0.2

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"


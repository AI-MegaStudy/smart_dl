from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "data"
MODEL_ROOT = PACKAGE_ROOT / "models"

DEFAULT_FRUIT_TYPE = "apple"
DEFAULT_LABELS = ("A", "B", "C")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MODEL_NAME = "resnet18"
DEFAULT_MODEL_VERSION = "freshness-resnet18-v0"

QC_LABEL_MAPPING = {
    "L": "A",
    "M": "B",
    "S": "C",
}

QC_GRADE_NAMES = {
    "A": "special",
    "B": "high",
    "C": "normal",
}

TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
TEST_RATIO = 0.2

"""SVM model configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # Project root
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = CACHE_DIR / "Dataset"


RANDOM_SEED = 42

# Embedding configuration
EMBEDDING_DIM = 300

# SVM hyperparameters
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"

# Cross-validation settings
NUM_FOLDS = 10

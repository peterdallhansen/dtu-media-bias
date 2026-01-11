"""Transformer model configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # Project root
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = CACHE_DIR / "Dataset"


RANDOM_SEED = 42
DEVICE = "auto"  # "cuda", "mps", "cpu", or "auto"

# Model
TRANSFORMER_MODEL = "distilbert-base-uncased"

# Extra features (metadata not present in article text)
USE_EXTRA_FEATURES = True
USE_DATE_FEATURES_T = True  # Publication date
USE_HYPERLINK_FEATURES_T = False  # Link structure
USE_SENTIMENT_FEATURES_T = True
USE_NER_FEATURES_T = True

NUM_EXTRA_FEATURES_T = (
    (
        (2 if USE_DATE_FEATURES_T else 0)
        + (4 if USE_HYPERLINK_FEATURES_T else 0)
        + (3 if USE_SENTIMENT_FEATURES_T else 0)
        + (5 if USE_NER_FEATURES_T else 0)
    )
    if USE_EXTRA_FEATURES
    else 0
)

# Training hyperparameters
BATCH_SIZE = 64  # Larger batch = smoother gradients (A100 has plenty of VRAM)
LEARNING_RATE = 5e-5  # Lower LR for more stable training
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40
DROPOUT = 0.5

# Cross-validation settings
NUM_FOLDS = 10
ENSEMBLE_TOP_K = 3
EARLY_STOPPING_PATIENCE = 5

"""BERT-MLP model configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # Project root
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = CACHE_DIR / "Dataset"


RANDOM_SEED = 42

# Model architecture
TRANSFORMER_MODEL = "distilbert-base-uncased"
EMBEDDING_DIM = 768  # DistilBERT hidden size
HIDDEN_DIM = 256
DROPOUT = 0.5

# Extra features from preprocessing
USE_EXTRA_FEATURES = True
USE_DATE_FEATURES = True
USE_HYPERLINK_FEATURES = False
USE_SENTIMENT_FEATURES = True
USE_NER_FEATURES = True

NUM_EXTRA_FEATURES = (
    (
        (2 if USE_DATE_FEATURES else 0)
        + (4 if USE_HYPERLINK_FEATURES else 0)
        + (3 if USE_SENTIMENT_FEATURES else 0)
        + (5 if USE_NER_FEATURES else 0)
    )
    if USE_EXTRA_FEATURES
    else 0
)

# Training
BATCH_SIZE = 64  # Larger batch = smoother gradients (A100 has plenty of VRAM)
LEARNING_RATE = 1e-4  # 1e-3 was too high, causing wild F1 oscillations
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 50

# Cross-validation settings
NUM_FOLDS = 10
ENSEMBLE_TOP_K = 3
EARLY_STOPPING_PATIENCE = 7

DEVICE = "auto"  # "cuda", "mps", "cpu", or "auto"

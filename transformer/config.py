"""Transformer model configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # Project root
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = CACHE_DIR / "Dataset"

ARTICLES_TRAIN = DATA_DIR / "articles-training-byarticle-20181122.xml"
LABELS_TRAIN = DATA_DIR / "ground-truth-training-byarticle-20181122.xml"
ARTICLES_TEST = DATA_DIR / "articles-test-byarticle-20181207.xml"
LABELS_TEST = DATA_DIR / "ground-truth-test-byarticle-20181207.xml"

ARTICLES_TEST_BYPUB = DATA_DIR / "articles-test-bypublisher-20181212.xml"
LABELS_TEST_BYPUB = DATA_DIR / "ground-truth-test-bypublisher-20181212.xml"

RANDOM_SEED = 42
DEVICE = "mps"  # Should be set to 'cuda' if on nvidia graphics card

# Model
TRANSFORMER_MODEL = "distilbert-base-uncased"

# Extra features (metadata not present in article text)
USE_EXTRA_FEATURES = True
USE_DATE_FEATURES_T = True  # Publication date
USE_HYPERLINK_FEATURES_T = True  # Link structure
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
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40
DROPOUT = 0.5

# Cross-validation settings
NUM_FOLDS = 10
ENSEMBLE_TOP_K = 3
EARLY_STOPPING_PATIENCE = 3

# Length debiasing (set to reduce length bias in predictions)
USE_LENGTH_DEBIASING = True
# Random truncation range: [min, max] percent of tokens to keep during training
TRUNCATION_RANGE = (0.3, 1.0)  # Randomly keep 30-100% of tokens
# Gradient reversal for length prediction (adversarial debiasing)
USE_GRADIENT_REVERSAL = True
GRADIENT_REVERSAL_LAMBDA = 0.1  # Strength of adversarial loss

# Fine-tuning configuration (end-to-end training with BERT)
USE_FINE_TUNING = True  # If True, use end-to-end training instead of frozen embeddings
FINE_TUNE_LAYERS = 1  # Number of transformer layers to unfreeze (from top)
FINE_TUNE_LR = 2e-5  # Learning rate for BERT layers
CLASSIFIER_LR = 1e-3  # Learning rate for classifier head (higher than BERT)
DISCRIMINATIVE_LR_FACTOR = 0.9  # Multiply LR by this for each lower layer
FREEZE_EPOCHS = 3  # Train classifier only for this many epochs before unfreezing
FINE_TUNE_EPOCHS = 10  # Additional epochs with BERT unfrozen
LABEL_SMOOTHING = 0.1  # Smooth labels to prevent overconfidence


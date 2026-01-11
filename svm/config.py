"""SVM model configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # Project root
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = CACHE_DIR / "Dataset"

# Data paths
ARTICLES_TRAIN = DATA_DIR / "articles-training-byarticle-20181122.xml"
LABELS_TRAIN = DATA_DIR / "ground-truth-training-byarticle-20181122.xml"
ARTICLES_TEST = DATA_DIR / "articles-test-byarticle-20181207.xml"
LABELS_TEST = DATA_DIR / "ground-truth-test-byarticle-20181207.xml"

ARTICLES_TEST_BYPUB = DATA_DIR / "articles-test-bypublisher-20181212.xml"
LABELS_TEST_BYPUB = DATA_DIR / "ground-truth-test-bypublisher-20181212.xml"

RANDOM_SEED = 42

# Embedding configuration
EMBEDDING_DIM = 300

# SVM hyperparameters
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"

# Cross-validation settings
NUM_FOLDS = 10

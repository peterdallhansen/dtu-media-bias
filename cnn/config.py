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

EMBEDDING_DIM = 300

MAX_SEQ_LEN = 512
MIN_WORD_FREQ = 2
VOCAB_SIZE = 50000

NUM_FILTERS = 128
KERNEL_SIZES = [3, 4, 5]
DROPOUT = 0.5

USE_DATE_FEATURES = True
USE_HYPERLINK_FEATURES = True
USE_SENTIMENT_FEATURES = True
USE_NER_FEATURES = True

NUM_EXTRA_FEATURES = (
    (2 if USE_DATE_FEATURES else 0)
    + (4 if USE_HYPERLINK_FEATURES else 0)
    + (3 if USE_SENTIMENT_FEATURES else 0)
    + (5 if USE_NER_FEATURES else 0)
)

BATCH_SIZE = 32
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 40

# Cross-validation settings
NUM_FOLDS = 10
ENSEMBLE_TOP_K = 3
EARLY_STOPPING_PATIENCE = 5

RANDOM_SEED = 42
DEVICE = "mps"  # Should be set to 'cuda' if on nvidia graphics card

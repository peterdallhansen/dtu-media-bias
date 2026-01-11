from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Dataset"
CACHE_DIR = BASE_DIR / "cache"

ARTICLES_TRAIN = DATA_DIR / "articles-training-byarticle-20181122.xml"
LABELS_TRAIN = DATA_DIR / "ground-truth-training-byarticle-20181122.xml"
ARTICLES_TEST = DATA_DIR / "articles-test-byarticle-20181207.xml"
LABELS_TEST = DATA_DIR / "ground-truth-test-byarticle-20181207.xml"

EMBEDDING_DIM = 300

MAX_SEQ_LEN = 512
MIN_WORD_FREQ = 2
VOCAB_SIZE = 50000

NUM_FILTERS = 128
KERNEL_SIZES = [3, 4, 5]
DROPOUT = 0.5

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20

RANDOM_SEED = 42
DEVICE = "cuda"

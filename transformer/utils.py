import pickle
import numpy as np
from tqdm import tqdm
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from cnn import config as cnn_config


def compute_embeddings(data, cache_path, model_name="distilbert-base-uncased"):
    """Pre-compute transformer document embeddings."""
    if cache_path.exists():
        print(f"Loading embeddings: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Computing {model_name} embeddings...")
    doc_embeddings = TransformerDocumentEmbeddings(model_name)

    embeddings = []
    for item in tqdm(data, desc="Embedding"):
        text = " ".join(item["tokens"][:512])
        if not text.strip():
            text = "empty"

        sentence = Sentence(text)
        doc_embeddings.embed(sentence)
        emb = sentence.embedding.cpu().numpy()
        embeddings.append(emb)
        sentence.clear_embeddings()

    embeddings = np.array(embeddings)

    print(f"Saving embeddings: {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


def compute_truncated_embeddings(
    data, cache_path, model_name="distilbert-base-uncased", truncation_range=(0.3, 1.0)
):
    """
    Compute embeddings with random truncation to reduce length bias.
    
    Each article is randomly truncated to keep between truncation_range[0] and
    truncation_range[1] percent of its tokens. This prevents the model from
    exploiting length as a shortcut.
    
    Args:
        data: List of article dictionaries with 'tokens' key.
        cache_path: Path to save cached embeddings.
        model_name: Transformer model name.
        truncation_range: Tuple (min_pct, max_pct) of tokens to keep.
    
    Returns:
        Tuple of (embeddings array, lengths array for gradient reversal).
    """
    if cache_path.exists():
        print(f"Loading truncated embeddings: {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
            if isinstance(cached, dict):
                return cached["embeddings"], cached["lengths"]
            return cached, None

    print(f"Computing {model_name} embeddings with random truncation...")
    print(f"Truncation range: {truncation_range[0]*100:.0f}% - {truncation_range[1]*100:.0f}%")
    doc_embeddings = TransformerDocumentEmbeddings(model_name)
    
    min_pct, max_pct = truncation_range

    embeddings = []
    lengths = []  # Store normalized lengths for gradient reversal
    
    for item in tqdm(data, desc="Embedding (truncated)"):
        tokens = item["tokens"][:512]
        original_len = len(tokens)
        
        # Random truncation
        keep_pct = np.random.uniform(min_pct, max_pct)
        keep_n = max(1, int(original_len * keep_pct))
        truncated_tokens = tokens[:keep_n]
        
        text = " ".join(truncated_tokens)
        if not text.strip():
            text = "empty"

        sentence = Sentence(text)
        doc_embeddings.embed(sentence)
        emb = sentence.embedding.cpu().numpy()
        embeddings.append(emb)
        
        # Store normalized length (0-1 scale) for adversarial training
        lengths.append(original_len / 512.0)
        
        sentence.clear_embeddings()

    embeddings = np.array(embeddings)
    lengths = np.array(lengths, dtype=np.float32)

    print(f"Saving truncated embeddings: {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "lengths": lengths}, f)

    return embeddings, lengths


def extract_features(data, config):
    """Extract metadata features based on config settings.

    Features in data were created by preprocess.py using cnn_config.
    We use cnn_config to know data structure, transformer config to select features.
    """
    if not config.USE_EXTRA_FEATURES:
        return None

    features = []
    for item in data:
        item_features = item.get("features", [])
        selected = []
        idx = 0

        # Date (2 dims) - advance idx if present in data, include if transformer wants it
        if cnn_config.USE_DATE_FEATURES:
            if config.USE_DATE_FEATURES_T:
                selected.extend(item_features[idx : idx + 2])
            idx += 2

        # Hyperlinks (4 dims)
        if cnn_config.USE_HYPERLINK_FEATURES:
            if config.USE_HYPERLINK_FEATURES_T:
                selected.extend(item_features[idx : idx + 4])
            idx += 4

        # Sentiment (3 dims)
        if cnn_config.USE_SENTIMENT_FEATURES:
            if config.USE_SENTIMENT_FEATURES_T:
                selected.extend(item_features[idx : idx + 3])
            idx += 3

        # NER (5 dims)
        if cnn_config.USE_NER_FEATURES:
            if config.USE_NER_FEATURES_T:
                selected.extend(item_features[idx : idx + 5])
            idx += 5

        features.append(selected)

    features = np.array(features, dtype=np.float32)
    return features if features.shape[1] > 0 else None

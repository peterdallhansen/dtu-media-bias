"""Utility functions for BERT-MLP classifier."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(predictions, labels, threshold=0.5):
    """Calculate classification metrics from predictions."""
    preds_binary = (np.array(predictions) >= threshold).astype(int)
    labels = np.array(labels).astype(int)
    
    return {
        "accuracy": accuracy_score(labels, preds_binary),
        "precision": precision_score(labels, preds_binary, zero_division=0),
        "recall": recall_score(labels, preds_binary, zero_division=0),
        "f1": f1_score(labels, preds_binary, zero_division=0),
    }


def extract_features(data, config):
    """Extract metadata features based on config settings.
    
    Features are created by preprocess.py. This function selects which
    features to use based on the bert_mlp config.
    """
    from cnn import config as cnn_config
    
    if not config.USE_EXTRA_FEATURES:
        return None

    features = []
    for item in data:
        item_features = item.get("features", [])
        selected = []
        idx = 0

        # Date (2 dims)
        if cnn_config.USE_DATE_FEATURES:
            if config.USE_DATE_FEATURES:
                selected.extend(item_features[idx : idx + 2])
            idx += 2

        # Hyperlinks (4 dims)
        if cnn_config.USE_HYPERLINK_FEATURES:
            if config.USE_HYPERLINK_FEATURES:
                selected.extend(item_features[idx : idx + 4])
            idx += 4

        # Sentiment (3 dims)
        if cnn_config.USE_SENTIMENT_FEATURES:
            if config.USE_SENTIMENT_FEATURES:
                selected.extend(item_features[idx : idx + 3])
            idx += 3

        # NER (5 dims)
        if cnn_config.USE_NER_FEATURES:
            if config.USE_NER_FEATURES:
                selected.extend(item_features[idx : idx + 5])
            idx += 5

        features.append(selected)

    features = np.array(features, dtype=np.float32)
    return features if features.shape[1] > 0 else None

"""Utility functions for SVM baseline."""

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_document_vectors(data, word2vec, dim=300):
    """
    Compute document vectors by averaging word embeddings.
    
    Unlike CNN/transformer approaches that truncate sequences, this method
    uses all tokens in each document to compute the representation.
    
    Args:
        data: List of article dictionaries containing 'tokens' key
        word2vec: Gensim word2vec model
        dim: Embedding dimension
        
    Returns:
        Array of shape (num_documents, dim)
    """
    doc_vectors = []
    coverage_stats = []

    for item in tqdm(data, desc="Document vectors"):
        tokens = item["tokens"]
        vectors = []

        for token in tokens:
            if token in word2vec:
                vectors.append(word2vec[token])

        if vectors:
            doc_vec = np.mean(vectors, axis=0)
            coverage_stats.append(len(vectors) / len(tokens) if tokens else 0)
        else:
            doc_vec = np.zeros(dim)
            coverage_stats.append(0)

        doc_vectors.append(doc_vec)

    avg_coverage = np.mean(coverage_stats)
    avg_tokens = np.mean([len(d["tokens"]) for d in data])
    print(f"  Coverage: {avg_coverage:.1%}, avg tokens: {avg_tokens:.0f}")

    return np.array(doc_vectors)


def calculate_metrics(y_true, y_pred):
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

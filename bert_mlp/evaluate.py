"""
BERT-MLP evaluation script.

Evaluates trained ensemble on test sets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from bert_mlp import config
from bert_mlp.model import BertMLP
from bert_mlp.dataset import BertEmbeddingDataset
from bert_mlp.utils import calculate_metrics, extract_features
from device import get_device
from preprocess import load_cached_data
from transformer.utils import compute_embeddings


def load_ensemble(device):
    """Load trained ensemble models."""
    info_path = config.CACHE_DIR / "bert_mlp_ensemble_info.pt"
    if not info_path.exists():
        return None, None

    info = torch.load(info_path, map_location=device, weights_only=False)
    top_indices = info["top_indices"]
    input_dim = info["input_dim"]
    num_extra_features = info.get("num_extra_features", 0)

    models = []
    for idx in top_indices:
        path = config.CACHE_DIR / f"bert_mlp_fold_{idx}.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = BertMLP(
                input_dim=input_dim,
                hidden_dim=config.HIDDEN_DIM,
                dropout=config.DROPOUT,
                num_extra_features=num_extra_features,
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)

    return models, info


def evaluate_ensemble(models, embeddings, labels, features, device):
    """Evaluate ensemble on a dataset."""
    num_extra = features.shape[1] if features is not None else 0
    dataset = BertEmbeddingDataset(embeddings, labels, features)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                emb = batch["embedding"].to(device)
                feat = (
                    batch["features"].to(device)
                    if batch["features"].numel() > 0
                    else None
                )
                outputs = model(emb, feat)
                preds.extend(outputs.cpu().numpy())
        all_preds.append(preds)

    ensemble_preds = np.mean(all_preds, axis=0)
    return calculate_metrics(ensemble_preds, labels)


def main():
    device = get_device(config.DEVICE)
    print(f"Device: {device}")

    # Load ensemble
    models, info = load_ensemble(device)
    if not models:
        print("No trained models found. Run train.py first.")
        return

    print(f"Loaded {len(models)} ensemble models")

    # By-article test set
    print("\n" + "=" * 50)
    print("By-Article Test Set")
    print("=" * 50)

    try:
        data = load_cached_data("test_byarticle")
        embeddings = compute_embeddings(
            data,
            config.CACHE_DIR / "test_byarticle_bert_mlp.pkl",
            config.TRANSFORMER_MODEL,
        )
        labels = np.array([item["label"] for item in data])
        features = extract_features(data, config)

        metrics = evaluate_ensemble(models, embeddings, labels, features, device)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
    except FileNotFoundError:
        print("Test set not found.")

    # By-publisher test set
    print("\n" + "=" * 50)
    print("By-Publisher Test Set")
    print("=" * 50)

    try:
        data = load_cached_data("test_bypublisher")
        embeddings = compute_embeddings(
            data,
            config.CACHE_DIR / "test_bypublisher_bert_mlp.pkl",
            config.TRANSFORMER_MODEL,
        )
        labels = np.array([item["label"] for item in data])
        features = extract_features(data, config)

        metrics = evaluate_ensemble(models, embeddings, labels, features, device)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
    except FileNotFoundError:
        print("Test set not found.")


if __name__ == "__main__":
    main()

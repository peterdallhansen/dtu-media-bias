"""
BERT-MLP training script.

Trains an MLP classifier on pre-computed DistilBERT embeddings using
10-fold cross-validation and ensemble selection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from bert_mlp import config
from bert_mlp.model import BertMLP
from bert_mlp.dataset import BertEmbeddingDataset
from bert_mlp.utils import calculate_metrics, extract_features
from device import get_device
from preprocess import load_cached_data
from transformer.utils import compute_embeddings


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        emb = batch["embedding"].to(device)
        features = (
            batch["features"].to(device) if batch["features"].numel() > 0 else None
        )
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(emb, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_preds, all_labels)
    return total_loss / len(loader), metrics


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            emb = batch["embedding"].to(device)
            features = (
                batch["features"].to(device) if batch["features"].numel() > 0 else None
            )
            labels = batch["label"].to(device)

            outputs = model(emb, features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_preds, all_labels)
    return total_loss / len(loader), metrics


def train_fold(
    fold_idx, train_indices, val_indices, embeddings, labels, features, device
):
    """Train a single fold."""
    # Create datasets for this fold
    train_emb = embeddings[train_indices]
    train_labels = labels[train_indices]
    train_features = features[train_indices] if features is not None else None

    val_emb = embeddings[val_indices]
    val_labels = labels[val_indices]
    val_features = features[val_indices] if features is not None else None

    train_dataset = BertEmbeddingDataset(train_emb, train_labels, train_features)
    val_dataset = BertEmbeddingDataset(val_emb, val_labels, val_features)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize model
    num_extra = config.NUM_EXTRA_FEATURES if features is not None else 0
    model = BertMLP(
        input_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        num_extra_features=num_extra,
    ).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["f1"])

        print(
            f"  Epoch {epoch+1}: train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    return best_model_state, best_val_f1


def main():
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    device = get_device(config.DEVICE)
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_data = load_cached_data("train")
    print(f"Training samples: {len(train_data)}")

    # Compute BERT embeddings
    embeddings = compute_embeddings(
        train_data,
        config.CACHE_DIR / "train_bert_mlp.pkl",
        config.TRANSFORMER_MODEL,
    )
    labels = np.array([item["label"] for item in train_data])

    # Extract extra features
    features = extract_features(train_data, config)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Extra features: {features.shape if features is not None else 'None'}")

    # Cross-validation
    config.CACHE_DIR.mkdir(exist_ok=True)
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )

    fold_scores = []
    print(f"\n{config.NUM_FOLDS}-Fold Cross Validation")

    for fold_idx, (train_indices, val_indices) in enumerate(
        skf.split(embeddings, labels)
    ):
        print(
            f"\nFold {fold_idx + 1}/{config.NUM_FOLDS} (train={len(train_indices)}, val={len(val_indices)})"
        )

        best_state, best_f1 = train_fold(
            fold_idx, train_indices, val_indices, embeddings, labels, features, device
        )
        fold_scores.append(best_f1)

        # Save fold model
        torch.save(
            {
                "model_state_dict": best_state,
                "fold": fold_idx,
                "val_f1": best_f1,
                "num_extra_features": (
                    config.NUM_EXTRA_FEATURES if features is not None else 0
                ),
            },
            config.CACHE_DIR / f"bert_mlp_fold_{fold_idx}.pt",
        )
        print(f"  Best val F1: {best_f1:.4f}")

    print(
        f"\nCV Results: mean F1={np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})"
    )

    # Select top-k models for ensemble
    top_k = min(config.ENSEMBLE_TOP_K, config.NUM_FOLDS)
    top_indices = np.argsort(fold_scores)[-top_k:]
    print(f"\nEnsemble: top {top_k} folds {[i+1 for i in top_indices]}")

    torch.save(
        {
            "num_folds": config.NUM_FOLDS,
            "fold_scores": fold_scores,
            "top_k": top_k,
            "top_indices": top_indices.tolist(),
            "num_extra_features": (
                config.NUM_EXTRA_FEATURES if features is not None else 0
            ),
            "input_dim": config.EMBEDDING_DIM,
        },
        config.CACHE_DIR / "bert_mlp_ensemble_info.pt",
    )

    print("Done. Run bert_mlp/evaluate.py or root evaluate.py for test results.")


if __name__ == "__main__":
    main()

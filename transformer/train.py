"""Transformer-based hyperpartisan news detection with K-fold cross-validation.

This version includes length debiasing to prevent the model from exploiting
article length as a shortcut. Two techniques are used:

1. Random truncation: Articles are randomly truncated to 30-100% of their
   original length during embedding computation, breaking length-label correlation.
   
2. Gradient reversal: An adversarial head tries to predict article length from
   the hidden representation. Gradient reversal ensures the main classifier
   cannot use length information.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import transformer.config as config
from preprocess import load_cached_data
from cnn.utils import calculate_metrics
from transformer.model import TransformerClassifier, DebiasedTransformerClassifier
from transformer.dataset import EmbeddingDataset
from transformer.utils import compute_embeddings, compute_truncated_embeddings, extract_features


class EmbeddingDatasetWithLength(EmbeddingDataset):
    """Dataset that includes normalized article length for adversarial training."""
    
    def __init__(self, embeddings, labels, features=None, lengths=None):
        super().__init__(embeddings, labels, features)
        self.lengths = lengths
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.lengths is not None:
            item["length"] = torch.tensor(self.lengths[idx], dtype=torch.float)
        return item


def train_epoch_debiased(model, loader, criterion, length_criterion, optimizer, device, use_features):
    """Training epoch with adversarial length debiasing."""
    model.train()
    total_loss = 0
    total_length_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)
        lengths = batch["length"].to(device)
        features = batch["features"].to(device) if use_features else None

        optimizer.zero_grad()
        
        # Get both classification and length predictions
        class_output, length_pred = model(
            embeddings, features, return_length_pred=True
        )
        
        # Main classification loss
        class_loss = criterion(class_output, labels)
        
        # Adversarial length prediction loss
        # (gradient reversal already applied in model, so we just minimize this)
        length_loss = length_criterion(length_pred, lengths)
        
        # Combined loss
        loss = class_loss + length_loss
        loss.backward()
        optimizer.step()

        total_loss += class_loss.item()
        total_length_loss += length_loss.item()
        all_preds.extend(class_output.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    avg_length_loss = total_length_loss / len(loader)
    return avg_loss, avg_length_loss, calculate_metrics(all_preds, all_labels)


def train_epoch(model, loader, criterion, optimizer, device, use_features):
    """Standard training epoch (no debiasing)."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)
        features = batch["features"].to(device) if use_features else None

        optimizer.zero_grad()
        outputs = model(embeddings, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), calculate_metrics(all_preds, all_labels)


def evaluate(model, loader, criterion, device, use_features):
    """Evaluation function."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)
            features = batch["features"].to(device) if use_features else None

            outputs = model(embeddings, features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), calculate_metrics(all_preds, all_labels), np.array(all_preds)


def get_predictions(model, loader, device, use_features):
    """Get predictions from model."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            features = batch["features"].to(device) if use_features else None
            outputs = model(embeddings, features)
            all_preds.extend(outputs.cpu().numpy())
    return np.array(all_preds)


def train_fold(train_idx, val_idx, embeddings, labels, features, device, lengths=None):
    """Train a single fold with optional debiasing."""
    use_features = features is not None
    use_debiasing = config.USE_LENGTH_DEBIASING and config.USE_GRADIENT_REVERSAL and lengths is not None

    # Create datasets
    if use_debiasing:
        train_dataset = EmbeddingDatasetWithLength(
            embeddings[train_idx], labels[train_idx],
            features[train_idx] if use_features else None,
            lengths[train_idx],
        )
        val_dataset = EmbeddingDatasetWithLength(
            embeddings[val_idx], labels[val_idx],
            features[val_idx] if use_features else None,
            lengths[val_idx],
        )
    else:
        train_dataset = EmbeddingDataset(
            embeddings[train_idx], labels[train_idx],
            features[train_idx] if use_features else None,
        )
        val_dataset = EmbeddingDataset(
            embeddings[val_idx], labels[val_idx],
            features[val_idx] if use_features else None,
        )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Create model (debiased or standard)
    if use_debiasing:
        model = DebiasedTransformerClassifier(
            input_dim=embeddings.shape[1],
            hidden_dim=256,
            dropout=config.DROPOUT,
            num_extra_features=config.NUM_EXTRA_FEATURES_T,
            gradient_reversal_lambda=config.GRADIENT_REVERSAL_LAMBDA,
        ).to(device)
    else:
        model = TransformerClassifier(
            input_dim=embeddings.shape[1],
            hidden_dim=256,
            dropout=config.DROPOUT,
            num_extra_features=config.NUM_EXTRA_FEATURES_T,
        ).to(device)

    criterion = nn.BCELoss()
    length_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        if use_debiasing:
            train_loss, length_loss, train_metrics = train_epoch_debiased(
                model, train_loader, criterion, length_criterion, optimizer, device, use_features
            )
            print(
                f"  Epoch {epoch+1}: "
                f"train_f1={train_metrics['f1']:.4f} "
                f"len_loss={length_loss:.4f} ",
                end=""
            )
        else:
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, use_features
            )
            print(
                f"  Epoch {epoch+1}: "
                f"train_f1={train_metrics['f1']:.4f} ",
                end=""
            )

        val_loss, val_metrics, _ = evaluate(
            model, val_loader, criterion, device, use_features
        )

        print(
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_model_state, best_val_f1, use_debiasing


def ensemble_predict(models, loader, device, use_features):
    """Ensemble prediction from multiple models."""
    all_preds = []
    for model in models:
        model.eval()
        preds = get_predictions(model, loader, device, use_features)
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)


def get_device():
    """Get compute device."""
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    device = get_device()
    print(f"Device: {device}")

    train_data = load_cached_data("train")
    print(f"Train: {len(train_data)} samples")

    config.CACHE_DIR.mkdir(exist_ok=True)
    
    # Choose embedding method based on debiasing config
    if config.USE_LENGTH_DEBIASING:
        print("\n[Debiasing] Using random truncation for embeddings")
        cache_path = config.CACHE_DIR / "train_transformer_truncated.pkl"
        train_embeddings, train_lengths = compute_truncated_embeddings(
            train_data, 
            cache_path, 
            config.TRANSFORMER_MODEL,
            config.TRUNCATION_RANGE,
        )
    else:
        print("\n[Standard] Using full-length embeddings")
        cache_path = config.CACHE_DIR / "train_transformer.pkl"
        train_embeddings = compute_embeddings(
            train_data, cache_path, config.TRANSFORMER_MODEL
        )
        train_lengths = None
    
    print(f"Embedding dim: {train_embeddings.shape[1]}")

    train_features = extract_features(train_data, config)
    use_features = train_features is not None

    if use_features:
        print(f"Extra features: {train_features.shape[1]} dims")
    else:
        print("Extra features: disabled")
    
    if config.USE_LENGTH_DEBIASING and config.USE_GRADIENT_REVERSAL:
        print(f"Gradient reversal lambda: {config.GRADIENT_REVERSAL_LAMBDA}")

    train_labels = np.array([item["label"] for item in train_data])

    num_folds = config.NUM_FOLDS
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.RANDOM_SEED)

    fold_models = []
    fold_scores = []

    print(f"\n{num_folds}-Fold Cross-Validation")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_embeddings, train_labels)):
        print(f"\nFold {fold_idx + 1}/{num_folds} (train={len(train_idx)}, val={len(val_idx)})")

        best_state, best_f1, used_debiasing = train_fold(
            train_idx, val_idx, train_embeddings, train_labels, train_features, device, train_lengths
        )

        fold_scores.append(best_f1)

        # Reconstruct model for saving (use standard classifier for inference compatibility)
        if used_debiasing:
            model = DebiasedTransformerClassifier(
                input_dim=train_embeddings.shape[1],
                hidden_dim=256,
                dropout=config.DROPOUT,
                num_extra_features=config.NUM_EXTRA_FEATURES_T,
                gradient_reversal_lambda=config.GRADIENT_REVERSAL_LAMBDA,
            ).to(device)
        else:
            model = TransformerClassifier(
                input_dim=train_embeddings.shape[1],
                hidden_dim=256,
                dropout=config.DROPOUT,
                num_extra_features=config.NUM_EXTRA_FEATURES_T,
            ).to(device)
        model.load_state_dict(best_state)
        fold_models.append(model)

        torch.save({
            "model_state_dict": best_state,
            "fold": fold_idx,
            "val_f1": best_f1,
            "input_dim": train_embeddings.shape[1],
            "num_extra_features": config.NUM_EXTRA_FEATURES_T,
            "debiased": used_debiasing,
        }, config.CACHE_DIR / f"transformer_model_fold_{fold_idx}.pt")

        print(f"  Best val F1: {best_f1:.4f}")

    print(f"\nCV Results: F1={np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

    # Select top-k models for ensemble based on validation scores
    top_k = min(config.ENSEMBLE_TOP_K, num_folds)
    top_indices = np.argsort(fold_scores)[-top_k:]

    print(f"\nEnsemble: top {top_k} folds {[i+1 for i in top_indices]}")

    torch.save({
        "num_folds": num_folds,
        "fold_scores": fold_scores,
        "top_k": top_k,
        "top_indices": top_indices.tolist(),
        "input_dim": train_embeddings.shape[1],
        "num_extra_features": config.NUM_EXTRA_FEATURES_T,
        "debiased": config.USE_LENGTH_DEBIASING,
    }, config.CACHE_DIR / "transformer_ensemble_info.pt")

    print("Done. Run evaluate.py for test results.")


if __name__ == "__main__":
    main()

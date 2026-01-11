import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from . import config
from preprocess import load_cached_data
from .utils import load_glove, build_vocab, create_embedding_matrix, calculate_metrics
from .dataset import HyperpartisanDataset
from .model import HyperpartisanCNN


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        extra_features = batch["extra_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, extra_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_preds, all_labels)
    return total_loss / len(loader), metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, extra_features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_preds, all_labels)
    return total_loss / len(loader), metrics, np.array(all_preds)


def get_predictions(model, loader, device):
    """Get raw predictions from model."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            extra_features = batch["extra_features"].to(device)
            outputs = model(input_ids, extra_features)
            all_preds.extend(outputs.cpu().numpy())
    return np.array(all_preds)


def train_fold(
    fold_idx,
    train_indices,
    val_indices,
    full_dataset,
    embedding_matrix,
    vocab,
    device,
    num_folds,
):
    """Train a single fold and return the best model state and validation metrics."""

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = HyperpartisanCNN(
        len(vocab), embedding_matrix, num_extra_features=config.NUM_EXTRA_FEATURES
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device)

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


def ensemble_predict(models, loader, device):
    """Average predictions from multiple models."""
    all_preds = []
    for model in models:
        model.eval()
        preds = get_predictions(model, loader, device)
        all_preds.append(preds)

    # Average predictions across all models
    ensemble_preds = np.mean(all_preds, axis=0)
    return ensemble_preds


def get_device():
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

    vocab = build_vocab(train_data, config.MIN_WORD_FREQ, config.VOCAB_SIZE)
    print(f"Vocab: {len(vocab)} words")

    word2vec = load_glove()
    embedding_matrix = create_embedding_matrix(vocab, word2vec, config.EMBEDDING_DIM)

    full_train_dataset = HyperpartisanDataset(train_data, vocab)
    labels = np.array([item["label"] for item in train_data])

    config.CACHE_DIR.mkdir(exist_ok=True)

    num_folds = config.NUM_FOLDS
    skf = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=config.RANDOM_SEED
    )

    fold_models = []
    fold_scores = []

    print(f"\n{num_folds}-Fold Cross Validation")

    for fold_idx, (train_indices, val_indices) in enumerate(
        skf.split(np.zeros(len(labels)), labels)
    ):
        print(
            f"\nFold {fold_idx + 1}/{num_folds} (train={len(train_indices)}, val={len(val_indices)})"
        )

        best_state, best_f1 = train_fold(
            fold_idx,
            train_indices,
            val_indices,
            full_train_dataset,
            embedding_matrix,
            vocab,
            device,
            num_folds,
        )

        fold_scores.append(best_f1)

        # Create model with best state for this fold
        model = HyperpartisanCNN(
            len(vocab), embedding_matrix, num_extra_features=config.NUM_EXTRA_FEATURES
        ).to(device)
        model.load_state_dict(best_state)
        fold_models.append(model)

        # Save individual fold model
        torch.save(
            {
                "model_state_dict": best_state,
                "vocab": vocab,
                "fold": fold_idx,
                "val_f1": best_f1,
                "num_extra_features": config.NUM_EXTRA_FEATURES,
            },
            config.CACHE_DIR / f"model_fold_{fold_idx}.pt",
        )

        print(f"  Best val F1: {best_f1:.4f}")

    print(
        f"\nCV Results: mean F1={np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})"
    )

    # Select top-k models for ensemble based on validation scores
    top_k = min(config.ENSEMBLE_TOP_K, num_folds)
    top_indices = np.argsort(fold_scores)[-top_k:]

    print(f"\nEnsemble: top {top_k} folds {[i+1 for i in top_indices]}")

    torch.save(
        {
            "vocab": vocab,
            "num_folds": num_folds,
            "fold_scores": fold_scores,
            "top_k": top_k,
            "top_indices": top_indices.tolist(),
            "num_extra_features": config.NUM_EXTRA_FEATURES,
        },
        config.CACHE_DIR / "ensemble_info.pt",
    )

    print("Done. Run evaluate.py for test results.")


if __name__ == "__main__":
    main()

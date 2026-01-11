"""Evaluate transformer ensemble on test sets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

import transformer.config as config
from preprocess import load_cached_data
from cnn.utils import calculate_metrics
from transformer.model import TransformerClassifier
from transformer.dataset import EmbeddingDataset
from transformer.utils import compute_embeddings, extract_features


def load_ensemble(device):
    ensemble_path = config.CACHE_DIR / "transformer_ensemble_info.pt"
    if not ensemble_path.exists():
        return None, None

    info = torch.load(ensemble_path, map_location=device, weights_only=False)
    top_indices = info["top_indices"]
    input_dim = info["input_dim"]
    num_extra_features = info.get("num_extra_features", 0)

    models = []
    for idx in top_indices:
        fold_path = config.CACHE_DIR / f"transformer_model_fold_{idx}.pt"
        if fold_path.exists():
            checkpoint = torch.load(fold_path, map_location=device, weights_only=False)
            model = TransformerClassifier(
                input_dim=input_dim,
                hidden_dim=256,
                dropout=config.DROPOUT,
                num_extra_features=num_extra_features,
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)

    return models, info


def get_predictions(model, loader, device, use_features):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            features = batch["features"].to(device) if use_features else None
            outputs = model(embeddings, features)
            all_preds.extend(outputs.cpu().numpy())
    return np.array(all_preds)


def ensemble_predict(models, loader, device, use_features):
    all_preds = []
    for model in models:
        preds = get_predictions(model, loader, device, use_features)
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)


def evaluate_dataset(models, embeddings, labels, features, device):
    use_features = features is not None
    dataset = EmbeddingDataset(embeddings, labels, features)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    preds = ensemble_predict(models, loader, device, use_features)
    metrics = calculate_metrics(preds, labels)

    preds_binary = (preds > 0.5).astype(int)
    return {
        "metrics": metrics,
        "gt_pos": int(labels.sum()),
        "gt_neg": len(labels) - int(labels.sum()),
        "pred_pos": int(preds_binary.sum()),
        "pred_neg": len(preds_binary) - int(preds_binary.sum()),
    }


def print_results(name, results, n_samples):
    m = results["metrics"]
    print(f"\n{name} (n={n_samples}):")
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  Distribution: {results['gt_pos']}/{results['gt_neg']} (actual), "
          f"{results['pred_pos']}/{results['pred_neg']} (predicted)")


def get_device():
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Device: {device}")

    models, info = load_ensemble(device)
    if models is None or len(models) == 0:
        print("No trained models found. Run: python -m transformer.train")
        return

    num_extra_features = info.get("num_extra_features", 0)
    use_features = num_extra_features > 0

    print(f"Ensemble: {len(models)} models (folds {[i+1 for i in info['top_indices']]})")
    print(f"CV scores: {[f'{s:.4f}' for s in info['fold_scores']]}")
    if use_features:
        print(f"Extra features: {num_extra_features} dims")

    # By-article test set
    try:
        print("\n" + "=" * 50)
        data = load_cached_data("test_byarticle")
        embeddings = compute_embeddings(
            data, config.CACHE_DIR / "test_transformer.pkl", config.TRANSFORMER_MODEL
        )
        labels = np.array([item["label"] for item in data])
        features = extract_features(data, config) if use_features else None

        results = evaluate_dataset(models, embeddings, labels, features, device)
        print_results("By-Article Test Set", results, len(data))
    except FileNotFoundError:
        print("\nBy-article test set not found.")

    # By-publisher test set
    try:
        print("\n" + "=" * 50)
        data = load_cached_data("test_bypublisher")
        embeddings = compute_embeddings(
            data, config.CACHE_DIR / "test_bypublisher_transformer.pkl", config.TRANSFORMER_MODEL
        )
        labels = np.array([item["label"] for item in data])
        features = extract_features(data, config) if use_features else None

        results = evaluate_dataset(models, embeddings, labels, features, device)
        print_results("By-Publisher Test Set", results, len(data))
    except FileNotFoundError:
        pass

    print("\n" + "=" * 50)
    print("Reference (SemEval-2019):")
    print("  By-Article:   Bertha von Suttner  Acc=0.822  F1=0.809")
    print("  By-Publisher: Tintin              Acc=0.706  F1=0.683")


if __name__ == "__main__":
    main()

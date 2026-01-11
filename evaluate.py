"""
Unified evaluation script for all models.

Evaluates CNN, Transformer, and SVM baselines on test sets and displays
results alongside reference values from SemEval-2019 Task 4.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

# Project imports
from preprocess import load_cached_data
from cnn.utils import (
    load_glove,
    build_vocab,
    create_embedding_matrix,
    calculate_metrics,
)
from cnn.dataset import HyperpartisanDataset
from cnn.model import HyperpartisanCNN
import cnn.config as cnn_config

from transformer.model import TransformerClassifier
from transformer.dataset import EmbeddingDataset
from transformer.utils import compute_embeddings, extract_features
import transformer.config as transformer_config

from svm.utils import compute_document_vectors
import svm.config as svm_config
from device import get_device

# BERT-MLP imports
from bert_mlp.model import BertMLP
from bert_mlp.dataset import BertEmbeddingDataset
from bert_mlp.utils import calculate_metrics as bert_calculate_metrics
from bert_mlp.utils import extract_features as bert_extract_features
import bert_mlp.config as bert_mlp_config


# Empty references (No reference paper for Kaggle dataset)
PAPER_RESULTS = {"by_article": [], "by_publisher": []}


def train_cnn_if_missing():
    """Train CNN if no trained model exists."""
    if not (cnn_config.CACHE_DIR / "ensemble_info.pt").exists():
        print("\n[Auto-train] CNN not found, training...")
        from cnn.train import main as train_cnn

        train_cnn()
        return True
    return False


def train_transformer_if_missing():
    """Train Transformer if no trained model exists."""
    if not (transformer_config.CACHE_DIR / "transformer_ensemble_info.pt").exists():
        print("\n[Auto-train] Transformer not found, training...")
        from transformer.train import main as train_transformer

        train_transformer()
        return True
    return False


def train_svm_if_missing():
    """Train SVM if no trained model exists."""
    if not (svm_config.CACHE_DIR / "svm_model.pkl").exists():
        print("\n[Auto-train] SVM not found, training...")
        from svm.train import main as train_svm

        train_svm()
        return True
    return False


def train_bert_mlp_if_missing():
    """Train BERT-MLP if no trained model exists."""
    if not (bert_mlp_config.CACHE_DIR / "bert_mlp_ensemble_info.pt").exists():
        print("\n[Auto-train] BERT-MLP not found, training...")
        from bert_mlp.train import main as train_bert_mlp

        train_bert_mlp()
        return True
    return False


def load_cnn_ensemble(device):
    """Load CNN ensemble models."""
    info_path = cnn_config.CACHE_DIR / "ensemble_info.pt"
    if not info_path.exists():
        return None, None

    info = torch.load(info_path, map_location=device, weights_only=False)
    vocab = info["vocab"]
    top_indices = info["top_indices"]
    num_extra_features = info.get("num_extra_features", 0)

    word2vec = load_glove()
    embedding_matrix = create_embedding_matrix(
        vocab, word2vec, cnn_config.EMBEDDING_DIM
    )

    models = []
    for idx in top_indices:
        path = cnn_config.CACHE_DIR / f"model_fold_{idx}.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = HyperpartisanCNN(
                len(vocab), embedding_matrix, num_extra_features=num_extra_features
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)

    return models, info


def load_transformer_ensemble(device):
    """Load transformer ensemble models."""
    info_path = transformer_config.CACHE_DIR / "transformer_ensemble_info.pt"
    if not info_path.exists():
        return None, None

    info = torch.load(info_path, map_location=device, weights_only=False)
    top_indices = info["top_indices"]
    input_dim = info["input_dim"]
    num_extra_features = info.get("num_extra_features", 0)

    models = []
    for idx in top_indices:
        path = transformer_config.CACHE_DIR / f"transformer_model_fold_{idx}.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = TransformerClassifier(
                input_dim=input_dim,
                hidden_dim=256,
                dropout=transformer_config.DROPOUT,
                num_extra_features=num_extra_features,
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)

    return models, info


def load_svm_model():
    """Load trained SVM model."""
    model_path = svm_config.CACHE_DIR / "svm_model.pkl"
    if not model_path.exists():
        return None, None

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    return data["model"], data["scaler"]


def load_bert_mlp_ensemble(device):
    """Load BERT-MLP ensemble models."""
    info_path = bert_mlp_config.CACHE_DIR / "bert_mlp_ensemble_info.pt"
    if not info_path.exists():
        return None, None

    info = torch.load(info_path, map_location=device, weights_only=False)
    top_indices = info["top_indices"]
    input_dim = info["input_dim"]
    num_extra_features = info.get("num_extra_features", 0)

    models = []
    for idx in top_indices:
        path = bert_mlp_config.CACHE_DIR / f"bert_mlp_fold_{idx}.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = BertMLP(
                input_dim=input_dim,
                hidden_dim=bert_mlp_config.HIDDEN_DIM,
                dropout=bert_mlp_config.DROPOUT,
                num_extra_features=num_extra_features,
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)

    return models, info


def evaluate_cnn(models, info, data, device):
    """Evaluate CNN ensemble on dataset."""
    vocab = info["vocab"]
    num_extra_features = info.get("num_extra_features", 0)

    dataset = HyperpartisanDataset(data, vocab)
    loader = DataLoader(dataset, batch_size=cnn_config.BATCH_SIZE, shuffle=False)
    labels = np.array([item["label"] for item in data])

    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                extra_features = batch["extra_features"].to(device)
                outputs = model(input_ids, extra_features)
                preds.extend(outputs.cpu().numpy())
        all_preds.append(preds)

    ensemble_preds = np.mean(all_preds, axis=0)
    return calculate_metrics(ensemble_preds, labels)


def evaluate_transformer(models, info, data, device, cache_name):
    """Evaluate transformer ensemble on dataset."""
    num_extra_features = info.get("num_extra_features", 0)
    use_features = num_extra_features > 0

    embeddings = compute_embeddings(
        data,
        transformer_config.CACHE_DIR / cache_name,
        transformer_config.TRANSFORMER_MODEL,
    )
    labels = np.array([item["label"] for item in data])
    features = extract_features(data, transformer_config) if use_features else None

    dataset = EmbeddingDataset(embeddings, labels, features)
    loader = DataLoader(
        dataset, batch_size=transformer_config.BATCH_SIZE, shuffle=False
    )

    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                emb = batch["embedding"].to(device)
                feat = batch["features"].to(device) if use_features else None
                outputs = model(emb, feat)
                preds.extend(outputs.cpu().numpy())
        all_preds.append(preds)

    ensemble_preds = np.mean(all_preds, axis=0)
    return calculate_metrics(ensemble_preds, labels)


def evaluate_svm(model, scaler, data, word2vec):
    """Evaluate SVM on dataset."""
    X = compute_document_vectors(data, word2vec, svm_config.EMBEDDING_DIM)
    y = np.array([item["label"] for item in data])

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    return {
        "accuracy": (y == y_pred).mean(),
        "precision": (
            (y_pred[y_pred == 1] == y[y_pred == 1]).mean() if y_pred.sum() > 0 else 0
        ),
        "recall": (y_pred[y == 1] == 1).mean() if y.sum() > 0 else 0,
        "f1": 0,  # Compute below
    }


def evaluate_svm_proper(model, scaler, data, word2vec):
    """Evaluate SVM with proper sklearn metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X = compute_document_vectors(data, word2vec, svm_config.EMBEDDING_DIM)
    y = np.array([item["label"] for item in data])

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }


def evaluate_bert_mlp(models, info, data, device, cache_name):
    """Evaluate BERT-MLP ensemble on dataset."""
    num_extra_features = info.get("num_extra_features", 0)
    use_features = num_extra_features > 0

    embeddings = compute_embeddings(
        data,
        bert_mlp_config.CACHE_DIR / cache_name,
        bert_mlp_config.TRANSFORMER_MODEL,
    )
    labels = np.array([item["label"] for item in data])
    features = bert_extract_features(data, bert_mlp_config) if use_features else None

    dataset = BertEmbeddingDataset(embeddings, labels, features)
    loader = DataLoader(dataset, batch_size=bert_mlp_config.BATCH_SIZE, shuffle=False)

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
    return bert_calculate_metrics(ensemble_preds, labels)


def print_table(title, results):
    """Print results table."""
    print(f"\n{title}")
    print("-" * 70)
    print(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 70)

    for name, metrics in results:
        if metrics is None:
            print(f"{name:<25} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        else:
            print(
                f"{name:<25} "
                f"{metrics['accuracy']:>8.3f} "
                f"{metrics['precision']:>8.3f} "
                f"{metrics['recall']:>8.3f} "
                f"{metrics['f1']:>8.3f}"
            )

    print("-" * 70)


def print_paper_reference(dataset_key):
    """Print reference values from paper."""
    print("\nSemEval-2019 Reference:")
    print("-" * 70)
    print(f"{'Team':<25} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 70)

    print("-" * 70)

    if not PAPER_RESULTS[dataset_key]:
        print("No reference results available for this dataset.")
        print("-" * 70)
        return

    for name, acc, prec, rec, f1 in PAPER_RESULTS[dataset_key]:
        print(f"{name:<25} {acc:>8.3f} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f}")

    print("-" * 70)


def generate_latex_table(title, our_results, paper_results, sort_by="f1"):
    """
    Generate a LaTeX table with smart ranking and bold best values.

    Args:
        title: Table title/caption
        our_results: List of (name, metrics_dict) tuples for our models
        paper_results: List of (name, acc, prec, rec, f1) tuples from paper
        sort_by: Metric to sort by ('accuracy', 'precision', 'recall', 'f1')

    Returns:
        LaTeX table string
    """
    # Combine all results into uniform format: (name, acc, prec, rec, f1)
    all_results = []

    # Add our results
    for name, metrics in our_results:
        if metrics is not None:
            all_results.append(
                (
                    name,
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                )
            )

    # Add paper results
    for name, acc, prec, rec, f1 in paper_results:
        all_results.append((name, acc, prec, rec, f1))

    if not all_results:
        return ""

    # Sort by specified metric (descending - best first)
    metric_idx = {"accuracy": 1, "precision": 2, "recall": 3, "f1": 4}
    sort_idx = metric_idx.get(sort_by, 4)
    all_results.sort(key=lambda x: x[sort_idx], reverse=True)

    # Find best values for each metric
    best_acc = max(r[1] for r in all_results)
    best_prec = max(r[2] for r in all_results)
    best_rec = max(r[3] for r in all_results)
    best_f1 = max(r[4] for r in all_results)

    def format_val(val, best_val, precision=3):
        """Format value with bold if it's the best."""
        formatted = f"{val:.{precision}f}"
        if abs(val - best_val) < 1e-6:  # Float comparison tolerance
            return f"\\textbf{{{formatted}}}"
        return formatted

    # Build LaTeX table
    lines = [
        f"% {title}",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{title}}}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & Accuracy & Precision & Recall & F1 \\\\",
        "\\midrule",
    ]

    for name, acc, prec, rec, f1 in all_results:
        # Escape underscores for LaTeX
        latex_name = name.replace("_", "\\_")

        acc_str = format_val(acc, best_acc)
        prec_str = format_val(prec, best_prec)
        rec_str = format_val(rec, best_rec)
        f1_str = format_val(f1, best_f1)

        lines.append(
            f"{latex_name} & {acc_str} & {prec_str} & {rec_str} & {f1_str} \\\\"
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def print_latex_tables(results_by_article, results_by_publisher):
    """Print LaTeX tables for all results."""
    print("\n" + "=" * 70)
    print("LATEX OUTPUT")
    print("=" * 70)

    if results_by_article:
        latex = generate_latex_table(
            "Performance on By-Article Test Set",
            results_by_article,
            PAPER_RESULTS["by_article"],
            sort_by="f1",
        )
        print("\n" + latex)

    if results_by_publisher:
        latex = generate_latex_table(
            "Performance on By-Publisher Test Set",
            results_by_publisher,
            PAPER_RESULTS["by_publisher"],
            sort_by="f1",
        )
        print("\n" + latex)


def main():
    device = get_device(transformer_config.DEVICE)
    print(f"Device: {device}")

    # Auto-train missing models
    print("\nChecking for trained models...")
    train_cnn_if_missing()
    train_transformer_if_missing()
    train_svm_if_missing()
    train_bert_mlp_if_missing()

    # Load all models
    print("\nLoading models...")
    cnn_models, cnn_info = load_cnn_ensemble(device)
    transformer_models, transformer_info = load_transformer_ensemble(device)
    svm_model, svm_scaler = load_svm_model()
    bert_mlp_models, bert_mlp_info = load_bert_mlp_ensemble(device)

    print(f"  CNN: {len(cnn_models) if cnn_models else 0} models")
    print(
        f"  Transformer: {len(transformer_models) if transformer_models else 0} models"
    )
    print(f"  SVM: {'loaded' if svm_model else 'not found'}")
    print(f"  BERT-MLP: {len(bert_mlp_models) if bert_mlp_models else 0} models")

    # Load word2vec for SVM
    word2vec = None
    if svm_model:
        word2vec = load_glove()

    # Store results for LaTeX output
    results_by_article = None
    results_by_publisher = None

    # By-article test set
    try:
        print("\nEvaluating on Test Set...")
        data = load_cached_data("test")

        results = []

        if cnn_models:
            metrics = evaluate_cnn(cnn_models, cnn_info, data, device)
            results.append(("CNN (Ours)", metrics))
        else:
            results.append(("CNN (Ours)", None))

        if transformer_models:
            metrics = evaluate_transformer(
                transformer_models,
                transformer_info,
                data,
                device,
                "test_transformer.pkl",
            )
            results.append(("Transformer (Ours)", metrics))
        else:
            results.append(("Transformer (Ours)", None))

        if svm_model:
            metrics = evaluate_svm_proper(svm_model, svm_scaler, data, word2vec)
            results.append(("SVM (Ours)", metrics))
        else:
            results.append(("SVM (Ours)", None))

        if bert_mlp_models:
            metrics = evaluate_bert_mlp(
                bert_mlp_models,
                bert_mlp_info,
                data,
                device,
                "test_byarticle_bert_mlp.pkl",
            )
            results.append(("BERT-MLP (Ours)", metrics))
        else:
            results.append(("BERT-MLP (Ours)", None))

        results_by_article = results
        print_table("Test Set Results", results)
        print_paper_reference("by_article")

    except FileNotFoundError:
        print("By-article test set not found.")

    # Print LaTeX tables at the end
    print_latex_tables(results_by_article, results_by_publisher)


if __name__ == "__main__":
    main()

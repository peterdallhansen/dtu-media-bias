"""Evaluate trained SVM model on test sets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np

import svm.config as config
from preprocess import load_cached_data
from cnn.utils import load_glove
from svm.utils import compute_document_vectors, calculate_metrics


def load_model():
    """Load trained SVM model and scaler from disk."""
    model_path = config.CACHE_DIR / "svm_model.pkl"
    if not model_path.exists():
        return None, None, None

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    return data["model"], data["scaler"], data.get("cv_metrics")


def evaluate_dataset(model, scaler, data, word2vec):
    """Evaluate model on a dataset and return metrics."""
    X = compute_document_vectors(data, word2vec, config.EMBEDDING_DIM)
    y = np.array([item["label"] for item in data])

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    metrics = calculate_metrics(y, y_pred)

    return {
        "metrics": metrics,
        "gt_pos": int(y.sum()),
        "gt_neg": len(y) - int(y.sum()),
        "pred_pos": int(y_pred.sum()),
        "pred_neg": len(y_pred) - int(y_pred.sum()),
    }


def print_results(name, results, n_samples):
    """Print evaluation results in a formatted table."""
    m = results["metrics"]
    print(f"\n{name} (n={n_samples}):")
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(
        f"  Distribution: {results['gt_pos']}/{results['gt_neg']} actual, "
        f"{results['pred_pos']}/{results['pred_neg']} predicted"
    )


def main():
    print("SVM Baseline Evaluation")
    print("-" * 40)

    # Load model
    model, scaler, cv_metrics = load_model()
    if model is None:
        print("No trained model found. Run: python -m svm.train")
        return

    if cv_metrics:
        print(f"CV metrics: Acc={cv_metrics['accuracy']:.4f}, F1={cv_metrics['f1']:.4f}")

    word2vec = load_glove()

    # By-article test set
    try:
        data = load_cached_data("test_byarticle")
        results = evaluate_dataset(model, scaler, data, word2vec)
        print_results("By-Article Test Set", results, len(data))
    except FileNotFoundError:
        print("By-article test set not available.")

    # By-publisher test set
    try:
        data = load_cached_data("test_bypublisher")
        results = evaluate_dataset(model, scaler, data, word2vec)
        print_results("By-Publisher Test Set", results, len(data))
    except FileNotFoundError:
        pass

    # Reference values from the original paper
    print("\nSemEval-2019 reference:")
    print("  Tom Jumbo Grumbo: Acc=0.806, F1=0.790")


if __name__ == "__main__":
    main()

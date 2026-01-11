"""
SVM baseline training for hyperpartisan news detection.

Replicates the Tom Jumbo Grumbo approach from SemEval-2019 Task 4:
- Document representation via averaged GloVe embeddings
- SVM classifier with RBF kernel
- 10-fold stratified cross-validation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import svm.config as config
from preprocess import load_cached_data
from cnn.utils import load_glove
from svm.utils import compute_document_vectors, calculate_metrics


def main():
    np.random.seed(config.RANDOM_SEED)

    print("SVM Baseline Training")
    print("-" * 40)

    # Load data
    train_data = load_cached_data("train")
    print(f"Training samples: {len(train_data)}")

    # Load embeddings and compute document vectors
    word2vec = load_glove()
    X_train = compute_document_vectors(train_data, word2vec, config.EMBEDDING_DIM)
    y_train = np.array([item["label"] for item in train_data])

    # Feature normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Cross-validation
    print(f"\n{config.NUM_FOLDS}-fold cross-validation:")
    
    svm = SVC(
        kernel=config.SVM_KERNEL,
        C=config.SVM_C,
        gamma=config.SVM_GAMMA,
        random_state=config.RANDOM_SEED,
    )

    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )
    cv_preds = cross_val_predict(svm, X_train_scaled, y_train, cv=skf)

    cv_metrics = calculate_metrics(y_train, cv_preds)
    print(f"  Accuracy:  {cv_metrics['accuracy']:.4f}")
    print(f"  Precision: {cv_metrics['precision']:.4f}")
    print(f"  Recall:    {cv_metrics['recall']:.4f}")
    print(f"  F1:        {cv_metrics['f1']:.4f}")

    # Train final model
    print("\nTraining final model...")
    svm.fit(X_train_scaled, y_train)

    # Save model
    config.CACHE_DIR.mkdir(exist_ok=True)
    model_path = config.CACHE_DIR / "svm_model.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": svm,
                "scaler": scaler,
                "cv_metrics": cv_metrics,
            },
            f,
        )
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()

"""
Multi-Model Ensemble for Hyperpartisan News Detection.

Combines predictions from CNN, Transformer, and SVM models to achieve
higher accuracy than any individual model.

Usage:
    python ensemble_evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast
from tqdm import tqdm

from preprocess import load_cached_data
from cnn.utils import calculate_metrics, load_glove
from cnn.dataset import HyperpartisanDataset
from cnn.model import HyperpartisanCNN
from cnn import config as cnn_config
from svm.utils import compute_document_vectors
from svm import config as svm_config
from transformer import config as transformer_config
from transformer.model import TransformerClassifier, DebiasedTransformerClassifier
from transformer.utils import compute_embeddings
from transformer.dataset import EmbeddingDataset


def get_device():
    """Get compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# CNN Predictions
# =============================================================================

def load_cnn_ensemble(device):
    """Load CNN ensemble models."""
    ensemble_path = cnn_config.CACHE_DIR / "ensemble_info.pt"
    if not ensemble_path.exists():
        return None, None
    
    info = torch.load(ensemble_path, map_location=device, weights_only=False)
    vocab = info['vocab']
    top_indices = info['top_indices']
    num_extra_features = info.get('num_extra_features', 0)
    
    models = []
    for idx in top_indices:
        fold_path = cnn_config.CACHE_DIR / f"model_fold_{idx}.pt"
        if fold_path.exists():
            checkpoint = torch.load(fold_path, map_location=device, weights_only=False)
            model = HyperpartisanCNN(len(vocab), num_extra_features=num_extra_features).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
    
    return models, vocab


def get_cnn_predictions(data, device):
    """Get CNN ensemble predictions."""
    models, vocab = load_cnn_ensemble(device)
    if models is None:
        print("  CNN: not available")
        return None
    
    dataset = HyperpartisanDataset(data, vocab)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                extra_features = batch['extra_features'].to(device)
                outputs = model(input_ids, extra_features)
                preds.extend(outputs.cpu().numpy())
        all_preds.append(preds)
    
    ensemble_preds = np.mean(all_preds, axis=0)
    print(f"  CNN: {len(models)} models")
    return ensemble_preds


# =============================================================================
# Transformer Predictions
# =============================================================================

def load_transformer_ensemble(device):
    """Load transformer ensemble models."""
    ensemble_path = transformer_config.CACHE_DIR / "transformer_ensemble_info.pt"
    if not ensemble_path.exists():
        return None, None
    
    info = torch.load(ensemble_path, map_location=device, weights_only=False)
    top_indices = info['top_indices']
    input_dim = info['input_dim']
    num_extra_features = info.get('num_extra_features', 0)
    is_debiased = info.get('debiased', False)
    
    models = []
    for idx in top_indices:
        fold_path = transformer_config.CACHE_DIR / f"transformer_model_fold_{idx}.pt"
        if fold_path.exists():
            checkpoint = torch.load(fold_path, map_location=device, weights_only=False)
            checkpoint_debiased = checkpoint.get('debiased', False)
            
            if checkpoint_debiased:
                model = DebiasedTransformerClassifier(
                    input_dim=input_dim,
                    hidden_dim=256,
                    dropout=0.5,
                    num_extra_features=num_extra_features,
                    gradient_reversal_lambda=transformer_config.GRADIENT_REVERSAL_LAMBDA,
                ).to(device)
            else:
                model = TransformerClassifier(
                    input_dim=input_dim,
                    hidden_dim=256,
                    dropout=0.5,
                    num_extra_features=num_extra_features,
                ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
    
    return models, info


def get_transformer_predictions(data, device):
    """Get transformer ensemble predictions."""
    models, info = load_transformer_ensemble(device)
    if models is None:
        print("  Transformer: not available")
        return None
    
    # Compute embeddings
    embeddings = compute_embeddings(
        data,
        transformer_config.CACHE_DIR / "test_transformer.pkl",
        transformer_config.TRANSFORMER_MODEL
    )
    labels = np.array([item['label'] for item in data])
    
    dataset = EmbeddingDataset(embeddings, labels, None)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                emb = batch['embedding'].to(device)
                outputs = model(emb, None)
                preds.extend(outputs.cpu().numpy())
        all_preds.append(preds)
    
    ensemble_preds = np.mean(all_preds, axis=0)
    print(f"  Transformer: {len(models)} models")
    return ensemble_preds


# =============================================================================
# SVM Predictions
# =============================================================================

def load_svm_model():
    """Load SVM model."""
    model_path = svm_config.CACHE_DIR / "svm_model.pkl"
    if not model_path.exists():
        return None, None
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['model'], data['scaler']


def get_svm_predictions(data):
    """Get SVM predictions as probabilities."""
    model, scaler = load_svm_model()
    if model is None:
        print("  SVM: not available")
        return None
    
    word2vec = load_glove()
    X = compute_document_vectors(data, word2vec, svm_config.EMBEDDING_DIM)
    X_scaled = scaler.transform(X)
    
    # Get probability predictions if available, otherwise use decision function
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        # Convert decision function to pseudo-probability with sigmoid
        scores = model.decision_function(X_scaled)
        probs = 1 / (1 + np.exp(-scores))
    
    print(f"  SVM: 1 model")
    return probs


# =============================================================================
# Ensemble
# =============================================================================

def ensemble_predictions(cnn_preds, transformer_preds, svm_preds, weights=None):
    """
    Combine predictions from multiple models.
    
    Args:
        cnn_preds: CNN predictions (or None)
        transformer_preds: Transformer predictions (or None)
        svm_preds: SVM predictions (or None)
        weights: Optional weights for each model [cnn, transformer, svm]
    
    Returns:
        Ensemble predictions
    """
    available = []
    preds_list = []
    
    if cnn_preds is not None:
        available.append("CNN")
        preds_list.append(cnn_preds)
    if transformer_preds is not None:
        available.append("Transformer")
        preds_list.append(transformer_preds)
    if svm_preds is not None:
        available.append("SVM")
        preds_list.append(svm_preds)
    
    if len(preds_list) == 0:
        raise ValueError("No predictions available")
    
    if weights is None:
        # Simple average
        ensemble = np.mean(preds_list, axis=0)
    else:
        # Weighted average
        w = [weights[i] for i, name in enumerate(["CNN", "Transformer", "SVM"]) if name in available]
        w = np.array(w) / sum(w)  # Normalize
        ensemble = np.average(preds_list, axis=0, weights=w)
    
    return ensemble, available


def evaluate_ensemble(data, device, weights=None):
    """Evaluate multi-model ensemble on a dataset."""
    print("\nLoading models...")
    
    cnn_preds = get_cnn_predictions(data, device)
    transformer_preds = get_transformer_predictions(data, device)
    svm_preds = get_svm_predictions(data)
    
    ensemble_preds, available = ensemble_predictions(
        cnn_preds, transformer_preds, svm_preds, weights
    )
    
    labels = np.array([item['label'] for item in data])
    metrics = calculate_metrics(ensemble_preds, labels)
    
    preds_binary = (ensemble_preds > 0.5).astype(int)
    
    return {
        'metrics': metrics,
        'models': available,
        'gt_pos': int(labels.sum()),
        'gt_neg': len(labels) - int(labels.sum()),
        'pred_pos': int(preds_binary.sum()),
        'pred_neg': len(preds_binary) - int(preds_binary.sum()),
        'individual': {
            'CNN': calculate_metrics(cnn_preds, labels) if cnn_preds is not None else None,
            'Transformer': calculate_metrics(transformer_preds, labels) if transformer_preds is not None else None,
            'SVM': calculate_metrics(svm_preds, labels) if svm_preds is not None else None,
        }
    }


def print_results(name, results, n_samples):
    """Print results with individual model comparison."""
    m = results['metrics']
    print(f"\n{'='*60}")
    print(f"{name} (n={n_samples})")
    print(f"{'='*60}")
    
    print(f"\nEnsemble ({' + '.join(results['models'])}):")
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    
    print(f"\nIndividual models:")
    for model_name in ['CNN', 'Transformer', 'SVM']:
        ind = results['individual'].get(model_name)
        if ind:
            print(f"  {model_name:12} Acc={ind['accuracy']:.4f}  F1={ind['f1']:.4f}")
    
    print(f"\nDistribution: {results['gt_pos']}/{results['gt_neg']} actual, "
          f"{results['pred_pos']}/{results['pred_neg']} predicted")


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print("MULTI-MODEL ENSEMBLE EVALUATION")
    print(f"{'='*60}")
    
    # Optional: set custom weights (CNN, Transformer, SVM)
    # Higher weight = more influence on final prediction
    weights = None  # Use equal weights
    # weights = [0.3, 0.5, 0.2]  # Example: favor Transformer
    
    # By-article test set
    try:
        data = load_cached_data("test_byarticle")
        results = evaluate_ensemble(data, device, weights)
        print_results("By-Article Test Set", results, len(data))
    except FileNotFoundError:
        print("By-article test set not found.")
    
    # By-publisher test set
    try:
        data = load_cached_data("test_bypublisher")
        results = evaluate_ensemble(data, device, weights)
        print_results("By-Publisher Test Set", results, len(data))
    except Exception as e:
        print(f"By-publisher: {e}")
    
    print(f"\n{'='*60}")
    print("Reference (SemEval-2019):")
    print("  By-Article:   Bertha von Suttner  Acc=0.822  F1=0.809")
    print("  By-Publisher: Tintin              Acc=0.706  F1=0.683")


if __name__ == "__main__":
    main()

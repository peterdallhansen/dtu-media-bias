import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from . import config
from preprocess import load_cached_data
from .dataset import HyperpartisanDataset
from .model import HyperpartisanCNN
from .utils import calculate_metrics


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint['vocab']
    num_extra_features = checkpoint.get('num_extra_features', 0)

    model = HyperpartisanCNN(len(vocab), num_extra_features=num_extra_features).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, vocab


def load_ensemble(device):
    """Load ensemble of fold models."""
    ensemble_path = config.CACHE_DIR / "ensemble_info.pt"
    if not ensemble_path.exists():
        return None, None

    info = torch.load(ensemble_path, map_location=device, weights_only=False)
    vocab = info['vocab']
    top_indices = info['top_indices']
    num_extra_features = info.get('num_extra_features', 0)

    models = []
    for idx in top_indices:
        fold_path = config.CACHE_DIR / f"model_fold_{idx}.pt"
        if fold_path.exists():
            checkpoint = torch.load(fold_path, map_location=device, weights_only=False)
            model = HyperpartisanCNN(len(vocab), num_extra_features=num_extra_features).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)

    return models, vocab


def get_predictions(model, loader, device):
    """Get raw predictions from model."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            extra_features = batch['extra_features'].to(device)
            outputs = model(input_ids, extra_features)
            all_preds.extend(outputs.cpu().numpy())
    return np.array(all_preds)


def ensemble_predict(models, loader, device):
    """Average predictions from multiple models."""
    all_preds = []
    for model in models:
        preds = get_predictions(model, loader, device)
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)


def evaluate_dataset(model, data, vocab, device, name):
    dataset = HyperpartisanDataset(data, vocab)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {name}", leave=False):
            input_ids = batch['input_ids'].to(device)
            extra_features = batch['extra_features'].to(device)
            outputs = model(input_ids, extra_features)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch['label'].numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    metrics = calculate_metrics(preds, labels)

    preds_binary = (preds > 0.5).astype(int)
    gt_hp = int(labels.sum())
    gt_nhp = len(labels) - gt_hp
    pred_hp = int(preds_binary.sum())
    pred_nhp = len(preds_binary) - pred_hp

    return metrics, gt_hp, gt_nhp, pred_hp, pred_nhp


def evaluate_ensemble_dataset(models, data, vocab, device, name):
    """Evaluate ensemble of models on a dataset."""
    dataset = HyperpartisanDataset(data, vocab)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    preds = ensemble_predict(models, loader, device)
    labels = np.array([item['label'] for item in data])
    metrics = calculate_metrics(preds, labels)

    preds_binary = (preds > 0.5).astype(int)
    gt_hp = int(labels.sum())
    gt_nhp = len(labels) - gt_hp
    pred_hp = int(preds_binary.sum())
    pred_nhp = len(preds_binary) - pred_hp

    return metrics, gt_hp, gt_nhp, pred_hp, pred_nhp


def print_metrics(name, metrics, n_samples):
    print(f"\n{name} (n={n_samples}):")
    print(f"  Acc={metrics['accuracy']:.3f}  P={metrics['precision']:.3f}  "
          f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")


def get_device():
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()

    # Try to load ensemble first
    ensemble_models, vocab = load_ensemble(device)
    use_ensemble = ensemble_models is not None and len(ensemble_models) > 0

    if use_ensemble:
        print(f"Ensemble: {len(ensemble_models)} models")
    else:
        checkpoint_path = config.CACHE_DIR / "best_model.pt"
        if not checkpoint_path.exists():
            print("No model found. Run train.py first.")
            return
        model, vocab = load_model(checkpoint_path, device)
        print(f"Single model, vocab={len(vocab)}")

    try:
        data = load_cached_data('test_byarticle')
        if use_ensemble:
            metrics, _, _, _, _ = evaluate_ensemble_dataset(
                ensemble_models, data, vocab, device, "by-article"
            )
        else:
            metrics, _, _, _, _ = evaluate_dataset(
                model, data, vocab, device, "by-article"
            )
        print_metrics("By-Article", metrics, len(data))
    except FileNotFoundError:
        print("\nBy-article test set not found.")

    try:
        data = load_cached_data('test_bypublisher')
        if use_ensemble:
            metrics, _, _, _, _ = evaluate_ensemble_dataset(
                ensemble_models, data, vocab, device, "by-publisher"
            )
        else:
            metrics, _, _, _, _ = evaluate_dataset(
                model, data, vocab, device, "by-publisher"
            )
        print_metrics("By-Publisher", metrics, len(data))
    except FileNotFoundError:
        pass

    print("\nReference (SemEval-2019 top teams):")
    print("  By-Article:   Bertha von Suttner Acc=0.822 F1=0.809")
    print("  By-Publisher: Tintin             Acc=0.706 F1=0.683")


if __name__ == "__main__":
    main()

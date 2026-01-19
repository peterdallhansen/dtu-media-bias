"""Evaluate fine-tuned transformer ensemble on test sets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

import transformer.config as config
from preprocess import load_cached_data
from cnn.utils import calculate_metrics
from transformer.finetune_model import FineTunedClassifier


class TextDataset(Dataset):
    """Dataset for evaluation with raw text."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", " ".join(item.get("tokens", [])))
        label = item["label"]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


def load_ensemble(device):
    """Load fine-tuned ensemble models."""
    ensemble_path = config.CACHE_DIR / "transformer_finetuned_ensemble_info.pt"
    if not ensemble_path.exists():
        return None, None

    info = torch.load(ensemble_path, map_location=device, weights_only=False)
    top_indices = info["top_indices"]
    fine_tune_layers = info.get("fine_tune_layers", 2)

    models = []
    for idx in top_indices:
        fold_path = config.CACHE_DIR / f"transformer_finetuned_fold_{idx}.pt"
        if fold_path.exists():
            checkpoint = torch.load(fold_path, map_location=device, weights_only=False)
            
            model = FineTunedClassifier(
                model_name=config.TRANSFORMER_MODEL,
                hidden_dim=256,
                dropout=0.0,  # No dropout during inference
                num_extra_features=0,
            ).to(device)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)

    return models, info


def get_predictions(model, loader, device):
    """Get predictions from a single model."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            all_preds.extend(outputs.cpu().numpy())
    return np.array(all_preds)


def ensemble_predict(models, loader, device):
    """Ensemble prediction from multiple models."""
    all_preds = []
    for model in models:
        preds = get_predictions(model, loader, device)
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)


def evaluate_dataset(models, data, tokenizer, device):
    """Evaluate ensemble on a dataset."""
    # Ensure text field exists
    for item in data:
        if "text" not in item:
            item["text"] = " ".join(item.get("tokens", []))
    
    dataset = TextDataset(data, tokenizer)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    labels = np.array([item["label"] for item in data])

    preds = ensemble_predict(models, loader, device)
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
    """Print evaluation results."""
    m = results["metrics"]
    print(f"\n{name} (n={n_samples}):")
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  Distribution: {results['gt_pos']}/{results['gt_neg']} (actual), "
          f"{results['pred_pos']}/{results['pred_neg']} (predicted)")


def get_device():
    """Get compute device."""
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
        print("No fine-tuned models found. Run: python -m transformer.finetune_train")
        return

    print(f"Ensemble: {len(models)} fine-tuned models")
    print(f"Folds: {[i+1 for i in info['top_indices']]}")
    print(f"CV scores: {[f'{s:.4f}' for s in info['fold_scores']]}")
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.TRANSFORMER_MODEL)

    # By-article test set
    try:
        print("\n" + "=" * 50)
        data = load_cached_data("test_byarticle")
        results = evaluate_dataset(models, data, tokenizer, device)
        print_results("By-Article Test Set (Fine-Tuned)", results, len(data))
    except FileNotFoundError:
        print("\nBy-article test set not found.")

    # By-publisher test set
    try:
        print("\n" + "=" * 50)
        data = load_cached_data("test_bypublisher")
        results = evaluate_dataset(models, data, tokenizer, device)
        print_results("By-Publisher Test Set (Fine-Tuned)", results, len(data))
    except FileNotFoundError:
        pass

    print("\n" + "=" * 50)
    print("Reference (SemEval-2019):")
    print("  By-Article:   Bertha von Suttner  Acc=0.822  F1=0.809")
    print("  By-Publisher: Tintin              Acc=0.706  F1=0.683")


if __name__ == "__main__":
    main()

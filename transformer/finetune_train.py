"""
End-to-end fine-tuning training for DistilBERT hyperpartisan classifier.

This script implements gradual unfreezing:
1. Phase 1 (FREEZE_EPOCHS): Train only the classifier head with BERT frozen
2. Phase 2 (FINE_TUNE_EPOCHS): Unfreeze top BERT layers with discriminative LR

This approach prevents catastrophic forgetting and overfitting on small datasets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

import transformer.config as config
from preprocess import load_cached_data
from cnn.utils import calculate_metrics
from transformer.finetune_model import FineTunedClassifier, LabelSmoothingBCELoss


class TextDataset(Dataset):
    """Dataset for end-to-end training with raw text."""
    
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


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), calculate_metrics(all_preds, all_labels)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), calculate_metrics(all_preds, all_labels)


def train_fold(train_data, val_data, tokenizer, device, fold_idx):
    """Train a single fold with gradual unfreezing."""
    
    train_dataset = TextDataset(train_data, tokenizer)
    val_dataset = TextDataset(val_data, tokenizer)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    # Create model
    model = FineTunedClassifier(
        model_name=config.TRANSFORMER_MODEL,
        hidden_dim=256,
        dropout=config.DROPOUT,
        num_extra_features=0,  # No extra features for fine-tuning
    ).to(device)
    
    # Loss with label smoothing
    criterion = LabelSmoothingBCELoss(smoothing=config.LABEL_SMOOTHING)
    
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    # =========================================================================
    # Phase 1: Train classifier only (BERT frozen)
    # =========================================================================
    print(f"\n  Phase 1: Training classifier head ({config.FREEZE_EPOCHS} epochs)")
    model.freeze_bert()
    
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=config.CLASSIFIER_LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    
    for epoch in range(config.FREEZE_EPOCHS):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(
            f"    Epoch {epoch+1}: "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
    
    # =========================================================================
    # Phase 2: Fine-tune BERT layers (gradual unfreezing)
    # =========================================================================
    print(f"\n  Phase 2: Fine-tuning top {config.FINE_TUNE_LAYERS} BERT layers ({config.FINE_TUNE_EPOCHS} epochs)")
    model.unfreeze_top_layers(config.FINE_TUNE_LAYERS)
    
    # Discriminative learning rates
    param_groups = model.get_optimizer_param_groups(
        bert_lr=config.FINE_TUNE_LR,
        classifier_lr=config.CLASSIFIER_LR * 0.1,  # Lower LR in phase 2
        weight_decay=config.WEIGHT_DECAY,
        discriminative_factor=config.DISCRIMINATIVE_LR_FACTOR,
    )
    
    optimizer = torch.optim.AdamW(param_groups)
    
    # Learning rate scheduler (linear warmup + decay)
    total_steps = len(train_loader) * config.FINE_TUNE_EPOCHS
    warmup_steps = total_steps // 10
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    patience_counter = 0  # Reset patience for phase 2
    
    for epoch in range(config.FINE_TUNE_EPOCHS):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"    Epoch {epoch+1}: "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"lr={current_lr:.2e}"
        )
        
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    return best_model_state, best_val_f1


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
    print(f"Fine-tuning: {config.FINE_TUNE_LAYERS} BERT layers")
    print(f"BERT LR: {config.FINE_TUNE_LR}, Classifier LR: {config.CLASSIFIER_LR}")
    print(f"Label smoothing: {config.LABEL_SMOOTHING}")

    # Load data
    train_data = load_cached_data("train")
    print(f"Train: {len(train_data)} samples")
    
    # Ensure text field exists
    for item in train_data:
        if "text" not in item:
            item["text"] = " ".join(item.get("tokens", []))
    
    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.TRANSFORMER_MODEL)
    
    # Labels
    train_labels = np.array([item["label"] for item in train_data])
    
    # K-fold CV
    num_folds = config.NUM_FOLDS
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.RANDOM_SEED)
    
    fold_scores = []
    
    print(f"\n{num_folds}-Fold Cross-Validation (Fine-Tuning)")
    config.CACHE_DIR.mkdir(exist_ok=True)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_data, train_labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{num_folds} (train={len(train_idx)}, val={len(val_idx)})")
        print("=" * 60)
        
        fold_train = [train_data[i] for i in train_idx]
        fold_val = [train_data[i] for i in val_idx]
        
        best_state, best_f1 = train_fold(fold_train, fold_val, tokenizer, device, fold_idx)
        fold_scores.append(best_f1)
        
        # Save checkpoint
        torch.save({
            "model_state_dict": best_state,
            "fold": fold_idx,
            "val_f1": best_f1,
            "fine_tuned": True,
            "fine_tune_layers": config.FINE_TUNE_LAYERS,
        }, config.CACHE_DIR / f"transformer_finetuned_fold_{fold_idx}.pt")
        
        print(f"\n  Best val F1: {best_f1:.4f}")
    
    print(f"\n{'='*60}")
    print(f"CV Results: F1={np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    
    # Select top-k models
    top_k = min(config.ENSEMBLE_TOP_K, num_folds)
    top_indices = np.argsort(fold_scores)[-top_k:]
    
    print(f"Ensemble: top {top_k} folds {[i+1 for i in top_indices]}")
    
    # Save ensemble info
    torch.save({
        "num_folds": num_folds,
        "fold_scores": fold_scores,
        "top_k": top_k,
        "top_indices": top_indices.tolist(),
        "fine_tuned": True,
        "fine_tune_layers": config.FINE_TUNE_LAYERS,
    }, config.CACHE_DIR / "transformer_finetuned_ensemble_info.pt")
    
    print("\nDone. Run finetune_evaluate.py for test results.")


if __name__ == "__main__":
    main()

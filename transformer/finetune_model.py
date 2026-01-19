"""
End-to-end fine-tuning model for hyperpartisan news detection.

This module provides a DistilBERT model with a classifier head that can be
fine-tuned end-to-end with gradual unfreezing and discriminative learning rates.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel

import transformer.config as config


class FineTunedClassifier(nn.Module):
    """
    DistilBERT with classifier head for end-to-end fine-tuning.
    
    Supports gradual unfreezing of transformer layers from top to bottom.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_extra_features: int = 0,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.num_extra_features = num_extra_features
        
        # Classifier head
        bert_dim = self.bert.config.dim  # 768 for distilbert-base
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_dim + num_extra_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initially freeze all BERT layers
        self.freeze_bert()

    def freeze_bert(self):
        """Freeze all BERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert(self):
        """Unfreeze all BERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self, n_layers: int):
        """
        Unfreeze the top n transformer layers.
        
        DistilBERT has 6 transformer layers (0-5).
        This unfreezes layers from the top (layer 5) downward.
        """
        # Always unfreeze pooler and embeddings output LayerNorm if present
        for name, param in self.bert.named_parameters():
            # Bottom layers stay frozen
            param.requires_grad = False
        
        # Unfreeze top n layers
        num_layers = len(self.bert.transformer.layer)  # 6 for DistilBERT
        for i in range(num_layers - n_layers, num_layers):
            for param in self.bert.transformer.layer[i].parameters():
                param.requires_grad = True
        
        # Count trainable params
        trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.bert.parameters())
        print(f"BERT: {trainable:,} / {total:,} params trainable ({n_layers} layers unfrozen)")

    def forward(self, input_ids, attention_mask, features=None):
        """Forward pass with optional extra features."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        if features is not None and self.num_extra_features > 0:
            cls_embedding = torch.cat([cls_embedding, features], dim=1)
        
        logits = self.classifier(cls_embedding)
        return torch.sigmoid(logits).squeeze(-1)

    def get_optimizer_param_groups(
        self,
        bert_lr: float,
        classifier_lr: float,
        weight_decay: float = 0.01,
        discriminative_factor: float = 0.9,
    ):
        """
        Get parameter groups with discriminative learning rates.
        
        Lower layers get progressively smaller learning rates.
        
        Args:
            bert_lr: Base learning rate for top BERT layer
            classifier_lr: Learning rate for classifier head
            weight_decay: Weight decay for regularization
            discriminative_factor: Multiply LR by this for each lower layer
        
        Returns:
            List of parameter group dicts for optimizer
        """
        param_groups = []
        
        # Classifier head (highest LR)
        param_groups.append({
            "params": list(self.classifier.parameters()),
            "lr": classifier_lr,
            "weight_decay": weight_decay,
        })
        
        # BERT embeddings (lowest LR)
        embed_lr = bert_lr * (discriminative_factor ** 6)
        param_groups.append({
            "params": list(self.bert.embeddings.parameters()),
            "lr": embed_lr,
            "weight_decay": weight_decay,
        })
        
        # BERT transformer layers (discriminative LR)
        num_layers = len(self.bert.transformer.layer)
        for i, layer in enumerate(self.bert.transformer.layer):
            # Layer 0 = bottom (lowest LR), Layer 5 = top (highest LR)
            layer_lr = bert_lr * (discriminative_factor ** (num_layers - 1 - i))
            param_groups.append({
                "params": list(layer.parameters()),
                "lr": layer_lr,
                "weight_decay": weight_decay,
            })
        
        return param_groups


class LabelSmoothingBCELoss(nn.Module):
    """Binary cross-entropy with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # Smooth labels: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy(pred, target_smooth)

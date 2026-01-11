"""
MLP classifier for pre-computed BERT embeddings.

Uses DistilBERT document embeddings with a simple MLP head for binary
classification. This approach leverages pre-trained contextual representations
which work better on small datasets than learning from scratch.
"""

import torch
import torch.nn as nn


class BertMLP(nn.Module):
    """MLP classifier on pre-computed BERT embeddings."""

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.5, num_extra_features=0):
        super().__init__()
        self.num_extra_features = num_extra_features
        
        # First layer: embedding + optional features → hidden
        self.fc1 = nn.Linear(input_dim + num_extra_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second layer: hidden → hidden/2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, features=None):
        # Concatenate extra features if provided
        if features is not None and self.num_extra_features > 0:
            x = torch.cat([x, features], dim=1)

        # First block
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # Output
        x = self.fc3(x)
        return torch.sigmoid(x).squeeze(1)

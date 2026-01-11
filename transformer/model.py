import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """MLP classifier on pre-computed transformer embeddings."""

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.5, num_extra_features=0):
        super().__init__()
        self.num_extra_features = num_extra_features
        self.fc1 = nn.Linear(input_dim + num_extra_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, features=None):
        if features is not None and self.num_extra_features > 0:
            x = torch.cat([x, features], dim=1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(1)

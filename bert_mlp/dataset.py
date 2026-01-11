"""Dataset classes for BERT-MLP classifier."""

import torch
from torch.utils.data import Dataset
import numpy as np


class BertEmbeddingDataset(Dataset):
    """Dataset for pre-computed BERT embeddings."""

    def __init__(self, embeddings, labels, features=None):
        """
        Args:
            embeddings: Pre-computed BERT embeddings (N, 768)
            labels: Binary labels (N,)
            features: Optional extra features (N, F)
        """
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.features = features.astype(np.float32) if features is not None else None

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        item = {
            "embedding": torch.tensor(self.embeddings[idx]),
            "label": torch.tensor(self.labels[idx]),
        }
        
        if self.features is not None:
            item["features"] = torch.tensor(self.features[idx])
        else:
            item["features"] = torch.tensor([])
            
        return item

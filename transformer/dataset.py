import torch
import numpy as np
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, features=None):
        self.embeddings = embeddings
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "embedding": torch.tensor(self.embeddings[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }

        if self.features is not None:
            item["features"] = torch.tensor(self.features[idx], dtype=torch.float)

        return item

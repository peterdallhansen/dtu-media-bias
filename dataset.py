import torch
from torch.utils.data import Dataset
from utils import tokens_to_ids
import config


class HyperpartisanDataset(Dataset):
    def __init__(self, data, vocab, max_len=None):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len or config.MAX_SEQ_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        token_ids = tokens_to_ids(item['tokens'], self.vocab, self.max_len)
        features = item.get('features', [0.0] * config.NUM_EXTRA_FEATURES)
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'extra_features': torch.tensor(features, dtype=torch.float),
            'label': torch.tensor(item['label'], dtype=torch.float)
        }

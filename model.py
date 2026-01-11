import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class HyperpartisanCNN(nn.Module):
    def __init__(self, vocab_size, embedding_matrix=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([
            nn.Conv1d(config.EMBEDDING_DIM, config.NUM_FILTERS, k)
            for k in config.KERNEL_SIZES
        ])

        self.batchnorms = nn.ModuleList([
            nn.BatchNorm1d(config.NUM_FILTERS)
            for _ in config.KERNEL_SIZES
        ])

        total_filters = config.NUM_FILTERS * len(config.KERNEL_SIZES)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(total_filters, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv, bn in zip(self.convs, self.batchnorms):
            c = conv(x)
            c = bn(c)
            c = F.relu(c)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outputs.append(c)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(1)

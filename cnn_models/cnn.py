#####
##### Import Packages
#####
# import os
import re
from collections import Counter
from tqdm import tqdm
import numpy as np
# import pandas as pd
# import nltk
# nltk.download("all")
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import random
# import time
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler, SequentialSampler)

EMBEDDING_DIM = 300
EPOCHS = 50
HP_BREAK = 60
SAVE_MODEL = True
BATCH_SIZE = 20

# Recursively check dimensions of list for testing
def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])

# For loading data
from preprocess import load_cached_data

# Load data
def load_data(breakat = 0.9):
    print("Loading dataset...")
    data_raw = load_cached_data()
    print(f"{len(data_raw)} data points found...")
    text = []
    labels = []
    for i in range(len(data_raw)):
        text.append(data_raw[i]['text'])
        labels.append(data_raw[i]['label'])
    breakat = (len(text)*breakat).__round__()
    return text[:breakat], text[breakat:], labels[:breakat], labels[breakat:]

# Load glove embedding
def load_glove(path = "cnn_models/glove.6B.300d.txt", dim=300):
    print("Loading embeddings...")
    f = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    embeddings = {}
    for step, line in tqdm(enumerate(f)):
        values = line.rstrip().split(" ")    
        word = values[0]
        vector = np.asarray(values[1:], dtype = "float32")
        embeddings[word] = vector
    return embeddings

# Tokenize and build vocab
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())
def build_vocab(texts, max_vocab_size = 50000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<PAD>':0, '<UNK>':1}
    for i, (word, _) in enumerate(counter.most_common(max_vocab_size-2)):
        vocab[word] = i + 2
    return vocab

# encode & pad
def encode(text, vocab):
    return [vocab.get(tok, vocab["<UNK>"]) for tok in tokenize(text)]
def pad(seq, max_len, vocab):
    return seq[:max_len] + [vocab["<PAD>"]] * (max_len - len(seq))

# Embedding matrix
def get_embedding_matrix(vocab, embedding, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    for word, idx in vocab.items():
        if word in embedding:
            embedding_matrix[idx] = embedding[word]
        elif word not in ("<PAD>", "<UNK>"):
            embedding_matrix[idx] = np.random.normal(scale = 0.6, size = embedding_dim)
    return embedding_matrix

# Device setup
def device_setup():
    if torch.cuda.is_available():
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        return torch.device("cpu")

# dataloader
def dataloader(train_inputs, test_inputs, train_labels, test_labels, batch_size = BATCH_SIZE):
    # Convert data type to torch.Tensor
    train_inputs, test_inputs, train_labels, test_labels =\
    tuple(torch.tensor(data) for data in [train_inputs, test_inputs, train_labels, test_labels])

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    test_data = TensorDataset(test_inputs, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader


# CNN class
class TextCNN(nn.Module):
    def __init__(self, embedding_layer):
        super(TextCNN, self).__init__()
        self.embedding = embedding_layer

        embed_dim = embedding_layer.embedding_dim

        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=5)

        self.pool = nn.AdaptiveMaxPool1d(1) # maxpool to prevent overfitting
        self.dropout = nn.Dropout(0.5) # prevent co-adaptation
        self.fc = nn.Linear(100, 2) # output

    def forward(self, x):               # BS = batch size, ML = max_len, ED = embedding_dim
        x                               # (BS, ML)
        x = self.embedding(x)           # (BS, ML, ED)
        x = x.transpose(1, 2)           # (BS, ED, ML)
        x = F.relu(self.conv(x))        # (BS, 100, ML)
        x = self.pool(x).squeeze(2)     # (BS, 100)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)    # (BS, 2)


def train_epoch(model, loader, optim, device, loss_fn, epoch_i):
    model.train()
    total_loss = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optim.zero_grad()

        logits = model(X)
        loss = loss_fn(logits, y)

        loss.backward()
        optim.step()

        total_loss += loss.item()
    return total_loss / len(loader), model


def evaluate(model, loader, device):
    model.eval()
    correct_total = total = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.sigmoid(logits)
            preds = [1 if pred[0] > 0.5 else 0 for pred in preds]
            
            correct = [1 if preds[i] == y[i] else 0 for i in range(len(preds))]
            correct_total =+ sum(correct)
            total += y.size(0)

    return correct_total / total, model


def predict(text, vocab, model, max_len, device):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids = ids[:max_len] + [vocab["<PAD>"]] * (max_len - len(ids))
    # Convert to PyTorch tensors
    input = torch.tensor(ids, dtype = torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input)
        probs = torch.softmax(logits, dim = 1).squeeze().tolist()
    
    #print(f"Probability of bias: {probs[1]*100:.4f}%")
    #print("Prediction:", "Biased" if probs[1] > 0.5 else "Not Biased")
    return 1 if probs[1] > 0.5 else 0, probs[1]*100


def test(model, vocab, max_len, device):
    open("cnn_models/logs/test_output.txt", "w").close()
    model.eval()
    print("loading test dataset...")
    test_data_article = load_cached_data("test_byarticle")
    print(f"{len(test_data_article)} by article data points found...")
    test_data_publisher = load_cached_data("test_bypublisher")
    print(f"{len(test_data_publisher)} by publisher data points found...")
    data_names = ['Article', 'Publisher']
    for j, k in enumerate([test_data_article, test_data_publisher]):
        test_text = []
        test_label = []
        for i in range(len(k)):
            test_text.append(k[i]['text'])
            test_label.append(k[i]['label'])
        predictions = []
        percentages = []
        for text in tqdm(test_text):
            pred, perc = (predict(text, vocab, model, max_len, device))
            percentages.append(perc)
            predictions.append(pred)
        correct = [0, 0, 0, 0]
        total = [len(k), 0, predictions.count(1), test_label.count(1)]
        for i in range(len(k)):
            if percentages[i] > HP_BREAK:
                total[1] += 1
                if test_label[i] == predictions[i]:
                    correct[1] += 1
            if test_label[i] == predictions[i]:
                correct[0] += 1
                if test_label[i] == 1:
                    correct[2] += 1
                    correct[3] += 1

        acc = correct[0] / total[0]
        prec = correct[2] / total[2]
        rec = correct[3] / total[3]
        f1 = 2 * (prec * rec) / (prec + rec)
        with open("cnn_models/logs/test_output.txt", "a") as f:
            print(f"\nBy-{data_names[j]} Test Result:")
            f.write(f"By-{data_names[j]} Test Results:")
            print(f"Test Accuracy: {acc*100:.2f}%")
            f.write(f"\nTest Accuracy: {acc*100:.2f}%")
            if total[1] > 0:
                print(f"High Percentage Accuracy: {correct[1] / total[1]*100:.2f}%")
                f.write(f"\nHigh Percentage Accuracy: {correct[1] / total[1]*100:.2f}%")
            else:
                print(f"No High Percentage Predictions")
                f.write(f"\nNo High Percentage Predictions")
            print(f"\n{'Model':^7}|{'Acc':^7}|{'Prec':^7}|{'Recall':^7}|{'F1':^7}")
            print("-"*40)
            print(f"{'CNN':^7}|{acc:^7.3f}|{prec:^7.3f}|{rec:^7.3f}|{f1:^7.3f}")
            f.write(f"\n\n{'Model':^7}|{'Acc':^7}|{'Prec':^7}|{'Recall':^7}|{'F1':^7}\n")
            f.write("-"*40)
            f.write(f"\n{'CNN':^7}|{acc:^7.3f}|{prec:^7.3f}|{rec:^7.3f}|{f1:^7.3f}\n\n\n")


def test_cnn():
    model = torch.load("./cnn_models/models/cnn_model.pt", weights_only = False)
    train_text, t, t, t = load_data()
    vocab = build_vocab(train_text)
    train_input = [encode(text, vocab) for text in train_text]
    max_len = 0
    for seq in train_input:
        max_len = max(len(seq), max_len)
    device = device_setup()
    test(model, vocab, max_len, device)


def main():
    train_text, test_text, train_label, test_label = load_data()
    embedding = load_glove()
    vocab = build_vocab(train_text) # Dictionary with all words id'ed by frequency
    train_input = [encode(text, vocab) for text in train_text] # [len(train_text)][words in text]
    test_input = [encode(text, vocab) for text in test_text]

    max_len = 0
    for seq in train_input:
        max_len = max(len(seq), max_len)
    train_input = [pad(data, max_len, vocab) for data in train_input] #[len(train_text)][max_len]
    test_input = [pad(data, max_len, vocab) for data in test_input] #[len(test_text)][max_len]

    # Embedding matrix
    embedding_matrix = get_embedding_matrix(vocab, embedding, EMBEDDING_DIM) # [ammount of different words in train_text][EMBEDDING_DIM]
    # 'the' has id 2 so embedding_matrix[2] is 300d vector for 'the'

    embedding_layer = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=True, padding_idx=vocab["<PAD>"])

    train_dataloader, test_dataloader = dataloader(train_input, test_input, train_label, test_label)
    device = device_setup()
    model = TextCNN(embedding_layer).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr = 0.1, rho = 0.95, weight_decay = 1e-3)
    
    loss_fn = nn.CrossEntropyLoss()
    with open("cnn_models/logs/train_output.txt", "w") as f:
        print(f"\n{'Epoch':^6}|{'Train Loss':^12}|{'Accuracy':^10}")
        print("-"*30)
        f.write(f"\n{'Epoch':^6}|{'Train Loss':^12}|{'Accuracy':^10}\n")
        f.write("-"*30)
        for epoch_i in range(EPOCHS):
            train_loss, model = train_epoch(model, train_dataloader, optimizer, device, loss_fn, epoch_i)
            val_acc, model = evaluate(model, test_dataloader, device)

            print(f"{epoch_i + 1:^6}|{train_loss:^12.4f}|{val_acc:^10.4f}")
            f.write(f"\n{epoch_i + 1:^6}|{train_loss:^12.4f}|{val_acc:^10.4f}")
    model.vocab = vocab
    model.max_len = max_len
    if SAVE_MODEL:
        torch.save(model, "./cnn_models/models/cnn_model.pt")
    #test(model, vocab, max_len, device) # for testing immediately after training model


#main()         # Train model
#test_cnn()     # Test model

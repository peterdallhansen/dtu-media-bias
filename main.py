
import os
import re
import html
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lxml import etree as ET
from torch.utils.data import IterableDataset, DataLoader
from collections import Counter
from tqdm import tqdm
from functools import partial
from nltk.tokenize import word_tokenize

# Ensure you have nltk data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace


def cleanQuotations(text):
    text = re.sub(r'[`‘’‛⸂⸃⸌⸍⸜⸝]', "'", text)
    text = re.sub(r'[„“”]|(\'\')|(,,)', '"', text)
    return text

def cleanText(text):
    text = re.sub(r'(www\S+)|(https?\S+)|(href)', ' ', text)
    text = re.sub(r'\{[^}]*\}|\[[^]]*\]|\([^)]*\)', ' ', text)
    text = re.sub(r'Getty [Ii]mages?|Getty|[Ff]ollow us on [Tt]witter|MORE:|ADVERTISEMENT|VIDEO', ' ', text)
    text = re.sub(r'@\S+|#\S+|\.{2,}', ' ', text)
    text = text.lstrip().replace('\n','')
    text = re.sub(r'  +', ' ', text)
    return text

def fixup(text):
    text = text.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'") \
               .replace('nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n") \
               .replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"') \
               .replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(' @-@ ', '-') \
               .replace('\\', ' \\ ')
    return html.unescape(text)

def textCleaning(title, text):
    title = cleanQuotations(title)
    text  = cleanQuotations(text)
    text  = cleanText(fixup(text))
    return (title + ". " + text).strip()

def preprocess_article(title, text):
    cleaned = textCleaning(title, text)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def parse_articles(article_path):
    for _, elem in ET.iterparse(article_path, events=("end",)):
        if elem.tag != "article":
            continue
        article_id = elem.get("id")
        title = elem.get("title", "") or ""
        paragraphs = [ (p.text or "").strip() for p in elem.findall("p") if p.text ]
        text = "\n".join(paragraphs)
        yield article_id, title, text
        elem.clear()

def parse_labels(label_path):
    labels = {}
    for _, elem in ET.iterparse(label_path, events=("end",)):
        if elem.tag != "article":
            continue
        # We extract 'bias' here for our target label
        labels[elem.get("id")] = {
            "hyperpartisan": elem.get("hyperpartisan"),
            "bias": elem.get("bias"),
        }
        elem.clear()
    return labels

def load_dataset(article_path, label_path, preprocess=None):
    labels = parse_labels(label_path)
    for article_id, title, text in parse_articles(article_path):
        if article_id not in labels:
            continue
        if preprocess:
            text = preprocess_article(title, text)
        yield {
            "id": article_id,
            "title": title,
            "text": text,
            "bias": labels[article_id]["bias"],
            "hyperpartisan": labels[article_id]["hyperpartisan"],
        }

class ArticleDataset(IterableDataset):
    def __init__(self, article_path, label_path):
        self.article_path = article_path
        self.label_path = label_path
    def __iter__(self):
        yield from load_dataset(self.article_path, self.label_path, preprocess=preprocess_article)


tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
tokenizer.pre_tokenizer = Whitespace()

def hf_tokenize(text):
    return tokenizer.pre_tokenizer.pre_tokenize_str(text)

def build_vocab_from_stream(dataset_stream, min_freq=3, max_size=None):
    counter = Counter()
    # Limit total for tqdm to avoid issues with indefinite streams, or remove total
    for sample in tqdm(dataset_stream, desc="Building vocab"):
        tokens = [w for w, _ in hf_tokenize(sample["text"])]
        counter.update(tokens)
        
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for token, freq in counter.most_common():
        if freq < min_freq:
            break
        if max_size and idx >= max_size:
            break
        vocab[token] = idx
        idx += 1
    return vocab

def numericalize(tokens, vocab):
    unk = vocab["<unk>"]
    return [vocab.get(t, unk) for t in tokens]


dataPath = "./Dataset"
train_article_path = os.path.join(dataPath, "train-articles.xml")
train_label_path   = os.path.join(dataPath, "ground-truth-training-bypublisher-20181122.xml")
val_article_path = os.path.join(dataPath, "val-articles.xml")
val_label_path   = os.path.join(dataPath, "ground-truth-validation-bypublisher-20181122.xml")

train_dataset = ArticleDataset(train_article_path, train_label_path)
val_dataset   = ArticleDataset(val_article_path, val_label_path)

print("Building vocab (this may take a moment)...")
# Note: In a real scenario with iterables, you might want to limit the stream 
# or use a subset to build vocab to save time. Here we run on full train.


BUILD_VOCAB = True

if BUILD_VOCAB:
    vocab = build_vocab_from_stream(
        load_dataset(train_article_path, train_label_path, preprocess_article),
        min_freq=3,
        max_size=50000
    )


    # Save vocab
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)
else:
    with open("vocab.json", "r") as f:
        vocab = json.load(f)

print("Vocab size:", len(vocab))


LABEL_MAP = {
    "left": 0,
    "left-center": 1,
    "least-biased": 2,
    "right-center": 3,
    "right": 4
}

def collate_batch(batch, vocab, max_len=256):
    texts = []
    labels = []

    for sample in batch:
        # Simple whitespace tokenization using NLTK or split
        tokens = word_tokenize(sample["text"])
        ids = numericalize(tokens[:max_len], vocab)

        pad = max_len - len(ids)
        if pad > 0:
            ids += [vocab["<pad>"]] * pad
        
        texts.append(ids)

        # --- MULTICLASS LOGIC ---
        bias_str = sample["bias"]
        # Default to 'least-biased' (2) if label missing or unknown
        label_idx = LABEL_MAP.get(bias_str, 2)
        labels.append(label_idx)

    return (
        torch.tensor(texts, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long)
    )

batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=partial(collate_batch, vocab=vocab, max_len=256)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=partial(collate_batch, vocab=vocab, max_len=256)
)



class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=5,
                 kernel_sizes=(3,4,5), num_filters=100,
                 padding_idx=0, pretrained_embeddings=None,
                 freeze_embeddings=False):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)           # (batch, seq, embed)
        x = x.transpose(1, 2)           # (batch, embed, seq)

        conv_outputs = []
        for conv in self.convs:
            c = conv(x)                 # (batch, num_filters, L_out)
            c = F.relu(c)
            # Max pooling over time
            c = F.max_pool1d(c, c.size(2))   # (batch, num_filters, 1)
            conv_outputs.append(c.squeeze(2))

        out = torch.cat(conv_outputs, dim=1)  # (batch, F*kernels)
        out = self.dropout(out)
        return self.fc(out)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

embed_dim = 300
# Updated for 5 classes (Left, Left-Center, Center, Right-Center, Right)
num_classes = 5 

model = TextCNN(
    vocab_size=len(vocab),
    embed_dim=embed_dim,
    num_classes=num_classes,
    padding_idx=vocab["<pad>"]
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


def train_one_epoch(loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for tokens, labels in tqdm(loader, desc="Training"):
        tokens = tokens.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0: return 0, 0
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)

        logits = model(tokens)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0: return 0, 0
    return total_loss / total, correct / total


epochs = 5

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss, train_acc = train_one_epoch(train_loader)
    print(f"  Train loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    
    val_loss, val_acc = eval_epoch(val_loader)
    print(f"  Val   loss: {val_loss:.4f} | Acc: {val_acc:.4f}")


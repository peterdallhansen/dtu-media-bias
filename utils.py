import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gensim.downloader as api
from tqdm import tqdm


def load_word2vec():
    print("Loading word2vec...")
    model = api.load('word2vec-google-news-300')
    return model


def build_vocab(articles, min_freq=2, max_size=50000):
    word_counts = Counter()
    for article in tqdm(articles, desc="Building vocab"):
        word_counts.update(article['tokens'])

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common(max_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def create_embedding_matrix(vocab, word2vec_model, dim=300):
    matrix = np.random.uniform(-0.25, 0.25, (len(vocab), dim)).astype(np.float32)
    matrix[0] = np.zeros(dim)

    found = 0
    for word, idx in tqdm(vocab.items(), desc="Creating embeddings"):
        if word in word2vec_model:
            matrix[idx] = word2vec_model[word]
            found += 1

    print(f"  {found}/{len(vocab)} words found in word2vec")
    return matrix


def tokens_to_ids(tokens, vocab, max_len):
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens[:max_len]]
    padding = [vocab['<PAD>']] * (max_len - len(ids))
    return ids + padding


def calculate_metrics(preds, labels):
    preds_binary = (np.array(preds) > 0.5).astype(int)
    labels = np.array(labels).astype(int)

    return {
        'accuracy': accuracy_score(labels, preds_binary),
        'precision': precision_score(labels, preds_binary, zero_division=0),
        'recall': recall_score(labels, preds_binary, zero_division=0),
        'f1': f1_score(labels, preds_binary, zero_division=0)
    }

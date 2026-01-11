import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from preprocess import load_cached_data
from utils import load_word2vec, build_vocab, create_embedding_matrix, calculate_metrics
from dataset import HyperpartisanDataset
from model import HyperpartisanCNN


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_preds, all_labels)
    return total_loss / len(loader), metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_preds, all_labels)
    return total_loss / len(loader), metrics


def main():
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading cached data...")
    train_data = load_cached_data('train')
    test_data = load_cached_data('test')
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    print("Building vocabulary...")
    vocab = build_vocab(train_data, config.MIN_WORD_FREQ, config.VOCAB_SIZE)
    print(f"Vocab size: {len(vocab)}")

    word2vec = load_word2vec()
    embedding_matrix = create_embedding_matrix(vocab, word2vec, config.EMBEDDING_DIM)

    train_dataset = HyperpartisanDataset(train_data, vocab)
    test_dataset = HyperpartisanDataset(test_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = HyperpartisanCNN(len(vocab), embedding_matrix).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    config.CACHE_DIR.mkdir(exist_ok=True)
    best_f1 = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'epoch': epoch,
                'test_f1': best_f1
            }, config.CACHE_DIR / "best_model.pt")
            print(f"Saved best model with F1: {best_f1:.4f}")

    print(f"\nTraining complete. Best Test F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

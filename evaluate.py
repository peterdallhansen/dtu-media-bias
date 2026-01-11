import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from preprocess import load_cached_data
from dataset import HyperpartisanDataset
from model import HyperpartisanCNN
from utils import calculate_metrics


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint['vocab']

    model = HyperpartisanCNN(len(vocab)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, vocab, checkpoint.get('epoch', 'N/A')


def evaluate_test(model, loader, device):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label']

            outputs = model(input_ids)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = config.CACHE_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"No model found at {checkpoint_path}. Run train.py first.")
        return

    print("Loading model...")
    model, vocab, best_epoch = load_model(checkpoint_path, device)

    print("Loading test data...")
    try:
        test_data = load_cached_data('test')
    except FileNotFoundError:
        print("Test cache not found. Run preprocess.py first.")
        return

    test_dataset = HyperpartisanDataset(test_data, vocab)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"\nDataset: by-article test set")
    print(f"Samples: {len(test_data)}")

    ground_truth_hp = sum(1 for d in test_data if d['label'] == 1)
    ground_truth_nhp = len(test_data) - ground_truth_hp
    print(f"Ground truth: {ground_truth_hp} hyperpartisan, {ground_truth_nhp} not ({ground_truth_hp/len(test_data)*100:.1f}% / {ground_truth_nhp/len(test_data)*100:.1f}%)")

    preds, labels = evaluate_test(model, test_loader, device)
    metrics = calculate_metrics(preds, labels)

    preds_binary = (preds > 0.5).astype(int)
    pred_hp = preds_binary.sum()
    pred_nhp = len(preds_binary) - pred_hp
    print(f"Predictions: {pred_hp} hyperpartisan, {pred_nhp} not")

    print("\n" + "=" * 50)
    print("RESULTS (SemEval-2019 Task 4 format)")
    print("=" * 50)
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print("=" * 50)

    print("\nReference (top teams on by-article test set):")
    print("  1. Bertha von Suttner: Acc=0.822, P=0.871, R=0.755, F1=0.809")
    print("  2. Vernon Fenwick:     Acc=0.820, P=0.815, R=0.828, F1=0.821")
    print("  4. Tom Jumbo Grumbo:   Acc=0.806, P=0.858, R=0.732, F1=0.790")


if __name__ == "__main__":
    main()

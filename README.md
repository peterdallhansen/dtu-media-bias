# Explaining Political Bias Classification in News Articles using CNNs and Transformers

A deep learning project for detecting and explaining hyperpartisan (politically biased) news articles. The models are trained on the SemEval-2019 Task 4 Hyperpartisan News Detection dataset.

This repository accompanies the paper _Explaining Political Bias Classification in News Articles using CNNs and Transformers_ (Jensen, Agbesi, Dall-Hansen, 2025).

## Abstract

Detecting political bias in news articles is an increasingly important task in the era of misinformation and polarized media. In this project, we implement multiple approaches for binary classification of news articles as hyperpartisan or mainstream: a CNN with GloVe embeddings, a transformer-based classifier using DistilBERT, and an SVM baseline. Beyond classification, we explore interpretability methods to identify which tokens contribute most to model decisions.

## Pipeline

![Pipeline Diagram](pipeline.svg)

## Project Structure

```
.
├── preprocess.py      # XML parsing and text preprocessing
├── evaluate.py        # Unified evaluation across all models
├── cnn/               # CNN with GloVe embeddings
│   ├── config.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── transformer/       # DistilBERT embeddings + classifier
│   ├── config.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── svm/               # SVM baseline (averaged GloVe + RBF kernel)
│   ├── config.py
│   ├── train.py
│   └── evaluate.py
└── cache/             # Preprocessed data and model checkpoints
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the [News articles for political bias classification](https://www.kaggle.com/datasets/gandpablo/news-articles-for-political-bias-classification) dataset from Kaggle.

The dataset is automatically downloaded/processed on the first run of `preprocess.py`.

### Prerequisites
- `kagglehub` must be installed:
  ```bash
  pip install kagglehub
  ```

## Usage

1. **Preprocess Data**
   Downloads and prepares the dataset (train/test split):
   ```bash
   python preprocess.py
   ```

2. **Evaluate (and Auto-Train)**
   The easiest way to run the project is via the unified evaluation script. It will automatically train any missing models (CNN, Transformer, SVM, BERT-MLP) and then evaluate them.
   ```bash
   python evaluate.py
   ```

### Individual Model Training
If you prefer to train models individually:

```bash
python -m cnn.train
python -m transformer.train
python -m svm.train
python -m bert_mlp.train
```

### Interpretability Visualization

Generate token-level attribution heatmaps:

```bash
python -m transformer.interpret --article-id "0000001" --output output/
python -m transformer.interpret --text "Your article text here" --output output/
```


## Results

| Model       | Approach                    | By-Article Acc |
| ----------- | --------------------------- | -------------- |
| CNN         | GloVe + multi-kernel CNN    | ~0.65          |
| Transformer | DistilBERT embeddings + MLP | ~0.82          |
| SVM         | Averaged GloVe + RBF kernel | ~0.81          |

### Reference (SemEval-2019)

| Team               | Approach    | By-Article Acc |
| ------------------ | ----------- | -------------- |
| Bertha von Suttner | ELMo + CNN  | 0.822          |
| Tom Jumbo Grumbo   | GloVe + SVM | 0.806          |

## Configuration

See `cnn/config.py`, `transformer/config.py`, and `svm/config.py` for hyperparameters.

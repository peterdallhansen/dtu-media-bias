# Explaining Political Bias Classification in News Articles using CNNs and Transformers

A deep learning project for detecting hyperpartisan (politically biased) news articles using Convolutional Neural Networks with Word2Vec embeddings. The model is trained on the SemEval-2019 Task 4 Hyperpartisan News Detection dataset.

## Abstract

Detecting political bias in news articles is an increasingly important task in the era of misinformation and polarized media. In this project, we implement a CNN-based approach for binary classification of news articles as hyperpartisan or mainstream. Using pre-trained Word2Vec embeddings and multi-kernel convolutions, we extract n-gram features from article text to capture linguistic patterns indicative of political bias. Our model is evaluated on the SemEval-2019 Hyperpartisan News Detection benchmark using accuracy, precision, recall, and F1-score metrics.

## Pipeline

<!-- TODO: Add LucidChart pipeline diagram -->

![Pipeline Diagram](Pipeline.png)]

_Pipeline diagram showing data preprocessing, model architecture, and evaluation flow._

## Project Structure

```
.
├── config.py          # Hyperparameters and paths
├── preprocess.py      # XML parsing and text preprocessing
├── dataset.py         # PyTorch Dataset class
├── model.py           # CNN model architecture
├── train.py           # Training loop
├── evaluate.py        # Model evaluation
├── utils.py           # Vocabulary, embeddings, metrics
├── requirements.txt   # Dependencies
└── Dataset/           # SemEval-2019 Task 4 data (not included)
```

## Installation

To install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the [SemEval-2019 Task 4](https://pan.webis.de/semeval19/semeval19-web/) Hyperpartisan News Detection dataset. Download the dataset and place the XML files in the `Dataset/` directory:

- `articles-training-byarticle-20181122.xml`
- `ground-truth-training-byarticle-20181122.xml`
- `articles-test-byarticle-20181207.xml`
- `ground-truth-test-byarticle-20181207.xml`

## Usage

### 1. Preprocess the data

```bash
python preprocess.py
```

This parses the XML files, cleans and tokenizes the text, and caches the processed data.

### 2. Train the model

```bash
python train.py
```

Trains the CNN model and saves the best checkpoint based on F1-score.

### 3. Evaluate

```bash
python evaluate.py
```

Loads the best model and evaluates on the test set.

## Model Architecture

The model uses a multi-kernel CNN architecture:

- **Embedding Layer**: Word2Vec (Google News 300d) with fine-tuning
- **Convolutional Layers**: Parallel convolutions with kernel sizes [3, 4, 5]
- **Batch Normalization**: Applied after each convolution
- **Pooling**: Global max pooling
- **Dropout**: 0.5 for regularization
- **Output**: Sigmoid activation for binary classification

## Configuration

Key hyperparameters (see `config.py`):

| Parameter           | Value     |
| ------------------- | --------- |
| Embedding Dimension | 300       |
| Max Sequence Length | 512       |
| Vocabulary Size     | 50,000    |
| CNN Filters         | 128       |
| Kernel Sizes        | [3, 4, 5] |
| Dropout             | 0.5       |
| Batch Size          | 32        |
| Learning Rate       | 1e-3      |
| Epochs              | 20        |

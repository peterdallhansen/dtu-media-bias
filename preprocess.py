import re
import csv
import pickle
import hashlib
import math
import random
import kagglehub
import requests
from pathlib import Path
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import StratifiedKFold
from cnn import config

# Optional imports
try:
    import spacy
except ImportError:
    spacy = None

# -------------------------------------------------------------------
# Utility Functions (Generic)
# -------------------------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text):
    return text.split()


def extract_sentiment_features(text, analyzer):
    scores = analyzer.polarity_scores(text[:5000])
    return [scores['compound'], scores['pos'], scores['neg']]


def extract_ner_features(text, nlp):
    doc = nlp(text[:10000])
    counts = {'PERSON': 0, 'ORG': 0, 'GPE': 0, 'NORP': 0, 'EVENT': 0}
    for ent in doc.ents:
        if ent.label_ in counts:
            counts[ent.label_] += 1
    return [math.log1p(counts[k]) for k in ['PERSON', 'ORG', 'GPE', 'NORP', 'EVENT']]


def get_config_hash():
    """Generate a hash based on feature configuration."""
    config_str = (
        f"dataset=kaggle,"
        f"date={config.USE_DATE_FEATURES},"
        f"sentiment={config.USE_SENTIMENT_FEATURES},"
        f"ner={config.USE_NER_FEATURES}"
    )
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


# -------------------------------------------------------------------
# Kaggle Specific Logic
# -------------------------------------------------------------------

def download_dataset():
    """Download the dataset using kagglehub."""
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("gandpablo/news-articles-for-political-bias-classification")
    print(f"Dataset downloaded to: {path}")
    return Path(path)


def map_label(bias_label):
    """
    Map 5-class bias label to binary hyperpartisan label.
    
    Mapping:
    - Hyperpartisan (1): left, right
    - Mainstream (0): leaning-left, center, leaning-right
    """
    bias_label = bias_label.lower().strip()
    if bias_label in ['left', 'right']:
        return 1
    elif bias_label in ['leaning-left', 'leaning-right', 'center']:
        return 0
    else:
        print(f"Warning: Unknown label '{bias_label}', treating as 0")
        return 0


def extract_date_features(date_str):
    """Extract date features from YYYY-MM-DD or MM/DD/YYYY formats."""
    if not date_str:
        return [0.5, 0.5]
    try:
        parts = date_str.split('-')
        if len(parts) != 3:
             parts = date_str.split('/')
        
        if len(parts) >= 1:
            if len(parts[0]) == 4: # YYYY-MM-DD
                 year = int(parts[0])
                 month = int(parts[1]) if len(parts) > 1 else 6
            else: # MM/DD/YYYY
                 year = int(parts[2]) if len(parts) > 2 else 2020
                 month = int(parts[0]) if len(parts) > 0 else 6
                 
            year_norm = (year - 2000) / 25.0
            year_norm = max(0.0, min(1.0, year_norm))
            month_norm = (month - 1) / 11.0
            return [month_norm, year_norm]
    except:
        pass
    return [0.5, 0.5]


def parse_csv_articles(csv_path):
    articles = []
    
    # Initialize analyzers
    analyzer = SentimentIntensityAnalyzer() if config.USE_SENTIMENT_FEATURES else None
    nlp = spacy.load('en_core_web_sm', disable=['parser']) if (config.USE_NER_FEATURES and spacy) else None
    
    print(f"Parsing {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        for i, row in enumerate(tqdm(rows, desc="Processing Articles")):
            article_id = hashlib.md5(row.get('url', str(i)).encode()).hexdigest()[:8]
            
            title = row.get('title', '')
            date_str = row.get('date', '')
            text_raw = row.get('page_text', '')
            bias_label = row.get('bias', '')
            
            cleaned_text = clean_text(title + ' ' + text_raw)
            tokens = tokenize(cleaned_text)
            
            label = map_label(bias_label)
            
            features = []
            if config.USE_DATE_FEATURES:
                features += extract_date_features(date_str)
            
            # No hyperlink features for Kaggle
            
            if config.USE_SENTIMENT_FEATURES:
                features += extract_sentiment_features(text_raw, analyzer)
                
            if config.USE_NER_FEATURES and nlp:
                features += extract_ner_features(text_raw, nlp)
            
            article_data = {
                'id': article_id,
                'title': title,
                'published': date_str,
                'text': cleaned_text,
                'tokens': tokens,
                'hyperlinks': [], # Empty
                'features': features,
                'label': label,
                'original_label': bias_label
            }
            articles.append(article_data)

    return articles


def preprocess_and_cache():
    """Download, parse, split, and cache the Kaggle dataset."""
    dataset_dir = download_dataset()
    csv_path = dataset_dir / "bias_clean.csv"
    
    if not csv_path.exists():
        found = list(dataset_dir.glob("**/*.csv"))
        if found:
            csv_path = found[0]
        else:
            raise FileNotFoundError(f"Could not find bias_clean.csv in {dataset_dir}")

    config.CACHE_DIR.mkdir(exist_ok=True)
    cfg_hash = get_config_hash()
    print(f"Config Hash: {cfg_hash}")
    
    all_data = parse_csv_articles(csv_path)
    print(f"Total articles parsed: {len(all_data)}")
    
    # Stratified Split (80/20)
    random.seed(config.RANDOM_SEED)
    labels = [d['label'] for d in all_data]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    train_idx, test_idx = next(skf.split(all_data, labels))
    
    train_data = [all_data[i] for i in train_idx]
    test_data = [all_data[i] for i in test_idx]
    
    # Stats
    train_pos = sum(d['label'] for d in train_data)
    train_neg = len(train_data) - train_pos
    test_pos = sum(d['label'] for d in test_data)
    test_neg = len(test_data) - test_pos
    
    print("\nSplit Statistics:")
    print(f"Train: {len(train_data)} samples ({train_pos} hyperpartisan, {train_neg} mainstream)")
    print(f"Test:  {len(test_data)} samples ({test_pos} hyperpartisan, {test_neg} mainstream)")
    
    # Cache
    train_cache_path = config.CACHE_DIR / f"train_data_{cfg_hash}.pkl"
    test_cache_path = config.CACHE_DIR / f"test_data_{cfg_hash}.pkl"
    
    with open(train_cache_path, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"Saved train data to {train_cache_path}")
    
    with open(test_cache_path, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"Saved test data to {test_cache_path}")
    
    return train_data, test_data


def load_cached_data(split='train'):
    """Load cached data."""
    # Map semeval keys to Kaggle keys if necessary
    if 'test' in split:
        split = 'test'
        
    cfg_hash = get_config_hash()
    cache_path = config.CACHE_DIR / f"{split}_data_{cfg_hash}.pkl"
    
    if not cache_path.exists():
        print(f"No cache found for config {cfg_hash}, preprocessing...")
        preprocess_and_cache()
        
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    preprocess_and_cache()

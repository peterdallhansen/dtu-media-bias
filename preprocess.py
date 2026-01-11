import re
import pickle
import zipfile
import math
import hashlib
import requests
from pathlib import Path
from lxml import etree
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from cnn import config

if config.USE_NER_FEATURES:
    import spacy


def get_config_hash():
    """Generate a hash based on feature configuration to detect config changes."""
    config_str = (
        f"date={config.USE_DATE_FEATURES},"
        f"hyperlink={config.USE_HYPERLINK_FEATURES},"
        f"sentiment={config.USE_SENTIMENT_FEATURES},"
        f"ner={config.USE_NER_FEATURES}"
    )
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

ZENODO_URL = "https://zenodo.org/api/records/5776081/files-archive"


def download_dataset():
    if config.ARTICLES_TRAIN.exists():
        return

    print("Dataset not found. Downloading from Zenodo...")
    config.DATA_DIR.mkdir(exist_ok=True)

    zip_path = config.DATA_DIR / "dataset.zip"
    response = requests.get(ZENODO_URL, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in tqdm(zf.namelist(), desc="Extracting"):
            if member.endswith('.zip'):
                inner_zip_path = config.DATA_DIR / member
                zf.extract(member, config.DATA_DIR)
                with zipfile.ZipFile(inner_zip_path, 'r') as inner_zf:
                    inner_zf.extractall(config.DATA_DIR)
                inner_zip_path.unlink()
            elif not member.endswith('/'):
                zf.extract(member, config.DATA_DIR)

    zip_path.unlink()
    print("Dataset ready.")


def extract_date_features(published_str):
    if not published_str:
        return [0.5, 0.5]
    try:
        parts = published_str.split('-')
        year = int(parts[0]) if len(parts) > 0 else 2010
        month = int(parts[1]) if len(parts) > 1 else 6
        year_norm = (year - 2000) / 20.0
        year_norm = max(0.0, min(1.0, year_norm))
        month_norm = (month - 1) / 11.0
        return [month_norm, year_norm]
    except:
        return [0.5, 0.5]


def extract_hyperlink_features(hyperlinks):
    external = sum(1 for h in hyperlinks if h.get('type') == 'external')
    internal = sum(1 for h in hyperlinks if h.get('type') == 'internal')
    total = external + internal
    external_log = math.log1p(external)
    internal_log = math.log1p(internal)
    link_ratio = external / (total + 1)
    has_links = 1.0 if total > 0 else 0.0
    return [external_log, internal_log, link_ratio, has_links]


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


def extract_text(element):
    text_parts = []
    if element.text:
        text_parts.append(element.text)
    for child in element:
        text_parts.append(extract_text(child))
        if child.tail:
            text_parts.append(child.tail)
    return ' '.join(text_parts)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text):
    return text.split()


def parse_articles(xml_path):
    articles = {}
    analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load('en_core_web_sm', disable=['parser']) if config.USE_NER_FEATURES else None
    context = etree.iterparse(xml_path, events=('end',), tag='article')

    for event, elem in tqdm(context, desc=f"Parsing {xml_path.name}"):
        article_id = elem.get('id')
        title = elem.get('title', '')
        published = elem.get('published-at', '')

        text = extract_text(elem)
        cleaned = clean_text(title + ' ' + text)
        tokens = tokenize(cleaned)

        hyperlinks = []
        for a in elem.findall('.//a'):
            href = a.get('href', '')
            link_type = a.get('type', 'external')
            if href:
                hyperlinks.append({'href': href, 'type': link_type})

        features = []
        if config.USE_DATE_FEATURES:
            features += extract_date_features(published)
        if config.USE_HYPERLINK_FEATURES:
            features += extract_hyperlink_features(hyperlinks)
        if config.USE_SENTIMENT_FEATURES:
            features += extract_sentiment_features(text, analyzer)
        if config.USE_NER_FEATURES:
            features += extract_ner_features(text, nlp)

        articles[article_id] = {
            'id': article_id,
            'title': title,
            'published': published,
            'text': cleaned,
            'tokens': tokens,
            'hyperlinks': hyperlinks,
            'features': features
        }

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return articles


def parse_labels(xml_path):
    labels = {}
    context = etree.iterparse(xml_path, events=('end',), tag='article')

    for event, elem in tqdm(context, desc=f"Labels [{xml_path.name}]"):
        article_id = elem.get('id')
        hyperpartisan = elem.get('hyperpartisan') == 'true'
        labels[article_id] = hyperpartisan

        elem.clear()

    return labels


def merge_data(articles, labels):
    data = []
    for article_id, article in articles.items():
        if article_id in labels:
            article['label'] = int(labels[article_id])
            data.append(article)
    return data


def preprocess_and_cache():
    download_dataset()
    config.CACHE_DIR.mkdir(exist_ok=True)

    cfg_hash = get_config_hash()
    print(f"Config hash: {cfg_hash}")

    print("Training set:")
    train_articles = parse_articles(config.ARTICLES_TRAIN)
    train_labels = parse_labels(config.LABELS_TRAIN)
    train_data = merge_data(train_articles, train_labels)

    pos = sum(d['label'] for d in train_data)
    neg = len(train_data) - pos
    print(f"  {len(train_data)} samples ({pos} hyperpartisan, {neg} not)")

    cache_path = config.CACHE_DIR / f"train_data_{cfg_hash}.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"  -> {cache_path}")

    if config.ARTICLES_TEST.exists() and config.LABELS_TEST.exists():
        print("\nTest set (by-article):")
        test_articles = parse_articles(config.ARTICLES_TEST)
        test_labels = parse_labels(config.LABELS_TEST)
        test_data = merge_data(test_articles, test_labels)

        pos = sum(d['label'] for d in test_data)
        neg = len(test_data) - pos
        print(f"  {len(test_data)} samples ({pos} hyperpartisan, {neg} not)")

        cache_path = config.CACHE_DIR / f"test_byarticle_data_{cfg_hash}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"  -> {cache_path}")

    if config.ARTICLES_TEST_BYPUB.exists() and config.LABELS_TEST_BYPUB.exists():
        print("\nTest set (by-publisher):")
        test_articles = parse_articles(config.ARTICLES_TEST_BYPUB)
        test_labels = parse_labels(config.LABELS_TEST_BYPUB)
        test_data = merge_data(test_articles, test_labels)

        pos = sum(d['label'] for d in test_data)
        neg = len(test_data) - pos
        print(f"  {len(test_data)} samples ({pos} hyperpartisan, {neg} not)")

        cache_path = config.CACHE_DIR / f"test_bypublisher_data_{cfg_hash}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"  -> {cache_path}")

    return train_data


def load_cached_data(split='train'):
    cfg_hash = get_config_hash()
    cache_path = config.CACHE_DIR / f"{split}_data_{cfg_hash}.pkl"
    if not cache_path.exists():
        print(f"No cache found for config {cfg_hash}, preprocessing...")
        preprocess_and_cache()
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    preprocess_and_cache()

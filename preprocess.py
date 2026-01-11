import re
import pickle
from lxml import etree
from tqdm import tqdm
import config


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

        articles[article_id] = {
            'id': article_id,
            'title': title,
            'published': published,
            'text': cleaned,
            'tokens': tokens,
            'hyperlinks': hyperlinks
        }

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return articles


def parse_labels(xml_path):
    labels = {}
    context = etree.iterparse(xml_path, events=('end',), tag='article')

    for event, elem in context:
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
    config.CACHE_DIR.mkdir(exist_ok=True)

    print("Processing training data...")
    train_articles = parse_articles(config.ARTICLES_TRAIN)
    train_labels = parse_labels(config.LABELS_TRAIN)
    train_data = merge_data(train_articles, train_labels)

    print(f"Train samples: {len(train_data)}")
    print(f"Hyperpartisan: {sum(d['label'] for d in train_data)}")
    print(f"Not hyperpartisan: {sum(1 - d['label'] for d in train_data)}")

    cache_path = config.CACHE_DIR / "train_data.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"Cached to {cache_path}")

    if config.ARTICLES_TEST.exists() and config.LABELS_TEST.exists():
        print("\nProcessing test data...")
        test_articles = parse_articles(config.ARTICLES_TEST)
        test_labels = parse_labels(config.LABELS_TEST)
        test_data = merge_data(test_articles, test_labels)

        print(f"Test samples: {len(test_data)}")

        cache_path = config.CACHE_DIR / "test_data.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"Cached to {cache_path}")

    return train_data


def load_cached_data(split='train'):
    cache_path = config.CACHE_DIR / f"{split}_data.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}. Run preprocess.py first.")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    preprocess_and_cache()

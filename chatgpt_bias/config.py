"""ChatGPT-based political bias detection configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = CACHE_DIR / "Dataset"

# Dataset paths (shared with other models)
ARTICLES_TRAIN = DATA_DIR / "articles-training-byarticle-20181122.xml"
LABELS_TRAIN = DATA_DIR / "ground-truth-training-byarticle-20181122.xml"
ARTICLES_TEST = DATA_DIR / "articles-test-byarticle-20181207.xml"
LABELS_TEST = DATA_DIR / "ground-truth-test-byarticle-20181207.xml"

ARTICLES_TEST_BYPUB = DATA_DIR / "articles-test-bypublisher-20181212.xml"
LABELS_TEST_BYPUB = DATA_DIR / "ground-truth-test-bypublisher-20181212.xml"

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Model parameters
TEMPERATURE = 0.1  # Low temperature for more deterministic predictions
MAX_TOKENS = 50  # Short responses for binary classification

# Cost control
MAX_ARTICLES_PER_SPLIT = (
    None  # Maximum articles to evaluate per dataset split (set to None for no limit)
)

# Caching
RESPONSE_CACHE_DIR = CACHE_DIR / "chatgpt_responses"
RESPONSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# System prompt for political bias detection
SYSTEM_PROMPT = """You are an expert analyst specialized in detecting political bias in news articles.

Your task is to classify news articles as either:
- HYPERPARTISAN: Articles showing extreme political bias, strong partisan language, or one-sided reporting
- MAINSTREAM: Balanced articles with neutral reporting and minimal political bias

Analyze the article's language, tone, framing, and overall objectivity."""

# User prompt template
USER_PROMPT_TEMPLATE = """Article Title: {title}

Article Text (truncated to first 2000 characters):
{text}

Based on this article, classify it as either HYPERPARTISAN or MAINSTREAM.
Respond with ONLY ONE WORD: either "HYPERPARTISAN" or "MAINSTREAM"."""

"""Utility functions for ChatGPT-based bias detection."""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import chatgpt_bias.config as config


def check_credentials():
    """Check if Azure OpenAI credentials are set."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please create a .env file with your Azure OpenAI credentials."
        )


def setup_llm():
    """Initialize Azure OpenAI LLM."""
    check_credentials()
    
    return AzureChatOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )


def get_cache_key(article_id: str, title: str, text: str) -> str:
    """Generate cache key for an article."""
    content = f"{article_id}|{title}|{text[:500]}"
    return hashlib.md5(content.encode()).hexdigest()


def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached response if it exists."""
    cache_file = config.RESPONSE_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None


def save_cached_response(cache_key: str, response_data: Dict[str, Any]):
    """Save response to cache."""
    cache_file = config.RESPONSE_CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, 'w') as f:
        json.dump(response_data, f, indent=2)


def classify_article(
    article: Dict[str, Any],
    llm: AzureChatOpenAI,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Classify a single article as hyperpartisan or mainstream.
    
    Args:
        article: Article dictionary containing 'id', 'title', 'text', 'label'
        llm: Azure OpenAI LLM instance
        use_cache: Whether to use cached responses
    
    Returns:
        Dictionary with prediction, true label, and metadata
    """
    article_id = article['id']
    title = article.get('title', '')
    text = article['text'][:2000]  # Truncate to control costs
    true_label = article.get('label', None)
    
    cache_key = get_cache_key(article_id, title, text)
    
    # Check cache first
    if use_cache:
        cached = get_cached_response(cache_key)
        if cached is not None:
            return cached
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", config.SYSTEM_PROMPT),
        ("user", config.USER_PROMPT_TEMPLATE)
    ])
    
    # Make API call
    chain = prompt | llm
    response = chain.invoke({
        "title": title,
        "text": text
    })
    
    # Parse response
    response_text = response.content.strip().upper()
    
    # Extract prediction (1 = hyperpartisan, 0 = mainstream)
    if "HYPERPARTISAN" in response_text:
        prediction = 1
    elif "MAINSTREAM" in response_text:
        prediction = 0
    else:
        # Default to mainstream if unclear
        print(f"Warning: Unclear response for article {article_id}: '{response_text}'")
        prediction = 0
    
    result = {
        "article_id": article_id,
        "prediction": prediction,
        "true_label": true_label,
        "response_text": response_text,
        "title": title,
        "text_preview": text[:200]
    }
    
    # Cache the result
    if use_cache:
        save_cached_response(cache_key, result)
    
    return result


def calculate_metrics(predictions: list, true_labels: list) -> Dict[str, float]:
    """Calculate classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def get_saved_metrics(split_name: str) -> Optional[Dict[str, float]]:
    """
    Load previously saved evaluation metrics for a split.
    
    Args:
        split_name: Name of the split ('test_byarticle' or 'test_bypublisher')
    
    Returns:
        Dictionary with metrics if saved results exist, None otherwise
    """
    results_path = config.RESPONSE_CACHE_DIR / f"results_{split_name}.json"
    
    if not results_path.exists():
        return None
    
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        return data.get('metrics')
    except Exception:
        return None


def batch_classify(
    articles: list,
    max_articles: Optional[int] = None,
    use_cache: bool = True,
    rate_limit_delay: float = 0.5
) -> Dict[str, Any]:
    """
    Classify a batch of articles with rate limiting.
    
    Args:
        articles: List of article dictionaries
        max_articles: Maximum number of articles to process (None for all)
        use_cache: Whether to use cached responses
        rate_limit_delay: Delay between API calls in seconds
    
    Returns:
        Dictionary with results and metrics
    """
    llm = setup_llm()
    
    # Limit number of articles if specified
    if max_articles is not None:
        articles = articles[:max_articles]
        print(f"Processing first {len(articles)} articles (limited by MAX_ARTICLES_PER_SPLIT)")
    
    results = []
    predictions = []
    true_labels = []
    
    print(f"Classifying {len(articles)} articles...")
    
    for i, article in enumerate(articles, 1):
        try:
            result = classify_article(article, llm, use_cache=use_cache)
            results.append(result)
            predictions.append(result['prediction'])
            true_labels.append(result['true_label'])
            
            # Progress update
            if i % 10 == 0 or i == len(articles):
                print(f"  Progress: {i}/{len(articles)} articles processed")
            
            # Rate limiting (skip if using cache)
            if not use_cache or get_cached_response(
                get_cache_key(article['id'], article.get('title', ''), article['text'][:2000])
            ) is None:
                time.sleep(rate_limit_delay)
                
        except Exception as e:
            print(f"Error processing article {article.get('id', 'unknown')}: {e}")
            # Default to mainstream on error
            results.append({
                "article_id": article.get('id', 'unknown'),
                "prediction": 0,
                "true_label": article.get('label', None),
                "error": str(e)
            })
            predictions.append(0)
            true_labels.append(article.get('label', 0))
    
    metrics = calculate_metrics(predictions, true_labels)
    
    return {
        "results": results,
        "metrics": metrics,
        "num_articles": len(articles)
    }

"""Evaluation script for ChatGPT-based bias detection model."""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess import load_cached_data
import chatgpt_bias.config as config
from chatgpt_bias.utils import batch_classify, calculate_metrics


def get_results_path(split_name: str):
    """Get path for saved results file."""
    return config.RESPONSE_CACHE_DIR / f"results_{split_name}.json"


def save_results(split_name: str, result: dict, max_articles: int):
    """Save evaluation results to JSON file."""
    results_path = get_results_path(split_name)
    
    save_data = {
        "split": split_name,
        "max_articles": max_articles,
        "num_articles_evaluated": result['num_articles'],
        "metrics": result['metrics'],
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Results saved to: {results_path}")


def load_results(split_name: str):
    """Load previously saved results."""
    results_path = get_results_path(split_name)
    
    if not results_path.exists():
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)


def display_results(split_name: str, metrics: dict, num_articles: int, from_cache: bool = False):
    """Display evaluation results."""
    cache_indicator = " (LOADED FROM SAVED RESULTS)" if from_cache else ""
    
    print(f"\n{'='*60}")
    print(f"ChatGPT Bias Detection - {split_name.upper()}{cache_indicator}")
    print(f"{'='*60}")
    print(f"Articles evaluated: {num_articles}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"{'='*60}\n")


def evaluate_on_split(split_name: str, max_articles: int = None, use_saved: bool = False):
    """
    Evaluate ChatGPT model on a specific data split.
    
    Args:
        split_name: Name of the split ('test_byarticle' or 'test_bypublisher')
        max_articles: Maximum number of articles to evaluate
        use_saved: If True, load and display previously saved results instead of re-running
    """
    # Try to load saved results if requested
    if use_saved:
        saved = load_results(split_name)
        if saved:
            print(f"\nLoading saved results for {split_name}...")
            display_results(split_name, saved['metrics'], saved['num_articles_evaluated'], from_cache=True)
            return saved['metrics']
        else:
            print(f"\nNo saved results found for {split_name}, running evaluation...")
    
    # Run new evaluation
    print(f"\nLoading {split_name} data...")
    data = load_cached_data(split_name)
    print(f"Loaded {len(data)} articles\n")
    
    # Apply max articles limit from config if not specified
    if max_articles is None:
        max_articles = config.MAX_ARTICLES_PER_SPLIT
    
    # Classify articles
    print(f"Running ChatGPT evaluation on {split_name}...")
    result = batch_classify(
        articles=data,
        max_articles=max_articles,
        use_cache=True,
        rate_limit_delay=0.5
    )
    
    # Display and save results
    display_results(split_name, result['metrics'], result['num_articles'], from_cache=False)
    save_results(split_name, result, max_articles)
    
    return result['metrics']


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ChatGPT-based political bias detection model"
    )
    parser.add_argument(
        "--split",
        choices=["test_byarticle", "test_bypublisher", "both"],
        default="both",
        help="Which test split to evaluate on"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help=f"Maximum number of articles to evaluate (default: {config.MAX_ARTICLES_PER_SPLIT})"
    )
    parser.add_argument(
        "--use-saved",
        action="store_true",
        help="Load and display previously saved results instead of re-running evaluation"
    )
    
    args = parser.parse_args()
    
    try:
        if args.split == "both":
            if not args.use_saved:
                print("Evaluating on both test splits...\n")
            evaluate_on_split("test_byarticle", args.max_articles, args.use_saved)
            evaluate_on_split("test_bypublisher", args.max_articles, args.use_saved)
        else:
            evaluate_on_split(args.split, args.max_articles, args.use_saved)
    
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

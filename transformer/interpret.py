import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import DistilBertModel, DistilBertTokenizerFast
from captum.attr import LayerIntegratedGradients

import transformer.config as config
from device import get_device
from preprocess import load_cached_data


class InterpretableClassifier(nn.Module):

    def __init__(self, classifier_state_dict, num_extra_features=0, hidden_dim=256):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(config.TRANSFORMER_MODEL)

        # Create a text-only classifier (no extra features)
        self.fc1 = nn.Linear(768, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Load weights, truncating fc1 if extra features were used during training
        trained_fc1_weight = classifier_state_dict["fc1.weight"]
        if trained_fc1_weight.shape[1] > 768:
            self.fc1.weight.data = trained_fc1_weight[:, :768]
        else:
            self.fc1.weight.data = trained_fc1_weight

        self.fc1.bias.data = classifier_state_dict["fc1.bias"]
        self.bn1.weight.data = classifier_state_dict["bn1.weight"]
        self.bn1.bias.data = classifier_state_dict["bn1.bias"]
        self.bn1.running_mean = classifier_state_dict["bn1.running_mean"]
        self.bn1.running_var = classifier_state_dict["bn1.running_var"]
        self.fc2.weight.data = classifier_state_dict["fc2.weight"]
        self.fc2.bias.data = classifier_state_dict["fc2.bias"]

        # Freeze parameters for inference
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS] token

        # MLP classifier head
        x = self.fc1(cls_embedding)
        x = self.bn1(x)
        x = torch.relu(x)
        logits = self.fc2(x)
        return logits

    def predict_proba(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits).squeeze(-1)


class TokenAttributor:

    def __init__(self, model, tokenizer, device, n_steps=50):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.n_steps = n_steps

        # Initialize Layer Integrated Gradients on word embeddings
        self.lig = LayerIntegratedGradients(
            self._forward_func, self.model.bert.embeddings.word_embeddings
        )

    def _forward_func(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def compute_attributions(self, text, target_class=1):
        # Tokenize input
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get model prediction
        with torch.no_grad():
            prediction = self.model.predict_proba(input_ids, attention_mask).item()

        # Baseline: padding tokens with CLS/SEP preserved
        baseline_ids = torch.zeros_like(input_ids)
        baseline_ids[:, 0] = self.tokenizer.cls_token_id
        baseline_ids[:, -1] = self.tokenizer.sep_token_id

        # Compute attributions w.r.t. logit output
        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=True,
            n_steps=self.n_steps,
        )

        # Sum attribution over embedding dimensions to get per-token scores
        token_attributions = attributions.sum(dim=-1).squeeze(0)

        # Negate for mainstream attribution
        if target_class == 0:
            token_attributions = -token_attributions

        # Get token strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

        return tokens, token_attributions.detach().cpu(), prediction


def merge_wordpieces(tokens, scores):
    merged_tokens = []
    merged_scores = []

    current_token = ""
    current_score = 0.0
    scores_list = scores.tolist()

    for token, score in zip(tokens, scores_list):
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            if current_token:
                merged_tokens.append(current_token)
                merged_scores.append(current_score)
                current_token = ""
                current_score = 0.0
            continue

        if token.startswith("##"):
            # Continuation of previous word
            current_token += token[2:]
            current_score += score
        else:
            # New word - save previous if exists
            if current_token:
                merged_tokens.append(current_token)
                merged_scores.append(current_score)
            current_token = token
            current_score = score

    # Append final token
    if current_token:
        merged_tokens.append(current_token)
        merged_scores.append(current_score)

    return merged_tokens, torch.tensor(merged_scores)


def normalize_scores(scores):
    max_abs = scores.abs().max()
    if max_abs > 0:
        return scores / max_abs
    return scores


def plot_text_heatmap(
    tokens,
    scores,
    title="Token Attribution Heatmap",
    max_tokens_per_line=15,
    figsize=(14, None),
):
    # Split into lines for better readability
    lines = []
    current_line_tokens = []
    current_line_scores = []

    for token, score in zip(tokens, scores.tolist()):
        current_line_tokens.append(token)
        current_line_scores.append(score)

        if len(current_line_tokens) >= max_tokens_per_line:
            lines.append((current_line_tokens, current_line_scores))
            current_line_tokens = []
            current_line_scores = []

    if current_line_tokens:
        lines.append((current_line_tokens, current_line_scores))

    # Calculate figure height based on number of lines
    n_lines = len(lines)
    height = figsize[1] if figsize[1] else max(2, 0.6 * n_lines + 1.5)

    fig, ax = plt.subplots(figsize=(figsize[0], height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # RdBu_r colormap: red = hyperpartisan, blue = mainstream
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    y_position = 0.9
    line_height = 0.8 / max(n_lines, 1)

    for line_tokens, line_scores in lines:
        x_position = 0.02

        for token, score in zip(line_tokens, line_scores):
            # Get background color based on score
            color = cmap(norm(score))

            # Determine text color for readability
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "white" if luminance < 0.5 else "black"

            # Add text with colored background
            text = ax.text(
                x_position,
                y_position,
                f" {token} ",
                fontsize=10,
                fontfamily="monospace",
                verticalalignment="center",
                bbox=dict(
                    facecolor=color, edgecolor="none", pad=2, boxstyle="round,pad=0.1"
                ),
                color=text_color,
            )

            # Get text width for positioning next token
            fig.canvas.draw()
            bbox = text.get_window_extent()
            display_to_data = ax.transData.inverted()
            bbox_data = display_to_data.transform(bbox)
            token_width = bbox_data[1, 0] - bbox_data[0, 0]
            x_position += token_width + 0.005

        y_position -= line_height

    # Add title
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, aspect=40
    )
    cbar.set_label("← Mainstream | Hyperpartisan →", fontsize=9)
    cbar.set_ticks([-1, 0, 1])

    plt.tight_layout()
    return fig


def generate_html_heatmap(tokens, scores, title="Token Attribution"):
    def score_to_color(score):
        if score > 0:
            # Red spectrum for positive (hyperpartisan)
            intensity = min(abs(score), 1.0)
            r = 255
            g = int(255 * (1 - intensity * 0.7))
            b = int(255 * (1 - intensity * 0.8))
        else:
            # Blue spectrum for negative (mainstream)
            intensity = min(abs(score), 1.0)
            r = int(255 * (1 - intensity * 0.8))
            g = int(255 * (1 - intensity * 0.6))
            b = 255
        return f"rgb({r},{g},{b})"

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; max-width: 1000px; margin: auto; }",
        "h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }",
        ".token { display: inline-block; padding: 4px 6px; margin: 2px; border-radius: 4px; cursor: pointer; }",
        ".token:hover { outline: 2px solid #333; }",
        ".legend { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px; }",
        ".legend-item { display: inline-block; margin-right: 20px; }",
        ".legend-color { display: inline-block; width: 20px; height: 20px; border-radius: 4px; vertical-align: middle; margin-right: 5px; }",
        "p.note { color: #666; font-style: italic; }",
        "</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        "<div class='text'>",
    ]

    scores_list = scores.tolist()
    for token, score in zip(tokens, scores_list):
        color = score_to_color(score)
        text_color = "#000" if abs(score) < 0.5 else "#fff" if score > 0 else "#fff"
        html_parts.append(
            f'<span class="token" style="background-color: {color}; color: {text_color};" '
            f'title="Score: {score:.3f}">{token}</span>'
        )

    html_parts.extend(
        [
            "</div>",
            "<div class='legend'>",
            "<div class='legend-item'><span class='legend-color' style='background: rgb(255,77,77);'></span> Hyperpartisan (+)</div>",
            "<div class='legend-item'><span class='legend-color' style='background: rgb(255,255,255); border: 1px solid #ccc;'></span> Neutral (0)</div>",
            "<div class='legend-item'><span class='legend-color' style='background: rgb(77,77,255);'></span> Mainstream (-)</div>",
            "</div>",
            "<p class='note'>Hover over tokens to see exact attribution scores.</p>",
            "</body></html>",
        ]
    )

    return "\n".join(html_parts)


def load_model(fold=None):
    # Load ensemble info to find best fold
    ensemble_path = config.CACHE_DIR / "transformer_ensemble_info.pt"
    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"Ensemble info not found at {ensemble_path}. Run transformer.train first."
        )

    ensemble_info = torch.load(ensemble_path, map_location="cpu", weights_only=True)

    if fold is None:
        top_indices = ensemble_info["top_indices"]
        fold = top_indices[-1]
        print(f"Fold {fold} (F1={ensemble_info['fold_scores'][fold]:.4f})")

    # Load the fold checkpoint
    checkpoint_path = config.CACHE_DIR / f"transformer_model_fold_{fold}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model = InterpretableClassifier(
        classifier_state_dict=checkpoint["model_state_dict"],
        num_extra_features=0,
    )

    return model


def load_article_by_id(article_id, split="test_byarticle"):
    data = load_cached_data(split)
    for article in data:
        if article["id"] == article_id:
            return article["text"], article.get("label")
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Generate token-level attribution heatmaps for hyperpartisan classification"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--article-id", type=str, help="Article ID from dataset")
    group.add_argument("--text", type=str, help="Raw text to analyze")

    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--format",
        choices=["matplotlib", "html", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific model fold to use (default: best)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=50, help="Number of integration steps for IG"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "cuda"],
        help="Override device (default: auto-detect)",
    )

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device(config.DEVICE)
    print(f"Device: {device}\n")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Loading model")
    model = load_model(fold=args.fold)
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.TRANSFORMER_MODEL)

    # Get input text
    if args.article_id:
        print(f"Loading article: {args.article_id}")
        text, label = load_article_by_id(args.article_id)
        if text is None:
            print(f"Article {args.article_id} not found in dataset")
            sys.exit(1)
        label_str = (
            "hyperpartisan" if label == 1 else "mainstream" if label == 0 else "unknown"
        )
        print(f"Ground truth: {label_str}")
        output_name = f"attribution_{args.article_id}"
    else:
        text = args.text
        output_name = "attribution_custom"

    preview = text[:200] + "..." if len(text) > 200 else text
    print(f"Input: {preview}")

    print(f"\nComputing attributions (n_steps={args.n_steps})")
    attributor = TokenAttributor(model, tokenizer, device, n_steps=args.n_steps)
    tokens, scores, prediction = attributor.compute_attributions(text, target_class=1)

    print(
        f"Prediction: {prediction:.3f} ({'hyperpartisan' if prediction > 0.5 else 'mainstream'})"
    )

    merged_tokens, merged_scores = merge_wordpieces(tokens, scores)
    normalized_scores = normalize_scores(merged_scores)

    print(f"Tokens: {len(merged_tokens)}")

    # Generate visualizations
    title = f"Token Attribution (P(hyperpartisan)={prediction:.2f})"

    if args.format in ["matplotlib", "both"]:
        fig = plot_text_heatmap(merged_tokens, normalized_scores, title=title)
        png_path = output_dir / f"{output_name}.png"
        pdf_path = output_dir / f"{output_name}.pdf"
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
        plt.close(fig)

    if args.format in ["html", "both"]:
        html = generate_html_heatmap(merged_tokens, normalized_scores, title=title)
        html_path = output_dir / f"{output_name}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved: {html_path}")

    print("\nTop tokens (hyperpartisan):")
    sorted_indices = torch.argsort(normalized_scores, descending=True)
    for i in sorted_indices[:10]:
        print(f"  {merged_tokens[i]:20s} {normalized_scores[i].item():+.3f}")

    print("\nTop tokens (mainstream):")
    for i in sorted_indices[-10:]:
        print(f"  {merged_tokens[i]:20s} {normalized_scores[i].item():+.3f}")


if __name__ == "__main__":
    main()

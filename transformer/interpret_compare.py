"""
Multi-method token attribution comparison for hyperpartisan classification.

Implements multiple interpretability methods for DistilBERT + MLP classifier:
- Integrated Gradients
- Integrated Gradients × Input
- Gradient
- Gradient × Input
- LIME (Local Interpretable Model-agnostic Explanations)
- Partition SHAP

IMPORTANT LIMITATION:
    The transformer model was trained with 14 extra features (date, hyperlinks,
    sentiment, NER) in addition to the 768-dim DistilBERT embedding. For token-level
    attribution, we use ONLY the text portion of the model (truncated fc1 weights).
    
    This means:
    - Predictions may differ from the full model
    - Attribution scores show which tokens influence the TEXT-ONLY decision
    - The full model's accuracy relies on extra features not attributable to tokens
    
    For accurate predictions, use the full model pipeline in transformer/train.py.
    This module is for understanding which tokens the model attends to.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from transformers import DistilBertModel, DistilBertTokenizerFast
from captum.attr import (
    LayerIntegratedGradients,
    LayerGradientXActivation,
    Saliency,
    InputXGradient,
)

import transformer.config as config


# =============================================================================
# Model Wrapper for Interpretability
# =============================================================================

class InterpretableClassifier(nn.Module):
    """
    DistilBERT + MLP classifier wrapped for interpretability.
    
    Loads trained MLP weights and truncates fc1 if extra features were used
    during training (since we only use text for attribution).
    """

    def __init__(self, classifier_state_dict: dict, hidden_dim: int = 256):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(config.TRANSFORMER_MODEL)
        
        # Text-only classifier (no extra features for attribution)
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
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        
        x = self.fc1(cls_embedding)
        x = self.bn1(x)
        x = torch.relu(x)
        logits = self.fc2(x)
        return logits

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return probability of hyperpartisan class."""
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits).squeeze(-1)


# =============================================================================
# Multi-Method Attributor
# =============================================================================

class MultiMethodAttributor:
    """
    Compute token-level attributions using multiple interpretability methods.
    
    Supports:
        - Integrated Gradients (IG)
        - Integrated Gradients × Input
        - Gradient (Saliency)
        - Gradient × Input
        - LIME
        - Partition SHAP
    """

    METHODS = [
        "Integrated Gradients",
        "Integrated Gradients × Input",
        "Gradient",
        "Gradient × Input",
        "LIME",
        "Partition SHAP",
    ]

    def __init__(
        self,
        model: InterpretableClassifier,
        tokenizer: DistilBertTokenizerFast,
        device: torch.device,
        n_steps: int = 50,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.n_steps = n_steps

        # Captum attributors
        self.lig = LayerIntegratedGradients(
            self._forward_func,
            self.model.bert.embeddings.word_embeddings
        )
        self.lg_x_act = LayerGradientXActivation(
            self._forward_func,
            self.model.bert.embeddings.word_embeddings
        )

    def _forward_func(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward function for Captum attributors."""
        return self.model(input_ids, attention_mask)

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Tokenize text and return tensors + token strings."""
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        return input_ids, attention_mask, tokens

    def _create_baseline(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create baseline (padding tokens with CLS/SEP preserved)."""
        baseline_ids = torch.zeros_like(input_ids)
        baseline_ids[:, 0] = self.tokenizer.cls_token_id
        seq_len = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
        for i, length in enumerate(seq_len):
            baseline_ids[i, length - 1] = self.tokenizer.sep_token_id
        return baseline_ids

    # -------------------------------------------------------------------------
    # Captum-based Methods
    # -------------------------------------------------------------------------

    def integrated_gradients(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        baseline_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Integrated Gradients attribution."""
        attributions, _ = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=True,
            n_steps=self.n_steps,
        )
        # Sum over embedding dimension
        return attributions.sum(dim=-1).squeeze(0)

    def integrated_gradients_x_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        baseline_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Integrated Gradients × Input (element-wise product)."""
        attributions, _ = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=True,
            n_steps=self.n_steps,
        )
        # Get input embeddings
        embeddings = self.model.bert.embeddings.word_embeddings(input_ids)
        # Element-wise multiplication
        attr_x_input = attributions * embeddings
        return attr_x_input.sum(dim=-1).squeeze(0)

    def gradient(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute simple gradient (saliency) attribution."""
        # Enable gradients temporarily
        embeddings = self.model.bert.embeddings.word_embeddings(input_ids)
        embeddings = embeddings.clone().detach().requires_grad_(True)
        
        # Forward through rest of model
        outputs = self.model.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        x = self.model.fc1(cls_embedding)
        x = self.model.bn1(x)
        x = torch.relu(x)
        logits = self.model.fc2(x)
        
        # Backward
        logits.backward()
        
        # Sum over embedding dimension
        return embeddings.grad.sum(dim=-1).squeeze(0)

    def gradient_x_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gradient × Input attribution."""
        embeddings = self.model.bert.embeddings.word_embeddings(input_ids)
        embeddings_detached = embeddings.clone().detach().requires_grad_(True)
        
        outputs = self.model.bert(inputs_embeds=embeddings_detached, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        x = self.model.fc1(cls_embedding)
        x = self.model.bn1(x)
        x = torch.relu(x)
        logits = self.model.fc2(x)
        
        logits.backward()
        
        # Gradient × Input
        attr = embeddings_detached.grad * embeddings_detached
        return attr.sum(dim=-1).squeeze(0)

    # -------------------------------------------------------------------------
    # LIME
    # -------------------------------------------------------------------------

    def lime_attributions(
        self,
        text: str,
        num_samples: int = 500,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Compute LIME attributions.
        
        Returns tokens and their attribution scores.
        """
        from lime.lime_text import LimeTextExplainer
        
        explainer = LimeTextExplainer(class_names=["mainstream", "hyperpartisan"])
        
        def predict_fn(texts: List[str]) -> np.ndarray:
            """Prediction function for LIME."""
            probs = []
            for t in texts:
                encoding = self.tokenizer(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                with torch.no_grad():
                    prob = self.model.predict_proba(input_ids, attention_mask).item()
                probs.append([1 - prob, prob])
            return np.array(probs)
        
        exp = explainer.explain_instance(
            text,
            predict_fn,
            num_features=100,
            num_samples=num_samples,
        )
        
        # Extract word-level attributions
        word_scores = dict(exp.as_list())
        words = text.split()
        scores = np.array([word_scores.get(w, 0.0) for w in words])
        
        return words, scores

    # -------------------------------------------------------------------------
    # Partition SHAP
    # -------------------------------------------------------------------------

    def partition_shap(
        self,
        text: str,
        max_evals: int = 500,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Compute Partition SHAP attributions.
        
        Returns tokens and their SHAP values.
        """
        import shap
        
        def predict_fn(texts: List[str]) -> np.ndarray:
            """Prediction function for SHAP."""
            probs = []
            for t in texts:
                if not t.strip():
                    probs.append([0.5, 0.5])
                    continue
                encoding = self.tokenizer(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                with torch.no_grad():
                    prob = self.model.predict_proba(input_ids, attention_mask).item()
                probs.append([1 - prob, prob])
            return np.array(probs)
        
        # Use partition explainer for hierarchical text
        masker = shap.maskers.Text(self.tokenizer)
        explainer = shap.Explainer(predict_fn, masker, output_names=["mainstream", "hyperpartisan"])
        
        shap_values = explainer([text], max_evals=max_evals)
        
        # Extract values for hyperpartisan class
        tokens = shap_values.data[0]
        values = shap_values.values[0][:, 1]  # hyperpartisan class
        
        return list(tokens), values

    # -------------------------------------------------------------------------
    # Unified Comparison
    # -------------------------------------------------------------------------

    def compute_all(
        self,
        text: str,
        methods: Optional[List[str]] = None,
        lime_samples: int = 500,
        shap_evals: int = 500,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Compute attributions using all specified methods.
        
        Args:
            text: Input text to analyze
            methods: List of method names to run (default: all)
            lime_samples: Number of LIME perturbation samples
            shap_evals: Max SHAP evaluations
        
        Returns:
            DataFrame with methods as rows and tokens as columns,
            plus the model's prediction probability.
        """
        if methods is None:
            methods = self.METHODS
        
        # Tokenize for Captum methods
        input_ids, attention_mask, tokens = self._tokenize(text)
        baseline_ids = self._create_baseline(input_ids)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model.predict_proba(input_ids, attention_mask).item()
        
        # Filter out special tokens for display
        valid_indices = [
            i for i, t in enumerate(tokens)
            if t not in ["[CLS]", "[SEP]", "[PAD]"]
        ]
        display_tokens = [tokens[i] for i in valid_indices]
        
        results = {}
        
        # Captum-based methods (token-level)
        if "Integrated Gradients" in methods:
            attr = self.integrated_gradients(input_ids, attention_mask, baseline_ids)
            scores = self._normalize(attr.detach().cpu().numpy())
            results["Integrated Gradients"] = [scores[i] for i in valid_indices]
        
        if "Integrated Gradients × Input" in methods:
            attr = self.integrated_gradients_x_input(input_ids, attention_mask, baseline_ids)
            scores = self._normalize(attr.detach().cpu().numpy())
            results["Integrated Gradients × Input"] = [scores[i] for i in valid_indices]
        
        if "Gradient" in methods:
            attr = self.gradient(input_ids, attention_mask)
            scores = self._normalize(attr.detach().cpu().numpy())
            results["Gradient"] = [scores[i] for i in valid_indices]
        
        if "Gradient × Input" in methods:
            attr = self.gradient_x_input(input_ids, attention_mask)
            scores = self._normalize(attr.detach().cpu().numpy())
            results["Gradient × Input"] = [scores[i] for i in valid_indices]
        
        # LIME (word-level, needs alignment)
        if "LIME" in methods:
            try:
                lime_words, lime_scores = self.lime_attributions(text, num_samples=lime_samples)
                lime_aligned = self._align_word_to_tokens(lime_words, lime_scores, display_tokens)
                results["LIME"] = self._normalize(lime_aligned)
            except Exception as e:
                print(f"LIME failed: {e}")
                results["LIME"] = [0.0] * len(display_tokens)
        
        # Partition SHAP
        if "Partition SHAP" in methods:
            try:
                shap_tokens, shap_values = self.partition_shap(text, max_evals=shap_evals)
                shap_aligned = self._align_shap_to_tokens(shap_tokens, shap_values, display_tokens)
                results["Partition SHAP"] = self._normalize(shap_aligned)
            except Exception as e:
                print(f"SHAP failed: {e}")
                results["Partition SHAP"] = [0.0] * len(display_tokens)
        
        # Create DataFrame
        df = pd.DataFrame(results, index=display_tokens).T
        df.columns = display_tokens
        
        return df, prediction

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [-1, 1] range."""
        max_abs = np.abs(scores).max()
        if max_abs > 0:
            return scores / max_abs
        return scores

    def _align_word_to_tokens(
        self,
        words: List[str],
        word_scores: np.ndarray,
        tokens: List[str],
    ) -> np.ndarray:
        """Align word-level scores (from LIME) to wordpiece tokens."""
        word_idx = 0
        token_scores = []
        word_score_map = {w.lower(): s for w, s in zip(words, word_scores)}
        
        for token in tokens:
            clean_token = token.replace("##", "").lower()
            # Find matching word
            score = word_score_map.get(clean_token, 0.0)
            token_scores.append(score)
        
        return np.array(token_scores)

    def _align_shap_to_tokens(
        self,
        shap_tokens: List[str],
        shap_values: np.ndarray,
        bert_tokens: List[str],
    ) -> np.ndarray:
        """Align SHAP tokens to BERT wordpiece tokens."""
        # Simple matching by position after cleaning
        aligned = []
        shap_idx = 0
        
        for bert_token in bert_tokens:
            if shap_idx < len(shap_values):
                aligned.append(shap_values[shap_idx])
                # Advance SHAP index for non-continuation tokens
                if not bert_token.startswith("##"):
                    shap_idx += 1
            else:
                aligned.append(0.0)
        
        return np.array(aligned)


# =============================================================================
# Visualization
# =============================================================================

def style_comparison_table(df: pd.DataFrame):
    """
    Style DataFrame as a heatmap similar to the reference image.
    
    Red = positive (hyperpartisan), Blue = negative (mainstream).
    """
    def color_cell(val: float) -> str:
        """Return background color based on value."""
        if pd.isna(val):
            return ""
        
        val = float(val)
        if val > 0:
            # Red spectrum
            intensity = min(abs(val), 1.0)
            r = 220
            g = int(220 * (1 - intensity * 0.6))
            b = int(220 * (1 - intensity * 0.6))
        else:
            # Blue spectrum
            intensity = min(abs(val), 1.0)
            r = int(220 * (1 - intensity * 0.6))
            g = int(220 * (1 - intensity * 0.6))
            b = 220
        
        return f"background-color: rgb({r},{g},{b})"
    
    styled = df.style.applymap(color_cell)
    styled = styled.format("{:.2f}")
    styled = styled.set_properties(**{
        "text-align": "center",
        "font-size": "11px",
        "border": "1px solid #ddd",
    })
    styled = styled.set_table_styles([
        {"selector": "th", "props": [("background-color", "#f5f5f5"), ("font-weight", "bold")]},
        {"selector": "td", "props": [("padding", "5px 10px")]},
    ])
    
    return styled


def plot_comparison_heatmaps(
    df: pd.DataFrame,
    prediction: float,
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Plot comparison heatmaps for all methods.
    
    Args:
        df: DataFrame from compute_all()
        prediction: Model prediction probability
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(len(df), 1, figsize=figsize, sharex=True)
    if len(df) == 1:
        axes = [axes]
    
    tokens = list(df.columns)
    max_tokens = min(30, len(tokens))  # Limit for readability
    
    for ax, (method_name, row) in zip(axes, df.iterrows()):
        values = row.values[:max_tokens].reshape(1, -1)
        
        sns.heatmap(
            values,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            cbar=False,
            xticklabels=tokens[:max_tokens],
            yticklabels=[method_name],
            annot=False,
        )
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
    
    axes[-1].tick_params(axis="x", rotation=45, labelsize=9)
    
    plt.suptitle(f"Token Attribution Comparison (P(hyperpartisan)={prediction:.2f})", fontsize=12)
    plt.tight_layout()
    
    return fig


# =============================================================================
# Model Loading Utilities
# =============================================================================

def get_device(override: Optional[str] = None) -> torch.device:
    """Get compute device."""
    if override:
        return torch.device(override)
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(fold: Optional[int] = None) -> InterpretableClassifier:
    """
    Load trained model from checkpoint.
    
    Args:
        fold: Specific fold to load (default: best performing fold)
    
    Returns:
        InterpretableClassifier instance
    """
    ensemble_path = config.CACHE_DIR / "transformer_ensemble_info.pt"
    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"Ensemble info not found at {ensemble_path}. Run transformer.train first."
        )
    
    ensemble_info = torch.load(ensemble_path, map_location="cpu", weights_only=True)
    
    if fold is None:
        top_indices = ensemble_info["top_indices"]
        fold = top_indices[-1]
        print(f"Using fold {fold} (F1={ensemble_info['fold_scores'][fold]:.4f})")
    
    checkpoint_path = config.CACHE_DIR / f"transformer_model_fold_{fold}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Check if model was trained with extra features and warn user
    num_extra = checkpoint.get("num_extra_features", 0)
    if num_extra > 0:
        print(f"WARNING: Model was trained with {num_extra} extra features.")
        print("         Interpretability uses text-only (truncated weights).")
        print("         Predictions may differ from the full model.")
    
    model = InterpretableClassifier(
        classifier_state_dict=checkpoint["model_state_dict"],
    )
    
    return model


def load_tokenizer() -> DistilBertTokenizerFast:
    """Load DistilBERT tokenizer."""
    return DistilBertTokenizerFast.from_pretrained(config.TRANSFORMER_MODEL)

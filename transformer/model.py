import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer for adversarial training.
    
    During forward pass: identity function.
    During backward pass: negates gradients, preventing the encoder from
    learning features that predict the adversarial target (length).
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    """Apply gradient reversal with configurable strength."""
    
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class TransformerClassifier(nn.Module):
    """MLP classifier on pre-computed transformer embeddings."""

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.5, num_extra_features=0):
        super().__init__()
        self.num_extra_features = num_extra_features
        self.fc1 = nn.Linear(input_dim + num_extra_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, features=None):
        if features is not None and self.num_extra_features > 0:
            x = torch.cat([x, features], dim=1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(1)


class DebiasedTransformerClassifier(nn.Module):
    """
    Transformer classifier with adversarial length debiasing.
    
    Uses gradient reversal to prevent the model from learning features
    that predict article length, forcing it to focus on semantic content.
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        dropout=0.5,
        num_extra_features=0,
        gradient_reversal_lambda=0.1,
    ):
        super().__init__()
        self.num_extra_features = num_extra_features
        
        # Main classifier
        self.fc1 = nn.Linear(input_dim + num_extra_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Adversarial length predictor (with gradient reversal)
        self.gradient_reversal = GradientReversal(gradient_reversal_lambda)
        self.length_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output normalized length (0-1)
        )

    def forward(self, x, features=None, return_length_pred=False):
        if features is not None and self.num_extra_features > 0:
            x = torch.cat([x, features], dim=1)

        # Shared hidden representation
        hidden = self.fc1(x)
        hidden = self.bn1(hidden)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        
        # Main classification output
        logits = self.fc2(hidden)
        class_output = torch.sigmoid(logits).squeeze(1)
        
        if return_length_pred:
            # Gradient reversal: prevents hidden from encoding length
            reversed_hidden = self.gradient_reversal(hidden)
            length_pred = self.length_predictor(reversed_hidden).squeeze(1)
            return class_output, length_pred
        
        return class_output

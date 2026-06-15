"""
src/training/losses.py
Focal Loss for multi-label classification.

Why Focal Loss?
  - Mental health dataset is heavily imbalanced (suicidal = ~8% of data)
  - Focal Loss down-weights easy examples and focuses on hard ones
  - Better than plain BCE for rare-class detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for multi-label classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    alpha: balances positive/negative examples
    gamma: focuses on hard examples (gamma=0 → standard BCE)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE loss (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # Compute p_t
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE Loss — simpler alternative to Focal Loss.
    Weights each label by inverse frequency.
    """

    def __init__(self, pos_weights: torch.Tensor = None):
        super().__init__()
        self.pos_weights = pos_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights, reduction="mean"
        )


def compute_class_weights(label_counts: dict, total: int, num_labels: int = 4, labels: list = None):
    """
    Compute positive class weights for imbalanced dataset.
    weight = (total - count) / count  (inverse frequency)
    """
    if labels is None:
        labels = ["depression", "anxiety", "suicidal", "stress"]
    
    weights = []
    for label in labels:
        count = label_counts.get(label, 1)
        weight = (total - count) / max(count, 1)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float)

"""
src/evaluation/metrics.py
Metrics for multi-label classification:
  - Per-label F1, Precision, Recall
  - Macro-averaged F1
  - AUC-ROC per label
  - Hamming Loss
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, hamming_loss, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list,
    thresholds: list = None,
) -> dict:
    """
    Args:
        y_true:     (N, num_labels) binary ground truth
        y_prob:     (N, num_labels) sigmoid probabilities
        labels:     list of label names
        thresholds: per-label decision thresholds (default 0.5 each)
    """
    if thresholds is None:
        thresholds = [0.5] * len(labels)

    # Apply thresholds
    y_pred = np.zeros_like(y_prob, dtype=int)
    for i, thresh in enumerate(thresholds):
        y_pred[:, i] = (y_prob[:, i] >= thresh).astype(int)

    metrics = {}

    # Per-label metrics
    for i, label in enumerate(labels):
        metrics[f"{label}_f1"]        = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[f"{label}_precision"] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[f"{label}_recall"]    = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        try:
            metrics[f"{label}_auc"] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            metrics[f"{label}_auc"] = 0.0

    # Macro averages
    metrics["macro_f1"]        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_recall"]    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["hamming_loss"]    = hamming_loss(y_true, y_pred)

    try:
        metrics["macro_auc"] = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        metrics["macro_auc"] = 0.0

    return metrics


def print_report(y_true, y_prob, labels, thresholds=None):
    if thresholds is None:
        thresholds = [0.5] * len(labels)

    y_pred = np.zeros_like(y_prob, dtype=int)
    for i, thresh in enumerate(thresholds):
        y_pred[:, i] = (y_prob[:, i] >= thresh).astype(int)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    for i, label in enumerate(labels):
        print(f"\n[{label.upper()}]")
        print(classification_report(y_true[:, i], y_pred[:, i], target_names=["No", "Yes"]))


def plot_confusion_matrices(y_true, y_prob, labels, thresholds=None, save_dir="outputs/plots/"):
    if thresholds is None:
        thresholds = [0.5] * len(labels)

    y_pred = np.zeros_like(y_prob, dtype=int)
    for i, thresh in enumerate(thresholds):
        y_pred[:, i] = (y_prob[:, i] >= thresh).astype(int)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4))
    for i, (label, ax) in enumerate(zip(labels, axes)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_title(label.capitalize())
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    path = save_dir / "confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrices saved to {path}")


def plot_training_history(history: list, save_dir="outputs/plots/"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["loss"] for h in history]
    val_f1     = [h["macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train Loss", marker="o")
    ax1.plot(epochs, val_loss,   label="Val Loss",   marker="s")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1, label="Val Macro F1", marker="D", color="green")
    ax2.set_title("Validation Macro F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {path}")

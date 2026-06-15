#!/usr/bin/env python3
"""
scripts/evaluate.py
Evaluate a saved checkpoint on the test set.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
"""
import argparse, json, sys, yaml
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.models.mental_bert import load_model
from src.data.dataset import get_dataloaders
from src.evaluation.metrics import compute_metrics, print_report, plot_confusion_matrices

LABELS = ["depression", "anxiety", "suicidal", "stress"]
THRESHOLDS = [0.5, 0.5, 0.35, 0.5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/train_config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model, device = load_model(args.checkpoint, config["model"]["name"])

    _, _, test_loader, _ = get_dataloaders(
        train_path=config["data"]["processed_dir"] + "train.csv",
        val_path=config["data"]["processed_dir"] + "val.csv",
        test_path=config["data"]["processed_dir"] + "test.csv",
        model_name=config["model"]["name"],
        max_len=config["model"]["max_len"],
        batch_size=32,
    )

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_logits.append(torch.sigmoid(logits).cpu())
            all_labels.append(batch["labels"].cpu())

    y_prob = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()

    metrics = compute_metrics(y_true, y_prob, LABELS, THRESHOLDS)
    logger.success(f"Macro F1:  {metrics['macro_f1']:.4f}")
    logger.success(f"Macro AUC: {metrics['macro_auc']:.4f}")
    logger.success(f"Hamming:   {metrics['hamming_loss']:.4f}")

    print_report(y_true, y_prob, LABELS, THRESHOLDS)
    plot_confusion_matrices(y_true, y_prob, LABELS, THRESHOLDS)

    with open("outputs/metrics/eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Results saved to outputs/metrics/eval_results.json")


if __name__ == "__main__":
    main()

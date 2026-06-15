#!/usr/bin/env python3
"""
scripts/train.py
Entry point for training MindSignal.

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mental_bert import MindSignalModel, get_device
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MindSignal Mental Health Detector")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("🧠 MindSignal — Mental Health Signal Detector")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Model:  {config['model']['name']}")
    logger.info(f"Labels: {config['model']['labels']}")

    # Device
    device = get_device()
    config["_device"] = str(device)

    # DataLoaders
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        train_path=config["data"]["processed_dir"] + "train.csv",
        val_path=config["data"]["processed_dir"] + "val.csv",
        test_path=config["data"]["processed_dir"] + "test.csv",
        model_name=config["model"]["name"],
        max_len=config["model"]["max_len"],
        batch_size=config["training"]["batch_size"],
    )

    # Model
    logger.info("Initializing model...")
    model = MindSignalModel(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
    )

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)

    # Train
    history = trainer.train()

    # Final test evaluation
    logger.info("\n📊 Running final evaluation on TEST set...")
    from src.evaluation.metrics import print_report, plot_confusion_matrices, plot_training_history
    import numpy as np

    test_metrics = trainer.evaluate(test_loader, "test")
    logger.success(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    logger.success(f"Test Macro AUC: {test_metrics['macro_auc']:.4f}")

    # Save test metrics
    metrics_path = Path(config["output"]["metrics_dir"]) / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")

    # Plot training curves
    plot_training_history(history, save_dir=config["output"]["plots_dir"])

    logger.success("🎉 Training pipeline complete!")
    logger.info(f"Best model saved at: {config['output']['checkpoint_dir']}best_model.pt")


if __name__ == "__main__":
    main()

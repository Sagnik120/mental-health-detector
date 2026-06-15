"""
src/training/trainer.py
Production-grade training loop with:
  - Gradient clipping
  - LR scheduling (linear warmup)
  - Early stopping
  - Checkpoint saving
  - TensorBoard logging
  - Per-epoch metric tracking
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from loguru import logger
from tqdm import tqdm

from src.training.losses import FocalLoss, compute_class_weights
from src.evaluation.metrics import compute_metrics

LABELS = ["depression", "anxiety", "suicidal", "stress"]


class Trainer:
    def __init__(self, model, train_loader, val_loader, config: dict, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Steps
        total_steps = len(train_loader) * config["training"]["epochs"]
        warmup_steps = int(total_steps * config["training"]["warmup_ratio"])

        # LR Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Loss function
        if config["training"]["focal_loss"]["use"]:
            self.criterion = FocalLoss(
                alpha=config["training"]["focal_loss"]["alpha"],
                gamma=config["training"]["focal_loss"]["gamma"],
            )
            logger.info("Using Focal Loss")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            logger.info("Using BCEWithLogitsLoss")

        # Paths
        self.checkpoint_dir = Path(config["output"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = Path(config["output"]["metrics_dir"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        if config["logging"]["tensorboard"]:
            self.writer = SummaryWriter(log_dir=config["logging"]["log_dir"])
        else:
            self.writer = None

        self.best_val_f1 = 0.0
        self.history = []

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels         = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask, token_type_ids)
            loss = self.criterion(logits, labels)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["gradient_clip"]
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / steps

    @torch.no_grad()
    def evaluate(self, loader, split="val"):
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0
        steps = 0

        for batch in tqdm(loader, desc=f"  [{split}]", leave=False):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels         = batch["labels"].to(self.device)

            logits = self.model(input_ids, attention_mask, token_type_ids)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            steps += 1
            all_logits.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Use per-label thresholds
        thresholds = self.config["training"]["thresholds"]
        thresh_list = [thresholds[l] for l in LABELS]

        metrics = compute_metrics(all_labels, all_logits, LABELS, thresh_list)
        metrics["loss"] = total_loss / steps
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state, path)
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.success(f"  ✅ New best model saved (val F1={metrics['macro_f1']:.4f})")

    def train(self):
        epochs = self.config["training"]["epochs"]
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        logger.info(f"Steps per epoch: {len(self.train_loader)}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, "val")
            elapsed = time.time() - t0

            val_f1 = val_metrics["macro_f1"]
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1

            # Log to console
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
            for label in LABELS:
                logger.info(f"  {label}: F1={val_metrics[f'{label}_f1']:.3f}")

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                self.writer.add_scalar("F1/macro_val", val_f1, epoch)
                for label in LABELS:
                    self.writer.add_scalar(f"F1/{label}_val", val_metrics[f"{label}_f1"], epoch)

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics
            })

        # Save full training history
        with open(self.metrics_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        if self.writer:
            self.writer.close()

        logger.success(f"Training complete! Best Val F1: {self.best_val_f1:.4f}")
        return self.history

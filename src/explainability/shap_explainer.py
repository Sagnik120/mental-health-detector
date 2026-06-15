"""
src/explainability/shap_explainer.py
SHAP-based token-level explanations for Mental-BERT.

For each prediction, shows which words pushed the model
toward or away from each label — the "novelty" feature.
"""

import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer
from loguru import logger


class MindSignalExplainer:
    """
    Wraps the trained model to produce token-level SHAP explanations.
    Uses shap.Explainer with a text masker for transformers.
    """

    LABELS = ["depression", "anxiety", "suicidal", "stress"]

    def __init__(self, model, tokenizer: AutoTokenizer, device, max_len: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.model.eval()

    def _predict_fn(self, texts):
        """Wrapper function SHAP calls to get model outputs."""
        encodings = self.tokenizer(
            list(texts),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(
                encodings["input_ids"].to(self.device),
                encodings["attention_mask"].to(self.device),
            )
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs  # shape: (batch, 4)

    def explain(self, text: str, n_samples: int = 50):
        """
        Generate SHAP values for a single text.
        Returns shap_values (list of 4 arrays, one per label).
        """
        logger.info(f"Computing SHAP explanations (n_samples={n_samples})...")
        masker = shap.maskers.Text(self.tokenizer)
        explainer = shap.Explainer(self._predict_fn, masker, output_names=self.LABELS)
        shap_values = explainer([text], max_evals=n_samples)
        return shap_values

    def plot_explanation(self, text: str, label: str = "depression", save_path: str = None):
        """
        Plot token importance bar chart for a specific label.
        """
        assert label in self.LABELS, f"label must be one of {self.LABELS}"
        label_idx = self.LABELS.index(label)

        shap_values = self.explain(text)
        values = shap_values.values[0, :, label_idx]
        tokens = shap_values.data[0]

        # Filter padding tokens
        non_pad = [i for i, t in enumerate(tokens) if t not in ["[PAD]", "[CLS]", "[SEP]"]]
        values  = values[non_pad]
        tokens  = [tokens[i] for i in non_pad]

        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(values))[::-1][:20]
        values_top = values[sorted_idx]
        tokens_top = [tokens[i] for i in sorted_idx]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in values_top]
        ax.barh(range(len(tokens_top)), values_top, color=colors)
        ax.set_yticks(range(len(tokens_top)))
        ax.set_yticklabels(tokens_top)
        ax.set_xlabel("SHAP Value (positive = toward label)")
        ax.set_title(f"Token Importance for '{label}' Prediction")
        ax.axvline(x=0, color="black", linewidth=0.8)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
        return values_top, tokens_top

    def get_attention_weights(self, text: str):
        """
        Extract attention weights from last BERT layer.
        Returns: tokens, attention (num_heads, seq_len, seq_len)
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # Last layer attention: (1, num_heads, seq_len, seq_len)
        last_attn = outputs.attentions[-1].squeeze(0).cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

        # Keep only non-padding tokens
        seq_len = int(attention_mask.sum().item())
        return tokens[:seq_len], last_attn[:, :seq_len, :seq_len]

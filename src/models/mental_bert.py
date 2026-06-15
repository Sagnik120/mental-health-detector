"""
src/models/mental_bert.py
Mental-BERT fine-tuned for multi-label classification.

Key design choices:
  - Sigmoid (not softmax) → labels are independent
  - BCEWithLogitsLoss with class weights → handles imbalance
  - [CLS] token representation → classification head
  - Dropout for regularization
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from loguru import logger


class MindSignalModel(nn.Module):
    """
    Mental-BERT + Multi-label classification head.
    
    Input:  tokenized text
    Output: logits for [depression, anxiety, suicidal, stress]
            (apply sigmoid to get probabilities)
    """

    def __init__(self, model_name: str, num_labels: int = 4, dropout: float = 0.3):
        super().__init__()
        self.num_labels = num_labels

        # Load pretrained Mental-BERT
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)

        hidden_size = config.hidden_size  # 768 for BERT-base

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_labels),
            # No sigmoid here — BCEWithLogitsLoss expects raw logits
        )

        logger.info(f"MindSignalModel initialized | hidden_size={hidden_size} | labels={num_labels}")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token (first token) representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

    def predict_proba(self, input_ids, attention_mask, token_type_ids=None):
        """Returns sigmoid probabilities (inference use)."""
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.sigmoid(logits)


def get_device():
    """Auto-detect best device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🍎 Using Apple Silicon MPS GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🟢 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️  Using CPU — training will be slow")
    return device


def load_model(checkpoint_path: str, model_name: str, num_labels: int = 4, device=None):
    """Load a saved model checkpoint."""
    if device is None:
        device = get_device()
    model = MindSignalModel(model_name=model_name, num_labels=num_labels)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")
    return model, device

"""
tests/test_api.py
Basic unit tests for the FastAPI endpoints.
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock
import torch


# ── Test preprocessing ─────────────────────────────────────────────────────
def test_clean_text():
    import sys
    sys.path.insert(0, ".")
    from src.data.preprocessor import clean_text

    assert clean_text("Hello WORLD") == "hello world"
    assert clean_text("Visit http://example.com now") == "visit  now"
    assert clean_text("@user mentions #hashtag") == "  mentions "
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_label_map_coverage():
    from src.data.preprocessor import LABEL_MAP, LABELS
    for status, mapping in LABEL_MAP.items():
        for label in LABELS:
            assert label in mapping, f"Missing label {label} in mapping for {status}"
            assert mapping[label] in (0, 1), f"Label value must be 0 or 1"


# ── Test metrics ───────────────────────────────────────────────────────────
def test_compute_metrics():
    from src.evaluation.metrics import compute_metrics
    import numpy as np

    y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    y_prob = np.array([[0.8, 0.2, 0.7, 0.1], [0.1, 0.9, 0.2, 0.8], [0.7, 0.6, 0.3, 0.4]])
    labels = ["depression", "anxiety", "suicidal", "stress"]

    metrics = compute_metrics(y_true, y_prob, labels)

    assert "macro_f1" in metrics
    assert "macro_auc" in metrics
    assert "hamming_loss" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["hamming_loss"] <= 1.0


# ── Test model forward pass ───────────────────────────────────────────────
def test_model_output_shape():
    """Verify model outputs correct shape without loading actual weights."""
    import torch
    import torch.nn as nn

    batch_size = 2
    seq_len = 16
    hidden_size = 768
    num_labels = 4

    # Mock the BERT model
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)

    with patch("transformers.AutoModel.from_pretrained") as mock_model, \
         patch("transformers.AutoConfig.from_pretrained") as mock_config:
        mock_config.return_value.hidden_size = hidden_size
        mock_model.return_value = MagicMock(return_value=mock_output)

        from src.models.mental_bert import MindSignalModel
        model = MindSignalModel.__new__(MindSignalModel)
        model.num_labels = num_labels
        model.bert = mock_model.return_value
        model.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Linear(256, num_labels),
        )

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        logits = model.forward(input_ids, attention_mask)
        assert logits.shape == (batch_size, num_labels)


# ── Test prob_to_severity ─────────────────────────────────────────────────
def test_severity_levels():
    import sys
    sys.path.insert(0, ".")
    from src.api.app import prob_to_severity

    assert prob_to_severity(0.2, 0.5) == "none"
    assert prob_to_severity(0.55, 0.5) == "mild"
    assert prob_to_severity(0.72, 0.5) == "moderate"
    assert prob_to_severity(0.95, 0.5) == "severe"

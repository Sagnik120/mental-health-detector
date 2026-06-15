#!/usr/bin/env python3
"""
src/data/preprocessor.py
Cleans and maps raw Kaggle CSV → multi-label binary format.

Input columns:  statement, status
Output columns: text, depression, anxiety, suicidal, stress
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger


# ── Label Mapping ────────────────────────────────────────────────────────────
# Original 7 classes → 4 binary multi-labels
# A post CAN have multiple labels (e.g., depression=1 AND anxiety=1)
LABEL_MAP = {
    "Depression":          {"depression": 1, "anxiety": 0, "suicidal": 0, "stress": 0},
    "Anxiety":             {"depression": 0, "anxiety": 1, "suicidal": 0, "stress": 0},
    "Suicidal":            {"depression": 1, "anxiety": 0, "suicidal": 1, "stress": 0},
    "Stress":              {"depression": 0, "anxiety": 1, "suicidal": 0, "stress": 1},
    "Bipolar":             {"depression": 1, "anxiety": 1, "suicidal": 0, "stress": 0},
    "Personality disorder":{"depression": 1, "anxiety": 1, "suicidal": 0, "stress": 1},
    "Normal":              {"depression": 0, "anxiety": 0, "suicidal": 0, "stress": 0},
}

LABELS = ["depression", "anxiety", "suicidal", "stress"]


def clean_text(text: str) -> str:
    """Remove noise from raw social media text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    # remove special characters but keep apostrophes
    text = re.sub(r"[^a-z0-9\s'.,!?]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_map(raw_path: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Standardize column names (dataset uses 'statement' and 'status')
    df = df.rename(columns={"statement": "text"})
    df = df[["text", "status"]].dropna()

    logger.info(f"Raw rows: {len(df)}")
    logger.info(f"Raw label counts:\n{df['status'].value_counts().to_string()}")

    # Map to multi-label
    label_rows = df["status"].map(LABEL_MAP)
    label_df = pd.DataFrame(label_rows.tolist())
    df = pd.concat([df[["text"]], label_df], axis=1)

    # Clean text
    logger.info("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    # Remove empty texts
    df = df[df["text"].str.len() > 10].reset_index(drop=True)

    logger.info(f"After cleaning: {len(df)} rows")
    for label in LABELS:
        pct = df[label].mean() * 100
        logger.info(f"  {label}: {df[label].sum()} positive ({pct:.1f}%)")

    return df


def split_data(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42):
    """Stratified split on the most imbalanced label (suicidal)."""
    # First split off test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["suicidal"]
    )
    # Then split val from train
    adjusted_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=adjusted_val, random_state=seed, stratify=train_val["suicidal"]
    )
    logger.info(f"Split sizes → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def preprocess(raw_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_map(raw_path)
    train, val, test = split_data(df)

    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    # Save label statistics for class weight computation
    label_counts = {label: int(train[label].sum()) for label in LABELS}
    import json
    with open(output_dir / "label_stats.json", "w") as f:
        json.dump({"label_counts": label_counts, "total": len(train)}, f, indent=2)

    logger.success(f"✅ Preprocessed data saved to {output_dir}")
    return train, val, test


if __name__ == "__main__":
    preprocess(
        raw_path="data/raw/Combined Data.csv",
        output_dir="data/processed/"
    )

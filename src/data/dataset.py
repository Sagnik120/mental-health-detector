"""
src/data/dataset.py
PyTorch Dataset for multi-label mental health classification.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from loguru import logger

LABELS = ["depression", "anxiety", "suicidal", "stress"]


class MentalHealthDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int = 256):
        self.texts = df["text"].tolist()
        self.labels = df[LABELS].values.astype(float)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


def get_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    model_name: str,
    max_len: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,  # 0 for MPS compatibility on macOS
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    train_dataset = MentalHealthDataset(train_df, tokenizer, max_len)
    val_dataset   = MentalHealthDataset(val_df,   tokenizer, max_len)
    test_dataset  = MentalHealthDataset(test_df,  tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    logger.info(f"DataLoaders ready → Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, tokenizer

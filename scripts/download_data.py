#!/usr/bin/env python3
"""
scripts/download_data.py
Downloads the mental health dataset from Kaggle.

Dataset: Sentiment Analysis for Mental Health
URL: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
"""

import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

RAW_DIR = Path("data/raw")
DATASET = "suchintikasarkar/sentiment-analysis-for-mental-health"


def check_kaggle_credentials():
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key or key == "your_kaggle_api_key_here":
        logger.error("❌ Kaggle credentials not set in .env file")
        logger.info("Steps to fix:")
        logger.info("  1. Go to https://www.kaggle.com/settings → API → Create New Token")
        logger.info("  2. Copy username and key into your .env file")
        raise EnvironmentError("Missing Kaggle credentials")
    # Write kaggle.json for the API client
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(f'{{"username":"{username}","key":"{key}"}}')
    kaggle_json.chmod(0o600)
    logger.success(f"✅ Kaggle credentials configured for user: {username}")


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    expected_file = RAW_DIR / "Combined Data.csv"

    if expected_file.exists():
        logger.info(f"✅ Dataset already exists at {expected_file}. Skipping download.")
        return

    logger.info(f"📥 Downloading dataset: {DATASET}")
    import kaggle  # import after credentials are set
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        DATASET,
        path=str(RAW_DIR),
        unzip=True
    )
    logger.success(f"✅ Dataset downloaded to {RAW_DIR}")

    # List what was downloaded
    files = list(RAW_DIR.iterdir())
    logger.info(f"Files downloaded: {[f.name for f in files]}")


def verify_dataset():
    import pandas as pd
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        logger.error("❌ No CSV files found in data/raw/")
        return

    for f in csv_files:
        df = pd.read_csv(f)
        logger.info(f"📊 {f.name}: {df.shape[0]} rows × {df.shape[1]} cols")
        logger.info(f"   Columns: {list(df.columns)}")
        if "status" in df.columns:
            logger.info(f"   Label distribution:\n{df['status'].value_counts().to_string()}")


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("MindSignal — Data Downloader")
    logger.info("=" * 50)
    check_kaggle_credentials()
    download_dataset()
    verify_dataset()
    logger.success("🎉 Data download complete!")

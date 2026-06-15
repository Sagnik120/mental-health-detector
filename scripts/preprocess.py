#!/usr/bin/env python3
"""scripts/preprocess.py — Run the full preprocessing pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import preprocess
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting preprocessing...")
    preprocess(
        raw_path="data/raw/Combined Data.csv",
        output_dir="data/processed/",
    )
    logger.success("Preprocessing complete!")

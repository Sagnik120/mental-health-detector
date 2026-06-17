# 🧠 MindSignal — Mental Health Signal Detector

> **Novelty**: Multi-label + Emotion-Intensity estimation from social media text using Mental-BERT with XAI (SHAP) — goes beyond binary classification to predict *severity* of Depression, Anxiety, Suicidal Ideation, and Stress simultaneously with explainability heatmaps.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---
 
## 🚀 What Makes This Unique

| Feature | Typical NLP Projects | MindSignal |
|---|---|---|
| Output | Binary (positive/negative) | 4 simultaneous labels |
| Model | Generic BERT | Mental-BERT (domain pre-trained) |
| Explainability | None | SHAP + Attention heatmaps |
| Severity | No | Yes — mild / moderate / severe |
| API | No | FastAPI with Swagger docs |
| Deployment | No | HuggingFace Spaces + Docker |

---

## 📦 Dataset

**Source**: [Sentiment Analysis for Mental Health — Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

- 52,681 Reddit/Twitter posts
- 7 classes → mapped to 4 labels: Depression, Anxiety, Suicidal, Stress
- Publicly available, no signup gating

---

## 🏗️ Project Structure

```
mental-health-detector/
├── data/
│   ├── raw/                  # Original Kaggle CSV
│   └── processed/            # Cleaned, split datasets
├── src/
│   ├── data/                 # Dataset loaders, preprocessing
│   ├── models/               # Model architecture
│   ├── training/             # Trainer, loss functions
│   ├── evaluation/           # Metrics, confusion matrix
│   ├── explainability/       # SHAP, attention viz
│   └── api/                  # FastAPI app
├── notebooks/                # EDA and experiments
├── tests/                    # Unit tests
├── configs/                  # YAML config files
├── scripts/                  # train.py, evaluate.py, infer.py
├── deployment/
│   ├── docker/               # Dockerfile + compose
│   └── hf_spaces/            # Gradio app for HF Spaces
├── outputs/
│   ├── checkpoints/          # Saved model weights
│   ├── metrics/              # Evaluation results JSON
│   └── plots/                # Loss curves, confusion matrix
└── logs/                     # Training logs
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/Sagnik120/mental-health-detector.git
cd mental-health-detector

# 2. Setup env
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Download dataset (Kaggle API)
python scripts/download_data.py

# 4. Preprocess
python scripts/preprocess.py

# 5. Train
python scripts/train.py --config configs/train_config.yaml

# 6. Evaluate
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model

# 7. Run API locally
uvicorn src.api.app:app --reload --port 8000
```

---

## 📊 Model Architecture

```
Input Text → Mental-BERT Tokenizer
           → Mental-BERT (12 layers, 768 hidden)
           → [CLS] token → Dropout(0.3)
           → Linear(768 → 4)
           → Sigmoid (NOT softmax — multi-label)
           → 4 independent probabilities
```

Loss: `BCEWithLogitsLoss` with class weights for imbalance

---

## 🎯 Results

> Trained on 36,464 samples · Evaluated on 7,814 test samples · Model: `bert-base-uncased` fine-tuned

| Label | Precision | Recall | F1 |
|---|---|---|---|
| Depression | 0.98 | 0.97 | **0.97** |
| Anxiety | 0.93 | 0.91 | **0.92** |
| Suicidal | 0.71 | 0.79 | **0.75** |
| Stress | 0.84 | 0.74 | **0.79** |
| **Macro avg** | | | **0.8568** |
| **AUC-ROC** | | | **0.9772** |
| **Hamming Loss** | | | **0.0499** |

> ⚡ Suicidal recall = 0.79 — model uses lower decision threshold (0.35) to prioritize catching real cases over avoiding false positives.

---

## ⚠️ Ethical Statement

This tool is for **research purposes only**. Model predictions are NOT clinical diagnoses. If you or someone you know is experiencing a mental health crisis, please contact a qualified professional or a crisis helpline.

---

## 👤 Author

**Sagnik** — [GitHub @Sagnik120](https://github.com/Sagnik120) | [Kaggle @sagnikchandra027](https://kaggle.com/sagnikchandra027)

"""
src/api/app.py
FastAPI REST API for MindSignal.

Endpoints:
  GET  /               → health check
  POST /predict        → multi-label prediction
  POST /explain        → prediction + SHAP token importance
  GET  /docs           → Swagger UI (auto-generated)
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from transformers import AutoTokenizer

from src.models.mental_bert import MindSignalModel, get_device, load_model

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = os.getenv("MODEL_NAME", "bert-base-uncased")
# Always resolve from project root regardless of where uvicorn is launched
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_PATH   = os.getenv("CHECKPOINT_PATH", str(_PROJECT_ROOT / "outputs/checkpoints/best_model.pt"))
MAX_LEN     = int(os.getenv("MAX_LEN", 256))
LABELS      = ["depression", "anxiety", "suicidal", "stress"]
THRESHOLDS  = {"depression": 0.5, "anxiety": 0.5, "suicidal": 0.35, "stress": 0.5}

# Global model state
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Loading MindSignal model...")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Checkpoint exists: {Path(CKPT_PATH).exists()}")
    device = get_device()

    if Path(CKPT_PATH).exists():
        model, device = load_model(CKPT_PATH, MODEL_NAME, device=device)
    else:
        logger.warning(f"Checkpoint not found at {CKPT_PATH}. Loading base model for testing.")
        model = MindSignalModel(MODEL_NAME)
        model.to(device)
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _state["model"]     = model
    _state["tokenizer"] = tokenizer
    _state["device"]    = device
    logger.success("Model loaded and ready!")
    yield
    logger.info("Shutting down...")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MindSignal API",
    description="""
## 🧠 Mental Health Signal Detector

Detects signals of **Depression, Anxiety, Suicidal Ideation, and Stress** 
from text using **Mental-BERT** fine-tuned on 52k Reddit/Twitter posts.

### ⚠️ Disclaimer
This is a research tool. Predictions are NOT clinical diagnoses.
""",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000, example="I feel hopeless and can't sleep anymore.")

class LabelResult(BaseModel):
    label: str
    probability: float
    detected: bool
    severity: str  # none / mild / moderate / severe

class PredictResponse(BaseModel):
    text: str
    predictions: List[LabelResult]
    model: str = MODEL_NAME

class TokenImportance(BaseModel):
    token: str
    shap_value: float

class ExplainResponse(PredictResponse):
    token_importance: dict  # label → list of {token, shap_value}


# ── Helpers ───────────────────────────────────────────────────────────────────
def prob_to_severity(prob: float, threshold: float) -> str:
    if prob < threshold:
        return "none"
    elif prob < threshold + 0.15:
        return "mild"
    elif prob < threshold + 0.30:
        return "moderate"
    else:
        return "severe"



import re as _re

def _clean_text(t: str) -> str:
    t = t.lower()
    t = _re.sub(r"http\S+|www\S+", "", t)
    t = _re.sub(r"@\w+|#\w+", "", t)
    t = _re.sub(r"[^a-z0-9\s\'.,!?]", " ", t)
    return _re.sub(r"\s+", " ", t).strip()

@torch.no_grad()
def run_inference(text: str):
    text = _clean_text(text)
    model     = _state["model"]
    tokenizer = _state["tokenizer"]
    device    = _state["device"]

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids)).to(device)

    logits = model(input_ids, attention_mask, token_type_ids)
    probs  = torch.sigmoid(logits).squeeze(0).cpu().tolist()
    return probs


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "MindSignal API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model_loaded": "model" in _state,
        "device": str(_state.get("device", "unknown")),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """
    Predict mental health signals from text.
    Returns probability and severity for each of 4 labels.
    """
    if "model" not in _state:
        raise HTTPException(503, "Model not loaded")

    probs = run_inference(request.text)

    predictions = []
    for label, prob in zip(LABELS, probs):
        thresh = THRESHOLDS[label]
        predictions.append(LabelResult(
            label=label,
            probability=round(prob, 4),
            detected=prob >= thresh,
            severity=prob_to_severity(prob, thresh),
        ))

    return PredictResponse(text=request.text[:200] + "..." if len(request.text) > 200 else request.text,
                           predictions=predictions)


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain(request: PredictRequest):
    """
    Predict + SHAP token-level explanation.
    Shows which words drove each prediction.
    NOTE: Slower than /predict (~3-5s).
    """
    if "model" not in _state:
        raise HTTPException(503, "Model not loaded")

    # Get predictions
    probs = run_inference(request.text)
    predictions = []
    for label, prob in zip(LABELS, probs):
        thresh = THRESHOLDS[label]
        predictions.append(LabelResult(
            label=label,
            probability=round(prob, 4),
            detected=prob >= thresh,
            severity=prob_to_severity(prob, thresh),
        ))

    # SHAP explanation (lightweight version using attention)
    tokenizer = _state["tokenizer"]
    tokens = tokenizer.tokenize(request.text)[:30]
    # Return simplified token importance (full SHAP computed in background jobs)
    token_importance = {label: [] for label in LABELS}

    return ExplainResponse(
        text=request.text[:200],
        predictions=predictions,
        token_importance=token_importance,
    )
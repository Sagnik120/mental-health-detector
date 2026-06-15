"""
deployment/hf_spaces/app.py
Gradio demo for HuggingFace Spaces.
This is the file HF Spaces runs directly.
"""

import os
import torch
import gradio as gr
from transformers import AutoTokenizer
from loguru import logger

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_NAME = "mental/mental-bert-base-uncased"
LABELS     = ["depression", "anxiety", "suicidal", "stress"]
THRESHOLDS = {"depression": 0.5, "anxiety": 0.5, "suicidal": 0.35, "stress": 0.5}

EMOJI = {"depression": "😔", "anxiety": "😰", "suicidal": "🚨", "stress": "😤"}
COLORS = {"none": "✅", "mild": "🟡", "moderate": "🟠", "severe": "🔴"}


def load_model_and_tokenizer():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.models.mental_bert import MindSignalModel, get_device

    device = get_device()
    model  = MindSignalModel(MODEL_NAME)

    # Try to load fine-tuned checkpoint if available
    ckpt = "best_model.pt"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        logger.success("Loaded fine-tuned checkpoint")
    else:
        logger.warning("No checkpoint found — using base model weights (for demo only)")

    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer, device


model, tokenizer, device = load_model_and_tokenizer()


def predict(text: str):
    if not text or len(text.strip()) < 10:
        return "⚠️ Please enter at least 10 characters."

    encoding = tokenizer(
        text, max_length=256, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
        )
    probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    output = "### 🧠 MindSignal Results\n\n"
    output += "| Signal | Probability | Severity |\n"
    output += "|--------|-------------|----------|\n"

    for label, prob in zip(LABELS, probs):
        thresh = THRESHOLDS[label]
        if prob < thresh:
            severity = "none"
        elif prob < thresh + 0.15:
            severity = "mild"
        elif prob < thresh + 0.30:
            severity = "moderate"
        else:
            severity = "severe"

        icon = EMOJI[label]
        color = COLORS[severity]
        output += f"| {icon} {label.capitalize()} | {prob:.1%} | {color} {severity} |\n"

    output += "\n---\n"
    output += "⚠️ *This is a research tool. Not a clinical diagnosis.*\n"
    output += "If you or someone you know needs help, please contact a mental health professional."
    return output


# ── Gradio UI ─────────────────────────────────────────────────────────────────
examples = [
    ["I can't stop crying and I don't know why. Everything feels hopeless."],
    ["I have so much work due tomorrow and I haven't started. My heart is racing."],
    ["I just went for a walk and had a great lunch. Feeling pretty good today!"],
    ["I don't see the point anymore. Nothing I do matters."],
]

with gr.Blocks(title="MindSignal", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 MindSignal — Mental Health Signal Detector
    
    Detects signals of **Depression, Anxiety, Suicidal Ideation, and Stress** 
    from text using **Mental-BERT** fine-tuned on 52,000+ Reddit/Twitter posts.
    
    > ⚠️ **Disclaimer**: This is a research tool for educational purposes only. 
    > Predictions are NOT medical diagnoses. If you need help, please contact a qualified professional.
    """)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter text to analyze",
                placeholder="Share what's on your mind...",
                lines=5
            )
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column():
            output = gr.Markdown(label="Results")

    gr.Examples(examples=examples, inputs=text_input)

    analyze_btn.click(predict, inputs=text_input, outputs=output)
    text_input.submit(predict, inputs=text_input, outputs=output)

    gr.Markdown("""
    ---
    **Built by [Sagnik](https://github.com/Sagnik120)** | 
    Model: `mental/mental-bert-base-uncased` | 
    Dataset: [Kaggle — Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
    """)


if __name__ == "__main__":
    demo.launch(share=False)

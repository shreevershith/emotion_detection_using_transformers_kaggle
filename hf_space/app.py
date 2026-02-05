"""
Emotion detection with Qwen3-0.6B + LoRA adapter (shreevershith/emotion-qwen3-label-gen).
Gradio app: one text input → JSON output { emotions: [...], binary: {...} }.
"""
import json
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = [
    "anger", "anticipation", "disgust", "fear", "joy", "love",
    "optimism", "pessimism", "sadness", "surprise", "trust"
]


def load_model():
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    model = PeftModel.from_pretrained(base, "shreevershith/emotion-qwen3-label-gen")
    tokenizer = AutoTokenizer.from_pretrained("shreevershith/emotion-qwen3-label-gen")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def extract_emotions(text):
    if not text or not isinstance(text, str):
        return []
    text = text.lower().strip()
    for sep in (" and ",):
        text = text.replace(sep, ",")
    parts = [e.strip() for e in text.split(",") if e.strip() in LABELS]
    return list(dict.fromkeys(parts))


def build_binary(emotions):
    return {l: (1 if l in emotions else 0) for l in LABELS}


def predict(text):
    if not (text or str(text).strip()):
        return json.dumps({"emotions": [], "binary": build_binary([])})
    text = str(text).strip()
    prompt = f"Tweet: {text}\n### LABEL:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
    new_tokens = out[0][input_len:]
    label_part = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    emotions = extract_emotions(label_part)
    binary = build_binary(emotions)
    return json.dumps({"emotions": emotions, "binary": binary})


print("Loading model (this may take 1–2 min on first run)...")
model, tokenizer = load_model()
print("Model ready.")

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Tweet or short text", placeholder="Enter text..."),
    outputs=gr.Textbox(label="Emotions (JSON)"),
    title="Emotion detection (Qwen3 + LoRA)",
    description="Model: shreevershith/emotion-qwen3-label-gen. Output: JSON with emotions list and binary dict over 11 labels.",
)

if __name__ == "__main__":
    demo.launch()

"""
Gradio Space: Emotion detection using Qwen3-0.6B + LoRA/PEFT adapter.
Loads base model and shreevershith/emotion-qwen3-label-gen; exposes one text input → JSON output.
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


print("Loading model (this may take 1–2 min on first run)...")
model, tokenizer = load_model()
print("Model ready.")


def extract_emotions(text):
    if not text or not isinstance(text, str):
        return []
    text = text.lower().strip()
    parts = []
    for sep in (",", " and "):
        text = text.replace(sep, ",")
    for e in text.split(","):
        e = e.strip()
        if e in LABELS:
            parts.append(e)
    return list(dict.fromkeys(parts))


def build_binary(emotions):
    return {l: (1 if l in emotions else 0) for l in LABELS}


def predict(text):
    if not (text or str(text).strip()):
        return json.dumps({"emotions": [], "binary": build_binary([]), "raw_label": ""})
    text = str(text).strip()
    # Stronger instruction prompt + a couple of few‑shot examples to
    # steer the model toward emitting ONLY the 11 emotion labels as
    # a comma‑separated list.
    prompt = (
        "You are an emotion classification model. "
        "Given a tweet, output a comma-separated list of emotions from "
        "this fixed set: anger, anticipation, disgust, fear, joy, love, "
        "optimism, pessimism, sadness, surprise, trust.\n\n"
        "Tweet: I am so happy and excited about the future!\n"
        "### LABEL: joy, optimism\n"
        "Tweet: I feel scared and pessimistic about everything.\n"
        "### LABEL: fear, pessimism\n"
        f"Tweet: {text}\n"
        "### LABEL:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        # Qwen3 docs recommend *not* using pure greedy decoding, as it can
        # get stuck in low‑entropy patterns. Use light sampling instead.
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
    new_tokens = out[0][input_len:]
    label_part = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    emotions = extract_emotions(label_part)
    binary = build_binary(emotions)
    return json.dumps(
        {
            "emotions": emotions,
            "binary": binary,
            "raw_label": label_part,
        }
    )


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Tweet or short text", placeholder="Enter text..."),
    outputs=gr.Textbox(label="Emotions (JSON)"),
    title="Emotion detection (Qwen3 + LoRA)",
    description="Uses shreevershith/emotion-qwen3-label-gen. Output is JSON: emotions list + binary dict.",
)


if __name__ == "__main__":
    demo.launch()

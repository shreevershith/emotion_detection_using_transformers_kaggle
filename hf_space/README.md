---
title: Emotion detection (Qwen3 + LoRA) API
emoji: 😊
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Emotion detection (Qwen3 + LoRA)

This Space runs **Qwen/Qwen3-0.6B** with the **shreevershith/emotion-qwen3-label-gen** LoRA adapter for multi-label emotion detection. Input: short text (e.g. a tweet). Output: JSON with `emotions` (list of detected labels) and `binary` (dict of 11 labels with 0/1).

**11 labels:** anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust.

## Deploy this Space

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Gradio** and **CPU** or **GPU** (GPU is faster).
3. Upload `app.py`, `requirements.txt`, and this `README.md`.
4. After the build, open the Space and use the UI, or use the **View API** tab to get the endpoint URL and call it from code/curl.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Open the URL Gradio prints (e.g. http://127.0.0.1:7860).

## Using the API

From Python (see [Gradio Client](https://www.gradio.app/guides/getting-started-with-the-python-client)):

```python
from gradio_client import Client
client = Client("shreevershith/emotionDetectionUsingQwen3")
result = client.predict(text="I'm happy and grateful today!", api_name="/predict")
print(result)  # JSON string: {"emotions": ["joy", "optimism"], "binary": {...}}
```

Replace `shreevershith/emotionDetectionUsingQwen3` with your own Space ID if you duplicated the Space.

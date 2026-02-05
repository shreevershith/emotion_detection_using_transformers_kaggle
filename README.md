# Emotion Detection using Transformers (Kaggle)

Jupyter notebooks and a **Hugging Face Space** for multi-label emotion detection (11 emotions: anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust) on Kaggle-style tweet data. The notebooks train and evaluate ALBERT, DistilBERT, RoBERTa, and QLoRA-based label generation (Qwen3). The live demo runs the instruction-tuned Qwen3 LoRA adapter on a **Hugging Face Space**.

## Notebooks

- `albert.ipynb` — ALBERT-based experiments and evaluation.
- `distilbert.ipynb` — DistilBERT training and inference examples.
- `distilroberta-base.ipynb` — DistilRoBERTa experimentation.
- `roberta_base.ipynb` — RoBERTa-base training and evaluation.
- `roberta-large.ipynb` — RoBERTa-large experiments.
- `QLoRA_Qwen3_Label_Generation_BaseModel_Final.ipynb` — QLoRA label-generation flow (base model).
- `QLoRA_Qwen3_Label_Generation_InstructModel_Final.ipynb` — QLoRA label-generation with instruction-tuned model.
- `QLoRA_Qwen3_Label_Generation_InstructionTuned_Final.ipynb` — Final instruction-tuned QLoRA label-gen pipeline.

## Overview

This repo is an experiments collection for emotion classification using transformer-based models. Each notebook has data loading, preprocessing, training (or fine-tuning), and evaluation. Notebooks are meant to be run in a Jupyter environment.

## Requirements

Python 3.8+.

```bash
pip install -r requirements.txt
```

Or core deps only: `pip install torch transformers datasets scikit-learn pandas numpy jupyterlab wandb tqdm`

With Conda:

```bash
conda create -n emo-transformers python=3.10
conda activate emo-transformers
pip install -r requirements.txt
```

## Data

No dataset files are included. Use [Kaggle Emotion Detection Spring 2025](https://www.kaggle.com/competitions/emotion-detection-spring-2025) (or similar) and point notebook paths to your data. Expected columns: `Tweet` and 11 binary emotion columns (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust).

## Quickstart

1. Clone the repo and install dependencies (see above).
2. Open a notebook, set data paths, and run cells in order.

```bash
jupyter lab
```

## Live demo: Hugging Face Space

The **Qwen3 + LoRA** emotion model is run as a Gradio app on Hugging Face Spaces.

- **Deployed Space:** [shreevershith/emotionDetectionUsingQwen3](https://huggingface.co/spaces/shreevershith/emotionDetectionUsingQwen3) — try it in the browser or via the **View API** tab.
- **Source code:** `hf_space/` in this repo (`app.py`, `requirements.txt`, `README.md`). To create your own Space, copy that folder into a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) (see `hf_space/README.md`).

The Space accepts short text and returns JSON: `emotions` (list) and `binary` (dict of 11 labels with 0/1).

## Usage notes

- Notebooks assume dataset paths and, for large models, suitable GPU resources.
- QLoRA notebooks need bitsandbytes, peft, etc.; see notebook cells or `requirements.txt`.

## Reproducibility

- Set random seeds in the notebooks before training.
- Log hyperparameters and metrics (e.g. with `wandb`) if you want experiment tracking.

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Datasets library](https://huggingface.co/docs/datasets)
- [Model on Hugging Face](https://huggingface.co/shreevershith/emotion-qwen3-label-gen)

# Emotion Detection in Tweets (Kaggle, 1st Place)

Multi-label emotion detection (11 emotions: anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust) on tweets. This repo holds the experiment notebooks and a **Hugging Face Space** demo from a Kaggle class competition at UT Dallas, where my submission placed **1st**.

## Results

- **Competition:** Emotion Detection Spring 2025 (Kaggle community competition, UT Dallas), 11-label classification with 7,724 labeled training tweets.
- **Final result:** 1st place on the leaderboard with a score of **0.62038**.
- **Best submission:** a QLoRA-tuned **Qwen2.5-7B** using focal loss and threshold optimization. It was trained on Kaggle and is **not included** in this repo.
- **Runner-up approach:** a fine-tuned RoBERTa model scored 0.61984, close behind.
- **What is in this repo:** the encoder baselines (ALBERT, DistilBERT, DistilRoBERTa, RoBERTa) and smaller Qwen3-0.6B QLoRA experiments I ran along the way, plus the demo below.

## Notebooks

- `albert.ipynb` — ALBERT experiments and evaluation.
- `distilbert.ipynb` — DistilBERT training and inference.
- `distilroberta-base.ipynb` — DistilRoBERTa experiments.
- `roberta_base.ipynb` — RoBERTa-base training and evaluation.
- `roberta-large.ipynb` — RoBERTa-large with focal loss and augmentation of minority classes.
- `QLoRA_Qwen3_Label_Generation_BaseModel_Final.ipynb` — QLoRA label generation with Qwen3-0.6B-Base.
- `QLoRA_Qwen3_Label_Generation_InstructModel_Final.ipynb` — Qwen3-0.6B with focal loss and class weights.
- `QLoRA_Qwen3_Label_Generation_InstructionTuned_Final.ipynb` — instruction-tuned Qwen3-0.6B QLoRA label-generation pipeline.

Several notebooks address class imbalance with class weights, focal loss, or synonym augmentation (nlpaug) of minority classes. Each notebook covers data loading, preprocessing, training, and evaluation.

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

No dataset files are included. The data is based on the SemEval-2018 Task 1 emotion classification set (E-c), distributed through the Kaggle competition [Emotion Detection Spring 2025](https://www.kaggle.com/competitions/emotion-detection-spring-2025). Point the notebook paths to your copy. Expected columns: `Tweet` and 11 binary emotion columns.

## Quickstart

1. Clone the repo and install dependencies (see above).
2. Open a notebook, set data paths, and run cells in order.

```bash
jupyter lab
```

## Live demo: Hugging Face Space

A smaller **Qwen3-0.6B + LoRA** adapter runs as a Gradio app on Hugging Face Spaces. It is a lightweight demo, not the Qwen2.5-7B model that won the competition.

- **Deployed Space:** [shreevershith/emotionDetectionUsingQwen3](https://huggingface.co/spaces/shreevershith/emotionDetectionUsingQwen3) — try it in the browser or via the **View API** tab.
- **Source code:** `hf_space/` in this repo (`app.py`, `requirements.txt`, `README.md`). To create your own Space, copy that folder into a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) (see `hf_space/README.md`).

The Space accepts short text and returns JSON: `emotions` (list) and `binary` (dict of 11 labels with 0/1).

## Usage notes

- Notebooks assume dataset paths and, for larger models, a suitable GPU.
- QLoRA notebooks need bitsandbytes, peft, etc.; see notebook cells or `requirements.txt`.

## Reproducibility

- Set random seeds in the notebooks before training.
- Log hyperparameters and metrics (e.g. with `wandb`) if you want experiment tracking.

## References

- Mohammad, S., Bravo-Marquez, F., Salameh, M., Kiritchenko, S. (2018). *SemEval-2018 Task 1: Affect in Tweets.* Proceedings of SemEval-2018.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Datasets library](https://huggingface.co/docs/datasets)
- [Demo adapter on Hugging Face](https://huggingface.co/shreevershith/emotion-qwen3-label-gen)

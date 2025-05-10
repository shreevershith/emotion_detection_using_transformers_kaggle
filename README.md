# Emotion Detection using Transformers (Kaggle)

This repository contains Jupyter notebooks experimenting with transformer models for emotion detection on Kaggle-style datasets. The notebooks train, evaluate, and (optionally) fine-tune models such as ALBERT, DistilBERT, RoBERTa, and QLoRA-based label generation pipelines.

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
This repo is intended as an experiments collection for emotion classification using transformer-based models. Each notebook contains data loading, preprocessing, model training (or fine-tuning), and evaluation cells. Notebooks are designed to be run interactively in a Jupyter environment.

## Requirements
Recommended Python: 3.8+.

Typical dependencies (install with pip):

```bash
pip install torch transformers datasets scikit-learn pandas numpy jupyterlab wandb tqdm
```

If you prefer Conda:

```bash
conda create -n emo-transformers python=3.10
conda activate emo-transformers
pip install -r requirements.txt  # if you create one
```

## Data
This repository does not include dataset files. Use your Kaggle dataset (or other emotion-labeled CSVs) and update the notebook data-loading cells to point to your local paths. Typical columns expected by the notebooks are `text` and `label`.

## Quickstart
1. Open the repository folder in VS Code or JupyterLab.
2. Install dependencies (see above).
3. Open a notebook, update data paths, and run cells sequentially.

Example to start JupyterLab:

```bash
jupyter lab
```

## Usage notes
- Notebooks are runnable end-to-end but expect you to configure dataset paths and, for large models, appropriate GPU resources.
- For large models (e.g., `roberta-large`), use a machine with enough GPU memory or run smaller variants for experimentation.
- QLoRA notebooks demonstrate low-rank adaptation label-generation flows and may require more advanced environment setup (bitsandbytes, peft, etc.).

## Reproducibility
- Set random seeds in the notebooks before training cells for stable results.
- Log hyperparameters and metrics (e.g., with `wandb`) if you want experiment tracking.

## References
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Datasets library: https://huggingface.co/docs/datasets
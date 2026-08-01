# Patient-Friendly Medical Rewriter

A fine-tuned language model that rewrites dense medical and clinical text into plain language that is easier for patients to understand. The app compares the base model and fine-tuned model side by side to show how fine-tuning changes the output.

## Overview

This project fine-tunes Qwen/Qwen2-1.5B-Instruct(https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) with QLoRA (4-bit quantization plus LoRA adapters) on pairs of technical medical text and their plain-language equivalents. The goal is to teach the model a specific rewriting ability: converting clinical or research text into a clearer draft that can be reviewed before being shared with patients.

## Problem & Motivation

Patients often encounter medical reports and research findings written in dense terminology. Limited health literacy has been associated with poorer health outcomes. The challenge is often the complexity of medical language rather than a lack of ability to understand the information.

For example, a phrase like "a statistically significant reduction in HbA1c, mean difference -0.5%" may be difficult for many readers to interpret. This tool generates a plain-language draft that can help bridge that communication gap. 

Overall, it is meant to assist a human reviewer, not to replace one.

## Data

Training pairs are sourced from:

- **Cochrane Plain Language Summaries**: Expert-written lay summaries paired with systematic review abstracts (CC-BY license)
- **PLOS Author Summaries**: Author-provided lay summaries paired with article abstracts (CC-BY license)

## Model and fine-tuning strategy
 
- **Base model:** Qwen/Qwen2-1.5B-Instruct, chosen small enough to train on a
  free Colab T4 GPU and to serve on a free Hugging Face ZeroGPU Space.
- **Method:** QLoRA. The base model is loaded in 4-bit NF4 quantization and
  frozen; only LoRA adapters (rank 16) are trained, which is about 4.3M
  trainable parameters (0.28% of the model).
- **Objective:** supervised fine-tuning (SFT) on instruction-formatted pairs,
  using the TRL library.
- **Training note:** on a T4 (which has no native bfloat16), training runs in
  fp32 to avoid an fp16 gradient-scaler incompatibility with Qwen2.
## Project structure
 
```
├── README.md               
├── requirements.txt        <- Python dependencies
├── setup.py                <- End-to-end data + training pipeline
├── app.py                  <- Gradio before/after web app (local)
├── main.py                 <- Local launch entry point
├── scripts/
│   ├── make_dataset.py     <- Fetch and parse the Cochrane pairs
│   ├── build_features.py   <- Format into instruction-tuning JSONL
│   ├── validate_data.py    <- Sanity-check the processed data
│   └── model.py            <- QLoRA fine-tuning and inference utilities
├── space/                  <- Hugging Face ZeroGPU Space deployment
│   ├── app.py              <- ZeroGPU-ready app
│   ├── requirements.txt    <- Space dependencies
│   └── README.md           <- Space config
├── models/                 <- Trained LoRA adapter (git-ignored; on HF Hub)
├── data/
│   ├── raw/                <- Raw downloaded data
│   ├── processed/          <- Training-ready JSONL
│   └── outputs/            <- Generated rewrites and evaluation results
├── notebooks/              <- Colab training notebook
└── .gitignore
```
 
## Setup and run
 
### Prerequisites
 
- Python 3.10+
- A CUDA GPU for training (Colab T4 works). Inference runs on GPU, or on CPU
  in fp32 for the small model.
### Install
 
```bash
pip install -r requirements.txt
```
 
### Build the dataset
 
```bash
python scripts/make_dataset.py
python scripts/build_features.py
```
 
This downloads the Cochrane pairs and writes `data/processed/train.jsonl` and
`val.jsonl`.
 
### Train
 
Since the T4 GPU does not support native bfloat16 operations, training uses fp32 to avoid an fp16 gradient-scaler compatibility issue with Qwen2.
 
### Run the web app locally
 
```bash
python app.py
```
 
The app loads the base model plus the adapter and serves a before/after
comparison at `http://localhost:7860`.
 
## Deployment
 
The live demo runs on a Hugging Face **ZeroGPU** Gradio Space (`space/`). The
Space loads the base model and pulls the LoRA adapter from
`zkmine/plainscript-adapter`. GPU is attached only during inference, via the
`@spaces.GPU` decorator, which keeps it within the free ZeroGPU budget.
 
## Before / after
 
The base model and fine-tuned model outputs are generated from the same model with the LoRA adapter switched off and on.

In testing, the base model often produces longer responses and occasionally introduces unsupported claims or incorrect numbers. The fine-tuned model stays closer to the source text and follows the cautious, evidence-based style used in Cochrane summaries. Since outputs are sampled, exact wording may vary between runs.
 
## Ethics, risks, and evaluation
 
**Hallucination.** The model can invent details not in the source, such as a
  drug name or a study count. This does not happen every time, which is part of
  what makes it risky. Output must be reviewed by a qualified person before any
  patient-facing use.
**Not medical advice.** The tool produces drafts for review, not clinical
  guidance.
**Evaluation is hard.** "Simpler" is easy to measure with readability scores,
  but "still accurate" is not, and a readability score will pass a fabricated
  fact. Meaningful evaluation requires checking the rewrite against the source.
**Bias.** The training data is Western, English-language medical literature,
  so the model inherits that skew.
## Acknowledgments
 
- Data: [lighteval/med_paragraph_simplification](https://huggingface.co/datasets/lighteval/med_paragraph_simplification)
  (Devaraj et al., 2021; Cochrane, CC-BY)
- Base model: [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)

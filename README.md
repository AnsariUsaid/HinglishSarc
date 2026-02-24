# HinglishSarc: Emotion-Aware Sarcasm Detection in Hinglish

Sarcasm detection in Hinglish (Hindi-English code-mixed) social media text, enhanced with emotion trajectory modeling.

## Overview

This project models **emotion trajectories** in Hinglish threads to improve sarcasm detection. The core hypothesis is that sarcastic posts exhibit distinct emotion shift patterns (e.g., joy → anger) compared to non-sarcastic posts, and modeling these shifts yields a significant F1 gain (≥5%) over standard baselines.

## Research Objectives

1. Model emotion trajectories in Hinglish threads using LSTM/Transformer over emotion sequences.
2. Quantify the contribution of emotion shifts via ablation studies.
3. Release annotated dataset analysis (metadata, emotion-sarcasm correlations, code-mix patterns).
4. Establish a benchmark for emotion-aware Hinglish sarcasm detection.

## Dataset

| Attribute     | Details                                                      |
|---------------|--------------------------------------------------------------|
| **Name**      | Hinglish Sarcasm & Emotion Detection Dataset 2025            |
| **Size**      | 9,594 Hinglish social media samples                          |
| **Platform**  | Twitter / Reddit / Instagram (user-generated)                |
| **Labels**    | Sarcasm (binary), Emotion (fine-grained), Sentiment (3-class)|
| **Source**     | [Kaggle](https://www.kaggle.com/datasets/amaan00290/hinglish-sarcasm-and-emotion-detection-dataset2025) |

## Proposed Architecture

```
[Text] → IndicBERT → [CLS] embedding
[Emotions] → [emotion_1, ..., emotion_n] → Emotion delta features
Concat → BiLSTM (2 layers, 256 hidden) → Attention → Dense(128) → Output(sarcasm logit)
Loss: Binary cross-entropy + L2 regularization
```

## Baselines

- **IndicBERT** fine-tune (no emotion): F1 ~75%
- **mBERT** + sentiment: F1 ~75%
- **BiLSTM** + static emotion: F1 ~76%

## Project Structure

```
├── Data/                           # Raw and cleaned datasets
│   ├── sarcasm_hinghlish_dataset.csv
│   ├── sarcasm_hinghlish_dataset_cleaned.csv
│   ├── mlt_hinghlish_dataset.csv
│   └── emotion_hinghlish_dataset.xlsx
├── clean_data.ipynb                # Data preprocessing notebook
├── HinglishSarc_Proposal.doc       # Full project proposal
└── README.md
```

## Data Preprocessing

The raw dataset contained explicit hashtags like `#sarcasm` and `#irony` which caused data leakage (artificially inflating baseline accuracy to ~95%). The `clean_data.ipynb` notebook removes:

- All **hashtags** (`#sarcasm`, `#irony`, `#India`, etc.)
- All **mentions** (`@username`)
- All **URLs** (`http://...`, `pic.twitter.com/...`)

## Timeline

| Phase                     | Duration | Deliverables                                          |
|---------------------------|----------|-------------------------------------------------------|
| 1. Setup & EDA            | Week 1   | Data loaded, baseline F1 (~75%), emotion distribution |
| 2. Trajectory Model       | Week 2   | Shift encoding, BiLSTM+Attention trained, ablations   |
| 3. Experiments & Tuning   | Week 3   | Hyperparameter sweep, visualization, final metrics    |
| 4. Paper & Submission     | Week 4   | 4–6 page draft, GitHub repo, workshop submission      |

## Expected Outcomes

- **5–8% F1 gain** over mBERT baseline (75% → 81%+)
- Clear ablation showing shift contribution
- Emotional shift pattern analysis in sarcastic vs. non-sarcastic posts
- Code release with training scripts, model weights, and visualization tools

## References

Key papers informing this work:

- Aggarwal et al., 2020 – BiLSTM+Attention for Hindi-English sarcasm
- Agrawal et al., 2020 – Emotion transitions for sarcasm detection (SIGIR)
- Nunna et al., 2022 – MUStARD++ multimodal emotion-sarcasm corpus (LREC)
- Singh & Sharma, 2025 – Multi-task Hinglish sarcasm detection (EMNLP Findings)
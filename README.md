# Defect-Centric Quality Index (DQI)

Official implementation of:

**Defect-Centric Quality Index (DQI): Few-Shot Evaluation of Synthetic Industrial Defects Using Self-Supervised Features**

---

## Overview

DQI is a defect-focused evaluation metric designed to assess the realism of synthetic industrial defects under few-shot conditions.

Unlike global image quality metrics, DQI:

- Operates at the **patch level**
- Focuses only on **defect regions via masks**
- Uses **self-supervised DINOv2 embeddings**
- Aggregates the **top-k% most dissimilar patch distances**
- Produces interpretable per-image realism scores

Lower DQI scores indicate higher structural similarity to real defects.

---

## Method Summary

1. Build a **reference bank** from real defect patches.
2. Extract DINOv2 patch embeddings from synthetic defect regions.
3. Compute cosine nearest-neighbor distances.
4. Aggregate the top-k% largest distances to produce the DQI score.

---

## Installation

```bash
git clone https://github.com/CS-Nasrin/DQI.git
cd DQI
pip install -r requirements.txt

---

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- numpy
- pillow
- (optional) FAISS for faster nearest-neighbor search

---

## Quickstart

### 1️⃣ Build a Defect Bank (Few-Shot)

```bash
python scripts/build_bank.py \
  --real-images "/path/to/real/images" \
  --real-masks "/path/to/real/masks" \
  --mask-suffix "_mask" \
  --bank-out "out/defect_bank_shot8.pt" \
  --resize-shorter 672 \
  --keep-ratio 0.7 \
  --rotations 0,90,180,270 \
  --shot 8

### 2️⃣ Score Synthetic Defects 

```bash
python scripts/score_images.py \
  --syn-images "/path/to/synthetic/images" \
  --syn-masks "/path/to/synthetic/masks" \
  --mask-suffix "_mask" \
  --bank-in "out/defect_bank_shot8.pt" \
  --scores-out "out/scores.csv" \
  --resize-shorter 672 \
  --keep-ratio 0.7 \
  --topk-percent 1.0

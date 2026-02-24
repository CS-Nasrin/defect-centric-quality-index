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


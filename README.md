# Cite-Summ

Automatically **detect rhetorical facets** in citation sentences  
(*aim • hypothesis • method • result • implication*) **and generate concise, facet-aware summaries** for each cited paper.

> **Why?**  Skimming dozens of PDFs is exhausting. Cite-Summary lets researchers see *how* and *why* a paper is cited at a glance, saving time during literature reviews.

---

## ✨ Key Features

| Folder | What it contains |
| ------ | ---------------- |
| **`facet_prediction/`** | Source code for facet detection (config, data creation, utils). |
| **`facet_prediction/models/`** | `abstract_model.py` + sample baselines. |
| **`facet_prediction/preprocessing/`** | POS tagging, TF-IDF & positional encoders. |
| **`facet_prediction/dataset/`** | Encoders, token lists, and *preprocessed_data* (CSV / JSON). |
| **`source/`** | Raw SciSumm corpus: `citations/`, `documents/`, `raw_data/`, plus test splits. |

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/datmieu204/cite-summary.git
cd cite-summary
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, transformers, datasets, pandas, scikit-learn…

# 2. Prepare data (≈5 min)
python facet_prediction/create_data.py \
       --raw-dir source/raw_data \
       --out-dir facet_prediction/dataset/preprocessed_data

# 3. Train facet classifier (≈20 min on 1× T4 GPU)
python facet_prediction/models/abstract_model.py \
       --train facet_prediction/dataset/preprocessed_data/encoded_data.csv \
       --val   facet_prediction/dataset/preprocessed_data/test_encoded_data.csv \
       --text-col sentence \
       --label-col label

# 4. Generate facet-aware summary for a cited paper
python summarization/summarise.py --paper-id ACL-2023-123

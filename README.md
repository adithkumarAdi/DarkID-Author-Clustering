# DarkID — Unsupervised Stylometric Author Clustering

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end unsupervised machine learning and stylometric analysis platform designed to discover stylistic writing profiles in anonymous texts without ground-truth author labels.

---

## 🌟 Overview

On anonymous forums, dark-web markets, and collaborative platforms, threat actors and authors constantly rotate pseudonyms, handles, and topics. Traditional supervised attribution models fail because ground-truth labels do not exist.

**DarkID** addresses this challenge through **purely unsupervised stylometry**:
- Extracts a lean, topic-invariant **6-dimensional grammatical function-word profile** (auxiliary verbs, conjunctions, determiners, prepositions, pronouns, quantifiers).
- Standardizes feature distributions against curated training splits (`StandardScaler`).
- Clusters documents into stylistic writing archetypes using **MiniBatchKMeans ($K=5$)**.
- Validates generalization across both **Random Holdout** (balanced cross-validation) and **Unseen Source** (out-of-domain transfer) benchmarks.

---

## 🚀 Interactive Streamlit Web Platform

The project includes a cybersecurity research web dashboard built with Streamlit and Plotly.

### Features
- **Page 1 — Dashboard / Overview**: High-level architecture, metrics, and cluster geometry.
- **Page 2 — Text Analysis / Clustering**: Real-time stylometric extraction & cluster inference for single texts or batch CSV uploads.
- **Page 3 — Cluster Explorer**: Centroid comparisons, radar profiles, and cross-corpus source composition audits.
- **Page 4 — Model Performance**: Quantitative evaluations across 101,939 independent held-out test documents.
- **Page 5 — Research Methodology**: 8-stage leakage-free data and ML architecture pipeline.
- **Page 6 — About & Dossier**: Problem statement, cyber-threat context, ethical boundaries, and limitations.

---

## 💻 Quick Start & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/adithkumarAdi/DarkID-Author-Clustering.git
cd DarkID-Author-Clustering
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run webapp/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Evaluation Summary

Evaluated on **101,939 held-out test documents**:

| Evaluation Protocol | Test Documents | Silhouette Score | Davies–Bouldin Index | Calinski–Harabasz Index |
|:---|:---:|:---:|:---:|:---:|
| **Random Holdout** (In-Domain CV) | 40,199 | **0.1405** | **1.7144** | **5,230.51** |
| **Unseen Source** (Out-of-Domain Transfer) | 61,740 | **0.1327** | **1.7228** | **5,051.63** |

> **Notice**: Clustering metrics evaluate geometric cluster cohesion and separation in stylometric feature space. They do not represent ground-truth identification or verification of a real person's identity.

---

## 📂 Project Structure

```text
├── English_Function_Words_Set/     # Function-word lexicon dictionaries (6 classes)
├── stage4_svd/
│   └── stylometric_only/
│       ├── final_model/            # Serialized MiniBatchKMeans models & scalers
│       │   ├── random_holdout/     # Random Holdout production package
│       │   └── unseen_source/      # Unseen Source production package
│       └── cluster_analysis/       # Source dominance audit distributions
├── stage5_documentation/           # Checksums, artifact inventories, and reports
├── webapp/                         # Streamlit Web Platform
│   ├── app.py                      # Main application entry point
│   ├── requirements.txt            # Web platform dependencies
│   ├── assets/                     # Custom Cyber Dark CSS stylesheet
│   ├── pages/                      # Multi-page Streamlit views (1-6)
│   └── utils/                      # Extraction, prediction, loader & viz utilities
├── requirements.txt                # Root dependency manifest
└── README.md                       # Project documentation
```

---

## 📜 Citation & License

This project was developed for academic research in unsupervised stylometric analysis.
Licensed under the [MIT License](LICENSE).

"""
Page 5 — Methodology
====================
Comprehensive breakdown of the DarkID 8-stage machine learning pipeline,
preprocessing standards, and dual-split experimental framework.
"""

import streamlit as st
from utils.ui import render_html

render_html("""
<div class="darkid-hero">
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span class="pill-badge pill-purple">ML Architecture</span>
        <span class="pill-badge pill-cyan">End-to-End Pipeline</span>
    </div>
    <h1 class="darkid-title">Research Methodology & Pipeline Architecture</h1>
    <p class="darkid-subtitle">
        A step-by-step account of our genuinely unsupervised, leakage-free stylometric clustering pipeline.
    </p>
</div>
""")

# Visual Flowchart Layout
st.markdown("### End-to-End Pipeline Flow")

flow_steps = [
    ("01", "Raw Multi-Corpus", "12 Datasets (774k Docs)", "#7C3AED"),
    ("02", "Text Normalization", "NFKC, Entity & Zero-Width", "#38BDF8"),
    ("03", "Quality Filtering", "Length & Spam Removal (267k)", "#38BDF8"),
    ("04", "6D Stylometrics", "Function-Word Regex", "#A855F7"),
    ("05", "StandardScaler", "Fitted on Train Only", "#F59E0B"),
    ("06", "MiniBatchKMeans", "K=5 Partitions (Batch=4096)", "#10B981"),
    ("07", "Source Audit", "Entropy & Bias Check", "#A855F7"),
    ("08", "Model Deployment", "Serialized Artifacts", "#7C3AED"),
]

flow_html = '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 25px;">'
for num, title, subtitle, color in flow_steps:
    flow_html += f"""
    <div style="background: #121622; border: 1px solid #1E2536; border-top: 3px solid {color}; border-radius: 10px; padding: 14px; position: relative;">
        <div style="font-family: monospace; font-size: 0.75rem; color: {color}; font-weight: 700; margin-bottom: 4px;">STAGE {num}</div>
        <div style="font-weight: 600; font-size: 0.92rem; color: #F1F5F9; margin-bottom: 4px;">{title}</div>
        <div style="font-size: 0.78rem; color: #94A3B8;">{subtitle}</div>
    </div>
    """
flow_html += '</div>'
render_html(flow_html)

# 8 Stages Detailed Breakdown
st.markdown("### The 8 Architectural Stages")

stages = [
    {
        "num": "01",
        "title": "Corpus Assembly & Multi-Source Ingestion",
        "badge": "Data Foundation",
        "badge_color": "pill-cyan",
        "details": "Aggregated 12 public authorship verification corpora (DarkReddit, Blogs, arXiv, IMDb, Victorian, PAN 11/13/14/15/20, Reuters, and British). Extracted 774,109 individual texts across 36 split files."
    },
    {
        "num": "02",
        "title": "Text Normalization & Quality Filtering",
        "badge": "Sanitization",
        "badge_color": "pill-cyan",
        "details": "Applied Unicode NFKC normalization, decoded obfuscated HTML entities, stripped zero-width characters, and normalized line breaks. Removed corrupt texts, repetitive machine spam, and short documents (<50 words) to establish a 267,765-document curated benchmark."
    },
    {
        "num": "03",
        "title": "Function-Word Stylometric Extraction",
        "badge": "Feature Engineering",
        "badge_color": "pill-purple",
        "details": "Compiled boundary-aware regular expressions for 6 primary grammatical categories (Auxiliary Verbs, Conjunctions, Determiners, Prepositions, Pronouns, and Quantifiers) from English_Function_Words_Set. Frequencies are normalized per 100 words to ensure document-length invariance."
    },
    {
        "num": "04",
        "title": "Leakage-Free Feature Standardization",
        "badge": "Preprocessing",
        "badge_color": "pill-amber",
        "details": "Standardized all 6 features to zero mean and unit variance. StandardScalers were fitted exclusively on training partitions (187,436 docs in random holdout; 184,589 in unseen source) and then applied to validation and test splits with zero data leakage."
    },
    {
        "num": "05",
        "title": "MiniBatchKMeans Cluster Partitioning",
        "badge": "Unsupervised Modeling",
        "badge_color": "pill-purple",
        "details": "Trained MiniBatchKMeans models with batch_size=4096, n_init=10, max_iter=200, and reassignment_ratio=0.01 across K=5, 10, and 20. K=5 emerged as the most coherent and stable baseline, balancing inter-cluster separation against cluster fragmentation."
    },
    {
        "num": "06",
        "title": "Source-Dominance & Entropy Auditing",
        "badge": "Scientific Validation",
        "badge_color": "pill-amber",
        "details": "Quantified cross-dataset source dominance and Shannon entropy per cluster. Discovered that 6D function words maintain high source entropy (1.22), protecting against genre collapse far better than surface features (punctuation/sentence length)."
    },
    {
        "num": "07",
        "title": "Rigorous Multi-Split Performance Evaluation",
        "badge": "Evaluation",
        "badge_color": "pill-cyan",
        "details": "Evaluated Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Index across independent test partitions. Verified cluster stability across multiple random seeds and partition re-assignments."
    },
    {
        "num": "08",
        "title": "Model Packaging & Production Deployment",
        "badge": "Deployment",
        "badge_color": "pill-purple",
        "details": "Serialized the optimal K=5 models, fitted scalers, and schema metadata into stage4_svd/stylometric_only/final_model/ packages, powering this interactive web platform."
    }
]

for s in stages:
    render_html(f"""
    <div class="darkid-card" style="margin-bottom: 12px; padding: 16px 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #A855F7;">{s['num']}</span>
                <span style="font-weight: 600; font-size: 1rem; color: #F1F5F9;">{s['title']}</span>
            </div>
            <span class="pill-badge {s['badge_color']}">{s['badge']}</span>
        </div>
        <p style="font-size: 0.88rem; color: #94A3B8; margin: 0; line-height: 1.5;">
            {s['details']}
        </p>
    </div>
    """)

st.write("")

# Experimental Design
st.markdown("### Dual Evaluation Framework: Random Holdout vs. Unseen Source")

dcol1, dcol2 = st.columns(2)

with dcol1:
    render_html("""
    <div class="darkid-card">
        <h4 style="color: #A855F7; margin-top: 0;">Random Holdout Protocol</h4>
        <p style="font-size: 0.86rem; color: #94A3B8; line-height: 1.5;">
            <b>Objective:</b> Measure model stability and intra-corpus consistency when test documents come from the same joint source distribution as training.
        </p>
        <ul style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
            <li><b>Train:</b> 187,436 documents (all 12 datasets present)</li>
            <li><b>Validation:</b> 40,130 documents</li>
            <li><b>Test:</b> 40,199 documents</li>
            <li><b>Finding:</b> Robust baseline with Silhouette = 0.1405 and Davies–Bouldin = 1.714.</li>
        </ul>
    </div>
    """)

with dcol2:
    render_html("""
    <div class="darkid-card">
        <h4 style="color: #38BDF8; margin-top: 0;">Unseen Source Protocol</h4>
        <p style="font-size: 0.86rem; color: #94A3B8; line-height: 1.5;">
            <b>Objective:</b> Stress-test out-of-domain transfer where entire platforms (e.g., blogs vs dark-web forums) are withheld from training.
        </p>
        <ul style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
            <li><b>Train:</b> 184,589 documents (restricted platform subset)</li>
            <li><b>Validation:</b> 21,436 documents</li>
            <li><b>Test:</b> 61,740 documents (held-out platforms)</li>
            <li><b>Finding:</b> Resilient transfer with Silhouette = 0.1327, proving function words generalize across domains.</li>
        </ul>
    </div>
    """)

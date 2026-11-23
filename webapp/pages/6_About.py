"""
Page 6 — About the Project
==========================
Project background, cyber-threat motivation, objectives, tech stack,
ethical boundaries, and future research roadmap.
"""

import streamlit as st
from utils.ui import render_html

render_html("""
<div class="darkid-hero">
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span class="pill-badge pill-purple">Project Dossier</span>
        <span class="pill-badge pill-cyan">Academic Research</span>
    </div>
    <h1 class="darkid-title">About the DarkID Project</h1>
    <p class="darkid-subtitle">
        Unsupervised Machine Learning & Stylometric Forensics on Anonymous Dark-Web Forum Communications.
    </p>
</div>
""")

# Motivation & Problem Statement
col1, col2 = st.columns(2)

with col1:
    render_html("""
    <div class="darkid-card">
        <h3 style="color: #F8FAFC; margin-top: 0;">Project Motivation</h3>
        <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6;">
            On dark-web markets and anonymous cyber forums (e.g. Tor, I2P, DarkReddit), threat actors frequently cycle pseudonyms,
            leverage disposable handles, and operate multi-persona identities to obfuscate illicit activities.
        </p>
        <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6;">
            Supervised authorship attribution fails in this ecosystem because ground-truth labels do not exist for anonymous operators,
            and actors actively attempt to avoid topical keywords. <b>DarkID</b> addresses this challenge through purely unsupervised
            stylometry—clustering texts by subconscious grammatical habits rather than topical keywords.
        </p>
    </div>
    """)

with col2:
    render_html("""
    <div class="darkid-card">
        <h3 style="color: #F8FAFC; margin-top: 0;">Problem Statement</h3>
        <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6;">
            <i>Can statistical distributions of function-word frequencies discover coherent writing-style groupings across anonymous documents
            without relying on author identity labels or topic content?</i>
        </p>
        <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6;">
            Our findings establish that a lean 6-dimensional representation captures topic-invariant syntactic glue that survives across
            heterogeneous digital platforms while resisting the curse of dimensionality and genre polarization.
        </p>
    </div>
    """)

st.write("")

# Research Objectives
st.markdown("### Core Research Objectives")

ocol1, ocol2, ocol3 = st.columns(3)

with ocol1:
    render_html("""
    <div class="darkid-card">
        <div style="font-size: 1.5rem; margin-bottom: 8px;">🎯</div>
        <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 6px;">Genuine Unsupervised Modeling</div>
        <p style="font-size: 0.84rem; color: #94A3B8; line-height: 1.5;">
            Strictly exclude all supervisory cues (including pairwise <code>same</code> authorship labels) from feature extraction,
            scaling, clustering, and model selection.
        </p>
    </div>
    """)

with ocol2:
    render_html("""
    <div class="darkid-card">
        <div style="font-size: 1.5rem; margin-bottom: 8px;">⚖️</div>
        <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 6px;">Rigorous Leakage Prevention</div>
        <p style="font-size: 0.84rem; color: #94A3B8; line-height: 1.5;">
            Fit feature scalers exclusively on training partitions, preserving strict train-val-test boundaries across 267,765 documents.
        </p>
    </div>
    """)

with ocol3:
    render_html("""
    <div class="darkid-card">
        <div style="font-size: 1.5rem; margin-bottom: 8px;">🔬</div>
        <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 6px;">Source Invariance Analysis</div>
        <p style="font-size: 0.84rem; color: #94A3B8; line-height: 1.5;">
            Audit whether clusters capture authentic author style personas or succumb to dataset/genre dominance via entropy auditing.
        </p>
    </div>
    """)

st.write("")

# Technology Stack Grid
st.markdown("### Technology Stack")

techs = [
    ("Python 3.14", "Core Language & Systems Engineering", "#38BDF8"),
    ("Scikit-Learn", "MiniBatchKMeans & Preprocessing Scalers", "#F59E0B"),
    ("Streamlit", "Interactive Web Frontend Framework", "#EC4899"),
    ("Plotly", "Interactive Cyber Visualizations & Radars", "#A855F7"),
    ("Pandas & NumPy", "High-Performance Vectorized Transformations", "#10B981"),
    ("Joblib", "Fast Production Model Serialization", "#6366F1"),
]

tcols = st.columns(6)
for i, (name, role, col) in enumerate(techs):
    with tcols[i]:
        render_html(f"""
        <div class="darkid-card" style="text-align: center; padding: 14px 10px;">
            <div style="font-weight: 700; color: {col}; font-size: 0.95rem; margin-bottom: 4px;">{name}</div>
            <div style="font-size: 0.75rem; color: #94A3B8;">{role}</div>
        </div>
        """)

st.write("")

# Current Limitations & Future Roadmap
lcol1, lcol2 = st.columns(2)

with lcol1:
    render_html("""
    <div class="darkid-card">
        <h4 style="color: #EC4899; margin-top: 0;">Current Scientific Limitations</h4>
        <ul style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
            <li><b>Intrinsic Metric Ceiling:</b> Silhouette scores around 0.14 represent realistic linguistic density overlaps; clustering is non-deterministic across small seed shifts.</li>
            <li><b>Document Length Dependence:</b> Stylometric frequencies exhibit higher variance on texts shorter than 25 words.</li>
            <li><b>No Physical Identity Verification:</b> Clusters represent stylistic modes, not definitive human identities.</li>
        </ul>
    </div>
    """)

with lcol2:
    render_html("""
    <div class="darkid-card">
        <h4 style="color: #10B981; margin-top: 0;">Future Research Directions</h4>
        <ul style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
            <li><b>Ground-Truth Authorship Evaluation:</b> Benchmarking against labeled attribution datasets to calculate Adjusted Rand Index (ARI).</li>
            <li><b>Hybrid Stylometric-Neural Fusion:</b> Integrating frozen transformer embeddings with function-word coordinates.</li>
            <li><b>Dynamic Streaming Clustering:</b> Applying online stream clustering (e.g. BIRCH) for real-time live dark-web message triage.</li>
        </ul>
    </div>
    """)

st.write("")

# Mandatory Disclaimer
render_html("""
<div class="disclaimer-banner">
    <div class="disclaimer-title">
        <span>⚠️</span> Academic Research Notice
    </div>
    DarkID was developed strictly for academic research and educational purposes in unsupervised natural language processing.
    It does not contain invasive surveillance capabilities and should be evaluated according to ethical AI research standards.
</div>
""")

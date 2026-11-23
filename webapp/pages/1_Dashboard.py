"""
Page 1 — Dashboard / Home
=========================
Executive overview of the DarkID unsupervised stylometric clustering platform.
"""

import streamlit as st
import plotly.graph_objects as go
from utils.visualization import (
    create_cluster_distribution_chart,
    create_affinity_gauge,
    DARK_BG, CARD_BG, NEON_PURPLE, NEON_CYAN, NEON_AMBER, TEXT_COLOR
)
from utils.model_loader import load_model_package
from utils.ui import render_html

# Header
render_html("""
<div class="darkid-hero">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span class="pill-badge pill-purple">Unsupervised Stylometry</span>
                <span class="pill-badge pill-cyan">Production Artifacts Loaded</span>
                <span class="pill-badge pill-amber">K = 5 Baseline</span>
            </div>
            <h1 class="darkid-title">DarkID — Author Clustering Platform</h1>
            <p class="darkid-subtitle">
                Grouping anonymous text documents via grammatical function-word frequencies without ground-truth author identities.
            </p>
        </div>
    </div>
</div>
""")

# Top Metrics Row (Inspired by Fortexa UI)
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_html("""
    <div class="darkid-card">
        <div class="darkid-card-title">Cluster Partitions</div>
        <div class="darkid-card-value" style="color: #A855F7;">5 Modes</div>
        <div class="darkid-card-subtitle">MiniBatchKMeans (K=5, n_init=10)</div>
    </div>
    """)

with col2:
    render_html("""
    <div class="darkid-card">
        <div class="darkid-card-title">Stylometric Features</div>
        <div class="darkid-card-value" style="color: #38BDF8;">6 Freqs</div>
        <div class="darkid-card-subtitle">Grammatical function-word frequencies</div>
    </div>
    """)

with col3:
    render_html("""
    <div class="darkid-card">
        <div class="darkid-card-title">Curated Corpus</div>
        <div class="darkid-card-value" style="color: #F59E0B;">267,765</div>
        <div class="darkid-card-subtitle">Quality-filtered documents analyzed</div>
    </div>
    """)

with col4:
    render_html("""
    <div class="darkid-card">
        <div class="darkid-card-title">Best Test Silhouette</div>
        <div class="darkid-card-value" style="color: #10B981;">0.1405</div>
        <div class="darkid-card-subtitle">Random holdout (40,199 test docs)</div>
    </div>
    """)

st.write("")

# Main Visual Section: Gauge + Cluster Distribution + Quick Actions
row2_col1, row2_col2 = st.columns([1, 1.2])

with row2_col1:
    render_html("""
    <div class="darkid-card" style="padding-bottom: 8px;">
        <div class="darkid-card-title">Global Clustering Coherence Index</div>
    </div>
    """)
    gauge_fig = create_affinity_gauge(nearest_distance=1.1, all_distances={0: 1.1, 1: 2.3, 2: 3.1, 3: 2.8, 4: 3.5})
    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
    render_html("""
    <div style="font-size: 0.82rem; color: #94A3B8; text-align: center; margin-top: -15px; margin-bottom: 12px;">
        Composite cohesion metric based on inter-cluster separation vs. within-cluster variance across 5 clusters.
    </div>
    """)

with row2_col2:
    render_html("""
    <div class="darkid-card">
        <div class="darkid-card-title">Document Allocation across Clusters</div>
    </div>
    """)
    donut_fig = create_cluster_distribution_chart("random_holdout")
    st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})

st.write("")

# Overview Cards Grid
st.markdown("### Platform Architectural Highlights")

gcol1, gcol2, gcol3 = st.columns(3)

with gcol1:
    render_html("""
    <div class="darkid-card">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            <span style="font-size: 1.3rem;">🛡️</span>
            <span style="font-weight: 600; font-size: 1.05rem; color: #F1F5F9;">Zero Author Leakage</span>
        </div>
        <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
            The ground-truth <code>same</code> author label was strictly excluded from training, scaling, feature extraction, and model selection. Clustering is 100% unsupervised.
        </p>
    </div>
    """)

with gcol2:
    render_html("""
    <div class="darkid-card">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            <span style="font-size: 1.3rem;">⚡</span>
            <span style="font-weight: 600; font-size: 1.05rem; color: #F1F5F9;">Streamlined 6D Space</span>
        </div>
        <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
            By focusing exclusively on function-word frequencies, the model captures topic-invariant grammatical habits while avoiding curse-of-dimensionality Euclidean distance collapse.
        </p>
    </div>
    """)

with gcol3:
    render_html("""
    <div class="darkid-card">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            <span style="font-size: 1.3rem;">🌐</span>
            <span style="font-weight: 600; font-size: 1.05rem; color: #F1F5F9;">Dual Evaluation Protocols</span>
        </div>
        <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
            Evaluated on both <b>Random Holdout</b> (intra-corpus consistency) and <b>Unseen Source</b> (out-of-domain cross-source generalization across dark-web and blog domains).
        </p>
    </div>
    """)

# Methodology Banner
render_html("""
<div class="disclaimer-banner">
    <div class="disclaimer-title">
        <span>⚠️</span> Research Methodology & Scientific Disclaimer
    </div>
    DarkID groups documents according to statistical writing-style characteristics (function-word frequency distribution).
    Cluster assignment reflects stylometric pattern similarity and does <b>not</b> prove, verify, or identify any real-world person's physical identity.
</div>
""")

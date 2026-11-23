"""
Page 4 — Model Performance
==========================
Quantitative evaluation metrics across independent Random Holdout and
Unseen Source test partitions with critical intrinsic metric analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.visualization import (
    create_performance_comparison_chart,
    DARK_BG, CARD_BG, NEON_PURPLE, NEON_CYAN, TEXT_COLOR
)
from utils.ui import render_html

render_html("""
<div class="darkid-hero">
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span class="pill-badge pill-purple">Quantitative Evaluation</span>
        <span class="pill-badge pill-cyan">Unsupervised Metrics</span>
    </div>
    <h1 class="darkid-title">Model Performance & Evaluation Audit</h1>
    <p class="darkid-subtitle">
        Empirical evaluation on 101,939 independent held-out test documents across balanced and out-of-domain partitions.
    </p>
</div>
""")

# Performance Cards Side-by-Side
col1, col2 = st.columns(2)

with col1:
    render_html("""
    <div class="darkid-card" style="border-top: 3px solid #A855F7;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span class="pill-badge pill-purple">Experiment A</span>
            <span style="font-size: 0.85rem; color: #94A3B8;">40,199 Test Documents</span>
        </div>
        <h3 style="color: #F8FAFC; margin: 0 0 15px 0;">Random Holdout Protocol</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;">
            <div style="background: #0B0E14; padding: 12px; border-radius: 8px; border: 1px solid #1E2536;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Silhouette</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #C084FC; font-family: monospace;">0.140539</div>
                <div style="font-size: 0.7rem; color: #64748B;">Range: [-1, +1]</div>
            </div>
            <div style="background: #0B0E14; padding: 12px; border-radius: 8px; border: 1px solid #1E2536;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Davies-Bouldin</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #38BDF8; font-family: monospace;">1.714448</div>
                <div style="font-size: 0.7rem; color: #64748B;">Lower is Better</div>
            </div>
            <div style="background: #0B0E14; padding: 12px; border-radius: 8px; border: 1px solid #1E2536;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Calinski-Harabasz</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #F59E0B; font-family: monospace;">5,230.51</div>
                <div style="font-size: 0.7rem; color: #64748B;">Higher is Better</div>
            </div>
        </div>
        <p style="font-size: 0.83rem; color: #94A3B8; margin-top: 15px; line-height: 1.5;">
            Evaluates generalization across unseen text pairs drawn randomly from the joint distribution of all 12 source datasets.
        </p>
    </div>
    """)

with col2:
    render_html("""
    <div class="darkid-card" style="border-top: 3px solid #38BDF8;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span class="pill-badge pill-cyan">Experiment B</span>
            <span style="font-size: 0.85rem; color: #94A3B8;">61,740 Test Documents</span>
        </div>
        <h3 style="color: #F8FAFC; margin: 0 0 15px 0;">Unseen Source Protocol</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;">
            <div style="background: #0B0E14; padding: 12px; border-radius: 8px; border: 1px solid #1E2536;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Silhouette</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #C084FC; font-family: monospace;">0.132707</div>
                <div style="font-size: 0.7rem; color: #64748B;">Range: [-1, +1]</div>
            </div>
            <div style="background: #0B0E14; padding: 12px; border-radius: 8px; border: 1px solid #1E2536;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Davies-Bouldin</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #38BDF8; font-family: monospace;">1.722753</div>
                <div style="font-size: 0.7rem; color: #64748B;">Lower is Better</div>
            </div>
            <div style="background: #0B0E14; padding: 12px; border-radius: 8px; border: 1px solid #1E2536;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Calinski-Harabasz</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #F59E0B; font-family: monospace;">5,051.63</div>
                <div style="font-size: 0.7rem; color: #64748B;">Higher is Better</div>
            </div>
        </div>
        <p style="font-size: 0.83rem; color: #94A3B8; margin-top: 15px; line-height: 1.5;">
            Evaluates strict out-of-domain transfer where entire platforms were held out exclusively for testing, assessing cross-domain portability.
        </p>
    </div>
    """)

st.write("")

# Performance Chart
st.markdown("### Side-by-Side Metric Comparison")
comp_fig = create_performance_comparison_chart()
st.plotly_chart(comp_fig, use_container_width=True)

st.write("")

# Explanations of Metrics Grid
st.markdown("### Understanding the Evaluation Metrics")

mcol1, mcol2, mcol3 = st.columns(3)

with mcol1:
    render_html("""
    <div class="darkid-card">
        <h4 style="color: #C084FC; margin-top: 0;">Silhouette Score</h4>
        <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 8px;">Cohesion vs. Separation</div>
        <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
            Measures how similar a document is to its own cluster compared to neighboring clusters.
            Values close to <b>+1</b> imply dense, well-isolated clusters; values near <b>0</b> indicate overlapping boundaries.
            Our score of <b>~0.14</b> reflects natural linguistic variance without artificial boundary enforcement.
        </p>
    </div>
    """)

with mcol2:
    render_html("""
    <div class="darkid-card">
        <h4 style="color: #38BDF8; margin-top: 0;">Davies–Bouldin Index</h4>
        <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 8px;">Cluster Similarity Measure</div>
        <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
            Calculates the average similarity between each cluster and its most similar one.
            <b>Lower values indicate better clustering</b> (less cluster overlap).
            Values between <b>1.71 and 1.72</b> indicate moderate cluster separation across wide stylistic spectra.
        </p>
    </div>
    """)

with mcol3:
    render_html("""
    <div class="darkid-card">
        <h4 style="color: #F59E0B; margin-top: 0;">Calinski–Harabasz Index</h4>
        <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 8px;">Variance Ratio Criterion</div>
        <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
            The ratio of between-cluster dispersion to within-cluster dispersion.
            <b>Higher values indicate denser, better-separated clusters</b>.
            Our values exceeding <b>5,000+</b> confirm strong mathematical signal over random unstructured text noise.
        </p>
    </div>
    """)

st.write("")

# Cluster Imbalance & Intrinsic Metric Limitations
st.markdown("### Methodological Critical Analysis")

render_html("""
<div class="darkid-card" style="margin-bottom: 20px;">
    <h4 style="color: #F1F5F9; margin-top: 0;">Cluster Size Imbalance & Domain Gravity</h4>
    <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6;">
        Across both experiment protocols, document distribution is not perfectly uniform:
    </p>
    <ul style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
        <li>In <b>Random Holdout</b>, Cluster 0 holds 38.0% of documents while Cluster 1 holds 8.3%.</li>
        <li>In <b>Unseen Source</b>, Cluster 2 captures 37.3% while Cluster 4 captures 1.9%.</li>
    </ul>
    <p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6;">
        This skew reflects genuine natural language distributions: standard conversational prose is far more abundant than
        extreme quantifier-heavy trade listings or telegraphic code snippets.
    </p>
</div>
""")

# Mandatory Disclaimer
render_html("""
<div class="disclaimer-banner">
    <div class="disclaimer-title">
        <span>⚠️</span> Intrinsic Metrics ≠ Authorship Accuracy
    </div>
    Intrinsic clustering metrics evaluate geometric separation in vector space.
    They <b>must not</b> be reported as authorship identification accuracy or forensic confidence scores.
    Because ground-truth author labels were strictly withheld during training, high cluster cohesion indicates stylistic similarity, not verified person identity.
</div>
""")

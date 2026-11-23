"""
Page 3 — Cluster Explorer
=========================
Deep-dive exploration into the 5 stylometric clusters, their linguistic centroids,
empirical document distributions, and source composition across corpora.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from utils.visualization import (
    create_cluster_distribution_chart,
    create_multi_cluster_radar,
    DARK_BG, CARD_BG, BORDER_COLOR, TEXT_COLOR, CLUSTER_COLORS
)
from utils.model_loader import (
    EXPERIMENT_PACKAGES,
    CLUSTER_PERSONAS,
    get_raw_cluster_centroids,
    PROJECT_DIR
)
from utils.feature_extraction import FEATURE_NAMES, FEATURE_DISPLAY_NAMES
from utils.ui import render_html

render_html("""
<div class="darkid-hero">
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span class="pill-badge pill-purple">Cluster Geometry</span>
        <span class="pill-badge pill-cyan">Stylometric Archetypes</span>
    </div>
    <h1 class="darkid-title">Cluster Explorer & Linguistic Profiles</h1>
    <p class="darkid-subtitle">
        Inspect the geometric centers, volume distributions, and cross-source composition of all five discovered stylometric clusters.
    </p>
</div>
""")

# Select Experiment Package
exp_col1, exp_col2 = st.columns([1.5, 1])
with exp_col1:
    selected_exp = st.selectbox(
        "Select Experiment Package to Inspect",
        options=list(EXPERIMENT_PACKAGES.keys()),
        format_func=lambda x: EXPERIMENT_PACKAGES[x]
    )

with exp_col2:
    render_html("""
    <div style="margin-top: 28px; font-size: 0.82rem; color: #94A3B8;">
        Algorithm: <b>MiniBatchKMeans (K=5)</b> | Features: <b>6 Grammatical Classes</b>
    </div>
    """)

st.write("")

# Section 1: Overview Charts Row
c_row1, c_row2 = st.columns([1, 1.2])

with c_row1:
    render_html("""
    <div class="darkid-card">
        <div class="darkid-card-title">Cluster Document Allocation</div>
    </div>
    """)
    donut_fig = create_cluster_distribution_chart(selected_exp)
    st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})

with c_row2:
    render_html("""
    <div class="darkid-card">
        <div class="darkid-card-title">Multi-Cluster Stylometric Centroids</div>
    </div>
    """)
    radar_fig = create_multi_cluster_radar(selected_exp)
    st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

st.write("")

# Section 2: Detailed Cluster Archetype Cards
st.markdown("### Discovered Stylometric Archetypes (Personas)")

personas = CLUSTER_PERSONAS.get(selected_exp, {})
raw_centers = get_raw_cluster_centroids(selected_exp)

tabs = st.tabs([f"Cluster {i}: {personas.get(i, {}).get('name', 'Profile')}" for i in range(5)])

for cid, tab in enumerate(tabs):
    with tab:
        p_info = personas.get(cid, {})
        c_vals = raw_centers[cid]

        col_a, col_b = st.columns([1.2, 1])

        with col_a:
            traits_html = "".join(f'<span style="background: #0B0E14; border: 1px solid #1E2536; padding: 4px 12px; border-radius: 6px; font-size: 0.82rem; color: #E2E8F0;">{t}</span>' for t in p_info['dominant_traits'])
            render_html(f"""
            <div class="darkid-card" style="border-left: 4px solid {p_info['color']}; margin-bottom: 15px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span class="pill-badge" style="background: rgba(168, 85, 247, 0.2); color: {p_info['color']}; border: 1px solid {p_info['color']}88;">
                        Cluster {cid} Archetype
                    </span>
                    <span style="font-size: 0.85rem; color: #94A3B8;">Partition ID: #{cid}</span>
                </div>
                <h3 style="color: #F8FAFC; margin: 4px 0 10px 0;">{p_info['name']}</h3>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                    {p_info['description']}
                </p>
                <div style="margin-top: 15px;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #64748B; font-weight: 600; margin-bottom: 8px;">
                        Dominant Stylometric Markers
                    </div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        {traits_html}
                    </div>
                </div>
            </div>
            """)

        with col_b:
            render_html("""
            <div class="darkid-card">
                <div class="darkid-card-title">Centroid Function Word Frequencies (% per 100 words)</div>
            </div>
            """)

            bar_df = pd.DataFrame({
                "Feature": [FEATURE_DISPLAY_NAMES[col] for col in FEATURE_NAMES],
                "Centroid Freq": [round(c_vals[i], 2) for i in range(len(FEATURE_NAMES))],
                "Corpus Avg": [round(raw_centers[:, i].mean(), 2) for i in range(len(FEATURE_NAMES))]
            })

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=bar_df["Feature"],
                y=bar_df["Centroid Freq"],
                name=f"Cluster {cid} Centroid",
                marker=dict(color=p_info["color"])
            ))
            fig_bar.add_trace(go.Bar(
                x=bar_df["Feature"],
                y=bar_df["Corpus Avg"],
                name="Corpus Average",
                marker=dict(color="#475569")
            ))
            fig_bar.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8", size=10),
                margin=dict(l=10, r=10, t=10, b=10),
                height=220,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

st.write("")

# Section 3: Cross-Corpus Source Composition
st.markdown("### Cross-Corpus Source Composition Analysis")
st.markdown("""
Our Stage 4.5.7.4 source-dominance audit investigated whether unsupervised clusters capture true stylistic personas
or reflect corpus origin. Below is the empirical dataset breakdown for each cluster.
""")

source_file = PROJECT_DIR / "stage4_svd" / "stylometric_only" / "cluster_analysis" / "source_dominance" / f"{selected_exp}_k5_cluster_source_composition.csv"

if source_file.exists():
    src_df = pd.read_csv(source_file)
    src_df = src_df.rename(columns={str(i): f"Cluster {i}" for i in range(5)})

    melted = src_df.melt(id_vars=["dataset"], var_name="Cluster", value_name="Percentage")
    
    fig_stack = px.bar(
        melted,
        x="Cluster",
        y="Percentage",
        color="dataset",
        title=f"Source Dataset Composition per Cluster ({selected_exp.replace('_', ' ').title()})",
        labels={"Percentage": "Contribution (%)", "dataset": "Source Dataset"},
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig_stack.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Inter"),
        yaxis=dict(gridcolor="#1E2536", title="Contribution to Cluster (%)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=380
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    with st.expander("🔍 View Raw Source Composition Data Table"):
        st.dataframe(src_df, use_container_width=True)

# Mandatory Disclaimer
render_html("""
<div class="disclaimer-banner">
    <div class="disclaimer-title">
        <span>⚠️</span> Cluster ID Semantics Notice
    </div>
    Cluster IDs (0 through 4) are arbitrary indices assigned by the MiniBatchKMeans optimization routine.
    They represent mathematical modes of function-word density and do <b>not</b> correspond to fixed author identities.
</div>
""")

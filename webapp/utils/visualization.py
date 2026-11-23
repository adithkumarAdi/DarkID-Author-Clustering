"""
DarkID Visualization Module
===========================
Generates responsive, dark-themed Plotly charts adhering to the
cyber-research aesthetic (deep charcoal, neon violet, cyan, and amber).
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .feature_extraction import FEATURE_NAMES, FEATURE_DISPLAY_NAMES
from .model_loader import CLUSTER_PERSONAS, get_raw_cluster_centroids

# Dark Cyber Palette
DARK_BG = "#0B0E14"
CARD_BG = "#131722"
BORDER_COLOR = "#232936"
TEXT_COLOR = "#E2E8F0"
MUTED_TEXT = "#94A3B8"
NEON_PURPLE = "#A855F7"
NEON_CYAN = "#38BDF8"
NEON_AMBER = "#F59E0B"
NEON_ROSE = "#EC4899"
NEON_EMERALD = "#10B981"

CLUSTER_COLORS = ["#38BDF8", "#F59E0B", "#94A3B8", "#A855F7", "#EC4899"]


def _base_layout(title: str = "") -> Dict[str, Any]:
    """Standard layout template for DarkID Plotly charts."""
    return dict(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=15, family="Inter, sans-serif")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED_TEXT, family="Inter, sans-serif"),
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(
            font=dict(color=TEXT_COLOR, size=11),
            bgcolor="rgba(19, 23, 34, 0.7)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
        ),
    )


def create_radar_comparison(
    extracted_features: Dict[str, float],
    predicted_cluster: int,
    experiment: str = "random_holdout",
) -> go.Figure:
    """
    Radar chart comparing the user's document stylometric profile against
    the assigned cluster centroid and overall corpus baseline.
    """
    categories = [FEATURE_DISPLAY_NAMES[col] for col in FEATURE_NAMES]
    doc_vals = [extracted_features[col] for col in FEATURE_NAMES]

    # Cluster centroid in raw feature units
    raw_centers = get_raw_cluster_centroids(experiment)
    centroid_vals = raw_centers[predicted_cluster].tolist()

    # Corpus overall mean
    corpus_means = raw_centers.mean(axis=0).tolist()

    # Close the radar loop
    r_categories = categories + [categories[0]]
    r_doc = doc_vals + [doc_vals[0]]
    r_centroid = centroid_vals + [centroid_vals[0]]
    r_corpus = corpus_means + [corpus_means[0]]

    fig = go.Figure()

    # Corpus Baseline
    fig.add_trace(go.Scatterpolar(
        r=r_corpus,
        theta=r_categories,
        name="Corpus Average",
        line=dict(color="#64748B", width=1.5, dash="dash"),
        fill=None,
    ))

    # Assigned Cluster Centroid
    fig.add_trace(go.Scatterpolar(
        r=r_centroid,
        theta=r_categories,
        name=f"Cluster {predicted_cluster} Centroid",
        line=dict(color=NEON_PURPLE, width=2.5),
        fill="toself",
        fillcolor="rgba(168, 85, 247, 0.15)",
    ))

    # User Input Document
    fig.add_trace(go.Scatterpolar(
        r=r_doc,
        theta=r_categories,
        name="Submitted Document",
        line=dict(color=NEON_CYAN, width=3),
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.25)",
        marker=dict(size=6, color=NEON_CYAN),
    ))

    fig.update_layout(
        **_base_layout("Stylometric Profile Radar (vs. Cluster Centroid)"),
        polar=dict(
            bgcolor="rgba(19, 23, 34, 0.4)",
            radialaxis=dict(
                visible=True,
                showline=False,
                gridcolor="#232936",
                tickfont=dict(color="#64748B", size=9),
            ),
            angularaxis=dict(
                gridcolor="#232936",
                linecolor="#232936",
                tickfont=dict(color=TEXT_COLOR, size=11, family="Inter, sans-serif"),
            ),
        ),
        height=380,
    )
    return fig


def create_affinity_gauge(
    nearest_distance: float,
    all_distances: Dict[int, float],
) -> go.Figure:
    """
    Radial gauge chart indicating Cluster Affinity (resemblance strength).
    Normalized inversely against centroid distances, styled like the Fortexa gauge.
    """
    # Normalized score: lower distance -> higher affinity score (capped between 10% and 98%)
    # Baseline distances in scaled space typically range from 0.5 to 5.0
    affinity_score = max(10, min(98, int(100 / (1.0 + nearest_distance * 0.7))))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=affinity_score,
        number=dict(suffix="%", font=dict(color=TEXT_COLOR, size=32, family="Inter")),
        title=dict(text="Cluster Affinity Index", font=dict(color=MUTED_TEXT, size=13)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=BORDER_COLOR, tickfont=dict(color="#64748B")),
            bar=dict(color=NEON_PURPLE, thickness=0.25),
            bgcolor=CARD_BG,
            borderwidth=1,
            bordercolor=BORDER_COLOR,
            steps=[
                dict(range=[0, 40], color="rgba(236, 72, 153, 0.2)"),
                dict(range=[40, 75], color="rgba(245, 158, 11, 0.2)"),
                dict(range=[75, 100], color="rgba(168, 85, 247, 0.3)"),
            ],
            threshold=dict(
                line=dict(color=NEON_CYAN, width=3),
                thickness=0.75,
                value=affinity_score,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        height=240,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


def create_centroid_distances_bar(distances: Dict[int, float], predicted_cluster: int) -> go.Figure:
    """Horizontal bar chart showing distance to all 5 cluster centers."""
    c_ids = list(distances.keys())
    c_labels = [f"Cluster {i}" for i in c_ids]
    c_dists = [distances[i] for i in c_ids]
    colors = [NEON_PURPLE if i == predicted_cluster else "#334155" for i in c_ids]

    fig = go.Figure(go.Bar(
        x=c_dists,
        y=c_labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color=BORDER_COLOR, width=1)),
        text=[f"{d:.2f}" for d in c_dists],
        textposition="auto",
        textfont=dict(color=TEXT_COLOR),
    ))

    fig.update_layout(
        **_base_layout("Scaled Euclidean Distance to Centroids (Lower = Closer)"),
        xaxis=dict(gridcolor="#232936", zerolinecolor="#232936"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        height=240,
    )
    return fig


def create_cluster_distribution_chart(experiment: str = "random_holdout") -> go.Figure:
    """Donut chart showing document distribution across the 5 clusters."""
    if experiment == "random_holdout":
        docs = [71249, 15575, 17198, 49716, 33698]
    else:
        docs = [44089, 52690, 68772, 15573, 3465]

    labels = [f"Cluster {i}" for i in range(5)]
    personas = CLUSTER_PERSONAS.get(experiment, {})
    hover_names = [personas.get(i, {}).get("name", f"Cluster {i}") for i in range(5)]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=docs,
        hole=0.55,
        marker=dict(colors=CLUSTER_COLORS, line=dict(color=DARK_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(color=TEXT_COLOR, size=12),
        hovertext=hover_names,
        hovertemplate="<b>%{label} (%{hovertext})</b><br>Documents: %{value:,}<br>Proportion: %{percent}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(f"Training Document Distribution ({experiment.replace('_', ' ').title()})"),
        showlegend=True,
        height=320,
    )
    return fig


def create_multi_cluster_radar(experiment: str = "random_holdout") -> go.Figure:
    """Multi-line radar chart displaying the stylometric fingerprints of all 5 clusters."""
    categories = [FEATURE_DISPLAY_NAMES[col] for col in FEATURE_NAMES]
    raw_centers = get_raw_cluster_centroids(experiment)
    personas = CLUSTER_PERSONAS.get(experiment, {})

    r_categories = categories + [categories[0]]
    fig = go.Figure()

    for c in range(5):
        vals = raw_centers[c].tolist() + [raw_centers[c][0]]
        p_name = personas.get(c, {}).get("name", f"Cluster {c}")
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=r_categories,
            name=f"Cluster {c} ({p_name})",
            line=dict(color=CLUSTER_COLORS[c], width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        **_base_layout("All 5 Cluster Centroid Fingerprints (% per 100 words)"),
        polar=dict(
            bgcolor="rgba(19, 23, 34, 0.4)",
            radialaxis=dict(visible=True, gridcolor="#232936", tickfont=dict(color="#64748B", size=8)),
            angularaxis=dict(gridcolor="#232936", tickfont=dict(color=TEXT_COLOR, size=10)),
        ),
        height=450,
    )
    return fig


def create_performance_comparison_chart() -> go.Figure:
    """Grouped bar chart comparing Random Holdout vs Unseen Source test metrics."""
    metrics = ["Silhouette Score (x10)", "Davies-Bouldin (Lower Better)", "Calinski-Harabasz (/1000)"]
    # Scaled for visually pleasing side-by-side comparison
    rh_vals = [0.140539 * 10, 1.714448, 5230.51 / 1000]
    us_vals = [0.132707 * 10, 1.722753, 5051.63 / 1000]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Random Holdout (40,199 test docs)",
        x=metrics,
        y=rh_vals,
        marker=dict(color=NEON_PURPLE, line=dict(color=BORDER_COLOR, width=1)),
        text=["0.1405", "1.7144", "5,230.5"],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR),
    ))
    fig.add_trace(go.Bar(
        name="Unseen Source (61,740 test docs)",
        x=metrics,
        y=us_vals,
        marker=dict(color=NEON_CYAN, line=dict(color=BORDER_COLOR, width=1)),
        text=["0.1327", "1.7228", "5,051.6"],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR),
    ))

    fig.update_layout(
        **_base_layout("Evaluation Metrics Comparison on Independent Test Sets"),
        barmode="group",
        yaxis=dict(gridcolor="#232936", zerolinecolor="#232936"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=360,
    )
    return fig

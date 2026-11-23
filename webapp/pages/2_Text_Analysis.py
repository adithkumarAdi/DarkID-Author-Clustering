"""
Page 2 — Text Analysis / Author Clustering
==========================================
Interactive inference engine for single document and batch multi-document
stylometric feature extraction and cluster assignment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

from utils.prediction import predict_single_document, predict_batch_documents
from utils.visualization import (
    create_radar_comparison,
    create_affinity_gauge,
    create_centroid_distances_bar,
    CLUSTER_COLORS
)
from utils.feature_extraction import FEATURE_NAMES, FEATURE_DISPLAY_NAMES
from utils.model_loader import EXPERIMENT_PACKAGES, CLUSTER_PERSONAS
from utils.ui import render_html

render_html("""
<div class="darkid-hero">
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span class="pill-badge pill-purple">Interactive Inference</span>
        <span class="pill-badge pill-cyan">Stylometric Feature Engine</span>
    </div>
    <h1 class="darkid-title">Text Stylometric Analysis & Cluster Assignment</h1>
    <p class="darkid-subtitle">
        Extract the 6 canonical function-word frequencies, standardize against training distributions, and infer cluster assignment.
    </p>
</div>
""")

# Model Package Configuration
cfg_col1, cfg_col2 = st.columns([1.5, 1])

with cfg_col1:
    selected_experiment = st.selectbox(
        "Select Model Package",
        options=list(EXPERIMENT_PACKAGES.keys()),
        format_func=lambda x: EXPERIMENT_PACKAGES[x],
        help="Choose which saved model package to use for inference. Random Holdout is trained across all balanced corpora; Unseen Source tests out-of-domain robustness."
    )

with cfg_col2:
    render_html("""
    <div style="margin-top: 28px; font-size: 0.82rem; color: #94A3B8;">
        Loaded: <b>model_k5.joblib</b> & <b>scaler.joblib</b> from <code>stage4_svd/stylometric_only/final_model/</code>
    </div>
    """)

# Analysis Mode Selector (Tabs)
analysis_tab1, analysis_tab2 = st.tabs(["📄 Single Document Analysis", "📁 Batch Documents Analysis (CSV / Multi)"])

# -----------------------------------------------------------------------------
# TAB 1: SINGLE DOCUMENT ANALYSIS
# -----------------------------------------------------------------------------
with analysis_tab1:
    PRESETS = {
        "— Select a Sample Preset (Optional) —": "",
        "Dark-Web Forum: OPSEC, PGP & VPN Advisory": (
            "Basically the whole issue with the vpn thing is because of the billing info, but if you use one that does "
            "not have to answer to the usa and also does not keep any billing info or logs then it is completely fine. "
            "Monopoly marketplace works great. It forces you to use pgp and you can never be too safe with your vendor keys."
        ),
        "Dark-Web Forum: Marketplace Vendor Inquiry": (
            "They can track through the blockchain transactions, if you wanna be able to use cashapp without being "
            "banned as quick send to a personal wallet then tumble into monero and then send to the market. "
            "What about research chemicals? Are the escrow times still 14 days or did they change the rules again?"
        ),
        "Academic Technical Prose (arXiv)": (
            "In this paper, we formulate an unsupervised framework for stylometric attribution across heterogeneous text domains. "
            "We examine the mathematical properties of function-word distributions and demonstrate that lexical diversity "
            "indices exhibit substantial variance when evaluated on cross-corpus validation partitions."
        ),
        "Victorian Literature Excerpt (Formal 19th Century)": (
            "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, "
            "it was the epoch of belief, it was the epoch of incredulity, it was the season of light, it was the season of darkness, "
            "we had everything before us, we had nothing before us, we were all going direct to heaven."
        ),
        "Informal Blog Entry (First-Person Conversational)": (
            "I ended up learning how to cook properly because all the restaurants were completely closed when everything started. "
            "I never bought anything fancy, but I really enjoyed spending time in my small kitchen trying out new pasta recipes with my friends."
        )
    }

    p_col1, p_col2 = st.columns([1.5, 1])
    with p_col1:
        preset_choice = st.selectbox("Load Sample Stylometric Document", list(PRESETS.keys()))
    
    with p_col2:
        uploaded_file = st.file_uploader("Or Upload .TXT File", type=["txt"], key="single_txt_upload")

    initial_text = ""
    if uploaded_file is not None:
        try:
            initial_text = uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif preset_choice and PRESETS[preset_choice]:
        initial_text = PRESETS[preset_choice]

    user_text = st.text_area(
        "Document Text Input",
        value=initial_text,
        height=180,
        placeholder="Type or paste document text here for stylometric analysis (minimum 15-25 words recommended)..."
    )

    t_words = len(user_text.split()) if user_text else 0
    t_chars = len(user_text) if user_text else 0
    render_html(f"""
    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748B; margin-top: -10px; margin-bottom: 15px;">
        <span>Word Count: <b>{t_words}</b> words</span>
        <span>Character Count: <b>{t_chars}</b> chars</span>
    </div>
    """)

    analyze_button = st.button("🚀 Analyze Document & Predict Cluster", use_container_width=True)

    if analyze_button or user_text:
        if not user_text.strip():
            st.warning("Please enter or load some text to perform stylometric clustering.")
        else:
            with st.spinner("Extracting stylometric features & inferring cluster assignment..."):
                res = predict_single_document(user_text, experiment=selected_experiment)

            if not res["success"]:
                st.error(f"Analysis failed: {res['error']}")
            else:
                cid = res["cluster_id"]
                persona = res["persona"]
                stats = res["stats"]

                st.write("")
                st.markdown("### Inference Results")

                res_col1, res_col2, res_col3, res_col4 = st.columns([1.2, 1, 1, 1])

                with res_col1:
                    render_html(f"""
                    <div class="darkid-card" style="border-left: 4px solid {persona['color']};">
                        <div class="darkid-card-title">Predicted Cluster</div>
                        <div class="darkid-card-value" style="color: {persona['color']};">Cluster {cid}</div>
                        <div class="darkid-card-subtitle" style="font-weight: 600; color: #E2E8F0;">{persona['name']}</div>
                    </div>
                    """)

                with res_col2:
                    render_html(f"""
                    <div class="darkid-card">
                        <div class="darkid-card-title">Document Length</div>
                        <div class="darkid-card-value">{stats['word_count']}</div>
                        <div class="darkid-card-subtitle">{stats['sentence_count']} detected sentences</div>
                    </div>
                    """)

                with res_col3:
                    render_html(f"""
                    <div class="darkid-card">
                        <div class="darkid-card-title">Avg Word Length</div>
                        <div class="darkid-card-value">{stats['avg_word_length']} <span style="font-size: 1rem;">ch</span></div>
                        <div class="darkid-card-subtitle">Lexical complexity proxy</div>
                    </div>
                    """)

                with res_col4:
                    runner_up = res["runner_up_cluster"]
                    runner_persona = CLUSTER_PERSONAS.get(selected_experiment, {}).get(runner_up, {}).get("name", f"Cluster {runner_up}")
                    render_html(f"""
                    <div class="darkid-card">
                        <div class="darkid-card-title">Runner-Up Cluster</div>
                        <div class="darkid-card-value" style="color: #94A3B8;">Cluster {runner_up}</div>
                        <div class="darkid-card-subtitle">{runner_persona[:20]}...</div>
                    </div>
                    """)

                if res.get("warning"):
                    st.warning(f"⚠️ {res['warning']}")

                st.write("")

                # Persona Description Box
                dominant_traits_html = "".join(f'<span style="font-size: 0.78rem; background: #0F131D; border: 1px solid #232936; padding: 3px 10px; border-radius: 6px; color: #CBD5E1;">{trait}</span>' for trait in persona['dominant_traits'])
                render_html(f"""
                <div style="background: rgba(19, 23, 34, 0.85); border: 1px solid #1E2536; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <span class="pill-badge" style="background: rgba(168, 85, 247, 0.2); color: {persona['color']}; border: 1px solid {persona['color']}88;">
                            {persona['badge']} Profile
                        </span>
                        <span style="font-weight: 600; font-size: 0.95rem; color: #F1F5F9;">{persona['name']}</span>
                    </div>
                    <div style="font-size: 0.87rem; color: #94A3B8; margin-bottom: 10px;">
                        {persona['description']}
                    </div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        {dominant_traits_html}
                    </div>
                </div>
                """)

                # Charts Row
                ch_col1, ch_col2 = st.columns([1.3, 1])

                with ch_col1:
                    radar_fig = create_radar_comparison(
                        res["features_raw"],
                        cid,
                        experiment=selected_experiment
                    )
                    st.plotly_chart(radar_fig, use_container_width=True)

                with ch_col2:
                    nearest_d = res["distances"][cid]
                    gauge_fig = create_affinity_gauge(nearest_d, res["distances"])
                    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

                    dist_fig = create_centroid_distances_bar(res["distances"], cid)
                    st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False})

                # Features Table
                st.markdown("#### Extracted 6D Feature Frequencies")
                feat_rows = []
                for col in FEATURE_NAMES:
                    feat_rows.append({
                        "Stylometric Feature": FEATURE_DISPLAY_NAMES[col],
                        "Raw Frequency (per 100 words)": f"{res['features_raw'][col]:.2f}%",
                        "Standardized Value (z-score)": f"{res['features_scaled'][col]:+.3f}",
                    })
                st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# TAB 2: BATCH DOCUMENTS ANALYSIS (CSV)
# -----------------------------------------------------------------------------
with analysis_tab2:
    st.markdown("#### Batch Document Processing")
    st.markdown("Upload a CSV file containing multiple documents. DarkID will extract features and predict cluster assignments for each row.")

    batch_file = st.file_uploader("Upload CSV File", type=["csv"], key="batch_csv_upload")

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            st.success(f"Successfully loaded {len(batch_df):,} rows.")

            text_candidates = [c for c in batch_df.columns if "text" in c.lower() or "content" in c.lower() or "body" in c.lower()]
            default_col = text_candidates[0] if text_candidates else batch_df.columns[0]
            
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                target_col = st.selectbox("Select Text Column", options=batch_df.columns, index=list(batch_df.columns).index(default_col))
            with b_col2:
                id_col = st.selectbox("Select ID Column (Optional)", options=["None (Auto-generate)"] + list(batch_df.columns), index=0)

            run_batch = st.button("⚡ Run Batch Clustering", use_container_width=True)

            if run_batch:
                texts = batch_df[target_col].astype(str).tolist()
                doc_ids = batch_df[id_col].astype(str).tolist() if id_col != "None (Auto-generate)" else None

                with st.spinner(f"Extracting stylometric features for {len(texts):,} documents..."):
                    results_df = predict_batch_documents(texts, doc_ids=doc_ids, experiment=selected_experiment)

                st.write("")
                st.markdown("### Batch Clustering Summary")

                valid_preds = results_df[results_df["Predicted Cluster"] >= 0]
                cluster_counts = valid_preds["Predicted Cluster"].value_counts().sort_index()

                scol1, scol2, scol3 = st.columns(3)
                with scol1:
                    st.metric("Total Processed", f"{len(results_df):,}")
                with scol2:
                    st.metric("Successfully Clustered", f"{len(valid_preds):,}")
                with scol3:
                    st.metric("Dominant Cluster", f"Cluster {cluster_counts.idxmax() if len(cluster_counts) > 0 else 'N/A'}")

                st.dataframe(results_df, use_container_width=True, hide_index=True)

                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Batch Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="darkid_batch_clustering_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Failed to process CSV file: {e}")

# Mandatory Disclaimer
render_html("""
<div class="disclaimer-banner">
    <div class="disclaimer-title">
        <span>⚠️</span> Unsupervised Attribution Notice
    </div>
    A cluster represents a statistical partition of texts sharing similar function-word frequency profiles.
    Unsupervised clustering does <b>not</b> constitute forensic proof of individual real-world identity.
</div>
""")

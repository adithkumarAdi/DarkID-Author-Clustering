"""
DarkID — Unsupervised Stylometric Author Clustering Platform
============================================================
Main Streamlit Application Entrypoint.
"""

from pathlib import Path
import streamlit as st
from utils.ui import render_html

# Configure page metadata
st.set_page_config(
    page_title="DarkID — Stylometric Author Clustering",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load and inject custom dark cyber stylesheet
CSS_PATH = Path(__file__).resolve().parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar Branding & Global Info
with st.sidebar:
    render_html("""
    <div style="padding: 12px 4px 18px 4px; border-bottom: 1px solid #1E2536; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: linear-gradient(135deg, #7C3AED, #38BDF8); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                🛡️
            </div>
            <div>
                <div style="font-weight: 700; font-size: 1.25rem; letter-spacing: -0.02em; color: #F8FAFC;">
                    Dark<span style="color: #A855F7;">ID</span>
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">
                    Author Clustering v2.0
                </div>
            </div>
        </div>
    </div>
    """)

    render_html("""
    <div style="background: #121622; border: 1px solid #1E2536; border-radius: 10px; padding: 12px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span style="width: 8px; height: 8px; background: #10B981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10B981;"></span>
            <span style="font-size: 0.8rem; font-weight: 600; color: #E2E8F0;">Inference Pipeline Ready</span>
        </div>
        <div style="font-size: 0.74rem; color: #94A3B8;">
            <b>Artifact:</b> MiniBatchKMeans (K=5)<br/>
            <b>Features:</b> 6 Grammatical Classes<br/>
            <b>Corpus:</b> 267,765 Documents
        </div>
    </div>
    """)

# Register navigation pages
pages = [
    st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊", default=True),
    st.Page("pages/2_Text_Analysis.py", title="Text Analysis", icon="🔍"),
    st.Page("pages/3_Cluster_Explorer.py", title="Cluster Explorer", icon="🌐"),
    st.Page("pages/4_Model_Performance.py", title="Model Performance", icon="📈"),
    st.Page("pages/5_Methodology.py", title="Methodology", icon="🧬"),
    st.Page("pages/6_About.py", title="About Project", icon="ℹ️"),
]

nav = st.navigation(pages)

# Sidebar footer
with st.sidebar:
    render_html("""
    <div style="margin-top: 40px; padding-top: 15px; border-top: 1px solid #1E2536; font-size: 0.72rem; color: #64748B; text-align: center;">
        DarkID Stylometric Forensics<br/>
        B.Tech Unsupervised Learning Project<br/>
        Strict Zero-Leakage Architecture
    </div>
    """)

# Execute active page
nav.run()

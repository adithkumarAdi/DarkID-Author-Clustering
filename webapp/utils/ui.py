"""
DarkID UI Utilities
===================
Safe HTML rendering helpers that prevent Streamlit's Markdown engine
from misinterpreting indented HTML as <pre><code> code blocks.
"""

import textwrap
import streamlit as st


def render_html(content: str) -> None:
    """
    Safely render raw HTML in Streamlit.
    Dedents the string to remove common leading indentation and uses st.html()
    to bypass Markdown's 4-space code block parser completely.
    """
    clean_html = textwrap.dedent(content).strip()
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)

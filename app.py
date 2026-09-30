"""Streamlit application for ECGRHYTHMIA wearable ECG evaluation.

Run with::

    streamlit run app.py

Workflow: Dataset → Annotation → Analysis → Interpretation, with Reconstruction
available independently to assemble canonical frames from a raw Raspberry Pi
recording and generate the website dataset.
"""

from __future__ import annotations

import streamlit as st

# Importing app_ui puts src/ on sys.path before any ecg_eval import happens.
from app_ui import PAGES
from app_ui import (
    page_analysis,
    page_annotation,
    page_dataset,
    page_interpretation,
    page_reconstruction,
)
from app_ui.common import REFERENCE, init_state, sidebar

st.set_page_config(
    page_title="ECGRHYTHMIA ECG Evaluation",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES_MAP = {
    "Dataset": page_dataset,
    "Annotation": page_annotation,
    "Analysis": page_analysis,
    "Interpretation": page_interpretation,
    "Reconstruction": page_reconstruction,
}

init_state()

ctx = sidebar()

page = st.radio(
    "Navigation", PAGES, horizontal=True, label_visibility="collapsed",
    key="nav_page",
)
st.divider()

try:
    PAGES_MAP[page].render(ctx)
except Exception as exc:  # noqa: BLE001 - the UI must not die on one bad frame
    st.error(f"The {page} page raised an error: {exc}")
    st.exception(exc)
    st.caption(
        "The scientific layer is importable from Python independently of this UI, so a "
        "rendering failure here does not affect the underlying computation."
    )

st.divider()
st.caption(f"Methodology reference: {REFERENCE}")

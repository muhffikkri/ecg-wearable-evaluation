"""Streamlit page modules. Presentation only -- no scientific logic here.

Importing this package puts ``src/`` on ``sys.path`` so the Streamlit entry
point works from a clean checkout without an editable install
(IDEA.md section 57, scope 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PAGES = ["Dataset", "Annotation", "Analysis", "Interpretation", "Reconstruction"]

__all__ = [
    "PAGES",
    "REPO_ROOT",
    "SRC",
    "common",
    "page_analysis",
    "page_annotation",
    "page_dataset",
    "page_interpretation",
    "page_reconstruction",
]

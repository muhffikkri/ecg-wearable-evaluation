"""ECGRHYTHMIA wearable ECG evaluation.

Signal-quality evaluation of ECG recorded by the ECGRHYTHMIA wearable prototype
under three static body positions (supine, sitting, standing), following the SQI
methodology of Zhao & Zhang (2018).

This package is the scientific layer. The Streamlit app in ``app.py`` is only
a presentation and orchestration layer and must not contain scientific
calculations (IDEA.md section 64).

This project evaluates ECG SIGNAL QUALITY. It is not a diagnostic tool and
must not be used to infer arrhythmia or any clinical condition.
"""

__version__ = "0.1.0"

REFERENCE = (
    "Zhao, Z., & Zhang, Y. (2018). SQI Quality Evaluation Mechanism of "
    "Single-Lead ECG Signal Based on Simple Heuristic Fusion and Fuzzy "
    "Comprehensive Evaluation. Frontiers in Physiology, 9, 727."
)

__all__ = ["__version__", "REFERENCE"]

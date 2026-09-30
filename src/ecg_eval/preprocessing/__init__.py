"""Preprocessing with independently toggleable filter stages.

Re-exports from :mod:`ecg_eval.preprocessing.pipeline`, whose DSP stages are
ported from the dashboard engine at ``templates/preprocessing.py``.
"""

from .pipeline import (
    DEFAULT_ENABLED,
    STAGE_ORDER,
    PreprocessedSignal,
    apply_butter_bandpass,
    apply_median_baseline,
    apply_notch,
    apply_poly_resample,
    apply_wavelet_denoising,
    apply_zscore_clip,
    default_config,
    describe_stages,
    enabled_stages,
    preprocess,
    resolve_config,
    sanitize_signal,
    stage_enabled,
)

__all__ = [
    "DEFAULT_ENABLED",
    "STAGE_ORDER",
    "PreprocessedSignal",
    "apply_butter_bandpass",
    "apply_median_baseline",
    "apply_notch",
    "apply_poly_resample",
    "apply_wavelet_denoising",
    "apply_zscore_clip",
    "default_config",
    "describe_stages",
    "enabled_stages",
    "preprocess",
    "resolve_config",
    "sanitize_signal",
    "stage_enabled",
]

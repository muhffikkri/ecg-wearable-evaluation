"""ECG preprocessing with individually toggleable stages.

The DSP stages are ported from the web dashboard's engine
(``templates/preprocessing.py``) so this application filters a signal exactly
the way the dashboard does. Every stage can be switched on or off independently
from the UI, and only the enabled stages run.

Defaults follow the dashboard's own pipeline order:

===============  =========  ==================================================
stage            default    rationale
===============  =========  ==================================================
wavelet          **on**     adaptive wavelet thresholding, db4 level 4
median baseline  **on**     median-filter baseline removal, kernel 51
butter bandpass  **on**     0.5-45 Hz, order 4
resample         off        the 250 Hz source is already the analysis rate
notch            off        the device already applied a 50 Hz notch on-device
zscore + clip    off        excluded on request: normalisation is not applied,
                             so amplitudes stay in mV and pSQI/basSQI keep
                             their physical meaning
===============  =========  ==================================================

Nothing here mutates the caller's array. When no stage is enabled the input
comes back as a read-only view, so the raw signal can never be overwritten by a
processing step (IDEA.md sections 13 and 44).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, medfilt, resample_poly

#: Stage order. Fixed so a run is reproducible regardless of toggle order.
STAGE_ORDER = ("wavelet", "baseline", "bandpass", "resample", "notch", "normalize")

#: Stages enabled by default. Everything else starts off.
DEFAULT_ENABLED: tuple[str, ...] = ("wavelet", "baseline", "bandpass")

DEFAULT_WAVELET = "db4"
DEFAULT_WAVELET_LEVEL = 4
DEFAULT_MEDIAN_KERNEL = 51
DEFAULT_BANDPASS_LOW_HZ = 0.5
DEFAULT_BANDPASS_HIGH_HZ = 45.0
DEFAULT_BANDPASS_ORDER = 4
DEFAULT_NOTCH_HZ = 50.0
DEFAULT_NOTCH_Q = 30.0
DEFAULT_CLIP_MIN = -5.0
DEFAULT_CLIP_MAX = 5.0
MIN_SIGNAL_SAMPLES = 32


@dataclass
class PreprocessedSignal:
    """A processed copy of a signal plus an exact record of what was applied."""

    signal: np.ndarray
    applied: bool
    config: dict[str, Any] = field(default_factory=dict)
    #: Stage name -> enabled, so the UI and the report show the full toggle
    #: state rather than only the stages that happened to do something.
    stages: dict[str, bool] = field(default_factory=dict)

    @property
    def enabled_stages(self) -> list[str]:
        return [name for name in STAGE_ORDER if self.stages.get(name)]

    @property
    def disabled_stages(self) -> list[str]:
        return [name for name in STAGE_ORDER if name in self.stages and not self.stages[name]]

    @property
    def summary(self) -> str:
        if not self.applied:
            return "not applied"
        return ", ".join(self.enabled_stages)


# ---------------------------------------------------------------------------
# Safety utilities (ported from templates/preprocessing.py)
# ---------------------------------------------------------------------------
def sanitize_signal(signal: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with zero before any DSP stage runs."""
    return np.nan_to_num(np.asarray(signal, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def validate_signal_shape(signal: np.ndarray) -> np.ndarray:
    """Require ``[timesteps, channels]`` and a workable length."""
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D [T, C], got shape {signal.shape}")
    if signal.shape[0] < MIN_SIGNAL_SAMPLES:
        raise ValueError(f"signal too short to process ({signal.shape[0]} samples)")
    return signal


def ensure_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Centre-crop when too long, zero-pad when too short."""
    signal = sanitize_signal(signal)
    current = signal.shape[0]
    if current == target_len:
        return signal
    if current > target_len:
        start = (current - target_len) // 2
        return signal[start : start + target_len, :]
    return np.pad(
        signal, ((0, target_len - current), (0, 0)),
        mode="constant", constant_values=0.0,
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def apply_wavelet_denoising(
    data: np.ndarray, wavelet: str = DEFAULT_WAVELET, level: int = DEFAULT_WAVELET_LEVEL
) -> np.ndarray:
    """Adaptive wavelet denoising, per channel.

    Threshold is ``std(approx_detail) / 2`` with soft thresholding, matching the
    dashboard engine. Each channel is decomposed independently so a channel
    with a different noise floor gets its own threshold.
    """
    data = sanitize_signal(data)
    out = np.zeros_like(data)
    max_level = pywt.dwt_max_level(data.shape[0], pywt.Wavelet(wavelet).dec_len)
    safe_level = max(1, min(int(level), max_level))

    for channel in range(data.shape[1]):
        coeffs = pywt.wavedec(data[:, channel], wavelet, level=safe_level)
        threshold = float(np.std(coeffs[-1])) / 2.0
        coeffs[1:] = [pywt.threshold(c, value=threshold, mode="soft") for c in coeffs[1:]]
        out[:, channel] = pywt.waverec(coeffs, wavelet)[: data.shape[0]]
    return sanitize_signal(out)


def apply_median_baseline(data: np.ndarray, kernel_size: int = DEFAULT_MEDIAN_KERNEL) -> np.ndarray:
    """Baseline wander removal by subtracting a running median.

    Chosen over a mean because it is robust to R-peak spikes and therefore
    preserves QRS morphology.
    """
    data = sanitize_signal(data)
    kernel = int(kernel_size)
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(3, min(kernel, data.shape[0]))

    out = np.zeros_like(data)
    for channel in range(data.shape[1]):
        out[:, channel] = data[:, channel] - medfilt(data[:, channel], kernel_size=kernel)
    return sanitize_signal(out)


def apply_butter_bandpass(
    signal: np.ndarray,
    sampling_rate: float,
    low_hz: float = DEFAULT_BANDPASS_LOW_HZ,
    high_hz: float = DEFAULT_BANDPASS_HIGH_HZ,
    order: int = DEFAULT_BANDPASS_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass with automatic Nyquist protection."""
    signal = sanitize_signal(signal)
    nyquist = 0.5 * float(sampling_rate)
    if nyquist <= 0:
        return signal

    low = max(low_hz / nyquist, 0.001)
    high = min(high_hz / nyquist, 0.99)
    if low >= high:
        return signal

    try:
        b, a = butter(int(order), [low, high], btype="band")
        return sanitize_signal(filtfilt(b, a, signal, axis=0))
    except ValueError:
        # Too short for the requested filter order: pass through unchanged
        # rather than failing the whole frame.
        return signal


def apply_poly_resample(signal: np.ndarray, src_fs: float, target_fs: float) -> np.ndarray:
    """Polyphase FIR resampling."""
    signal = sanitize_signal(signal)
    if src_fs == target_fs:
        return signal
    src, target = int(src_fs), int(target_fs)
    divisor = int(np.gcd(src, target))
    return sanitize_signal(resample_poly(signal, target // divisor, src // divisor, axis=0))


def apply_notch(
    signal: np.ndarray, sampling_rate: float, frequency: float, quality: float = DEFAULT_NOTCH_Q
) -> np.ndarray:
    """Single-frequency notch. Off by default: see the module docstring."""
    from scipy.signal import iirnotch

    signal = sanitize_signal(signal)
    nyquist = 0.5 * float(sampling_rate)
    if nyquist <= 0 or frequency >= nyquist:
        return signal
    try:
        b, a = iirnotch(frequency / nyquist, quality)
        return sanitize_signal(filtfilt(b, a, signal, axis=0))
    except ValueError:
        return signal


def apply_zscore_clip(
    data: np.ndarray,
    epsilon: float = 1e-8,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX,
) -> np.ndarray:
    """Per-channel z-score with clipping.

    Available but OFF by default: the requested analysis keeps amplitudes in
    mV, and z-scoring would remove the amplitude information that pSQI and
    basSQI depend on.
    """
    data = sanitize_signal(data)
    std = np.std(data, axis=0)
    normalised = (data - np.mean(data, axis=0)) / (std + epsilon)
    return sanitize_signal(np.clip(normalised, clip_min, clip_max))


def extract_signal_statistics(signal: np.ndarray) -> dict[str, Any]:
    """Lightweight statistics, used by the reconstruction manifest."""
    signal = sanitize_signal(signal)
    return {
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "shape": tuple(int(v) for v in signal.shape),
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def default_config(**overrides: Any) -> dict[str, Any]:
    """Default stage configuration.

    Single source of truth for the toggles: the UI reads these keys to render
    its switches and :func:`preprocess` reads them to decide what to run, so
    the displayed state and the executed state cannot drift apart.
    """
    config: dict[str, Any] = {
        # master switch; the UI may still override it per run
        "applied": True,
        "wavelet": {
            "enabled": "wavelet" in DEFAULT_ENABLED,
            "wavelet": DEFAULT_WAVELET,
            "level": DEFAULT_WAVELET_LEVEL,
        },
        "baseline": {
            "enabled": "baseline" in DEFAULT_ENABLED,
            "kernel_size": DEFAULT_MEDIAN_KERNEL,
        },
        "bandpass": {
            "enabled": "bandpass" in DEFAULT_ENABLED,
            "low_hz": DEFAULT_BANDPASS_LOW_HZ,
            "high_hz": DEFAULT_BANDPASS_HIGH_HZ,
            "order": DEFAULT_BANDPASS_ORDER,
        },
        "resample": {
            "enabled": "resample" in DEFAULT_ENABLED,
            "target_fs": None,
        },
        "notch": {
            "enabled": "notch" in DEFAULT_ENABLED,
            "frequency_hz": DEFAULT_NOTCH_HZ,
            "quality": DEFAULT_NOTCH_Q,
        },
        "normalize": {
            "enabled": "normalize" in DEFAULT_ENABLED,
            "clip_min": DEFAULT_CLIP_MIN,
            "clip_max": DEFAULT_CLIP_MAX,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def stage_enabled(config: dict[str, Any] | None, stage: str) -> bool:
    """Whether one stage is switched on."""
    if not config:
        return False
    spec = config.get(stage)
    if spec is None:
        return False
    if isinstance(spec, dict):
        return bool(spec.get("enabled", False))
    return bool(spec)


def enabled_stages(config: dict[str, Any] | None) -> list[str]:
    """Enabled stages in canonical execution order."""
    if not config:
        return []
    return [stage for stage in STAGE_ORDER if stage_enabled(config, stage)]


def resolve_config(config: dict[str, Any] | None, enabled: bool | None = None) -> dict[str, Any]:
    """Apply the master switch.

    ``applied`` gates the whole chain. When the caller passes ``enabled``
    explicitly it wins, which is how the UI's "execute preprocessing" switch
    overrides configuration for one run without editing the config file.
    """
    resolved = dict(config or {})
    if "applied" in resolved:
        switch = bool(resolved.pop("applied"))
    else:
        switch = bool(resolved)
    if enabled is not None:
        switch = bool(enabled)
    return resolved if switch else {}


def preprocess(
    signal: np.ndarray,
    sampling_rate: float,
    config: dict[str, Any] | None,
    *,
    enabled: bool | None = None,
) -> PreprocessedSignal:
    """Run the enabled preprocessing stages, in order.

    Only the stages switched on in ``config`` run. The returned
    :class:`PreprocessedSignal` records both the effective parameters and the
    full on/off state of every stage, so downstream analysis and the report can
    state exactly which filters produced the signal they measured.

    The returned array is ``[timesteps, channels]``; pass a single lead as
    ``signal[:, None]``.
    """
    original = np.asarray(signal, dtype=np.float32)
    if original.ndim == 1:
        original = original[:, None]

    resolved = resolve_config(config, enabled)
    record: dict[str, Any] = {}
    stages = {stage: stage_enabled(resolved, stage) for stage in STAGE_ORDER}

    if not any(stages.values()):
        # Nothing enabled: hand back a read-only view so a downstream stage
        # cannot write through to the caller's array.
        view = original.view()
        view.flags.writeable = False
        return PreprocessedSignal(signal=view, applied=False, config=record, stages=stages)

    result = sanitize_signal(validate_signal_shape(original))
    src_fs = float(sampling_rate)

    # 1. wavelet denoising
    if stages["wavelet"]:
        spec = resolved.get("wavelet", {}) or {}
        name = str(spec.get("wavelet", DEFAULT_WAVELET))
        level = int(spec.get("level", DEFAULT_WAVELET_LEVEL))
        result = apply_wavelet_denoising(result, wavelet=name, level=level)
        record["wavelet"] = {"wavelet": name, "level": level}

    # 2. baseline wander removal
    if stages["baseline"]:
        spec = resolved.get("baseline", {}) or {}
        kernel = int(spec.get("kernel_size", DEFAULT_MEDIAN_KERNEL))
        result = apply_median_baseline(result, kernel_size=kernel)
        record["baseline"] = {"method": "median_filter_subtraction", "kernel_size": kernel}

    # 3. bandpass
    if stages["bandpass"]:
        spec = resolved.get("bandpass", {}) or {}
        low = float(spec.get("low_hz", DEFAULT_BANDPASS_LOW_HZ))
        high = float(spec.get("high_hz", DEFAULT_BANDPASS_HIGH_HZ))
        order = int(spec.get("order", DEFAULT_BANDPASS_ORDER))
        result = apply_butter_bandpass(result, src_fs, low_hz=low, high_hz=high, order=order)
        record["bandpass"] = {"low_hz": low, "high_hz": high, "order": order}

    # 4. resample
    if stages["resample"]:
        spec = resolved.get("resample", {}) or {}
        target_fs = spec.get("target_fs")
        if target_fs:
            target_fs = float(target_fs)
            result = apply_poly_resample(result, src_fs, target_fs)
            record["resample"] = {"from_hz": src_fs, "to_hz": target_fs}
            src_fs = target_fs

    # 5. notch
    if stages["notch"]:
        spec = resolved.get("notch", {}) or {}
        frequency = float(spec.get("frequency_hz", DEFAULT_NOTCH_HZ))
        quality = float(spec.get("quality", DEFAULT_NOTCH_Q))
        result = apply_notch(result, src_fs, frequency=frequency, quality=quality)
        record["notch"] = {"frequency_hz": frequency, "quality": quality}

    # 6. normalisation (off by default, on request only)
    if stages["normalize"]:
        spec = resolved.get("normalize", {}) or {}
        clip_min = float(spec.get("clip_min", DEFAULT_CLIP_MIN))
        clip_max = float(spec.get("clip_max", DEFAULT_CLIP_MAX))
        result = apply_zscore_clip(result, clip_min=clip_min, clip_max=clip_max)
        record["normalize"] = {"method": "zscore_clip", "clip": [clip_min, clip_max]}

    return PreprocessedSignal(signal=result, applied=True, config=record, stages=stages)


def describe_stages(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Toggle state of every stage, for the UI and for the report."""
    return [{"stage": stage, "enabled": stage_enabled(config, stage)} for stage in STAGE_ORDER]


__all__ = [
    "DEFAULT_BANDPASS_HIGH_HZ",
    "DEFAULT_BANDPASS_LOW_HZ",
    "DEFAULT_BANDPASS_ORDER",
    "DEFAULT_CLIP_MAX",
    "DEFAULT_CLIP_MIN",
    "DEFAULT_ENABLED",
    "DEFAULT_MEDIAN_KERNEL",
    "DEFAULT_WAVELET",
    "DEFAULT_WAVELET_LEVEL",
    "MIN_SIGNAL_SAMPLES",
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
    "ensure_length",
    "extract_signal_statistics",
    "preprocess",
    "resolve_config",
    "sanitize_signal",
    "stage_enabled",
    "validate_signal_shape",
]

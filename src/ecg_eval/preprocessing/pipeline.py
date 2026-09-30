"""Preprocessing.

The wearable data already carries a 50 Hz notch applied on-device, visible in
``model_ready/frame_*_input.json -> source_processing``. This module therefore
does NOT re-apply a notch by default, and it never mutates the caller's array:
every stage produces a new array, and an untouched run hands back a read-only
view of the input (IDEA.md sections 13 and 44).

Every stage is optional and recorded in the returned config dict, so the
report can state exactly what was done to the signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


@dataclass
class PreprocessedSignal:
    """A processed copy of a signal plus a record of what was applied."""

    signal: np.ndarray
    applied: bool
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if not self.applied:
            return "not applied"
        return ", ".join(f"{k}={v}" for k, v in self.config.items() if v not in (None, False))


def baseline_correct(signal: np.ndarray, method: str = "median") -> np.ndarray:
    """Remove the DC / slow baseline component.

    This is an explicit ADAPTATION: the reference method assumes database
    records, while the ECGRHYTHMIA leads carry a large DC offset (the raw
    metadata reports baselines of roughly -2.7, +9.1 and +11.8 mV). Kurtosis
    and power ratios are both affected by such an offset, so it is removed
    before analysis. Configured in ``configs/zhao_zhang.yaml``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if method == "median":
        return signal - np.median(signal)
    if method == "mean":
        return signal - np.mean(signal)
    if method in (None, "none"):
        return signal
    raise ValueError(f"unknown baseline method {method!r}")


def bandpass(
    signal: np.ndarray,
    sampling_rate: float,
    low_hz: float | None,
    high_hz: float | None,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase bandpass. Both limits may be None to skip that side."""
    if low_hz is None and high_hz is None:
        return np.asarray(signal, dtype=np.float64)
    nyquist = sampling_rate / 2.0
    low_n = (low_hz / nyquist) if low_hz else None
    high_n = (high_hz / nyquist) if high_hz else None
    if low_n is not None and high_n is not None:
        sos = butter(order, [low_n, high_n], btype="band", output="sos")
    elif low_n is not None:
        sos = butter(order, low_n, btype="high", output="sos")
    else:
        sos = butter(order, high_n, btype="low", output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))


def notch(signal: np.ndarray, sampling_rate: float, frequency: float, quality: float = 30.0) -> np.ndarray:
    nyquist = sampling_rate / 2.0
    b, a = iirnotch(frequency / nyquist, quality)
    return filtfilt(b, a, np.asarray(signal, dtype=np.float64))


def resolve_config(config: dict[str, Any] | None, enabled: bool | None = None) -> dict[str, Any]:
    """Resolve the *effective* preprocessing chain for one run.

    ``preprocessing.applied`` is the master switch required by IDEA.md section
    13: a configured chain is only executed when preprocessing has been
    explicitly enabled, so the raw signal stays the default view. Individual
    stages stay configured, they are simply not run.

    ``enabled`` overrides the switch for an explicit, user-initiated run. When
    the key is absent entirely, the chain runs iff any stage is configured,
    which keeps direct library use behaving as written.
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
    """Run the configured preprocessing chain.

    Returns the original signal unchanged when nothing is enabled, so the UI
    can honestly show "preprocessing not applied".
    """
    original = np.asarray(signal, dtype=np.float64)
    resolved = resolve_config(config, enabled)
    record: dict[str, Any] = {}
    result = original

    if resolved.get("remove_baseline"):
        method = str(resolved.get("baseline_method", "median"))
        result = baseline_correct(result, method)
        record["baseline_method"] = method

    low = resolved.get("highpass_hz")
    high = resolved.get("lowpass_hz")
    if low or high:
        result = bandpass(result, sampling_rate, low, high)
        record["bandpass_hz"] = [low, high]

    notch_hz = resolved.get("notch_hz")
    if notch_hz:
        result = notch(result, sampling_rate, float(notch_hz))
        record["notch_hz"] = float(notch_hz)

    applied = bool(record)
    if not applied:
        # Hand back a read-only view so a downstream stage cannot write through
        # to the caller's array while still reporting the signal unaltered.
        result = original.view()
        result.flags.writeable = False
    return PreprocessedSignal(signal=result, applied=applied, config=record)


__all__ = [
    "PreprocessedSignal",
    "bandpass",
    "baseline_correct",
    "notch",
    "preprocess",
    "resolve_config",
]

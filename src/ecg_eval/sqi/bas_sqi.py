"""basSQI: relative power in the baseline (very low frequency) band.

    basSQI = power in the baseline band / power in the total analysis band

A low value means the signal is dominated by QRS-frequency content rather
than by baseline wander. Frequency limits come from configuration and are
stored on the result (IDEA.md section 26).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models.result import SpectralResult
from .p_sqi import band_power, compute_psd


def bas_sqi(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    baseline_band_hz: tuple[float, float] = (0.0, 1.0),
    total_band_hz: tuple[float, float] = (0.0, 40.0),
    nperseg: int = 500,
    method: str = "welch",
) -> tuple[float, SpectralResult]:
    """Compute basSQI and return the stored spectral intermediates."""
    frequencies, power = compute_psd(signal, sampling_rate, nperseg=nperseg, method=method)
    baseline_power = band_power(frequencies, power, *baseline_band_hz)
    total_power = band_power(frequencies, power, *total_band_hz)
    value = (baseline_power / total_power) if total_power > 0 else 0.0

    result = SpectralResult(
        frequencies=[float(f) for f in frequencies],
        psd=[float(p) for p in power],
        baseline_band_power=baseline_power,
        bas_sqi=float(value),
        bas_total_power=total_power,
        psd_method=method,
    )
    return float(value), result


def bas_sqi_config(config: Any) -> dict[str, Any]:
    return {
        "baseline_band_hz": tuple(config.get("bas_sqi.baseline_band_hz", (0.0, 1.0))),
        "total_band_hz": tuple(config.get("bas_sqi.total_band_hz", (0.0, 40.0))),
        "nperseg": int(config.get("bas_sqi.psd_nperseg", 500)),
        "method": str(config.get("p_sqi.psd_method", "welch")),
    }


__all__ = ["bas_sqi", "bas_sqi_config"]

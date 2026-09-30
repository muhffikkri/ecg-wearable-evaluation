"""pSQI: relative power in the QRS-related frequency band.

    pSQI = power in the QRS band / power in the total analysis band

Both frequency limits come from configuration and are recorded on the result.
The intermediate PSD and both band powers are stored, never just the ratio
(IDEA.md section 24).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import welch

from ..models.result import SpectralResult


def compute_psd(
    signal: np.ndarray,
    sampling_rate: float,
    nperseg: int = 500,
    method: str = "welch",
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density of one lead.

    Welch with a 2 s window (500 samples at 250 Hz) as configured; the method
    and window length are part of the stored provenance.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size < 8:
        return np.empty(0), np.empty(0)
    nperseg = int(min(nperseg, signal.size))
    if method == "periodogram":
        from scipy.signal import periodogram

        return periodogram(signal, fs=sampling_rate, nfft=max(256, nperseg))
    frequencies, power = welch(signal, fs=sampling_rate, nperseg=nperseg)
    return frequencies, power


def band_power(
    frequencies: np.ndarray,
    power: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Integrate the PSD over [low_hz, high_hz] using the trapezoid rule."""
    if frequencies.size == 0:
        return 0.0
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], frequencies[mask])) if hasattr(np, "trapezoid") else float(
        np.trapz(power[mask], frequencies[mask])
    )


def p_sqi(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    qrs_band_hz: tuple[float, float] = (5.0, 15.0),
    total_band_hz: tuple[float, float] = (0.5, 40.0),
    nperseg: int = 500,
    method: str = "welch",
) -> tuple[float, SpectralResult]:
    """Compute pSQI and return the stored spectral intermediates."""
    frequencies, power = compute_psd(signal, sampling_rate, nperseg=nperseg, method=method)
    qrs_power = band_power(frequencies, power, *qrs_band_hz)
    total_power = band_power(frequencies, power, *total_band_hz)
    value = (qrs_power / total_power) if total_power > 0 else 0.0

    result = SpectralResult(
        frequencies=[float(f) for f in frequencies],
        psd=[float(p) for p in power],
        qrs_band_power=qrs_power,
        p_sqi=float(value),
        p_total_power=total_power,
        psd_method=method,
    )
    return float(value), result


def p_sqi_config(config: Any) -> dict[str, Any]:
    """Extract pSQI parameters from a :class:`~ecg_eval.config.Config`."""
    return {
        "qrs_band_hz": tuple(config.get("p_sqi.qrs_band_hz", (5.0, 15.0))),
        "total_band_hz": tuple(config.get("p_sqi.total_band_hz", (0.5, 40.0))),
        "nperseg": int(config.get("p_sqi.psd_nperseg", 500)),
        "method": str(config.get("p_sqi.psd_method", "welch")),
    }


__all__ = ["band_power", "compute_psd", "p_sqi", "p_sqi_config"]

"""pSQI: relative power in the QRS-related frequency band.

Rumus (Zhao & Zhang 2018, Eq 3)::

    pSQI = integral(f=5..15 Hz) P(f) df  /  integral(f=5..40 Hz) P(f) df

Pembilang adalah pita QRS (5-15 Hz, sekitar 99 % energi ECG, terpusat di dekat
10 Hz); penyebut adalah energi keseluruhan sinyal pada pita analisis yang
dimulai dari 5 Hz -- bukan dari 0.5 Hz. Kriteria penerimaannya bergantung pada
detak jantung dan ada di :func:`p_sqi_acceptance`.

Both frequency limits come from configuration and are recorded on the result.
The intermediate PSD and both band powers are stored, never just the ratio
(IDEA.md section 24).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import welch

from ..models.result import (
    OPTIMAL,
    SUSPICIOUS,
    UNQUALIFIED,
    UNDEFINED,
    Acceptance,
    SpectralResult,
)
from ._params import unwrap


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
    total_band_hz: tuple[float, float] = (5.0, 40.0),
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
        "total_band_hz": tuple(config.get("p_sqi.total_band_hz", (5.0, 40.0))),
        "nperseg": int(config.get("p_sqi.psd_nperseg", 500)),
        "method": str(config.get("p_sqi.psd_method", "welch")),
    }


#: Heart-rate bands of Eq (5), each with its own l1, l2 and l3.
#:
#: The article calibrated these limits on heart rates from 60 to 160 bpm only.
#: A frame outside that range has no defined criterion, which is reported rather
#: than resolved by borrowing another band's numbers.
_DEFAULT_HEART_RATE_BANDS: tuple[dict[str, float], ...] = (
    {"min_bpm": 60.0, "max_bpm": 130.0, "l1": 0.5, "l2": 0.8, "l3": 0.4},
    {"min_bpm": 130.0, "max_bpm": 160.0, "l1": 0.4, "l2": 0.7, "l3": 0.3},
)

ACCEPTANCE_RULE = "Zhao & Zhang (2018) Eq (4)-(5)"

_BAND_KEYS = ("min_bpm", "max_bpm", "l1", "l2", "l3")


def heart_rate_bands(config: Any = None) -> tuple[dict[str, float], ...]:
    """The heart-rate-dependent limits of Eq (5), read from configuration."""
    section = (config.section("p_sqi") if config is not None else {}) or {}
    spec = section.get("acceptance", {}) or {}
    raw = spec.get("heart_rate_bands")
    if not raw:
        return _DEFAULT_HEART_RATE_BANDS
    bands: list[dict[str, float]] = []
    for band in raw:
        bands.append({key: float(unwrap(band.get(key))) for key in _BAND_KEYS})
    return tuple(bands)


def p_sqi_acceptance(
    value: float, heart_rate_bpm: float, config: Any = None
) -> Acceptance:
    """Judge one pSQI value against Zhao & Zhang (2018) Eq (4) and Eq (5).

    The limits depend on heart rate::

        60-130 bpm    l1 = 0.5   l2 = 0.8   l3 = 0.4
        130-160 bpm   l1 = 0.4   l2 = 0.7   l3 = 0.3

        optimal       l1 <= pSQI <= l2
        suspicious    l3 <= pSQI <  l1
        unqualified   pSQI > l2  or  pSQI < l3

    This index is band-pass-like: a value *above* l2 is unqualified, not
    optimal, because too much of the total power sitting in the QRS band is as
    abnormal as too little.

    A heart rate outside the calibrated 60-160 bpm range yields UNDEFINED. The
    article publishes no limits there, and reusing another band's numbers would
    invent a criterion.
    """
    value = float(value)
    heart_rate_bpm = float(heart_rate_bpm)
    bands = heart_rate_bands(config)

    if not np.isfinite(value):
        return Acceptance(
            level=UNDEFINED,
            value=value,
            rule=ACCEPTANCE_RULE,
            reason="pSQI is not a finite value",
        )
    if not np.isfinite(heart_rate_bpm):
        return Acceptance(
            level=UNDEFINED,
            value=value,
            rule=ACCEPTANCE_RULE,
            reason="heart rate could not be estimated from the R peaks",
        )

    band = next(
        (item for item in bands if item["min_bpm"] <= heart_rate_bpm <= item["max_bpm"]),
        None,
    )
    if band is None:
        return Acceptance(
            level=UNDEFINED,
            value=value,
            rule=ACCEPTANCE_RULE,
            limits={
                "min_bpm": bands[0]["min_bpm"],
                "max_bpm": bands[-1]["max_bpm"],
                "heart_rate_bpm": heart_rate_bpm,
            },
            reason=(
                f"heart rate {heart_rate_bpm:.1f} bpm is outside the calibrated "
                f"{bands[0]['min_bpm']:.0f}-{bands[-1]['max_bpm']:.0f} bpm range"
            ),
        )

    limits = {key: band[key] for key in _BAND_KEYS}
    limits["heart_rate_bpm"] = heart_rate_bpm
    if band["l1"] <= value <= band["l2"]:
        level = OPTIMAL
    elif band["l3"] <= value < band["l1"]:
        level = SUSPICIOUS
    else:
        level = UNQUALIFIED
    return Acceptance(level=level, value=value, rule=ACCEPTANCE_RULE, limits=limits)


__all__ = [
    "ACCEPTANCE_RULE",
    "band_power",
    "compute_psd",
    "heart_rate_bands",
    "p_sqi",
    "p_sqi_acceptance",
    "p_sqi_config",
]

"""basSQI: relative power in the baseline (very low frequency) band.

Rumus (Zhao & Zhang 2018, Eq 11)::

    basSQI = 1 - integral(f=0..1 Hz) P(f) df / integral(f=0..40 Hz) P(f) df

Awalan "1 -" adalah bagian dari definisi: nilai yang mendekati 1 berarti hampir
tidak ada baseline wander, dan nilai rendah berarti daya pada pita 0-1 Hz
abnormal tinggi terhadap pita 0-40 Hz. Rasio tanpa "1 -" akan membalik arti
setiap pita penerimaan pada Eq (12). Kriteria penerimaan (Eq 12) ada di
:func:`bas_sqi_acceptance`.

Frequency limits come from configuration and are stored on the result
(IDEA.md section 26).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models.result import (
    OPTIMAL,
    SUSPICIOUS,
    UNQUALIFIED,
    UNDEFINED,
    Acceptance,
    SpectralResult,
)
from ._params import unwrap
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
    if total_power > 0:
        # Zhao & Zhang (2018) Eq (11). The subtraction is part of the
        # definition, not a normalisation step applied afterwards.
        value = 1.0 - (baseline_power / total_power)
    else:
        # No power at all in the analysis band, so the ratio is undefined.
        # Report the worst possible value rather than a missing number: a flat
        # or dead lead must not pass the acceptance check by default.
        value = 0.0

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


#: Default limits of Eq (12).
_DEFAULT_ACCEPTANCE = {
    "optimal_from": 0.95,
    "optimal_to": 1.00,
    "suspicious_from": 0.90,
    "suspicious_to": 0.95,
    "unqualified_below": 0.90,
}

ACCEPTANCE_RULE = "Zhao & Zhang (2018) Eq (12)"


def acceptance_limits(config: Any = None) -> dict[str, float]:
    """Acceptance limits for this index, read from the configuration section."""
    section = (config.section("bas_sqi") if config is not None else {}) or {}
    spec = section.get("acceptance", {}) or {}
    limits = dict(_DEFAULT_ACCEPTANCE)
    for key in limits:
        if key in spec:
            limits[key] = float(unwrap(spec[key]))
    return limits


def bas_sqi_acceptance(value: float, config: Any = None) -> Acceptance:
    """Judge one basSQI value against Zhao & Zhang (2018) Eq (12).

    ::

        optimal      0.95 <= basSQI <= 1
        suspicious   0.90 <= basSQI <  0.95
        unqualified  basSQI < 0.90

    The article's worked examples give 0.966 for a high-quality sample and 0.5
    for a low-quality one.
    """
    limits = acceptance_limits(config)
    value = float(value)
    if not np.isfinite(value):
        return Acceptance(
            level=UNDEFINED,
            value=value,
            rule=ACCEPTANCE_RULE,
            limits=limits,
            reason="basSQI is not a finite value",
        )
    if limits["optimal_from"] <= value <= limits["optimal_to"]:
        level = OPTIMAL
    elif limits["suspicious_from"] <= value < limits["suspicious_to"]:
        level = SUSPICIOUS
    else:
        level = UNQUALIFIED
    return Acceptance(level=level, value=value, rule=ACCEPTANCE_RULE, limits=limits)


__all__ = [
    "ACCEPTANCE_RULE",
    "acceptance_limits",
    "bas_sqi",
    "bas_sqi_acceptance",
    "bas_sqi_config",
]

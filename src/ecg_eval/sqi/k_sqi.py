"""kSQI: kurtosis of the analysis signal.

The convention is stated explicitly in configuration so a library default can
never silently define the scientific meaning (IDEA.md section 25):

* ``fisher``  -- E[(x - mean)^4] / sigma^4 - 3. A clean ECG is leptokurtic
  (kurtosis well above 3) because of the sharp QRS complexes. High values mean
  a QRS-dominated, "clean-looking" amplitude distribution.
* ``pearson`` -- E[(x - mean)^4] / sigma^4, i.e. Fisher + 3.

The mean and standard deviation are stored alongside the value.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FISHER = "fisher"
PEARSON = "pearson"


def kurtosis(signal: np.ndarray, definition: str = FISHER) -> float:
    """Kurtosis of a 1-D signal under the requested convention."""
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size < 4:
        return float("nan")
    if not np.all(np.isfinite(signal)):
        signal = signal[np.isfinite(signal)]
        if signal.size < 4:
            return float("nan")

    std = float(np.std(signal))
    if std == 0.0:
        return float("nan")

    # Direct fourth-moment formulation. scipy's default (fisher=True,
    # bias=True) is not used so the definition is visible in the code.
    centered = signal - np.mean(signal)
    value = float(np.mean(centered ** 4) / (std ** 4))
    if definition == FISHER:
        return value - 3.0
    if definition == PEARSON:
        return value
    raise ValueError(f"unknown kurtosis definition {definition!r}; use 'fisher' or 'pearson'")


def k_sqi(
    signal: np.ndarray,
    definition: str = FISHER,
) -> tuple[float, float, float]:
    """Return ``(kSQI, mean, std)`` for the given signal."""
    signal = np.asarray(signal, dtype=np.float64).ravel()
    finite = signal[np.isfinite(signal)] if not np.all(np.isfinite(signal)) else signal
    mean = float(np.mean(finite)) if finite.size else float("nan")
    std = float(np.std(finite)) if finite.size else float("nan")
    return kurtosis(signal, definition=definition), mean, std


def k_sqi_config(config: Any) -> dict[str, Any]:
    return {
        "definition": str(config.get("k_sqi.definition", FISHER)),
        "compute_on": str(config.get("k_sqi.compute_on", "preprocessed")),
    }


__all__ = ["FISHER", "PEARSON", "k_sqi", "k_sqi_config", "kurtosis"]

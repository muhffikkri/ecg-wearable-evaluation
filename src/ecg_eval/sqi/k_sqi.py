"""kSQI: kurtosis of the analysis signal.

Rumus (Zhao & Zhang 2018, Eq 9)::

    kSQI = mu4 = E{(x - mu_x)^4} / sigma^4

Ini momen terstandardisasi keempat (konvensi Pearson), sehingga sinyal Gaussian
bernilai 3 -- bukan 0. Kriteria penerimaan Eq (10) dinyatakan untuk nilai nu4
tersebut, jadi konvensinya adalah bagian dari ilmu, bukan pilihan pustaka, dan
:func:`k_sqi_acceptance` menolak menilai nilai berkonvensi fisher. Kriteria
penerimaan (Eq 10) ada di :func:`k_sqi_acceptance`.

The convention is stated explicitly in configuration so a library default can
never silently define the scientific meaning (IDEA.md section 25):

* ``pearson`` -- E[(x - mean)^4] / sigma^4, the article's nu4. A clean ECG is
  leptokurtic (well above 3) because of the sharp QRS complexes. High values
  mean a QRS-dominated, "clean-looking" amplitude distribution.
* ``fisher``  -- the same minus 3, so a Gaussian sits at 0. Kept for comparison
  plots only: Eq (10) thresholds the value 5 on the nu4 scale, and applying it
  to an excess kurtosis would be wrong by exactly three.

The mean and standard deviation are stored alongside the value.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models.result import (
    OPTIMAL,
    UNQUALIFIED,
    UNDEFINED,
    Acceptance,
)
from ._params import unwrap

FISHER = "fisher"
PEARSON = "pearson"


def kurtosis(signal: np.ndarray, definition: str = PEARSON) -> float:
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
    definition: str = PEARSON,
) -> tuple[float, float, float]:
    """Return ``(kSQI, mean, std)`` for the given signal."""
    signal = np.asarray(signal, dtype=np.float64).ravel()
    finite = signal[np.isfinite(signal)] if not np.all(np.isfinite(signal)) else signal
    mean = float(np.mean(finite)) if finite.size else float("nan")
    std = float(np.std(finite)) if finite.size else float("nan")
    return kurtosis(signal, definition=definition), mean, std


def k_sqi_config(config: Any) -> dict[str, Any]:
    return {
        "definition": str(config.get("k_sqi.definition", PEARSON)),
        "compute_on": str(config.get("k_sqi.compute_on", "preprocessed")),
    }


#: Default limits of Eq (10).
_DEFAULT_ACCEPTANCE = {"optimal_above": 5.0, "unqualified_at_or_below": 5.0}

ACCEPTANCE_RULE = "Zhao & Zhang (2018) Eq (10)"


def acceptance_limits(config: Any = None) -> dict[str, float]:
    """Acceptance limits for this index, read from the configuration section."""
    section = (config.section("k_sqi") if config is not None else {}) or {}
    spec = section.get("acceptance", {}) or {}
    limits = dict(_DEFAULT_ACCEPTANCE)
    for key in limits:
        if key in spec:
            limits[key] = float(unwrap(spec[key]))
    return limits


def k_sqi_acceptance(
    value: float, definition: str = PEARSON, config: Any = None
) -> Acceptance:
    """Judge one kSQI value against Zhao & Zhang (2018) Eq (10).

    ::

        optimal      kSQI > 5
        unqualified  kSQI <= 5

    Eq (10) is binary: the article defines no suspicious band for kurtosis, and
    its Eq (29) membership assignment is binary in the same way, so the fusion
    never receives a suspicious verdict from this index.

    The threshold 5 is stated for nu4, the fourth standardized moment, where a
    Gaussian signal sits at 3. A value computed under the excess-kurtosis
    (fisher) convention is reported as UNDEFINED rather than compared against 5,
    because that comparison would be wrong by exactly three.
    """
    limits = acceptance_limits(config)
    value = float(value)
    if str(definition) != PEARSON:
        return Acceptance(
            level=UNDEFINED,
            value=value,
            rule=ACCEPTANCE_RULE,
            limits=limits,
            reason=(
                f"Eq (10) thresholds nu4, where a Gaussian is 3; this value uses "
                f"the {definition!r} convention and is not comparable with it"
            ),
        )
    if not np.isfinite(value):
        return Acceptance(
            level=UNDEFINED,
            value=value,
            rule=ACCEPTANCE_RULE,
            limits=limits,
            reason="kSQI is not a finite value",
        )
    level = OPTIMAL if value > limits["optimal_above"] else UNQUALIFIED
    return Acceptance(level=level, value=value, rule=ACCEPTANCE_RULE, limits=limits)


__all__ = [
    "ACCEPTANCE_RULE",
    "FISHER",
    "PEARSON",
    "acceptance_limits",
    "k_sqi",
    "k_sqi_acceptance",
    "k_sqi_config",
    "kurtosis",
]

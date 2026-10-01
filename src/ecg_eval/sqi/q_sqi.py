"""qSQI: agreement between two independent R-peak detectors.

Rumus (Zhao & Zhang 2018, Eq 1)::

    qSQI = 2N / (Na + Nb)

    N  = puncak R yang cocok satu-lawan-satu di dalam jendela toleransi
    Na = puncak R dari detektor A (Hilbert + ambang adaptif dinamis)
    Nb = puncak R dari detektor B (transformasi wavelet)

Penyebutnya adalah JUMLAH kedua cacah puncak, bukan cacah yang terkecil.
Kriteria penerimaan (Eq 2) ada di :func:`q_sqi_acceptance`.

Primary reproduction (IDEA.md section 20):

    Hilbert + dynamic adaptive threshold   vs   wavelet R-wave detection

Adapted / experimental mode (section 21):

    Hilbert + dynamic adaptive threshold   vs   Pan-Tompkins

The matching tolerance comes from configuration and is recorded on the result,
so a substituted tolerance can never be silent. All match details are retained
for visual debugging (section 23).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..detectors import DetectionResult, get_detector
from ..models.result import (
    OPTIMAL,
    SUSPICIOUS,
    UNQUALIFIED,
    UNDEFINED,
    Acceptance,
    PeakMatchResult,
)
from ._params import unwrap

LABEL_ZHAO = "zhao_zhang"
LABEL_ADAPTED = "pan_tompkins_adapted"


def match_peaks(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    sampling_rate: float,
    tolerance_ms: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """One-to-one nearest matching inside a symmetric tolerance window.

    Each peak may be matched at most once. Candidates are taken in order of
    increasing absolute distance so the closest pair wins, which avoids a
    greedy left-to-right pass letting a distant pair steal a peak.
    """
    a = np.asarray(peaks_a, dtype=int)
    b = np.asarray(peaks_b, dtype=int)
    tolerance_samples = tolerance_ms * sampling_rate / 1000.0

    if a.size == 0 or b.size == 0 or tolerance_samples <= 0:
        return [], a.tolist(), b.tolist()

    candidates: list[tuple[float, int, int]] = []
    for i, pa in enumerate(a):
        for j, pb in enumerate(b):
            distance = abs(pa - pb)
            if distance <= tolerance_samples:
                candidates.append((distance, i, j))
    candidates.sort()

    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((int(a[i]), int(b[j])))

    pairs.sort()
    unmatched_a = [int(v) for i, v in enumerate(a) if i not in used_a]
    unmatched_b = [int(v) for j, v in enumerate(b) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


def q_sqi(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    detector_a: str = "zhao_hilbert",
    detector_b: str = "zhao_wavelet",
    tolerance_ms: float = 150.0,
    adaptive: bool = False,
    detector_kwargs_a: dict[str, Any] | None = None,
    detector_kwargs_b: dict[str, Any] | None = None,
) -> tuple[float, DetectionResult, DetectionResult, PeakMatchResult]:
    """Compute qSQI plus both detections and the full match record."""
    det_a = get_detector(detector_a, **(detector_kwargs_a or {}))
    det_b = get_detector(detector_b, **(detector_kwargs_b or {}))
    result_a = det_a.detect(signal, sampling_rate)
    result_b = det_b.detect(signal, sampling_rate)

    tolerance = float(tolerance_ms)
    if adaptive:
        # Adaptive mode: scale the window with the rate implied by detector B,
        # bounded so it can never collapse or explode.
        interval = np.diff(result_b.peaks) if result_b.peaks.size > 1 else np.array([])
        if interval.size:
            mean_rr = float(np.mean(interval)) / sampling_rate
            if mean_rr > 0:
                tolerance = float(np.clip(0.25 * mean_rr * 1000.0, 80.0, 250.0))

    pairs, unmatched_a, unmatched_b = match_peaks(
        result_a.peaks, result_b.peaks, sampling_rate, tolerance
    )

    # Zhao & Zhang (2018) Eq (1): agreement is counted against the SUM of both
    # detection counts, not against the smaller one:
    #
    #     qSQI = 2N / (Na + Nb)
    #
    # Dividing by min(Na, Nb) instead -- which this module did before -- reports
    # 1.0 whenever one detector's peaks are a subset of the other's, so a
    # detector that missed half the beats still scored a perfect match.
    n_matched = len(pairs)
    total_detected = int(result_a.peaks.size + result_b.peaks.size)
    value = (2.0 * n_matched / total_detected) if total_detected > 0 else 0.0

    label = LABEL_ADAPTED if "pan_tompkins" in {detector_a, detector_b} else LABEL_ZHAO
    match = PeakMatchResult(
        n_peaks_a=int(result_a.peaks.size),
        n_peaks_b=int(result_b.peaks.size),
        n_matched=len(pairs),
        q_sqi=float(value),
        peaks_a=result_a.as_list(),
        peaks_b=result_b.as_list(),
        matched_pairs=pairs,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        tolerance_ms=tolerance,
        detector_a=detector_a,
        detector_b=detector_b,
        label=label,
    )
    return float(value), result_a, result_b, match


#: Default limits of Eq (2). The configuration carries the same numbers with
#: their provenance; these are only the fallback for a caller with no config.
_DEFAULT_ACCEPTANCE = {
    "optimal_above": 0.90,
    "suspicious_from": 0.60,
    "suspicious_to": 0.90,
    "unqualified_below": 0.60,
}

ACCEPTANCE_RULE = "Zhao & Zhang (2018) Eq (2)"


def acceptance_limits(config: Any = None) -> dict[str, float]:
    """Acceptance limits for this index, read from the configuration section."""
    section = (config.section("q_sqi") if config is not None else {}) or {}
    spec = section.get("acceptance", {}) or {}
    limits = dict(_DEFAULT_ACCEPTANCE)
    for key in limits:
        if key in spec:
            limits[key] = float(unwrap(spec[key]))
    return limits


def q_sqi_acceptance(value: float, config: Any = None) -> Acceptance:
    """Judge one qSQI value against Zhao & Zhang (2018) Eq (2).

    ::

        optimal      qSQI > 90 %
        suspicious   60 % <= qSQI <= 90 %
        unqualified  qSQI < 60 %

    The outer rules are strict, and that is what resolves the shared boundaries:
    exactly 90 % falls to suspicious and exactly 60 % also falls to suspicious,
    which is what Eq (2) prints.
    """
    limits = acceptance_limits(config)
    value = float(value)
    if not np.isfinite(value):
        return Acceptance(
            level=UNDEFINED,
            value=value,
            rule=ACCEPTANCE_RULE,
            limits=limits,
            reason="qSQI is not a finite value",
        )
    if value > limits["optimal_above"]:
        level = OPTIMAL
    elif value >= limits["suspicious_from"]:
        level = SUSPICIOUS
    else:
        level = UNQUALIFIED
    return Acceptance(level=level, value=value, rule=ACCEPTANCE_RULE, limits=limits)


__all__ = [
    "ACCEPTANCE_RULE",
    "LABEL_ADAPTED",
    "LABEL_ZHAO",
    "acceptance_limits",
    "match_peaks",
    "q_sqi",
    "q_sqi_acceptance",
]

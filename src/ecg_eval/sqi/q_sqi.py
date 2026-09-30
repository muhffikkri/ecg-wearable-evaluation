"""qSQI: agreement between two independent R-peak detectors.

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
from ..models.result import PeakMatchResult

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

    denominator = min(result_a.peaks.size, result_b.peaks.size)
    value = (len(pairs) / denominator) if denominator > 0 else 0.0

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


__all__ = ["LABEL_ADAPTED", "LABEL_ZHAO", "match_peaks", "q_sqi"]

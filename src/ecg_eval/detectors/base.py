"""R-peak detector interface.

A single interface so additional detector families (Hamilton, Christov,
Engzee, Two Average, stationary wavelet) can be added later without touching
qSQI (IDEA.md section 22).

Every detector returns sample indices into the frame it was given, plus a
dict of diagnostic info that the UI shows next to the waveform.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectionResult:
    """R-peak locations and the parameters that produced them."""

    peaks: np.ndarray
    detector: str
    label: str = "zhao_zhang"     # "zhao_zhang" | "pan_tompkins_adapted"
    params: dict[str, Any] = field(default_factory=dict)

    def as_list(self) -> list[int]:
        return [int(p) for p in self.peaks]

    def times_s(self, sampling_rate: float) -> np.ndarray:
        return self.peaks / float(sampling_rate)


class RPeakDetector(abc.ABC):
    """Base class for all R-peak detectors."""

    name: str = "base"
    #: Which qSQI pairing this detector belongs to.
    label: str = "zhao_zhang"

    @abc.abstractmethod
    def detect(self, signal: np.ndarray, sampling_rate: float) -> DetectionResult:
        """Detect R peaks in a single-lead signal.

        ``signal`` is a 1-D array in mV. Implementations must be deterministic:
        the same input and config must always give the same peaks, otherwise
        the analysis cache is invalid.
        """

    # -- shared helpers -------------------------------------------------
    @staticmethod
    def _empty(detector: str, label: str) -> DetectionResult:
        return DetectionResult(peaks=np.empty(0, dtype=int), detector=detector, label=label)

    @staticmethod
    def _search_backward(
        peaks: np.ndarray,
        signal: np.ndarray,
        sampling_rate: float,
        search_backward_ms: float,
        refractory_ms: float,
    ) -> np.ndarray:
        """Promote a sloped maximum back to the true R peak.

        Standard practice in QRS detection: when a threshold is crossed on a
        rising edge, walk back while the slope stays positive. The refractory
        window then suppresses the double detection this would otherwise cause.
        """
        if peaks.size == 0:
            return peaks
        back = max(1, int(round(search_backward_ms * sampling_rate / 1000.0)))
        refractory = max(1, int(round(refractory_ms * sampling_rate / 1000.0)))
        corrected: list[int] = []
        for peak in peaks:
            p = int(peak)
            limit = max(0, p - back)
            while p - 1 >= limit and signal[p - 1] < signal[p]:
                p -= 1
            if corrected and p - corrected[-1] < refractory:
                # Keep the larger of the two candidates.
                if signal[p] > signal[corrected[-1]]:
                    corrected[-1] = p
                continue
            corrected.append(p)
        return np.asarray(corrected, dtype=int)

    @staticmethod
    def _refractory_filter(peaks: np.ndarray, signal: np.ndarray, refractory_ms: int) -> np.ndarray:
        if peaks.size <= 1:
            return peaks
        kept = [int(peaks[0])]
        for peak in peaks[1:]:
            p = int(peak)
            if p - kept[-1] >= refractory_ms:
                kept.append(p)
            elif signal[p] > signal[kept[-1]]:
                kept[-1] = p
        return np.asarray(kept, dtype=int)


__all__ = ["DetectionResult", "RPeakDetector"]

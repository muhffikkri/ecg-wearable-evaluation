"""Detector B of the primary Zhao & Zhang qSQI: wavelet-based R-wave detection.

Uses the discrete wavelet transform modulo-3 maximum correspondence: at the
coarsest scale whose detail coefficients carry the QRS energy, R peaks are the
sample positions of the largest absolute modulus-3 phase.

This is the reproduction path required by IDEA.md section 20. It is
deliberately a *different algorithm* from :mod:`zhao_hilbert`, because qSQI
measures the agreement between two independent detectors.

VALIDATION STATUS: neither detector is validated against beat annotations. The
recorded dataset carries no R-peak labels, so on the 29-09-2026 session this
detector finds roughly a third as many peaks as the Hilbert detector, and the
resulting qSQI ranges from 0.29 to 1.00 across frames. That spread is currently
dominated by this detector's under-detection rather than by genuine algorithmic
disagreement. The numbers are reported as measured, and no parameter is tuned to
close the gap without beat labels to tune against.
"""

from __future__ import annotations

import numpy as np
import pywt

from .base import DetectionResult, RPeakDetector


class ZhaoWaveletDetector(RPeakDetector):
    name = "zhao_wavelet"
    label = "zhao_zhang"

    def __init__(
        self,
        *,
        wavelet: str = "db1",
        decomposition_level: int | None = None,
        search_backward_ms: float = 30.0,
        refractory_ms: float = 200.0,
        min_distance_ms: float = 200.0,
    ) -> None:
        self.wavelet = wavelet
        self.decomposition_level = decomposition_level
        self.search_backward_ms = search_backward_ms
        self.refractory_ms = refractory_ms
        self.min_distance_ms = min_distance_ms

    def detect(self, signal: np.ndarray, sampling_rate: float) -> DetectionResult:
        signal = np.asarray(signal, dtype=np.float64).ravel()
        if signal.size < int(0.5 * sampling_rate):
            return self._empty(self.name, self.label)

        finite = np.isfinite(signal)
        if not finite.all():
            idx = np.arange(signal.size)
            signal = np.interp(idx, idx[finite], signal[finite])

        signal = signal - np.median(signal)
        signal = signal / (np.max(np.abs(signal)) or 1.0)

        level = self.decomposition_level or self._choose_level(signal)
        if level < 1:
            level = 1

        coeffs = pywt.wavedec(signal, self.wavelet, level=level)
        # wavedec returns [cA_L, d_L, d_(L-1), ..., d_1], so coeffs[1] is the
        # detail band of the chosen level, not D1. At 250 Hz with db1 that is
        # roughly 31-62 Hz for level 2 and 16-31 Hz for level 3; the QRS
        # complex sits mostly below 40 Hz, which is why _choose_level usually
        # lands on 2 or 3.
        detail = np.asarray(coeffs[1], dtype=np.float64)
        if detail.size < 3:
            return self._empty(self.name, self.label)

        # Modulo-3 maximum correspondence (Mallat's modulus maxima method).
        abs_detail = np.abs(detail)
        candidates = np.where(
            (abs_detail[1:-1] > abs_detail[:-2]) & (abs_detail[1:-1] >= abs_detail[2:])
        )[0] + 1
        if candidates.size == 0:
            return self._empty(self.name, self.label)

        threshold = 0.05 * float(np.max(abs_detail))
        candidates = candidates[abs_detail[candidates] >= threshold]
        if candidates.size == 0:
            return self._empty(self.name, self.label)

        peaks = np.unique(candidates)
        min_distance = max(1, int(round(self.min_distance_ms * sampling_rate / 1000.0)))
        if peaks.size > 1 and min_distance > 1:
            order = peaks[np.argsort(abs_detail[peaks])[::-1]]
            kept: list[int] = []
            for candidate in order:
                if all(abs(int(candidate) - k) >= min_distance for k in kept):
                    kept.append(int(candidate))
            peaks = np.asarray(sorted(kept), dtype=int)

        refractory = max(1, int(round(self.refractory_ms * sampling_rate / 1000.0)))
        peaks = self._refractory_filter(peaks, signal, refractory)
        peaks = self._search_backward(
            peaks, signal, sampling_rate, self.search_backward_ms, self.refractory_ms
        )

        return DetectionResult(
            peaks=peaks,
            detector=self.name,
            label=self.label,
            params={
                "method": "modulo-3 maximum correspondence on DWT detail",
                "wavelet": self.wavelet,
                "decomposition_level": int(level),
                "refractory_ms": self.refractory_ms,
                "min_distance_ms": self.min_distance_ms,
                "n_peaks": int(peaks.size),
            },
        )

    @staticmethod
    def _choose_level(signal: np.ndarray) -> int:
        """Smallest level whose detail still carries meaningful energy."""
        max_level = pywt.dwt_max_level(len(signal), "db1")
        if max_level < 1:
            return 1
        coeffs = pywt.wavedec(signal, "db1", level=min(5, max_level))
        best_level = 1
        best_energy = 0.0
        for index, detail in enumerate(coeffs[1:], start=1):
            energy = float(np.sum(np.asarray(detail) ** 2))
            if energy > best_energy:
                best_energy, best_level = energy, index
        return best_level


__all__ = ["ZhaoWaveletDetector"]

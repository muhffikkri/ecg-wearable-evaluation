"""Detector A of the primary Zhao & Zhang qSQI: Hilbert transform with a
dynamic adaptive threshold.

The envelope of the analytic signal (Hilbert magnitude) is thresholded with a
level that adapts to the local signal amplitude, then refined by the standard
sloped search-backward step.

This is the reproduction path required by IDEA.md section 20.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, hilbert

from .base import DetectionResult, RPeakDetector


class ZhaoHilbertDetector(RPeakDetector):
    name = "zhao_hilbert"
    label = "zhao_zhang"

    def __init__(
        self,
        *,
        initial_threshold_frac: float = 0.10,
        threshold_rate: float = 0.02,
        signal_peak_frac: float = 0.25,
        search_backward_ms: float = 30.0,
        refractory_ms: float = 200.0,
        smooth_window_ms: float = 100.0,
    ) -> None:
        self.initial_threshold_frac = initial_threshold_frac
        self.threshold_rate = threshold_rate
        self.signal_peak_frac = signal_peak_frac
        self.search_backward_ms = search_backward_ms
        self.refractory_ms = refractory_ms
        self.smooth_window_ms = smooth_window_ms

    def detect(self, signal: np.ndarray, sampling_rate: float) -> DetectionResult:
        signal = np.asarray(signal, dtype=np.float64).ravel()
        if signal.size < int(0.5 * sampling_rate):
            return self._empty(self.name, self.label)

        finite = np.isfinite(signal)
        if not finite.all():
            # Fill gaps by nearest neighbour so the transform stays defined.
            idx = np.arange(signal.size)
            signal = np.interp(idx, idx[finite], signal[finite])

        signal = signal - np.median(signal)

        envelope = np.abs(hilbert(signal))
        amplitude = float(np.max(envelope)) or 1.0

        # The adaptive level runs on the smoothed envelope: smoothing suppresses
        # the T wave relative to the QRS complex, so the T wave cannot clear a
        # threshold derived from QRS-dominated energy. Peak positions are then
        # refined on the raw envelope, where the true R peak is sharpest.
        window = max(3, int(round(self.smooth_window_ms * sampling_rate / 1000.0)))
        smoothed = np.convolve(envelope, np.ones(window) / window, mode="same")
        smoothed_amplitude = float(np.max(smoothed)) or 1.0

        # Dynamic adaptive threshold: a low initial level that rises whenever a
        # signal peak is accepted and decays between beats, so the threshold
        # tracks the current signal amplitude instead of a single global level.
        initial_threshold = self.initial_threshold_frac * smoothed_amplitude
        rate = self.threshold_rate * smoothed_amplitude

        signal_peaks, _ = find_peaks(
            smoothed, height=self.signal_peak_frac * smoothed_amplitude, distance=1
        )
        is_signal_peak = np.zeros(signal.size, dtype=bool)
        is_signal_peak[signal_peaks] = True

        threshold = np.empty(signal.size, dtype=np.float64)
        threshold[0] = initial_threshold
        for index in range(1, signal.size):
            previous = threshold[index - 1]
            if is_signal_peak[index]:
                threshold[index] = previous + rate
            else:
                threshold[index] = max(initial_threshold, previous - rate / 8.0)

        refractory = max(1, int(round(self.refractory_ms * sampling_rate / 1000.0)))
        # Candidates are local maxima of the SMOOTHED envelope, already
        # separated by the refractory window, so neither a rising QRS slope nor
        # a T wave can produce a second candidate inside one beat.
        candidates, _ = find_peaks(smoothed, distance=refractory)
        if candidates.size == 0:
            return self._empty(self.name, self.label)

        # A candidate is accepted only if it clears BOTH the adaptive level
        # (tracked over the record) and a fraction of the strongest envelope
        # value seen (the "signal peak" criterion). The second test is what
        # rejects the T wave, whose envelope is far below the QRS complex.
        clears_level = smoothed[candidates] > threshold[candidates]
        is_qrs = envelope[candidates] > self.signal_peak_frac * amplitude
        peaks = candidates[clears_level & is_qrs]
        if peaks.size == 0:
            return self._empty(self.name, self.label)
        peaks = self._refractory_filter(peaks, smoothed, refractory)
        # Refine each candidate onto the true R peak of the raw envelope.
        peaks = self._search_backward(
            peaks, envelope, sampling_rate, self.search_backward_ms, self.refractory_ms
        )

        return DetectionResult(
            peaks=peaks,
            detector=self.name,
            label=self.label,
            params={
                "method": "Hilbert envelope + dynamic adaptive threshold",
                "initial_threshold_frac": self.initial_threshold_frac,
                "threshold_rate": self.threshold_rate,
                "signal_peak_frac": self.signal_peak_frac,
                "search_backward_ms": self.search_backward_ms,
                "refractory_ms": self.refractory_ms,
                "smooth_window_ms": self.smooth_window_ms,
                "n_peaks": int(peaks.size),
            },
        )


__all__ = ["ZhaoHilbertDetector"]

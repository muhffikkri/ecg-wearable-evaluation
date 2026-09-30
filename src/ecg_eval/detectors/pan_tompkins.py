"""Pan-Tompkins R-peak detector.

Used as an *adaptation*: substituting this for the wavelet detector gives the
"Experimental / Adapted qSQI" mode of IDEA.md section 21. It never replaces
the primary Zhao-Zhang pairing in the default configuration, and every result
it produces is tagged ``pan_tompkins_adapted``.

Classic implementation: bandpass, derivative, squaring, moving-window
integration, adaptive dual thresholds, search-back.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from .base import DetectionResult, RPeakDetector


class PanTompkinsDetector(RPeakDetector):
    name = "pan_tompkins"
    label = "pan_tompkins_adapted"

    def __init__(
        self,
        *,
        bandpass_hz: tuple[float, float] = (5.0, 15.0),
        filter_order: int = 2,
        window_ms: float = 150.0,
        refractory_ms: float = 200.0,
        search_backward_ms: float = 30.0,
        initial_threshold_frac: float = 0.25,
        threshold_rate: float = 0.01,
        signal_peak_frac: float = 0.30,
    ) -> None:
        self.bandpass_hz = bandpass_hz
        self.filter_order = filter_order
        self.window_ms = window_ms
        self.refractory_ms = refractory_ms
        self.search_backward_ms = search_backward_ms
        self.initial_threshold_frac = initial_threshold_frac
        self.threshold_rate = threshold_rate
        self.signal_peak_frac = signal_peak_frac

    def detect(self, signal: np.ndarray, sampling_rate: float) -> DetectionResult:
        signal = np.asarray(signal, dtype=np.float64).ravel()
        if signal.size < int(0.5 * sampling_rate):
            return self._empty(self.name, self.label)

        finite = np.isfinite(signal)
        if not finite.all():
            idx = np.arange(signal.size)
            signal = np.interp(idx, idx[finite], signal[finite])

        signal = signal - np.median(signal)

        low, high = self.bandpass_hz
        nyquist = sampling_rate / 2.0
        high = min(high, nyquist * 0.95)
        low = max(low, 0.5)
        if high <= low:
            return self._empty(self.name, self.label)
        sos = butter(self.filter_order, [low / nyquist, high / nyquist], btype="band", output="sos")
        try:
            filtered = sosfiltfilt(sos, signal)
        except ValueError:
            return self._empty(self.name, self.label)

        derivative = np.gradient(filtered) * sampling_rate
        squared = derivative ** 2

        window = max(1, int(round(self.window_ms * sampling_rate / 1000.0)))
        integrated = np.convolve(squared, np.ones(window) / window, mode="same")

        # Dual adaptive thresholds over the integrated signal.
        scale = float(np.max(integrated)) or 1.0
        primary_threshold = self.initial_threshold_frac * scale
        secondary_threshold = 0.5 * primary_threshold
        rate = self.threshold_rate * scale

        signal_peaks, _ = find_peaks(integrated, height=self.signal_peak_frac * scale, distance=1)

        threshold = np.full(integrated.size, primary_threshold)
        for index in range(1, integrated.size):
            previous = threshold[index - 1]
            if index in signal_peaks:
                threshold[index] = previous + rate
            else:
                threshold[index] = max(primary_threshold * 0.5, previous - rate / 8.0)

        refractory = max(1, int(round(self.refractory_ms * sampling_rate / 1000.0)))
        candidates, _ = find_peaks(integrated, distance=refractory)
        if candidates.size == 0:
            return self._empty(self.name, self.label)

        above = integrated[candidates] > threshold[candidates]
        peaks = candidates[above]
        if peaks.size == 0:
            # Fall back to the search-back rule: take the highest local maxima
            # within one refractory window instead of returning nothing.
            peaks = candidates[integrated[candidates] > secondary_threshold]
        if peaks.size == 0:
            return self._empty(self.name, self.label)

        peaks = self._refractory_filter(peaks, integrated, refractory)
        peaks = self._search_backward(
            peaks, filtered, sampling_rate, self.search_backward_ms, self.refractory_ms
        )

        return DetectionResult(
            peaks=peaks,
            detector=self.name,
            label=self.label,
            params={
                "method": "Pan-Tompkins (bandpass-derivative-square-integrate)",
                "bandpass_hz": list(self.bandpass_hz),
                "integration_window_ms": self.window_ms,
                "refractory_ms": self.refractory_ms,
                "search_backward_ms": self.search_backward_ms,
                "n_peaks": int(peaks.size),
            },
        )


__all__ = ["PanTompkinsDetector"]

"""R-peak detector registry.

Adding a detector family means adding one entry here; qSQI is unchanged
(IDEA.md section 22).
"""

from __future__ import annotations

from .base import DetectionResult, RPeakDetector
from .pan_tompkins import PanTompkinsDetector
from .zhao_hilbert import ZhaoHilbertDetector
from .zhao_wavelet import ZhaoWaveletDetector

#: name -> factory. Extend as new classical families are implemented.
REGISTRY: dict[str, type[RPeakDetector]] = {
    ZhaoHilbertDetector.name: ZhaoHilbertDetector,
    ZhaoWaveletDetector.name: ZhaoWaveletDetector,
    PanTompkinsDetector.name: PanTompkinsDetector,
}


def get_detector(name: str, **kwargs) -> RPeakDetector:
    """Instantiate a detector by name."""
    try:
        factory = REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown R-peak detector {name!r}; available: {available}") from exc
    return factory(**kwargs)


def available_detectors() -> list[str]:
    return sorted(REGISTRY)


__all__ = [
    "DetectionResult",
    "PanTompkinsDetector",
    "REGISTRY",
    "RPeakDetector",
    "ZhaoHilbertDetector",
    "ZhaoWaveletDetector",
    "available_detectors",
    "get_detector",
]

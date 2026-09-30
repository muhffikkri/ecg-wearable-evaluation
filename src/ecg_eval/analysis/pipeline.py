"""Per-frame SQI pipeline with deterministic caching.

    ECG frame
        -> validation
        -> preprocessing (explicit, recorded)
        -> R-peak detection (two detectors)
        -> qSQI
        -> PSD -> pSQI
        -> statistics -> kSQI
        -> baseline power -> basSQI
        -> fuzzy comprehensive evaluation
        -> quality class

Every intermediate is retained on the :class:`FrameResult`; nothing returns
only a class label (IDEA.md section 19).

The cache key is ``frame content hash + config fingerprint + annotation``,
so re-visiting the UI does not recompute and a configuration change
invalidates stale results (sections 50 and 41).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .. import REFERENCE, __version__
from ..annotation import storage as annotation_storage
from ..config import Config
from ..detectors import DetectionResult
from ..models.annotation import STATIC_POSITIONS
from ..models.frame import ECGFrame
from ..models.result import (
    EXCELLENT,
    FrameResult,
    FuzzyResult,
    PeakMatchResult,
    RunProvenance,
    SpectralResult,
)
from ..preprocessing import preprocess, resolve_config as resolve_preprocessing
from ..sqi import bas_sqi as bas_sqi_fn
from ..sqi import fuzzy as fuzzy_module
from ..sqi import k_sqi as k_sqi_fn
from ..sqi import p_sqi as p_sqi_fn
from ..sqi import q_sqi as q_sqi_fn
from .frame_analysis import Segment10s, segment_frame

logger = logging.getLogger(__name__)

PROGRESS_EPS = 0.0


class ResultCache:
    """JSON-backed cache keyed by frame hash + config + annotation.

    Stored under ``processed/`` so nothing under ``data/`` is written to
    (IDEA.md sections 44 and 50).
    """

    def __init__(self, directory: Path | str | None, *, enabled: bool = True) -> None:
        self.directory = Path(directory) if directory else None
        self.enabled = enabled
        self._memory: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.enabled and self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.path = self.directory / "frame_cache.json"
            if self.path.exists():
                try:
                    self._memory = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("result cache unreadable, starting empty: %s", exc)
                    self._memory = {}
        else:
            self.path = None

    def key(self, frame: ECGFrame, config_fingerprint: str, annotation_key: str) -> str:
        payload = f"{frame.content_hash()}|{config_fingerprint}|{annotation_key}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        hit = self._memory.get(key)
        if hit is None:
            self.misses += 1
        else:
            self.hits += 1
        return hit

    def put(self, key: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._memory[key] = payload
        if self.misses > 0:
            self.misses -= 1

    def flush(self) -> None:
        if self.enabled and self.path is not None:
            self.path.write_text(json.dumps(self._memory, indent=1, default=str), encoding="utf-8")

    def clear(self) -> None:
        self._memory.clear()
        if self.path is not None and self.path.exists():
            self.path.unlink()

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._memory)}


def _detector_kwargs(config: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    search_backward = float(config.get("rpeak.search_backward_ms", 30.0))
    refractory = float(config.get("rpeak.refractory_ms", 200.0))
    common = {"search_backward_ms": search_backward, "refractory_ms": refractory}
    return dict(common), dict(common)


def _unwrap(param: Any, default: Any) -> Any:
    if isinstance(param, dict) and "value" in param:
        return param["value"]
    return default if param is None else param


class FrameAnalyzer:
    """Runs the full SQI pipeline for single frames, reusing cached results."""

    def __init__(
        self,
        config: Config,
        *,
        cache: ResultCache | None = None,
        preprocessing_enabled: bool | None = None,
    ) -> None:
        self.config = config
        self.cache = cache or ResultCache(None, enabled=False)
        self.fingerprint = config.scientific_fingerprint()
        self.detector_a = str(config.get("rpeak.primary_detector_a", "zhao_hilbert"))
        self.detector_b = str(config.get("rpeak.primary_detector_b", "zhao_wavelet"))
        self.analysis_lead = str(config.get("signal.analysis_lead", "Lead II"))
        self.frame_duration = float(config.get("signal.frame_duration_s", 10.0))
        self.preprocessing_enabled = preprocessing_enabled
        self._kwargs_a, self._kwargs_b = _detector_kwargs(config)

    # -- configuration views -------------------------------------------
    @property
    def q_sqi_params(self) -> dict[str, Any]:
        return {
            "detector_a": self.detector_a,
            "detector_b": self.detector_b,
            "tolerance_ms": float(_unwrap(self.config.get("q_sqi.match_tolerance_ms"), 150.0)),
            "adaptive": bool(self.config.get("q_sqi.adaptive.enabled", False)),
        }

    @property
    def p_sqi_params(self) -> dict[str, Any]:
        return {
            "qrs_band_hz": tuple(self.config.get("p_sqi.qrs_band_hz", (5.0, 15.0))),
            "total_band_hz": tuple(self.config.get("p_sqi.total_band_hz", (0.5, 40.0))),
            "nperseg": int(self.config.get("p_sqi.psd_nperseg", 500)),
            "method": str(self.config.get("p_sqi.psd_method", "welch")),
        }

    @property
    def bas_sqi_params(self) -> dict[str, Any]:
        return {
            "baseline_band_hz": tuple(self.config.get("bas_sqi.baseline_band_hz", (0.0, 1.0))),
            "total_band_hz": tuple(self.config.get("bas_sqi.total_band_hz", (0.0, 40.0))),
            "nperseg": int(self.config.get("bas_sqi.psd_nperseg", 500)),
            "method": str(self.config.get("p_sqi.psd_method", "welch")),
        }

    @property
    def k_sqi_params(self) -> dict[str, Any]:
        return {
            "definition": str(_unwrap(self.config.get("k_sqi.definition"), "fisher")),
            "compute_on": str(_unwrap(self.config.get("k_sqi.compute_on"), "preprocessed")),
        }

    @property
    def fuzzy_config(self) -> dict[str, Any]:
        return self.config.section("fuzzy")

    @property
    def preprocessing_config(self) -> dict[str, Any]:
        """The preprocessing chain exactly as configured, master switch included."""
        return self.config.section("preprocessing")

    @property
    def effective_preprocessing_config(self) -> dict[str, Any]:
        """The chain that will actually run, after the ``applied`` switch."""
        return resolve_preprocessing(self.preprocessing_config, self.preprocessing_enabled)

    @property
    def is_adaptation(self) -> bool:
        """True when detector B has been swapped for Pan-Tompkins."""
        return self.detector_b == "pan_tompkins"

    @property
    def pending_parameters(self) -> list[str]:
        """Scientific parameters still unverified against the published paper."""
        return [path for path, _ in self.config.pending_parameters()]

    @property
    def is_verified_reproduction(self) -> bool:
        """True only when the reference pairing is used *and* nothing is pending.

        Using the reference detector pairing is not on its own grounds for
        calling a run a reproduction: every membership-function parameter in
        configs/zhao_zhang.yaml is still marked pending_verification, so a run
        using that pairing reproduces the method's structure and none of its
        confirmed constants.
        """
        return not self.is_adaptation and self.config.is_strict_reproduction and not self.pending_parameters

    @property
    def method_label(self) -> str:
        if self.is_adaptation:
            return "Pan-Tompkins adaptation"
        if self.is_verified_reproduction:
            return "Zhao-Zhang verified reproduction"
        return "Zhao-Zhang structured reproduction (not verified)"

    @property
    def verification_note(self) -> str:
        """Why a run is not a verified reproduction, in plain language."""
        if self.is_adaptation:
            return (
                "Detector B is Pan-Tompkins instead of the reference wavelet detector, so "
                "absolute qSQI values are not comparable with the reference."
            )
        pending = self.pending_parameters
        if not self.config.is_strict_reproduction or pending:
            shown = ", ".join(pending[:6])
            more = f" and {len(pending) - 6} more" if len(pending) > 6 else ""
            return (
                f"{len(pending)} scientific parameter(s) are still marked "
                f"pending_verification ({shown}{more}), so this run reproduces the "
                f"structure of the method and not its confirmed constants."
            )
        return "All scientific parameters are verified against the published paper."

    # -- the pipeline ---------------------------------------------------
    def analyze_frame(self, frame: ECGFrame, position: str) -> FrameResult:
        """Full pipeline for one canonical frame at one body position."""
        # The preprocessing switch is a run-time choice rather than a config
        # value, so it is part of the cache key: toggling it must not reuse
        # results computed on the other signal.
        annotation_key = f"{position}:{frame.internal_id}:pp={self.preprocessing_enabled}"
        cache_key = self.cache.key(frame, self.fingerprint, annotation_key)
        cached = self.cache.get(cache_key)
        if cached is not None:
            result = _result_from_payload(cached)
            if result is not None:
                return result

        result = self._compute(frame, position)
        self.cache.put(cache_key, result.run_provenance.get("payload", {}) or _payload(result))
        return result

    def _compute(self, frame: ECGFrame, position: str) -> FrameResult:
        lead_name = self.analysis_lead
        signal = frame.lead(lead_name)

        result = FrameResult(
            subject_id=frame.subject_id,
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            internal_id=frame.internal_id,
            position=position,
            start_time=frame.timestamp,
            end_time=frame.timestamp,
            sample_count=frame.n_samples,
            sampling_rate=frame.sampling_rate,
        )

        issues = frame.signal_issues()
        if issues:
            result.valid = False
            result.invalid_reason = "; ".join(issues)
            result.quality_class = "INVALID"
            return result

        # Preprocessing: explicit and recorded, never overwriting the raw view.
        processed = preprocess(
            signal, frame.sampling_rate, self.preprocessing_config,
            enabled=self.preprocessing_enabled,
        )
        result.preprocessing_applied = processed.applied
        result.preprocessing_config = dict(processed.config)
        analysis_signal = processed.signal if processed.applied else signal

        # qSQI
        params = self.q_sqi_params
        try:
            q_value, det_a, det_b, match = q_sqi_fn(
                analysis_signal, frame.sampling_rate,
                detector_a=params["detector_a"],
                detector_b=params["detector_b"],
                tolerance_ms=params["tolerance_ms"],
                adaptive=params["adaptive"],
                detector_kwargs_a=self._kwargs_a,
                detector_kwargs_b=self._kwargs_b,
            )
        except Exception as exc:  # noqa: BLE001 - recorded on the result
            result.valid = False
            result.invalid_reason = f"qSQI failed: {exc}"
            result.quality_class = "INVALID"
            return result

        result.q_sqi = q_value
        result.n_peaks_a = match.n_peaks_a
        result.n_peaks_b = match.n_peaks_b
        result.n_matched = match.n_matched
        result.peaks_a = match.peaks_a
        result.peaks_b = match.peaks_b
        result.q_sqi_label = match.label

        # pSQI
        p_value, spectral = p_sqi_fn(analysis_signal, frame.sampling_rate, **self.p_sqi_params)
        result.p_sqi = p_value

        # kSQI
        k_value, k_mean, k_std = k_sqi_fn(analysis_signal, definition=self.k_sqi_params["definition"])
        result.k_sqi = k_value
        result.kurtosis_mean = k_mean
        result.kurtosis_std = k_std
        result.kurtosis_definition = self.k_sqi_params["definition"]

        # basSQI
        bas_value, bas_spectral = bas_sqi_fn(analysis_signal, frame.sampling_rate, **self.bas_sqi_params)
        result.bas_sqi = bas_value

        # Fuzzy comprehensive evaluation
        sqi_values = {"qSQI": q_value, "pSQI": p_value, "kSQI": k_value, "basSQI": bas_value}
        try:
            fuzzy_result = fuzzy_module.evaluate(sqi_values, self.fuzzy_config)
        except ValueError as exc:
            result.valid = False
            result.invalid_reason = f"fuzzy evaluation failed: {exc}"
            result.quality_class = "INVALID"
            return result

        result.fuzzy_excellent = fuzzy_result.excellent
        result.fuzzy_barely_acceptable = fuzzy_result.barely_acceptable
        result.fuzzy_unacceptable = fuzzy_result.unacceptable
        result.quality_class = fuzzy_result.quality_class

        result.run_provenance = {
            "software_version": __version__,
            "config_version": self.config.version,
            "config_name": self.config.name,
            "config_fingerprint": self.fingerprint,
            "detector_a": params["detector_a"],
            "detector_b": params["detector_b"],
            "q_sqi_method": self.method_label,
            "q_sqi_label": match.label,
            "match_tolerance_ms": match.tolerance_ms,
            "analysis_lead": lead_name,
            "preprocessing": dict(processed.config),
            "preprocessing_requested": self.effective_preprocessing_config,
            "fuzzy": fuzzy_result.to_dict(),
            "spectral": {
                "qrs_band_power": spectral.qrs_band_power,
                "p_total_power": spectral.p_total_power,
                "baseline_band_power": bas_spectral.baseline_band_power,
                "bas_total_power": bas_spectral.bas_total_power,
                "psd_method": spectral.psd_method,
            },
            "reference": REFERENCE,
        }
        result.run_provenance["payload"] = _payload(result)
        return result

    def analyze_segment(self, segment: Segment10s) -> FrameResult:
        """Analyze a pre-segmented window by wrapping it as a frame."""
        frame = segment.source_frame
        proxy = ECGFrame(
            subject_id=segment.subject_id,
            session_id=segment.session_id,
            frame_id=f"{frame.frame_id}#{segment.metadata.get('window_index', 0)}",
            source_file=frame.source_file,
            source_format=frame.source_format,
            timestamp=segment.start_time,
            sampling_rate=segment.sampling_rate,
            duration_s=segment.sample_count / segment.sampling_rate,
            signal=segment.signal,
            metadata={**frame.metadata, "segmented_from": frame.internal_id},
            provenance=frame.provenance,
            date_folder=frame.date_folder,
        )
        return self.analyze_frame(proxy, segment.position)

    def inspect_frame(self, frame: ECGFrame, position: str = "") -> dict[str, Any]:
        """Compute without caching, returning full intermediates for the UI."""
        result = self._compute(frame, position)
        payload = dict(result.run_provenance)
        payload.pop("payload", None)
        return {
            "result": result,
            "raw": frame.signal,
            "lead_names": list(frame.channel_order()),
            "detail": payload,
        }


def _payload(result: FrameResult) -> dict[str, Any]:
    """Serialisable cache payload (intermediates included)."""
    return {
        "row": result.to_row(),
        "peaks_a": result.peaks_a,
        "peaks_b": result.peaks_b,
        "provenance": {k: v for k, v in result.run_provenance.items() if k != "payload"},
    }


def _result_from_payload(payload: dict[str, Any]) -> FrameResult | None:
    row = payload.get("row")
    if not isinstance(row, dict):
        return None
    result = FrameResult(
        subject_id=row.get("subject_id", ""),
        session_id=row.get("session_id", ""),
        frame_id=row.get("frame_id", ""),
        internal_id=row.get("internal_id", ""),
        position=row.get("position", ""),
        start_time=row.get("start_time"),
        end_time=row.get("end_time"),
        sample_count=int(row.get("sample_count", 0) or 0),
        sampling_rate=float(row.get("sampling_rate", 0.0) or 0.0),
        q_sqi=float(row.get("qSQI", float("nan"))),
        p_sqi=float(row.get("pSQI", float("nan"))),
        k_sqi=float(row.get("kSQI", float("nan"))),
        bas_sqi=float(row.get("basSQI", float("nan"))),
        fuzzy_excellent=float(row.get("fuzzy_excellent", float("nan"))),
        fuzzy_barely_acceptable=float(row.get("fuzzy_barely_acceptable", float("nan"))),
        fuzzy_unacceptable=float(row.get("fuzzy_unacceptable", float("nan"))),
        quality_class=row.get("quality_class", ""),
        kurtosis_mean=float(row.get("kurtosis_mean", float("nan"))),
        kurtosis_std=float(row.get("kurtosis_std", float("nan"))),
        kurtosis_definition=row.get("kurtosis_definition", ""),
        n_peaks_a=int(row.get("n_peaks_a", 0) or 0),
        n_peaks_b=int(row.get("n_peaks_b", 0) or 0),
        n_matched=int(row.get("n_matched", 0) or 0),
        q_sqi_label=row.get("q_sqi_label", ""),
        peaks_a=list(payload.get("peaks_a", []) or []),
        peaks_b=list(payload.get("peaks_b", []) or []),
        valid=bool(row.get("valid", True)),
        invalid_reason=str(row.get("invalid_reason", "")),
        preprocessing_applied=bool(row.get("preprocessing_applied", False)),
    )
    result.run_provenance = dict(payload.get("provenance", {}))
    return result


def run_analysis(
    dataset,
    annotations: Iterable,
    config: Config,
    *,
    cache: ResultCache | None = None,
    progress: Callable[[int, int, FrameResult], None] | None = None,
    include_raw: bool = True,
    preprocessing_enabled: bool | None = None,
) -> dict[str, Any]:
    """Analyze every statically-labelled frame in the dataset.

    TRANSITION and UNLABELED frames are never analysed, so movement artifacts
    during posture changes cannot be counted as stationary-position quality
    (IDEA.md section 16).
    """
    from ..annotation import manager as annotation_manager

    annotations = list(annotations)
    analyzer = FrameAnalyzer(config, cache=cache, preprocessing_enabled=preprocessing_enabled)

    tasks: list[tuple[ECGFrame, str]] = []
    for (subject, session), frames in dataset.frames_by_session.items():
        labels = annotation_manager.labels_for_frames(annotations, subject, session, len(frames))
        for frame, label in zip(frames, labels):
            if label in STATIC_POSITIONS:
                tasks.append((frame, label))

    results: list[FrameResult] = []
    for index, (frame, label) in enumerate(tasks, start=1):
        result = analyzer.analyze_frame(frame, label)
        results.append(result)
        if progress is not None and (index % 5 == 0 or index == len(tasks)):
            progress(index, len(tasks), result)

    return {
        "results": results,
        "analyzer": analyzer,
        "provenance": RunProvenance(
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            software_version=__version__,
            config_version=config.version,
            config_name=config.name,
            config_fingerprint=analyzer.fingerprint,
            methodology=config.methodology,
            sqi_method=analyzer.method_label,
            detector_a=analyzer.detector_a,
            detector_b=analyzer.detector_b,
            sampling_rate=float(config.get("signal.sampling_rate_hz", 250.0)),
            preprocessing=analyzer.preprocessing_config,
            preprocessing_applied=analyzer.effective_preprocessing_config,
            fuzzy_config=analyzer.fuzzy_config,
            dataset_source=str(dataset.inventory.coverage() if dataset.inventory else {}),
            reference=REFERENCE,
        ).to_dict(),
        "counts": {
            "tasks": len(tasks),
            "valid": sum(1 for r in results if r.valid),
            "invalid": sum(1 for r in results if not r.valid),
        },
    }


__all__ = [
    "EXCELLENT",
    "DetectionResult",
    "FuzzyResult",
    "FrameAnalyzer",
    "PeakMatchResult",
    "ResultCache",
    "RunProvenance",
    "SpectralResult",
    "annotation_storage",
    "run_analysis",
    "segment_frame",
]

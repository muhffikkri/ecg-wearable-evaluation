"""Result models for the SQI pipeline.

Every intermediate quantity is retained. Nothing returns only a class label
(IDEA.md sections 19, 23, 24, 25, 26, 29, 40, 41).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EXCELLENT = "Excellent"
BARELY_ACCEPTABLE = "Barely Acceptable"
UNACCEPTABLE = "Unacceptable"

QUALITY_CLASSES = (EXCELLENT, BARELY_ACCEPTABLE, UNACCEPTABLE)

#: Per-index acceptance levels, as named by Zhao & Zhang (2018).
#:
#: Every index is judged against its own criterion *before* any fusion happens,
#: so these levels are reported next to the index value and are never folded
#: into the fused quality class.
OPTIMAL = "optimal"
SUSPICIOUS = "suspicious"
UNQUALIFIED = "unqualified"

#: Returned when an index's criterion cannot be applied to a frame at all --
#: for example the QRS spectral power index when the heart rate lies outside the
#: 60-160 bpm range the reference calibrated its limits on. Deliberately not one
#: of the three real levels, so an inapplicable criterion can never be counted
#: as suspicious or unqualified by accident.
UNDEFINED = "undefined"

ACCEPTANCE_LEVELS = (OPTIMAL, SUSPICIOUS, UNQUALIFIED)

SQI_KEYS = ("qSQI", "pSQI", "kSQI", "basSQI")

#: Display names for each signal-quality index.
#:
#: ``SQI_KEYS`` names stay as the column names in the results table, the config
#: keys and the exported CSV, because those are data consumed by other code and
#: written into report artefacts. Only the label a reader sees is spelled out,
#: so no figure, table or generated paragraph has to make the reader guess what
#: a code such as "pSQI" stood for.
SQI_LABELS: dict[str, str] = {
    "qSQI": "Deteksi R-peak",
    "pSQI": "Distribusi Daya Spektral QRS",
    "kSQI": "Kurtosis Sinyal",
    "basSQI": "Daya Relatif Baseline",
}


def sqi_label(key: str) -> str:
    """Human-readable name for an SQI key, falling back to the key itself."""
    return SQI_LABELS.get(key, key)


@dataclass
class Acceptance:
    """One signal-quality index judged against its own acceptance criterion.

    ``level`` is one of :data:`OPTIMAL`, :data:`SUSPICIOUS` or
    :data:`UNQUALIFIED`, or :data:`UNDEFINED` when the criterion does not apply
    to this frame. The limits that produced the verdict are kept, so a reader
    can re-check the comparison instead of trusting the label.
    """

    level: str = UNDEFINED
    value: float = float("nan")
    rule: str = ""
    limits: dict[str, float] = field(default_factory=dict)
    #: Set when ``level`` is UNDEFINED, saying what stopped the criterion.
    reason: str = ""

    @property
    def defined(self) -> bool:
        return self.level in ACCEPTANCE_LEVELS

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["defined"] = self.defined
        return payload


@dataclass
class HeuristicFusionResult:
    """Simple heuristic fusion of the per-index acceptance levels.

    The reference builds the fused class from the *counts* of optimal,
    suspicious and unqualified indices rather than from the index values, so the
    counts are stored next to the class that they produced.
    """

    quality_class: str = UNDEFINED
    n_factors: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    levels: dict[str, str] = field(default_factory=dict)
    rule: str = ""
    #: False when at least one index had no applicable criterion, in which case
    #: no fused class is defined and ``quality_class`` is UNDEFINED.
    applied: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PeakMatchResult:
    """Everything needed to visually debug qSQI (IDEA.md section 23)."""

    n_peaks_a: int
    n_peaks_b: int
    n_matched: int
    q_sqi: float
    peaks_a: list[int] = field(default_factory=list)
    peaks_b: list[int] = field(default_factory=list)
    matched_pairs: list[tuple[int, int]] = field(default_factory=list)
    unmatched_a: list[int] = field(default_factory=list)
    unmatched_b: list[int] = field(default_factory=list)
    tolerance_ms: float = 0.0
    detector_a: str = ""
    detector_b: str = ""
    label: str = ""  # "zhao_zhang" | "pan_tompkins_adapted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpectralResult:
    """Stored PSD and band powers feeding pSQI and basSQI."""

    frequencies: list[float] = field(default_factory=list)
    psd: list[float] = field(default_factory=list)
    qrs_band_power: float = 0.0
    p_sqi: float = 0.0
    baseline_band_power: float = 0.0
    bas_sqi: float = 0.0
    p_total_power: float = 0.0
    bas_total_power: float = 0.0
    psd_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FuzzyResult:
    """Evaluation matrix, weight vector, membership vector and decision.

    The synthesised membership vector ``S`` is the output of the configured
    operator and is reported exactly as computed. It is not rescaled: the
    reference's rows of R are raw membership degrees that need not sum to 1 (the
    Cauchy rows of qSQI and basSQI do not), so rescaling S would invent numbers
    no operator in the literature produces.

    ``score`` is the defuzzified value ``v`` of Eq (33), the quantity the
    reported class is actually decided from. It lies between the smallest and
    largest rating value, so it reads as "this frame sits near level v".
    """

    membership: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    evaluation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    quality_class: str = UNACCEPTABLE
    synthesis: str = ""
    config_version: str = ""
    #: Defuzzified score v of Eq (33); NaN when no rating could be formed.
    score: float = float("nan")
    #: Numerical value j of each rating level, as used by Eq (33).
    rating_values: dict[str, float] = field(default_factory=dict)
    #: SQI factors whose membership row was identically zero, so they carried no
    #: weight into the synthesis. Recorded rather than dropped in silence.
    out_of_support_factors: list[str] = field(default_factory=list)

    @property
    def excellent(self) -> float:
        return float(self.membership.get(EXCELLENT, 0.0))

    @property
    def barely_acceptable(self) -> float:
        return float(self.membership.get(BARELY_ACCEPTABLE, 0.0))

    @property
    def unacceptable(self) -> float:
        return float(self.membership.get(UNACCEPTABLE, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "membership": self.membership,
            "weights": self.weights,
            "evaluation_matrix": self.evaluation_matrix,
            "quality_class": self.quality_class,
            "synthesis": self.synthesis,
            "config_version": self.config_version,
            "score": self.score,
            "rating_values": self.rating_values,
            "out_of_support_factors": self.out_of_support_factors,
        }


@dataclass
class FrameResult:
    """Complete per-frame analysis output, including run provenance."""

    subject_id: str
    session_id: str
    frame_id: str
    internal_id: str = ""
    position: str = ""
    start_time: str | None = None
    end_time: str | None = None
    sample_count: int = 0
    sampling_rate: float = 0.0

    q_sqi: float = float("nan")
    p_sqi: float = float("nan")
    k_sqi: float = float("nan")
    bas_sqi: float = float("nan")

    #: Heart rate used by the acceptance criterion that depends on it,
    #: estimated as 60 / mean R-R from the reference detections.
    heart_rate_bpm: float = float("nan")
    #: Per-index acceptance level: one of ACCEPTANCE_LEVELS, or UNDEFINED.
    q_sqi_acceptance: str = ""
    p_sqi_acceptance: str = ""
    k_sqi_acceptance: str = ""
    bas_sqi_acceptance: str = ""
    #: Simple heuristic fusion (the reference's step 1), kept beside the fuzzy
    #: class (step 2) so the two can be compared the way the article compares
    #: them in its Table 4 and Table 5.
    fusion_class: str = ""
    fusion_optimal: int = 0
    fusion_suspicious: int = 0
    fusion_unqualified: int = 0

    fuzzy_excellent: float = float("nan")
    fuzzy_barely_acceptable: float = float("nan")
    fuzzy_unacceptable: float = float("nan")
    quality_class: str = ""

    kurtosis_mean: float = float("nan")
    kurtosis_std: float = float("nan")
    kurtosis_definition: str = ""
    n_peaks_a: int = 0
    n_peaks_b: int = 0
    n_matched: int = 0
    q_sqi_label: str = ""
    peaks_a: list[int] = field(default_factory=list)
    peaks_b: list[int] = field(default_factory=list)

    valid: bool = True
    invalid_reason: str = ""
    preprocessing_applied: bool = False
    preprocessing_config: dict[str, Any] = field(default_factory=dict)
    run_provenance: dict[str, Any] = field(default_factory=dict)

    def sqi_values(self) -> dict[str, float]:
        return {key: getattr(self, key.lower()) for key in SQI_KEYS}

    def to_row(self) -> dict[str, Any]:
        """Flat row for frame_results.csv (IDEA.md section 40)."""
        return {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "position": self.position,
            "frame_id": self.frame_id,
            "internal_id": self.internal_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sample_count": self.sample_count,
            "sampling_rate": self.sampling_rate,
            "qSQI": self.q_sqi,
            "pSQI": self.p_sqi,
            "kSQI": self.k_sqi,
            "basSQI": self.bas_sqi,
            "heart_rate_bpm": self.heart_rate_bpm,
            "qSQI_acceptance": self.q_sqi_acceptance,
            "pSQI_acceptance": self.p_sqi_acceptance,
            "kSQI_acceptance": self.k_sqi_acceptance,
            "basSQI_acceptance": self.bas_sqi_acceptance,
            "fusion_class": self.fusion_class,
            "fusion_optimal": self.fusion_optimal,
            "fusion_suspicious": self.fusion_suspicious,
            "fusion_unqualified": self.fusion_unqualified,
            "fuzzy_excellent": self.fuzzy_excellent,
            "fuzzy_barely_acceptable": self.fuzzy_barely_acceptable,
            "fuzzy_unacceptable": self.fuzzy_unacceptable,
            "quality_class": self.quality_class,
            "kurtosis_mean": self.kurtosis_mean,
            "kurtosis_std": self.kurtosis_std,
            "kurtosis_definition": self.kurtosis_definition,
            "n_peaks_a": self.n_peaks_a,
            "n_peaks_b": self.n_peaks_b,
            "n_matched": self.n_matched,
            "q_sqi_label": self.q_sqi_label,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "preprocessing_applied": self.preprocessing_applied,
        }


@dataclass
class RunProvenance:
    """Recorded for every analysis run (IDEA.md section 41)."""

    analysis_timestamp: str
    software_version: str
    config_version: str
    config_name: str
    config_fingerprint: str
    methodology: str
    sqi_method: str
    detector_a: str
    detector_b: str
    sampling_rate: float
    preprocessing: dict[str, Any]
    #: The preprocessing chain that actually ran, after the ``applied`` switch.
    preprocessing_applied: dict[str, Any] = field(default_factory=dict)
    fuzzy_config: dict[str, Any] = field(default_factory=dict)
    dataset_source: str = ""
    reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ACCEPTANCE_LEVELS",
    "Acceptance",
    "BARELY_ACCEPTABLE",
    "EXCELLENT",
    "FrameResult",
    "FuzzyResult",
    "HeuristicFusionResult",
    "OPTIMAL",
    "PeakMatchResult",
    "QUALITY_CLASSES",
    "RunProvenance",
    "SpectralResult",
    "SQI_KEYS",
    "SUSPICIOUS",
    "UNACCEPTABLE",
    "UNDEFINED",
    "UNQUALIFIED",
]

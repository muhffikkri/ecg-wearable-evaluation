"""Interpretation engine.

Generates a structured scientific interpretation entirely from calculated
results. No conclusion is hard-coded (IDEA.md sections 36, 37, 38, 52).

Two rules are enforced throughout:

1. **Measurement vs cause.** An SQI indicates a possible artifact. It does
   not establish its cause, and the wording keeps that distinction.
2. **Signal quality, not patient health.** Nothing here infers arrhythmia,
   cardiac disease or any clinical condition (sections 38, 39, 56).
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..models.result import BARELY_ACCEPTABLE, EXCELLENT, UNACCEPTABLE, sqi_label
from ..analysis.statistics import SQI_COLUMNS, class_distribution, describe

POSITIONS = ("SUPINE", "SITTING", "STANDING")

#: What a low value on each SQI measures, and what it may suggest.
#: "measurement" is a statement about the signal; "possible cause" is explicitly
#: hedged (IDEA.md section 37).
SQI_INTERPRETATION: dict[str, dict[str, str]] = {
    "qSQI": {
        "name": "R-peak detection agreement",
        "low_measurement": (
            "The two independent R-peak detectors disagree on more than a small "
            "fraction of beats."
        ),
        "low_possible_cause": (
            "This may reflect waveform distortion, motion artifact, or reduced QRS "
            "clarity. It does not by itself identify which of these applies."
        ),
    },
    "pSQI": {
        "name": "QRS-band relative power",
        "low_measurement": (
            "Less of the signal's spectral energy lies in the QRS-related band "
            "relative to the analysis band."
        ),
        "low_possible_cause": (
            "This may suggest high-frequency interference or spectral distortion."
        ),
    },
    "kSQI": {
        "name": "Kurtosis",
        "low_measurement": (
            "The amplitude distribution is less sharply peaked than a clean ECG "
            "would be."
        ),
        "low_possible_cause": (
            "This may indicate an increased noise contribution to the signal."
        ),
    },
    "basSQI": {
        "name": "Baseline relative power",
        "low_measurement": (
            "The signal contains relatively more low-frequency energy than a "
            "stable recording would."
        ),
        "low_possible_cause": (
            "This may suggest baseline wander or electrode-motion effects."
        ),
    },
}

#: Phrases that would turn a signal-quality statement into a clinical claim.
FOREIGN_CLAIMS_BLOCKED = (
    "arrhythmia",
    "cardiac disease",
    "clinically validated",
    "clinical equivalence",
    "equivalent to hospital",
    "patient abnormality",
    "diagnos",
)

#: Conclusions IDEA.md sections 52 and 56 forbid the application from stating
#: because no criterion in this project defines them.
UNDECLARED_CONCLUSIONS_BLOCKED = (
    "the wearable performed well",
    "the wearable performed poorly",
    "the wearable is good",
    "the wearable is bad",
    "clinically better",
    "clinically worse",
)

#: Constructions that turn a blocked phrase into an explicit disclaimer,
#: e.g. "not a clinically validated classifier".
_NEGATION_CUES = re.compile(
    r"\b(?:not|no|never|without|cannot|rather than|"
    r"does not|do not|did not|must not|is not|are not|was not|were not|"
    r"implies no|draws no|makes no)\b"
)

#: Clause boundaries. Claims are asserted per clause, so a disclaimer inside a
#: clause does not license a claim in the same clause. The comma is included
#: because it ends the concessive clause in "although not validated, it detects
#: arrhythmia", leaving the assertion to stand on its own without a negation.
_CLAUSE_SPLIT = re.compile(r"[.;:!?,()\n]")

#: Adversative connectives are also boundaries. Without them, "the device is not
#: clinically validated but it detects arrhythmias" is read as a disclaimer
#: because a negation appears somewhere before the blocked phrase, when in fact
#: the phrase is asserted after the contrast.
#:
#: Only adversative connectives qualify. Subordinating ones (although, though,
#: while, despite) introduce a concessive clause in which a preceding negation
#: still governs the rest of the sentence, so treating them as boundaries would
#: wrongly license the very claims this guard exists to catch.
#:
#: The choice is deliberately biased towards flagging: a false positive rewrites
#: a sentence into a disclaimer, while a false negative lets a clinical claim
#: through.
_CONTRAST_SPLIT = re.compile(
    r"\b(?:but|however|yet|whereas|nevertheless|nonetheless|instead)\b",
    re.IGNORECASE,
)


def _clauses(text: str) -> list[str]:
    """Split into assertion units: sentences, then contrastive clauses."""
    out: list[str] = []
    for sentence in _CLAUSE_SPLIT.split(text):
        if not sentence.strip():
            continue
        out.extend(part for part in _CONTRAST_SPLIT.split(sentence) if part.strip())
    return out


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.1f}%"


def _asserted_claims(text: str) -> list[str]:
    """Blocked phrases that the text ASSERTS, clause by clause.

    A blocked phrase inside a negated clause is a disclaimer and is not a
    claim. Negation is detected with whole-word cues rather than by deleting
    the cue, because deleting it would turn the disclaimer into an assertion,
    and because bare substring deletion corrupts unrelated words ("noisy",
    "normal", "another", "notes").
    """
    asserted: list[str] = []
    blocked = FOREIGN_CLAIMS_BLOCKED + UNDECLARED_CONCLUSIONS_BLOCKED
    for clause in _clauses(text):
        lowered = " ".join(clause.lower().split())
        if not lowered:
            continue
        for phrase in blocked:
            index = lowered.find(phrase)
            if index < 0:
                continue
            if _NEGATION_CUES.search(lowered[:index]):
                continue  # disclaimer, e.g. "no diagnostic inference is made"
            asserted.append(phrase)
    return asserted


def _guard(text: str) -> str:
    """Fail loudly if generated text drifts into clinical claims.

    The engine's own scope statements legitimately contain these words inside
    a negation ("no clinical interpretation is made"), so a negated mention is
    allowed and only an assertion raises.
    """
    asserted = _asserted_claims(text)
    if asserted:
        raise AssertionError(
            f"interpretation text contains a clinical claim ({asserted[0]!r}); "
            "this project evaluates signal quality only"
        )
    return text


def low_sqi_factors(sqi_values: dict[str, float], threshold: float) -> list[str]:
    """SQI factors below the threshold, largest deficiency first."""
    shortfalls = []
    for factor, value in sqi_values.items():
        if value is None or not math.isfinite(value):
            shortfalls.append((factor, float("inf")))
        elif value < threshold:
            shortfalls.append((factor, threshold - value))
    shortfalls.sort(key=lambda item: item[1], reverse=True)
    return [factor for factor, _ in shortfalls]


def _attr(result: Any, column: str) -> float:
    """Read an SQI column from a FrameResult or a DataFrame row.

    The DataFrame column names (``qSQI``) and the dataclass field names
    (``q_sqi``) differ, so the mapping lives in one place.
    """
    if hasattr(result, "to_row"):
        return float(result.to_row()[column])
    return float(result[column])


def describe_frame(result: Any) -> str:
    """Per-frame diagnostic paragraph for a single frame."""
    if not result.valid:
        return _guard(
            f"Frame {result.frame_id} ({result.subject_id}, {result.position}) was excluded "
            f"from analysis: {result.invalid_reason or 'validation failed'}."
        )

    sqi_values = {column: _attr(result, column) for column in SQI_COLUMNS}
    excellent = result.fuzzy_excellent
    barely = result.fuzzy_barely_acceptable
    unacceptable = result.fuzzy_unacceptable

    lines = [
        f"Frame {result.frame_id} ({result.subject_id}, {result.position}) was classified "
        f"**{result.quality_class}** with membership "
        f"Excellent {_fmt(excellent)}, Barely Acceptable {_fmt(barely)}, "
        f"Unacceptable {_fmt(unacceptable)}."
    ]
    lines.append("")
    lines.append(
        "SQI values: "
        f"{sqi_label('qSQI')} {_fmt(result.q_sqi)}, "
        f"{sqi_label('pSQI')} {_fmt(result.p_sqi)}, "
        f"{sqi_label('kSQI')} {_fmt(result.k_sqi)}, "
        f"{sqi_label('basSQI')} {_fmt(result.bas_sqi)}."
    )

    if result.quality_class == UNACCEPTABLE:
        # Rank factors by how far they sit below the Excellent anchor, so the
        # "primarily associated with" list reflects the actual numbers.
        low = sorted(
            SQI_COLUMNS,
            key=lambda column: sqi_values[column] if np.isfinite(sqi_values[column]) else -np.inf,
        )[:2]
        lines.append("")
        lines.append("Unacceptable quality primarily associated with:")
        for column in low:
            info = SQI_INTERPRETATION[column]
            lines.append(f"- **{sqi_label(column)}** ({info['name']}): {_fmt(sqi_values[column])}. {info['low_measurement']}")
    return _guard("\n".join(lines))


def overall_finding(df: pd.DataFrame) -> str:
    """Section 1: overall signal quality, from the class distribution."""
    if df.empty:
        return "No frames were available for analysis."

    valid = df[df["valid"]] if "valid" in df else df
    total = int(len(df))
    if valid.empty:
        return f"All {total} analysed frames failed validation and none were classified."

    distribution = class_distribution(valid["quality_class"].tolist())
    parts = [
        f"Of the {len(valid)} analysed ECG segments, "
        f"{_pct(distribution.get(EXCELLENT, 0.0))} were classified as {EXCELLENT}, "
        f"{_pct(distribution.get(BARELY_ACCEPTABLE, 0.0))} as {BARELY_ACCEPTABLE}, and "
        f"{_pct(distribution.get(UNACCEPTABLE, 0.0))} as {UNACCEPTABLE}."
    ]
    if total != len(valid):
        parts.append(f"A further {total - len(valid)} frame(s) were excluded as invalid.")
    return _guard(" ".join(parts))


def position_comparison_text(position_df: pd.DataFrame, comparison: dict[str, Any] | None) -> str:
    """Section 2: behaviour per body position, with the numbers behind it."""
    if position_df.empty:
        return "No position-level results are available."

    lines = ["Position comparison", "-------------------"]
    for _, row in position_df.iterrows():
        position = row["position"]
        lines.append(
            f"- **{position}**: n={int(row['n_frames'])} frames; "
            f"{sqi_label('qSQI')} mean {_fmt(row.get('qSQI_mean'))} / "
            f"median {_fmt(row.get('qSQI_median'))}; "
            f"{sqi_label('pSQI')} mean {_fmt(row.get('pSQI_mean'))}; "
            f"{sqi_label('kSQI')} mean {_fmt(row.get('kSQI_mean'))}; "
            f"{sqi_label('basSQI')} mean {_fmt(row.get('basSQI_mean'))}; "
            f"{_pct(row.get(f'pct_{EXCELLENT}', float('nan')))} {EXCELLENT}, "
            f"{_pct(row.get(f'pct_{BARELY_ACCEPTABLE}', float('nan')))} {BARELY_ACCEPTABLE}, "
            f"{_pct(row.get(f'pct_{UNACCEPTABLE}', float('nan')))} {UNACCEPTABLE}."
        )

    if comparison and comparison.get("tests"):
        lines.append("")
        lines.append("Repeated-measures analysis (Friedman test across the three positions):")
        for column, outcome in comparison["tests"].items():
            if outcome.get("status") != "computed":
                lines.append(
                    f"- {column}: not computable ({outcome.get('note', outcome.get('status'))})."
                )
                continue
            alpha = comparison.get("alpha", 0.05)
            detectable = outcome["p_value"] < alpha
            lines.append(
                f"- {column}: statistic {_fmt(outcome['statistic'])}, "
                f"p = {_fmt(outcome['p_value'], 4)}, "
                f"n = {outcome['n_subjects']} subjects, "
                f"Kendall's W = {_fmt(outcome.get('kendall_w'))}. "
                + (
                    f"A difference at alpha = {alpha} is statistically detectable."
                    if detectable
                    else f"No difference statistically detectable at alpha = {alpha}."
                )
            )
        lines.append("")
        lines.append(
            "These results describe the recorded sample. They do not establish that any "
            "position is clinically better or worse."
        )
    return _guard("\n".join(lines))


def sqi_interpretation_text(df: pd.DataFrame) -> str:
    """Section 3: per-SQI behaviour across the whole dataset."""
    if df.empty:
        return "No SQI results are available."

    valid = df[df["valid"]] if "valid" in df else df
    if valid.empty:
        return "No valid frames are available for SQI interpretation."

    lines = ["SQI interpretation", "-------------------"]
    for column in SQI_COLUMNS:
        info = SQI_INTERPRETATION[column]
        stats = describe(valid[column].tolist())
        line = (
            f"- **{column}** ({info['name']}): mean {_fmt(stats['mean'])}, "
            f"median {_fmt(stats['median'])}, SD {_fmt(stats['sd'])}, "
            f"IQR {_fmt(stats['iqr'])}, range {_fmt(stats['min'])} to {_fmt(stats['max'])} "
            f"(n = {stats['n']})."
        )
        lines.append(line)
    return _guard("\n".join(lines))


def artifact_text(df: pd.DataFrame, threshold: float) -> str:
    """Section 4: what the low-quality frames look like in SQI terms."""
    if df.empty:
        return "No frames are available for artifact inspection."

    valid = df[df["valid"]] if "valid" in df else df
    if valid.empty:
        return "No valid frames are available for artifact inspection."

    counts = {column: int((valid[column] < threshold).sum()) for column in SQI_COLUMNS}
    lines = ["Potential artifact characteristics", "-------------------------------"]
    for column, count in counts.items():
        info = SQI_INTERPRETATION[column]
        share = 100.0 * count / len(valid)
        if count == 0:
            lines.append(
                f"- {column}: no frame fell below {threshold:g}. {info['low_measurement']} "
                "was therefore not observed in this dataset."
            )
            continue
        lines.append(
            f"- Frames with {column} below {threshold:g} ({count} of {len(valid)}, "
            f"{_pct(share)}): {info['low_measurement']} {info['low_possible_cause']}"
        )
    return _guard("\n".join(lines))


def variability_text(subject_df: pd.DataFrame) -> str:
    """Section 5: inter-subject variability, reported as dispersion."""
    if subject_df.empty:
        return "Not enough annotated subjects to assess inter-subject variability."

    lines = ["Inter-subject variability", "--------------------------"]
    for column in SQI_COLUMNS:
        mean_column = f"{column}_mean"
        if mean_column not in subject_df:
            continue
        values = [v for v in subject_df[mean_column].tolist() if v is not None and np.isfinite(v)]
        if len(values) < 2:
            continue
        spread = float(np.std(values, ddof=1))
        lines.append(
            f"- {column}: subject-level means range from {_fmt(min(values))} to "
            f"{_fmt(max(values))} (SD across subject means {_fmt(spread)}, "
            f"n = {len(values)} subject-position groups)."
        )
    if len(lines) == 1:
        return "Not enough annotated subjects to assess inter-subject variability."
    return _guard("\n".join(lines))


def acquisition_text(dataset_summary: dict[str, Any] | None, coverage: dict[str, Any] | None) -> str:
    """Question 1 of IDEA.md section 55: continuity of acquisition.

    The expected frame count is read from the annotation coverage, which is
    itself derived from the configured dataset expectation. Nothing about the
    experiment design is hard-coded here (IDEA.md sections 31 and 62).
    """
    if not dataset_summary and not coverage:
        return "Dataset summary is unavailable."

    lines = ["Acquisition completeness", "-----------------------"]
    if dataset_summary:
        # ``subjects`` arrives as a count from analysis.statistics.overall_summary
        # but as a list of ids elsewhere, so accept either rather than assuming
        # one shape and crashing on the other.
        subjects = dataset_summary.get("subjects", 0)
        n_subjects = len(subjects) if isinstance(subjects, (list, tuple, set)) else int(subjects or 0)
        lines.append(
            f"- Ingested {dataset_summary.get('frames', 0)} frame(s) from "
            f"{dataset_summary.get('sessions', 0)} session(s) across "
            f"{n_subjects} subject(s); "
            f"source format(s): {', '.join(dataset_summary.get('source_formats', [])) or 'n/a'}; "
            f"sampling rate(s): {', '.join(str(r) for r in dataset_summary.get('sampling_rates', [])) or 'n/a'} Hz."
        )
        if dataset_summary.get("malformed"):
            lines.append(
                f"- {dataset_summary['malformed']} malformed record(s) were reported and "
                "excluded rather than silently discarded."
            )
    if coverage:
        design = coverage.get("expected_design")
        prefix = f"Expected frames ({design}): " if design else "Expected frames: "
        lines.append(
            f"{prefix}{coverage.get('expected_frames', 'n/a')}; "
            f"analyzable annotated frames: {coverage.get('analyzable_frames', 'n/a')}; "
            f"missing: {coverage.get('missing_frames', 'n/a')}; "
            f"excluded transitions: {coverage.get('excluded_transitions', 'n/a')}; "
            f"invalid: {dataset_summary.get('invalid_frames', 'n/a') if dataset_summary else 'n/a'}."
        )
    return _guard("\n".join(lines))


def limitations_text(config_pending: Sequence[tuple[str, Any]], config_name: str) -> str:
    """Section 7: experimental limitations, stated explicitly."""
    lines = ["Limitations", "------------"]
    lines.append(
        "- The fuzzy evaluation follows Zhao & Zhang (2018), which was developed and "
        "evaluated on PhysioNet databases. Applying it to ECGRHYTHMIA recordings is an "
        "application of an established signal-quality methodology to a new dataset. It is "
        "not a classifier that has been validated for this wearable, and the class "
        "distribution it produces is not a performance claim about the device."
    )
    if config_pending:
        preview = ", ".join(path for path, _ in config_pending[:8])
        more = f" (+{len(config_pending) - 8} more)" if len(config_pending) > 8 else ""
        lines.append(
            f"- Configuration '{config_name}' has {len(config_pending)} parameter(s) still "
            f"pending verification against the published paper: {preview}{more}. Results "
            "should be read as a structured reproduction until those values are confirmed."
        )
    lines.append(
        "- Signal quality describes the recording, not the subject's health. The report "
        "makes no inference about any medical condition."
    )
    lines.append(
        "- TRANSITION frames are excluded from the position comparison, so the results "
        "describe static positions only and exclude posture-change periods."
    )
    lines.append(
        "- Body-position labels come from the experimenter's manual notes; label accuracy is "
        "not independently verified by the application."
    )
    return _guard("\n".join(lines))


def build_report(
    *,
    frame_df: pd.DataFrame,
    position_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    comparison: dict[str, Any] | None = None,
    dataset_summary: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
    config_pending: Sequence[tuple[str, Any]] = (),
    config_name: str = "",
    provenance: dict[str, Any] | None = None,
    problematic_threshold: float = 0.5,
) -> str:
    """Assemble the full structured Markdown interpretation.

    ``problematic_threshold`` is supplied by the caller from
    ``interpretation.problematic_unacceptable_threshold`` in the configuration,
    so no protocol value is hidden in this module (IDEA.md section 62).
    """
    sections = [
        "# ECGRHYTHMIA Wearable ECG Signal Quality Evaluation",
        "",
        "> Scope statement: this report evaluates ECG **signal quality** acquired by the "
        "ECGRHYTHMIA wearable prototype. It is not a diagnostic assessment and contains no "
        "clinical interpretation.",
        "",
        acquisition_text(dataset_summary, coverage),
        "",
        overall_finding(frame_df),
        "",
        position_comparison_text(position_df, comparison),
        "",
        sqi_interpretation_text(frame_df),
        "",
        artifact_text(frame_df, problematic_threshold),
        "",
        variability_text(subject_df),
        "",
        limitations_text(config_pending, config_name),
    ]

    if provenance:
        sections.extend(
            [
                "",
                "Reproducibility record",
                "-----------------------",
                f"- Analysis timestamp: {provenance.get('analysis_timestamp', 'n/a')}",
                f"- Software version: {provenance.get('software_version', 'n/a')}",
                f"- Configuration: {provenance.get('config_name', 'n/a')} "
                f"v{provenance.get('config_version', 'n/a')} "
                f"(fingerprint {provenance.get('config_fingerprint', 'n/a')})",
                f"- SQI method: {provenance.get('sqi_method', 'n/a')}",
                f"- R-peak detectors: {provenance.get('detector_a', 'n/a')} vs "
                f"{provenance.get('detector_b', 'n/a')}",
                f"- Sampling rate: {provenance.get('sampling_rate', 'n/a')} Hz",
                f"- Reference: {provenance.get('reference', 'n/a')}",
            ]
        )

    return _guard("\n".join(sections) + "\n")


def problematic_frames(frame_df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Frames worth inspecting, worst first."""
    if frame_df.empty:
        return frame_df
    valid = frame_df[frame_df["valid"]].copy() if "valid" in frame_df else frame_df.copy()
    if valid.empty:
        return valid
    valid = valid.sort_values("fuzzy_unacceptable", ascending=False)
    return valid.head(limit)


__all__ = [
    "FOREIGN_CLAIMS_BLOCKED",
    "SQI_INTERPRETATION",
    "UNDECLARED_CONCLUSIONS_BLOCKED",
    "acquisition_text",
    "artifact_text",
    "build_report",
    "describe_frame",
    "limitations_text",
    "low_sqi_factors",
    "overall_finding",
    "position_comparison_text",
    "problematic_frames",
    "sqi_interpretation_text",
    "variability_text",
]

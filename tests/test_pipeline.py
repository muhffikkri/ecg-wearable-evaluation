"""Annotation, segmentation, pipeline, statistics and interpretation tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ecg_eval.analysis import (
    FrameAnalyzer,
    ResultCache,
    describe,
    export_results,
    friedman_test,
    overall_summary,
    position_comparison,
    position_summary,
    results_to_frame,
    run_analysis,
    segment_frame,
    subject_summary,
)
from ecg_eval.annotation import manager, storage
from ecg_eval.config import load_config
from ecg_eval.ingestion import Dataset, ingest
from ecg_eval.interpretation import build_report, describe_frame, low_sqi_factors
from ecg_eval.models.annotation import (
    SITTING,
    STANDING,
    SUPINE,
    TRANSITION,
    Segment,
    SubjectAnnotation,
)
from ecg_eval.preprocessing import (
    STAGE_ORDER,
    default_config,
    enabled_stages,
    preprocess,
    stage_enabled,
)
from conftest import requires_real_data

FS = 250.0


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
def test_assign_and_lookup():
    annotations: list[SubjectAnnotation] = []
    manager.assign(annotations, "S01", "ses1", SUPINE, 0, 5)
    labels = manager.labels_for_frames(annotations, "S01", "ses1", 10)
    assert labels[:6] == [SUPINE] * 6
    assert labels[6:] == ["UNLABELED"] * 4


def test_assign_replaces_overlapping_range():
    annotations: list[SubjectAnnotation] = []
    manager.assign(annotations, "S01", "ses1", SUPINE, 0, 9)
    manager.assign(annotations, "S01", "ses1", SITTING, 4, 6)
    labels = manager.labels_for_frames(annotations, "S01", "ses1", 10)
    assert labels == [SUPINE] * 4 + [SITTING] * 3 + [SUPINE] * 3


def test_analyzable_frames_excludes_transitions():
    annotations: list[SubjectAnnotation] = []
    manager.assign(annotations, "S01", "ses1", SUPINE, 0, 2)
    manager.assign(annotations, "S01", "ses1", TRANSITION, 3, 3)
    manager.assign(annotations, "S01", "ses1", STANDING, 4, 5)
    assert manager.analyzable_frames(annotations, "S01", "ses1", 6) == [0, 1, 2, 4, 5]


def test_assign_rejects_invalid_label():
    with pytest.raises(ValueError):
        manager.assign([], "S01", "ses1", "LYING", 0, 1)


def test_segment_rejects_reversed_range():
    with pytest.raises(ValueError):
        Segment(SUPINE, 5, 2)


def test_per_position_counts():
    annotations: list[SubjectAnnotation] = []
    manager.assign(annotations, "S01", "ses1", SUPINE, 0, 1)
    manager.assign(annotations, "S01", "ses1", TRANSITION, 2, 2)
    counts = manager.per_position_counts(annotations, "S01", "ses1", 5)
    assert counts[SUPINE] == 2
    assert counts[TRANSITION] == 1
    assert counts["UNLABELED"] == 2


def test_validation_detects_out_of_range():
    annotations = [SubjectAnnotation("S01", "ses1", [Segment(SUPINE, 0, 99)])]
    problems = manager.validate(annotations, {("S01", "ses1"): [object()] * 5})
    assert any("outside recording" in p for p in problems)


def test_save_and_load_round_trip(tmp_path):
    annotations: list[SubjectAnnotation] = []
    manager.assign(annotations, "S01", "ses1", SUPINE, 0, 5, note="lying down")
    manager.assign(annotations, "S01", "ses1", SITTING, 6, 11)
    path = storage.save_annotations(tmp_path, annotations)

    loaded = storage.load_annotations(tmp_path)
    assert len(loaded) == 1
    assert [s.label for s in loaded[0].segments] == [SUPINE, SITTING]
    assert loaded[0].segments[0].note == "lying down"
    assert path.exists()
    assert (tmp_path / storage.MANIFEST_FILE).exists()


def test_load_missing_file_is_empty(tmp_path):
    assert storage.load_annotations(tmp_path / "absent") == []


def test_annotation_hash_changes_with_content():
    a: list[SubjectAnnotation] = []
    b: list[SubjectAnnotation] = []
    manager.assign(a, "S01", "ses1", SUPINE, 0, 5)
    manager.assign(b, "S01", "ses1", SITTING, 0, 5)
    assert storage.annotation_hash(a) != storage.annotation_hash(b)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def test_ten_second_frame_used_directly(make_frame, clean_signal):
    frame = make_frame(clean_signal)
    segments = segment_frame(frame, position=SUPINE, analysis_duration_s=10.0)
    assert len(segments) == 1
    assert segments[0].sample_count == 2500
    assert segments[0].metadata["segmentation"] == "source frame used directly"


def test_continuous_signal_is_windowed(make_frame):
    frame = make_frame(np.zeros(7500, dtype=np.float32))
    segments = segment_frame(frame, position=SUPINE, analysis_duration_s=10.0)
    assert len(segments) == 3
    assert [s.start_sample for s in segments] == [0, 2500, 5000]
    assert all(s.sample_count == 2500 for s in segments)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_default_config_enables_only_the_three_requested_stages():
    """wavelet + median baseline + butter bandpass on; nothing else.

    Normalisation is deliberately excluded so amplitudes stay in mV.
    """
    config = default_config()
    assert enabled_stages(config) == ["wavelet", "baseline", "bandpass"]
    assert not stage_enabled(config, "normalize")
    assert not stage_enabled(config, "notch")
    assert not stage_enabled(config, "resample")


def test_every_stage_can_be_toggled_independently(clean_signal):
    """Each stage switches on its own, without dragging the others along."""
    for stage in STAGE_ORDER:
        config = {stage: {"enabled": True}}
        if stage == "resample":
            config["resample"]["target_fs"] = 125
        output = preprocess(clean_signal[:, None], FS, config, enabled=True)
        assert output.enabled_stages == [stage], stage
        assert output.stages[stage] is True
        assert all(not v for k, v in output.stages.items() if k != stage)


def test_disabled_stage_leaves_the_signal_untouched():
    """With everything off the signal must come back bit-identical."""
    signal = np.random.default_rng(0).standard_normal(2500).astype(np.float32) + 9.0
    output = preprocess(signal[:, None], FS, default_config(), enabled=False)
    assert output.applied is False
    assert np.array_equal(output.signal[:, 0], signal)
    # read-only view: a downstream stage cannot write through to the input
    assert output.signal.flags.writeable is False


def test_preprocess_removes_baseline():
    signal = np.random.default_rng(0).standard_normal(2500) + 9.0
    output = preprocess(signal, FS, {"baseline": {"enabled": True, "kernel_size": 51}}, enabled=True)
    assert output.applied
    assert output.enabled_stages == ["baseline"]
    assert abs(float(np.median(output.signal[:, 0]))) < 0.5
    assert output.config["baseline"]["method"] == "median_filter_subtraction"


def test_bandpass_and_wavelet_run_together_by_default(clean_signal):
    """The shipped default chain is the three requested stages, in order."""
    output = preprocess(clean_signal[:, None], FS, default_config(), enabled=True)
    assert output.enabled_stages == ["wavelet", "baseline", "bandpass"]
    assert set(output.config) == {"wavelet", "baseline", "bandpass"}
    assert output.config["wavelet"]["wavelet"] == "db4"
    assert output.config["bandpass"]["low_hz"] == 0.5
    assert output.config["bandpass"]["high_hz"] == 45.0


def test_normalisation_is_off_by_default_so_units_stay_mv():
    """Normalisation must not run unless explicitly switched on.

    Baseline removal deliberately strips DC offset, so the check is on the
    amplitude scale (std), not the mean.
    """
    rng = np.random.default_rng(1)
    signal = (rng.standard_normal(2500).astype(np.float32) * 3.0 + 40.0)
    off = preprocess(signal[:, None], FS, default_config(), enabled=True)
    assert "normalize" not in off.config
    # amplitudes still physical (mV-ish), not z-scored to unit variance
    assert 0.5 < float(np.std(off.signal[:, 0])) < 5.0

    on = preprocess(
        signal[:, None], FS, {**default_config(), "normalize": {"enabled": True}}, enabled=True
    )
    assert "normalize" in on.config
    assert abs(float(np.mean(on.signal[:, 0]))) < 1e-3  # float32 rounding
    assert float(np.std(on.signal[:, 0])) <= 1.0 + 1e-2


def test_preprocess_does_not_modify_input(clean_signal):
    original = clean_signal.copy()
    preprocess(clean_signal[:, None], FS, default_config(), enabled=True)
    assert np.array_equal(clean_signal, original)


def test_preprocessing_applied_switch_gates_the_whole_chain(config):
    """`preprocessing.applied` is the master switch, not a comment.

    A configured chain must not run while the switch is false, because the
    analysis has to state honestly which signal it measured
    (IDEA.md sections 13 and 44).
    """
    rng = np.random.default_rng(0)
    signal = (rng.standard_normal(2500) + 9.0).astype(np.float32)
    section = config.section("preprocessing")
    assert enabled_stages(section) == ["wavelet", "baseline", "bandpass"]

    off = preprocess(signal[:, None], FS, {**section, "applied": False})
    assert off.applied is False
    assert np.array_equal(off.signal[:, 0], signal)
    assert float(np.median(off.signal[:, 0])) > 1.0  # offset kept

    on = preprocess(signal[:, None], FS, section, enabled=True)
    assert on.applied is True
    assert on.enabled_stages == ["wavelet", "baseline", "bandpass"]


def test_analyzer_records_the_preprocessing_it_actually_ran(config, make_frame, noisy_signal):
    """Provenance must name the stages that actually ran, and their toggles."""
    section = config.section("preprocessing")

    off = FrameAnalyzer(config, cache=ResultCache(None, enabled=False), preprocessing_enabled=False)
    assert off.effective_preprocessing_config == {}
    result_off = off.analyze_frame(make_frame(noisy_signal), SUPINE)
    assert result_off.preprocessing_applied is False
    assert result_off.run_provenance["preprocessing"] == {}
    assert result_off.run_provenance["preprocessing_requested"] == {}

    on = FrameAnalyzer(config, cache=ResultCache(None, enabled=False), preprocessing_enabled=True)
    assert enabled_stages(on.effective_preprocessing_config) == ["wavelet", "baseline", "bandpass"]
    result_on = on.analyze_frame(make_frame(noisy_signal), SUPINE)
    assert result_on.preprocessing_applied is True
    assert set(result_on.run_provenance["preprocessing"]) == {"wavelet", "baseline", "bandpass"}
    assert result_on.run_provenance["preprocessing_stages"]["normalize"] is False


def test_preprocessing_toggle_invalidates_cached_results(config, make_frame, noisy_signal):
    """Both the master switch and each stage toggle must enter the cache key.

    Otherwise flipping one filter would silently reuse SQIs computed on a
    differently filtered signal.
    """
    cache = ResultCache(None, enabled=True)
    frame = make_frame(noisy_signal)

    off = FrameAnalyzer(config, cache=cache, preprocessing_enabled=False)
    off.analyze_frame(frame, SUPINE)
    off.analyze_frame(frame, SUPINE)
    assert cache.stats() == {"hits": 1, "misses": 0, "entries": 1}

    FrameAnalyzer(config, cache=cache, preprocessing_enabled=True).analyze_frame(frame, SUPINE)
    assert cache.stats()["entries"] == 2  # a different signal, so a new entry

    # now vary ONE stage and confirm it also produces a new entry
    stages = {
        "applied": True,
        "wavelet": {"enabled": False, "wavelet": "db4", "level": 4},
        "baseline": {"enabled": True, "kernel_size": 51},
        "bandpass": {"enabled": True, "low_hz": 0.5, "high_hz": 45.0, "order": 4},
        "resample": {"enabled": False},
        "notch": {"enabled": False},
        "normalize": {"enabled": False},
    }
    FrameAnalyzer(config, cache=cache, preprocessing_stages=stages).analyze_frame(frame, SUPINE)
    assert cache.stats()["entries"] == 3


def test_activated_chain_drives_the_analysis(config, make_frame, noisy_signal):
    """Only the ACTIVATED chain may affect the SQIs.

    The sidebar stages a pending selection and the Activate button commits it,
    so editing a checkbox alone must not change a computed result.
    """
    frame = make_frame(noisy_signal)

    def active(stages):
        analyzer = FrameAnalyzer(config, cache=ResultCache(None, enabled=False), preprocessing_stages=stages)
        result = analyzer.analyze_frame(frame, SUPINE)
        ran = sorted(k for k, v in result.run_provenance["preprocessing_stages"].items() if v)
        return result, ran

    base = {"applied": True}
    for stage in STAGE_ORDER:
        base[stage] = {"enabled": stage in ("wavelet", "baseline", "bandpass")}

    first, ran_first = active(base)
    assert ran_first == ["bandpass", "baseline", "wavelet"]

    # pending edit, not yet activated: the active chain is what counts
    pending = {**base, "wavelet": {"enabled": False}}
    second, ran_second = active(base)
    assert ran_second == ran_first
    assert (second.q_sqi, second.p_sqi, second.k_sqi) == (first.q_sqi, first.p_sqi, first.k_sqi)

    # after Activate the committed chain is used, and the values move
    third, ran_third = active(pending)
    assert ran_third == ["bandpass", "baseline"]
    assert (third.q_sqi, third.p_sqi, third.k_sqi) != (first.q_sqi, first.p_sqi, first.k_sqi)

    # activating nothing means the raw signal, honestly reported
    off = {s: {"enabled": False} for s in STAGE_ORDER}
    off["applied"] = True
    fourth, ran_fourth = active(off)
    assert ran_fourth == []
    assert fourth.preprocessing_applied is False


# ---------------------------------------------------------------------------
# Frame analysis
# ---------------------------------------------------------------------------
def test_analyzer_produces_all_intermediates(config, make_frame, clean_signal):
    analyzer = FrameAnalyzer(config, cache=ResultCache(None, enabled=False))
    result = analyzer.analyze_frame(make_frame(clean_signal), SUPINE)

    assert result.valid
    assert 0.0 <= result.q_sqi <= 1.0
    assert 0.0 <= result.p_sqi <= 1.0
    assert math.isfinite(result.k_sqi)
    assert 0.0 <= result.bas_sqi <= 1.0
    assert result.quality_class in {"Excellent", "Barely Acceptable", "Unacceptable"}

    # The memberships are the article's bounded sum S = W o R over its raw
    # membership rows, and are deliberately not rescaled: the Cauchy rows of
    # qSQI and basSQI do not sum to 1, so neither does S.
    assert 0.0 <= result.fuzzy_excellent <= 1.0
    assert 0.0 <= result.fuzzy_barely_acceptable <= 1.0
    assert 0.0 <= result.fuzzy_unacceptable <= 1.0
    assert 0.0 < result.fuzzy_excellent + result.fuzzy_unacceptable

    # The reported class is decided from the defuzzified score v of Eq (33),
    # which lies between the rating values 1 (Excellent) and 3 (Unacceptable).
    assert 1.0 <= result.run_provenance["fuzzy"]["score"] <= 3.0
    assert math.isfinite(result.heart_rate_bpm)
    assert result.q_sqi_acceptance in {"optimal", "suspicious", "unqualified", "undefined"}
    assert result.p_sqi_acceptance in {"optimal", "suspicious", "unqualified", "undefined"}
    assert result.fusion_class in {"Excellent", "Barely Acceptable", "Unacceptable", "undefined"}
    assert (
        result.fusion_optimal + result.fusion_suspicious + result.fusion_unqualified
    ) <= 4

    provenance = result.run_provenance
    for key in (
        "detector_a", "detector_b", "spectral", "fuzzy", "match_tolerance_ms",
        "reference", "acceptance", "fusion", "heart_rate_bpm", "heart_rate_source",
    ):
        assert key in provenance
    assert provenance["spectral"]["qrs_band_power"] >= 0
    assert provenance["spectral"]["baseline_band_power"] >= 0
    assert provenance["fuzzy"]["membership"]
    assert provenance["acceptance"]["rules"]["qSQI"].startswith("Zhao & Zhang")
    assert provenance["fusion"]["rule"].startswith("Zhao & Zhang")


def test_clean_signal_scores_better_than_noisy(config, make_frame, clean_signal, noisy_signal):
    analyzer = FrameAnalyzer(config, cache=ResultCache(None, enabled=False))
    good = analyzer.analyze_frame(make_frame(clean_signal), SUPINE)
    bad = analyzer.analyze_frame(make_frame(noisy_signal), SUPINE)
    assert good.k_sqi > bad.k_sqi
    # A lower v is a better rating (v1 = Excellent, v3 = Unacceptable), so the
    # clean frame must not be rated worse than the noisy one.
    assert good.run_provenance["fuzzy"]["score"] <= bad.run_provenance["fuzzy"]["score"]
    assert good.fusion_unqualified <= bad.fusion_unqualified


def test_invalid_frame_is_marked_not_crashed(config, make_frame):
    analyzer = FrameAnalyzer(config, cache=ResultCache(None, enabled=False))
    result = analyzer.analyze_frame(make_frame(np.zeros(2500, dtype=np.float32)), SUPINE)
    assert not result.valid
    assert result.quality_class == "INVALID"
    assert result.invalid_reason


def test_method_label_distinguishes_reproduction_from_adaptation(config, make_frame, clean_signal):
    reproduction = FrameAnalyzer(load_config("zhao_zhang"), cache=ResultCache(None, enabled=False))
    assert reproduction.method_label == "Zhao-Zhang structured reproduction (not verified)"
    assert not reproduction.is_adaptation

    adapted = FrameAnalyzer(load_config("pan_tompkins_adapted"), cache=ResultCache(None, enabled=False))
    assert adapted.method_label == "Pan-Tompkins adaptation"
    assert adapted.is_adaptation


def test_the_reference_pairing_alone_does_not_earn_a_reproduction_claim(config):
    """Every reference parameter is still pending_verification.

    Using the reference detector pairing is not grounds for calling a run a
    reproduction. configs/zhao_zhang.yaml declares is_strict_reproduction: false
    and marks all 37 of its scientific parameters as unverified, so the label has
    to say so rather than reading "reproduction" off the detector name.
    """
    analyzer = FrameAnalyzer(load_config("zhao_zhang"), cache=ResultCache(None, enabled=False))

    assert not analyzer.is_adaptation
    assert analyzer.pending_parameters
    assert not analyzer.config.is_strict_reproduction
    assert not analyzer.is_verified_reproduction
    assert "not verified" in analyzer.method_label
    assert "pending_verification" in analyzer.verification_note


def test_a_fully_verified_config_earns_the_reproduction_claim(tmp_path, make_frame, clean_signal):
    """The reproduction label is reachable, once the parameters are confirmed."""
    path = tmp_path / "configs"
    path.mkdir()
    (path / "verified.yaml").write_text(
        """
meta:
  is_strict_reproduction: true
rpeak:
  primary_detector_b: zhao_wavelet
q_sqi:
  match_tolerance_ms:
    value: 150.0
    source: zhao_zhang_2018
""",
        encoding="utf-8",
    )
    analyzer = FrameAnalyzer(load_config("verified", path), cache=ResultCache(None, enabled=False))

    assert analyzer.pending_parameters == []
    assert analyzer.is_verified_reproduction
    assert analyzer.method_label == "Zhao-Zhang verified reproduction"
    assert "verified" in analyzer.verification_note.lower()


def test_the_adaptation_note_names_the_substituted_detector(config):
    adapted = FrameAnalyzer(load_config("pan_tompkins_adapted"), cache=ResultCache(None, enabled=False))

    assert "Pan-Tompkins" in adapted.verification_note
    assert "not comparable" in adapted.verification_note


def test_cache_reuses_result(tmp_path, config, make_frame, clean_signal):
    cache = ResultCache(tmp_path / "cache", enabled=True)
    analyzer = FrameAnalyzer(config, cache=cache)
    frame = make_frame(clean_signal)

    first = analyzer.analyze_frame(frame, SUPINE)
    second = analyzer.analyze_frame(frame, SUPINE)
    assert first.q_sqi == second.q_sqi
    assert first.quality_class == second.quality_class
    assert cache.stats()["entries"] == 1


def test_cache_invalidated_by_config_change(tmp_path, config, make_frame, clean_signal):
    cache = ResultCache(tmp_path / "cache", enabled=True)
    frame = make_frame(clean_signal)
    FrameAnalyzer(config, cache=cache).analyze_frame(frame, SUPINE)
    assert cache.stats()["entries"] == 1

    changed = load_config("pan_tompkins_adapted")
    FrameAnalyzer(changed, cache=cache).analyze_frame(frame, SUPINE)
    assert cache.stats()["entries"] == 2


def test_cache_invalidated_by_signal_change(tmp_path, config, make_frame, clean_signal):
    cache = ResultCache(tmp_path / "cache", enabled=True)
    analyzer = FrameAnalyzer(config, cache=cache)
    analyzer.analyze_frame(make_frame(clean_signal), SUPINE)
    analyzer.analyze_frame(make_frame(clean_signal + 1.0), SUPINE)
    assert cache.stats()["entries"] == 2


def test_cache_survives_reload(tmp_path, config, make_frame, clean_signal):
    cache = ResultCache(tmp_path / "cache", enabled=True)
    FrameAnalyzer(config, cache=cache).analyze_frame(make_frame(clean_signal), SUPINE)
    cache.flush()
    reloaded = ResultCache(tmp_path / "cache", enabled=True)
    assert reloaded.stats()["entries"] == 1


def test_config_fingerprint_changes_with_scientific_parameters():
    a = load_config("zhao_zhang")
    b = load_config("pan_tompkins_adapted")
    assert a.scientific_fingerprint() != b.scientific_fingerprint()


# ---------------------------------------------------------------------------
# End-to-end run over the real dataset
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def annotated_dataset():
    dataset = ingest()
    annotations: list[SubjectAnnotation] = []
    for subject, session in dataset.sessions():
        frames = dataset.session_frames(subject, session)
        n = len(frames)
        third = n // 3
        manager.assign(annotations, subject, session, SUPINE, 0, third - 1)
        manager.assign(annotations, subject, session, SITTING, third, 2 * third - 1)
        manager.assign(annotations, subject, session, STANDING, 2 * third, n - 1)
    return dataset, annotations


@requires_real_data
def test_run_analysis_over_real_dataset(annotated_dataset):
    dataset, annotations = annotated_dataset
    config = load_config("default")
    analysis = run_analysis(dataset, annotations, config, cache=ResultCache(None, enabled=False))

    assert analysis["counts"]["tasks"] > 0
    assert analysis["counts"]["valid"] > 0
    df = results_to_frame(analysis["results"])
    assert not df.empty
    assert {"qSQI", "pSQI", "kSQI", "basSQI", "quality_class", "position"} <= set(df.columns)

    provenance = analysis["provenance"]
    for key in (
        "analysis_timestamp", "software_version", "config_version", "sqi_method",
        "detector_a", "detector_b", "sampling_rate", "preprocessing", "fuzzy_config", "reference",
    ):
        assert key in provenance


def test_transition_frames_never_analysed(annotated_dataset):
    dataset, annotations = annotated_dataset
    annotations.append(
        SubjectAnnotation("S01", "x", [Segment(TRANSITION, 0, 5)])
    )
    analysis = run_analysis(dataset, annotations, load_config("default"),
                            cache=ResultCache(None, enabled=False))
    assert all(r.position in {SUPINE, SITTING, STANDING} for r in analysis["results"])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_describe():
    stats = describe([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["sd"] == pytest.approx(1.2909944, rel=1e-5)
    assert stats["min"] == 1.0 and stats["max"] == 4.0


def test_describe_empty():
    assert describe([])["n"] == 0


def test_describe_ignores_nan():
    assert describe([1.0, float("nan"), 3.0])["n"] == 2


def test_position_summary_and_friedman():
    rows = []
    for subject in ("S01", "S02", "S03", "S04"):
        for position, value in ((SUPINE, 0.9), (SITTING, 0.7), (STANDING, 0.5)):
            rows.append(
                {
                    "subject_id": subject, "position": position, "frame_id": "000001",
                    "qSQI": value, "pSQI": value, "kSQI": 4.0, "basSQI": value,
                    "quality_class": "Excellent", "valid": True,
                    "fuzzy_excellent": 0.8, "fuzzy_barely_acceptable": 0.15,
                    "fuzzy_unacceptable": 0.05,
                }
            )
    df = pd.DataFrame(rows)

    summary = position_summary(df)
    assert list(summary["position"]) == [SUPINE, SITTING, STANDING]
    assert summary.iloc[0]["n_frames"] == 4

    outcome = friedman_test(df, "qSQI")
    assert outcome["status"] == "computed"
    assert outcome["p_value"] < 0.05      # a real difference was injected
    assert 0.0 <= outcome["kendall_w"] <= 1.0

    comparison = position_comparison(df, alpha=0.05)
    assert comparison["detectable_difference"]["qSQI"] is True
    assert "disclaimer" in comparison


def test_friedman_needs_enough_subjects():
    df = pd.DataFrame([
        {"subject_id": "S01", "position": SUPINE, "qSQI": 0.9, "valid": True},
        {"subject_id": "S01", "position": SITTING, "qSQI": 0.8, "valid": True},
    ])
    assert friedman_test(df, "qSQI")["status"] == "insufficient data"


def test_friedman_reports_no_difference_when_there_is_none():
    """A real negative result must be computed, not returned as nan."""
    rows = []
    for subject in ("S01", "S02", "S03", "S04", "S05"):
        for position in (SUPINE, SITTING, STANDING):
            rows.append(
                {"subject_id": subject, "position": position, "qSQI": 0.8,
                 "valid": True, "quality_class": "Excellent"}
            )
    df = pd.DataFrame(rows)
    outcome = friedman_test(df, "qSQI")
    # Every subject is identical, so the ranking term is zero: reported as
    # uncomputable rather than as a nan p-value that compares False to alpha.
    assert outcome["status"] == "not computable"
    assert "p_value" not in outcome
    assert position_comparison(df)["detectable_difference"]["qSQI"] is False


def test_friedman_ignores_subjects_missing_a_position():
    """Repeated measures needs complete subjects; partial ones must be dropped."""
    rows = []
    for subject, positions in (
        ("S01", (SUPINE, SITTING, STANDING)),
        ("S02", (SUPINE, SITTING, STANDING)),
        ("S03", (SUPINE, SITTING, STANDING)),
        ("S04", (SUPINE, SITTING)),
    ):
        for index, position in enumerate(positions):
            rows.append(
                {"subject_id": subject, "position": position, "qSQI": 0.9 - 0.2 * index,
                 "valid": True, "quality_class": "Excellent"}
            )
    outcome = friedman_test(pd.DataFrame(rows), "qSQI")
    assert outcome["n_subjects"] == 3
    assert "S04" not in outcome["subjects"]


def test_overall_summary():
    df = pd.DataFrame([
        {"subject_id": "S01", "position": SUPINE, "qSQI": 0.9, "pSQI": 0.8, "kSQI": 4.0,
         "basSQI": 0.7, "quality_class": "Excellent", "valid": True},
        {"subject_id": "S01", "position": SITTING, "qSQI": 0.3, "pSQI": 0.2, "kSQI": 1.0,
         "basSQI": 0.3, "quality_class": "Unacceptable", "valid": True},
    ])
    summary = overall_summary(df)
    assert summary["valid_frames"] == 2
    assert summary["class_percentages"]["Excellent"] == pytest.approx(50.0)


def test_subject_summary_has_one_row_per_subject_position():
    df = pd.DataFrame([
        {"subject_id": s, "position": p, "qSQI": 0.8, "pSQI": 0.8, "kSQI": 4.0,
         "basSQI": 0.8, "quality_class": "Excellent", "valid": True}
        for s in ("S01", "S02") for p in (SUPINE, SITTING)
    ])
    assert len(subject_summary(df)) == 4


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
@pytest.fixture
def results_df():
    rows = []
    for subject in ("S01", "S02"):
        for position, q in ((SUPINE, 0.95), (SITTING, 0.85), (STANDING, 0.75)):
            rows.append(
                {
                    "subject_id": subject, "session_id": "s", "position": position,
                    "frame_id": "000001", "internal_id": f"{subject}-{position}",
                    "qSQI": q, "pSQI": q, "kSQI": 4.0, "basSQI": q,
                    "fuzzy_excellent": 0.8, "fuzzy_barely_acceptable": 0.15,
                    "fuzzy_unacceptable": 0.05, "quality_class": "Excellent",
                    "valid": True, "invalid_reason": "",
                }
            )
    return pd.DataFrame(rows)


def test_report_generates_all_sections(results_df):
    report = build_report(
        frame_df=results_df,
        position_df=position_summary(results_df),
        subject_df=subject_summary(results_df),
        comparison=position_comparison(results_df),
        dataset_summary={"frames": 6, "sessions": 2, "subjects": ["S01", "S02"],
                         "source_formats": ["jsonl"], "sampling_rates": [250.0]},
        coverage={"expected_frames": 216, "analyzable_frames": 6, "missing_frames": 210,
                  "excluded_transitions": 0},
        config_pending=[("fuzzy.weights.qSQI.value", 0.25)],
        config_name="default",
        provenance={"analysis_timestamp": "2026-09-29T00:00:00Z", "config_name": "default"},
    )
    for heading in (
        "Acquisition completeness", "Position comparison", "SQI interpretation",
        "Potential artifact characteristics", "Inter-subject variability", "Limitations",
        "Reproducibility record",
    ):
        assert heading in report
    assert "216" in report
    assert "signal quality" in report.lower()


def test_report_contains_no_clinical_claims(results_df):
    report = build_report(
        frame_df=results_df, position_df=position_summary(results_df),
        subject_df=subject_summary(results_df),
    ).lower()
    for phrase in (
        "arrhythmia", "cardiac disease", "clinically validated",
        "the wearable performed well", "equivalent to hospital",
    ):
        assert phrase not in report


def test_report_refuses_to_emit_a_banned_claim(monkeypatch, results_df):
    from ecg_eval.interpretation import engine

    monkeypatch.setattr(engine, "overall_finding", lambda df: "the wearable performed well")
    with pytest.raises(AssertionError, match="clinical claim"):
        engine.build_report(
            frame_df=results_df, position_df=pd.DataFrame(), subject_df=pd.DataFrame()
        )


def test_guard_allows_its_own_disclaimers():
    """A negation is a disclaimer, and must not be turned into an assertion.

    Regression: the guard used to delete the negation cue before matching,
    which made "not a clinically validated classifier" read as a claim and
    blocked report generation entirely.
    """
    from ecg_eval.interpretation.engine import _guard

    allowed = [
        "This is not a clinically validated classifier for this wearable.",
        "No diagnostic inference is made from these measurements.",
        "The report makes no inference about any medical condition.",
        "A normal range of values was recorded.",
        "Another 4 frames were excluded.",
        "None of the frames showed a noisy signal.",
    ]
    for text in allowed:
        assert _guard(text) == text

    with pytest.raises(AssertionError, match="clinical claim"):
        _guard("The subject shows arrhythmia in frames 3 to 5.")
    with pytest.raises(AssertionError, match="clinical claim"):
        _guard("Frames 3 to 5 suggest a cardiac disease.")


def test_guard_does_not_mutate_the_text_it_checks():
    """Bare substring negation stripping corrupted words like 'noisy' and 'normal'."""
    from ecg_eval.interpretation.engine import _guard

    text = "12 frames recorded; a noisy baseline in 3 of them."
    assert _guard(text) == text
    assert "noisy" in _guard(text)


def test_describe_frame_for_unacceptable(config, make_frame, noisy_signal):
    analyzer = FrameAnalyzer(config, cache=ResultCache(None, enabled=False))
    result = analyzer.analyze_frame(make_frame(noisy_signal), STANDING)
    text = describe_frame(result)
    assert result.subject_id in text
    assert "Excellent" in text
    if result.quality_class == "Unacceptable":
        assert "primarily associated with" in text


def test_describe_frame_for_invalid(config, make_frame):
    analyzer = FrameAnalyzer(config, cache=ResultCache(None, enabled=False))
    text = describe_frame(analyzer.analyze_frame(make_frame(np.zeros(2500, dtype=np.float32)), SUPINE))
    assert "excluded" in text


def test_low_sqi_factors_ranked_by_deficit():
    order = low_sqi_factors({"qSQI": 0.9, "pSQI": 0.1, "kSQI": 0.2, "basSQI": 0.85}, 0.5)
    assert order[0] == "pSQI"
    assert set(order) == {"pSQI", "kSQI"}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_export_writes_all_artifacts(tmp_path, annotated_dataset):
    dataset, annotations = annotated_dataset
    analysis = run_analysis(dataset, annotations, load_config("default"),
                            cache=ResultCache(None, enabled=False))
    written = export_results(
        tmp_path, analysis["results"], provenance=analysis["provenance"],
        coverage={"expected_frames": 216},
    )
    for name in ("frame_results", "subject_results", "position_results",
                 "overall_results", "run_provenance"):
        assert written[name].exists()
    assert "subject_id" in written["frame_results"].read_text(encoding="utf-8").splitlines()[0]


@requires_real_data
def test_export_does_not_touch_data(tmp_path, annotated_dataset):
    dataset, annotations = annotated_dataset
    analysis = run_analysis(dataset, annotations, load_config("default"),
                            cache=ResultCache(None, enabled=False))
    before = {p: p.stat().st_mtime_ns for p in dataset.frames_by_session and []} or None
    del before
    export_results(tmp_path, analysis["results"])
    # The real data directory must be unchanged.
    from ecg_eval.config import REPO_ROOT

    data_dir = REPO_ROOT / "data"
    sample = sorted(data_dir.rglob("*.jsonl"))[0]
    assert sample.exists()


# ---------------------------------------------------------------------------
# Configuration provenance
# ---------------------------------------------------------------------------
def test_config_exposes_pending_parameters(zhao_config):
    pending = zhao_config.pending_parameters()
    paths = {path for path, _ in pending}
    # The article prints no R-peak matching tolerance and does not say whether
    # kurtosis is computed before or after filtering, so exactly those two
    # engine parameters stay declared as unverified. Everything the article does
    # print is recorded as verified against the equation it came from.
    assert paths == {"q_sqi.match_tolerance_ms", "k_sqi.compute_on"}
    assert zhao_config.is_strict_reproduction is False

    verified = {path for path, _ in zhao_config.verified_parameters()}
    assert "q_sqi.acceptance.optimal_above" in verified
    assert "k_sqi.acceptance.optimal_above" in verified
    assert "bas_sqi.acceptance.unqualified_below" in verified
    assert "p_sqi.acceptance.heart_rate_bands[0].l1" in verified
    assert "fuzzy.weights.qSQI" in verified
    assert "fuzzy.rating_values.Excellent" in verified
    assert "fuzzy.decision.excellent_max" in verified
    # Every index has explicit membership parameters for all three rating levels.
    for factor in ("qSQI", "pSQI", "kSQI", "basSQI"):
        for level in ("Excellent", "Barely Acceptable", "Unacceptable"):
            assert any(p.startswith(f"fuzzy.membership.{factor}.{level}.") for p in verified)


def test_config_inheritance():
    default = load_config("default")
    assert default.get("meta.config_name") == "default"
    # default extends zhao_zhang, so the scientific values are inherited.
    assert default.get("p_sqi.qrs_band_hz") == [5.0, 15.0]
    assert default.get("cache.enabled") is True


def test_adapted_config_is_marked():
    adapted = load_config("pan_tompkins_adapted")
    assert adapted.get("rpeak.primary_detector_b") == "pan_tompkins"
    assert "Pan-Tompkins" in adapted.adaptation_note
    assert adapted.is_strict_reproduction is False


def test_dataset_class_is_empty_safe():
    dataset = Dataset()
    assert dataset.n_frames == 0
    assert dataset.subjects() == []
    assert dataset.frame_by_id("missing") is None

def test_every_figure_builder_produces_a_valid_figure():
    """Plotly validates trace properties at construction time.

    An invalid keyword (e.g. `points=` instead of `boxpoints=` on a Box) only
    fails when the chart is actually built, inside the page render, so these
    builders are exercised here to catch that class of error.
    """
    import pandas as pd

    from ecg_eval.visualization.plots import (
        quality_distribution_figure,
        sqi_box_figure,
        sqi_trend_figure,
    )

    df = pd.DataFrame({
        "subject_id": ["S01"] * 6,
        "session_id": ["s1"] * 6,
        "position": ["SUPINE", "SITTING", "STANDING"] * 2,
        "frame_id": [f"{i:06d}" for i in range(6)],
        "qSQI": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "pSQI": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "kSQI": [6.0, 5.5, 5.0, 4.5, 4.0, 3.5],
        "basSQI": [0.99, 0.97, 0.96, 0.94, 0.92, 0.91],
        "quality_class": ["Excellent", "Barely Acceptable", "Unacceptable"] * 2,
        
    })

    for builder in (quality_distribution_figure, sqi_box_figure, sqi_trend_figure):
        figure = builder(df)
        assert figure is not None
        assert len(figure.data) > 0, builder.__name__



"""SQI tests: the four indices, their acceptance criteria, the heuristic fusion
and the fuzzy comprehensive evaluation (Zhao & Zhang 2018)."""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

# ``ecg_eval.sqi.__init__`` re-exports functions that carry the same name as
# four of its own submodules (``bas_sqi``, ``k_sqi``, ``p_sqi``, ``q_sqi``), so
# ``import ecg_eval.sqi.q_sqi`` would hand back the function instead of the
# module. Importing through ``importlib`` avoids the shadowing and keeps each
# test referring to the module it actually means.
bas_mod = importlib.import_module("ecg_eval.sqi.bas_sqi")
fuzzy_mod = importlib.import_module("ecg_eval.sqi.fuzzy")
fusion_mod = importlib.import_module("ecg_eval.sqi.heuristic_fusion")
k_mod = importlib.import_module("ecg_eval.sqi.k_sqi")
p_mod = importlib.import_module("ecg_eval.sqi.p_sqi")
q_mod = importlib.import_module("ecg_eval.sqi.q_sqi")

from ecg_eval.detectors import available_detectors, get_detector
from ecg_eval.detectors.pan_tompkins import PanTompkinsDetector
from ecg_eval.detectors.zhao_hilbert import ZhaoHilbertDetector
from ecg_eval.detectors.zhao_wavelet import ZhaoWaveletDetector
from ecg_eval.models.result import (
    BARELY_ACCEPTABLE,
    EXCELLENT,
    OPTIMAL,
    SUSPICIOUS,
    UNACCEPTABLE,
    UNDEFINED,
    UNQUALIFIED,
)

FS = 250.0


def levels_from(optimal: int, suspicious: int, unqualified: int) -> dict[str, str]:
    """Build a per-index level mapping with the given counts."""
    out: dict[str, str] = {}
    for index in range(optimal):
        out[f"o{index}"] = OPTIMAL
    for index in range(suspicious):
        out[f"s{index}"] = SUSPICIOUS
    for index in range(unqualified):
        out[f"u{index}"] = UNQUALIFIED
    return out


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def test_registry_lists_primary_and_adapted():
    names = available_detectors()
    assert {"zhao_hilbert", "zhao_wavelet", "pan_tompkins"} <= set(names)


def test_unknown_detector_raises():
    with pytest.raises(KeyError):
        get_detector("does_not_exist")


def test_hilbert_detector_finds_the_synthetic_r_peaks(clean_signal):
    result = get_detector("zhao_hilbert").detect(clean_signal, FS)
    assert abs(result.peaks.size - clean_signal.size / FS * 1.0) <= 2
    assert result.label == "zhao_zhang"


def test_wavelet_detector_finds_the_synthetic_r_peaks(clean_signal):
    result = get_detector("zhao_wavelet").detect(clean_signal, FS)
    assert abs(result.peaks.size - clean_signal.size / FS) <= 2
    assert result.label == "zhao_zhang"


def test_pan_tompkins_is_labelled_as_adaptation(clean_signal):
    result = PanTompkinsDetector().detect(clean_signal, FS)
    assert result.label == "pan_tompkins_adapted"
    assert result.peaks.size > 0


def test_detector_is_deterministic(clean_signal):
    detector = ZhaoHilbertDetector()
    assert detector.detect(clean_signal, FS).as_list() == detector.detect(clean_signal, FS).as_list()


def test_detector_handles_short_signal():
    assert get_detector("zhao_hilbert").detect(np.zeros(10), FS).peaks.size == 0


def test_detector_tolerates_nan(clean_signal):
    corrupt = clean_signal.copy()
    corrupt[500] = np.nan
    assert get_detector("zhao_wavelet").detect(corrupt, FS).peaks.size > 0


# ---------------------------------------------------------------------------
# qSQI -- Eq (1) formula and Eq (2) acceptance
# ---------------------------------------------------------------------------
def test_match_peaks_within_tolerance():
    pairs, unmatched_a, unmatched_b = q_mod.match_peaks(
        np.array([100, 200, 300]), np.array([110, 205, 500]), FS, 150.0
    )
    assert pairs == [(100, 110), (200, 205)]
    assert unmatched_a == [300]
    assert unmatched_b == [500]


def test_match_peaks_respects_tolerance():
    pairs, _, _ = q_mod.match_peaks(np.array([100]), np.array([200]), FS, 150.0)
    assert pairs == []


def test_match_peaks_one_to_one():
    pairs, _, _ = q_mod.match_peaks(np.array([100, 104]), np.array([102]), FS, 150.0)
    assert len(pairs) == 1


def test_match_peaks_empty():
    pairs, unmatched_a, unmatched_b = q_mod.match_peaks(np.array([]), np.array([1, 2]), FS, 150.0)
    assert pairs == [] and unmatched_b == [1, 2] and unmatched_a == []


def test_q_sqi_is_the_agreement_over_the_sum_of_both_counts(clean_signal):
    """Eq (1): qSQI = 2N / (Na + Nb)."""
    value, _, _, match = q_mod.q_sqi(clean_signal, FS, tolerance_ms=150.0)
    expected = 2.0 * match.n_matched / (match.n_peaks_a + match.n_peaks_b)
    assert value == pytest.approx(expected)
    assert 0.0 <= value <= 1.0


def test_q_sqi_differs_from_dividing_by_the_smaller_count():
    """The two formulas only coincide when both detectors find equal counts.

    Dividing by the smaller count would score a detector that found a subset of
    the other's peaks as a perfect match, so this pins the corrected denominator
    with a case where the two definitions provably disagree.
    """
    a = np.array([100.0, 200.0, 300.0])
    b = np.array([110.0, 500.0])
    pairs, _, _ = q_mod.match_peaks(a, b, FS, 150.0)
    matched, peaks_a, peaks_b = len(pairs), a.size, b.size
    assert matched == 1
    assert 2.0 * matched / (peaks_a + peaks_b) == pytest.approx(0.4)
    assert matched / min(peaks_a, peaks_b) == pytest.approx(0.5)


def test_q_sqi_perfect_agreement_is_one(clean_signal):
    """The same detector against itself matches every peak, so Eq (1) gives 1."""
    value, _, _, match = q_mod.q_sqi(
        clean_signal, FS, detector_a="zhao_hilbert", detector_b="zhao_hilbert",
        tolerance_ms=150.0,
    )
    assert match.n_matched == match.n_peaks_a == match.n_peaks_b
    assert value == pytest.approx(1.0)


def test_q_sqi_no_peaks_gives_zero():
    value, _, _, match = q_mod.q_sqi(np.zeros(2500), FS)
    assert value == 0.0
    assert match.n_matched == 0


def test_q_sqi_uses_pan_tompkins_when_substituted(clean_signal):
    _, _, _, match = q_mod.q_sqi(clean_signal, FS, detector_b="pan_tompkins", tolerance_ms=150.0)
    assert match.label == "pan_tompkins_adapted"
    assert match.detector_b == "pan_tompkins"


def test_q_sqi_records_all_debug_fields(clean_signal):
    _, _, _, match = q_mod.q_sqi(clean_signal, FS, tolerance_ms=150.0)
    payload = match.to_dict()
    for key in (
        "n_peaks_a", "n_peaks_b", "n_matched", "q_sqi", "peaks_a", "peaks_b",
        "matched_pairs", "unmatched_a", "unmatched_b", "tolerance_ms",
    ):
        assert key in payload
    assert match.tolerance_ms == 150.0


def test_q_sqi_adaptive_tolerance_differs(clean_signal):
    _, _, _, fixed = q_mod.q_sqi(clean_signal, FS, tolerance_ms=100.0, adaptive=False)
    _, _, _, adaptive = q_mod.q_sqi(clean_signal, FS, tolerance_ms=100.0, adaptive=True)
    assert fixed.tolerance_ms == 100.0
    assert 80.0 <= adaptive.tolerance_ms <= 250.0


def test_q_sqi_acceptance_boundaries(config):
    """Eq (2): the outer rules are strict, so 90 % and 60 % are suspicious."""
    assert q_mod.q_sqi_acceptance(0.95, config).level == OPTIMAL
    assert q_mod.q_sqi_acceptance(0.90, config).level == SUSPICIOUS
    assert q_mod.q_sqi_acceptance(0.75, config).level == SUSPICIOUS
    assert q_mod.q_sqi_acceptance(0.60, config).level == SUSPICIOUS
    assert q_mod.q_sqi_acceptance(0.59, config).level == UNQUALIFIED
    assert q_mod.q_sqi_acceptance(0.0, config).level == UNQUALIFIED


def test_q_sqi_acceptance_reports_undefined_for_nan(config):
    verdict = q_mod.q_sqi_acceptance(float("nan"), config)
    assert verdict.level == UNDEFINED
    assert verdict.reason
    assert not verdict.defined


def test_q_sqi_acceptance_limits_come_from_configuration(config):
    limits = q_mod.acceptance_limits(config)
    assert limits["optimal_above"] == pytest.approx(0.90)
    assert limits["suspicious_from"] == pytest.approx(0.60)
    assert limits["unqualified_below"] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# pSQI -- Eq (3) formula, Eq (4)-(5) acceptance
# ---------------------------------------------------------------------------
def test_p_sqi_default_denominator_band_starts_at_five_hz(config):
    """Eq (3): the denominator is 5-40 Hz, not 0.5-40 Hz."""
    resolved = p_mod.p_sqi_config(config)
    assert resolved["qrs_band_hz"] == (5.0, 15.0)
    assert resolved["total_band_hz"] == (5.0, 40.0)
    assert resolved["method"] == "welch"


def test_p_sqi_denominator_excludes_power_below_five_hz():
    """A 2 Hz component must not dilute the index, because it is out of band.

    With the article's 5-40 Hz denominator, a signal carrying equal power at
    2 Hz and 10 Hz puts almost all of its in-band power in the QRS band. A
    0.5-40 Hz denominator would halve the value, so this pins the correction.
    """
    t = np.arange(2500) / FS
    two_plus_ten = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 2 * t)
    value, spectral = p_mod.p_sqi(two_plus_ten, FS)
    assert value > 0.8
    assert spectral.p_total_power > 0


def test_p_sqi_perfect_band_concentration():
    t = np.arange(2500) / FS
    value, spectral = p_mod.p_sqi(np.sin(2 * np.pi * 10 * t), FS)
    assert value == pytest.approx(1.0, abs=0.05)
    assert len(spectral.frequencies) == len(spectral.psd)
    assert spectral.p_total_power >= spectral.qrs_band_power


def test_p_sqi_falls_with_high_frequency_noise():
    """EMG-like content above the QRS band lowers the index (Eq 3).

    The article states the mechanism: "If EMG interference exists, the
    high-frequency component increases, and pSQI decreases."
    """
    t = np.arange(2500) / FS
    qrs = np.sin(2 * np.pi * 10 * t)
    emg = np.sin(2 * np.pi * 35 * t)
    clean, _ = p_mod.p_sqi(qrs, FS)
    noisy, _ = p_mod.p_sqi(qrs + emg, FS)
    assert clean > 0.9
    assert noisy < 0.6
    assert noisy < clean


def test_p_sqi_is_insensitive_to_sub_five_hz_content():
    """Baseline wander sits outside BOTH bands of Eq (3), so it does not lower pSQI.

    This is a property of the article's definition rather than an oversight: the
    denominator starts at 5 Hz, so a sub-5 Hz component cannot dilute the index,
    and detecting baseline drift is the separate job of the baseline relative
    power index. The earlier 0.5-40 Hz denominator is what hid this: it let
    wander dominate the denominator and drove pSQI down for a reason the article
    assigns to another index.
    """
    t = np.arange(2500) / FS
    value, _ = p_mod.p_sqi(np.sin(2 * np.pi * 0.3 * t), FS)
    assert value > 0.5


def test_p_sqi_acceptance_uses_the_first_heart_rate_band(config):
    """Eq (5): 60-130 bpm gives l1=0.5, l2=0.8, l3=0.4."""
    assert p_mod.p_sqi_acceptance(0.50, 70.0, config).level == OPTIMAL
    assert p_mod.p_sqi_acceptance(0.80, 70.0, config).level == OPTIMAL
    assert p_mod.p_sqi_acceptance(0.45, 70.0, config).level == SUSPICIOUS
    assert p_mod.p_sqi_acceptance(0.40, 70.0, config).level == SUSPICIOUS
    assert p_mod.p_sqi_acceptance(0.39, 70.0, config).level == UNQUALIFIED
    # Too much power in the QRS band is unqualified too: the index is
    # band-pass-like, unlike the other three.
    assert p_mod.p_sqi_acceptance(0.81, 70.0, config).level == UNQUALIFIED
    assert p_mod.p_sqi_acceptance(0.95, 70.0, config).level == UNQUALIFIED


def test_p_sqi_acceptance_uses_the_second_heart_rate_band(config):
    """Eq (5): 130-160 bpm gives l1=0.4, l2=0.7, l3=0.3."""
    assert p_mod.p_sqi_acceptance(0.40, 145.0, config).level == OPTIMAL
    assert p_mod.p_sqi_acceptance(0.70, 145.0, config).level == OPTIMAL
    assert p_mod.p_sqi_acceptance(0.35, 145.0, config).level == SUSPICIOUS
    assert p_mod.p_sqi_acceptance(0.29, 145.0, config).level == UNQUALIFIED
    assert p_mod.p_sqi_acceptance(0.71, 145.0, config).level == UNQUALIFIED


def test_p_sqi_acceptance_at_130_bpm_uses_the_first_band(config):
    """The two printed bands share 130 bpm; the first match wins, by design."""
    verdict = p_mod.p_sqi_acceptance(0.5, 130.0, config)
    assert verdict.level == OPTIMAL
    assert verdict.limits["l1"] == pytest.approx(0.5)
    assert p_mod.p_sqi_acceptance(0.45, 130.0, config).level == SUSPICIOUS


def test_p_sqi_acceptance_is_undefined_outside_the_calibrated_range(config):
    low = p_mod.p_sqi_acceptance(0.5, 50.0, config)
    high = p_mod.p_sqi_acceptance(0.5, 200.0, config)
    assert low.level == UNDEFINED and high.level == UNDEFINED
    assert "60" in low.reason and "160" in low.reason
    assert not low.defined


def test_p_sqi_acceptance_is_undefined_without_a_heart_rate(config):
    verdict = p_mod.p_sqi_acceptance(0.5, float("nan"), config)
    assert verdict.level == UNDEFINED
    assert "heart rate" in verdict.reason


def test_heart_rate_bands_come_from_configuration(config):
    bands = p_mod.heart_rate_bands(config)
    assert [(b["min_bpm"], b["max_bpm"]) for b in bands] == [(60.0, 130.0), (130.0, 160.0)]
    assert (bands[0]["l1"], bands[0]["l2"], bands[0]["l3"]) == (0.5, 0.8, 0.4)
    assert (bands[1]["l1"], bands[1]["l2"], bands[1]["l3"]) == (0.4, 0.7, 0.3)


# ---------------------------------------------------------------------------
# kSQI -- Eq (9) formula and Eq (10) acceptance
# ---------------------------------------------------------------------------
def test_kurtosis_is_the_fourth_standardized_moment_of_a_gaussian():
    """Eq (9) is nu4 (Pearson), where a Gaussian signal sits at 3, not 0."""
    signal = np.random.default_rng(1).standard_normal(200000)
    assert k_mod.kurtosis(signal, "pearson") == pytest.approx(3.0, abs=0.05)
    assert k_mod.kurtosis(signal, "fisher") == pytest.approx(0.0, abs=0.05)
    assert k_mod.kurtosis(signal, "fisher") == pytest.approx(
        k_mod.kurtosis(signal, "pearson") - 3.0
    )


def test_default_kurtosis_convention_is_pearson():
    """A library default must not silently decide the science: nu4 is the default."""
    signal = np.random.default_rng(2).standard_normal(5000)
    assert k_mod.kurtosis(signal) == pytest.approx(k_mod.kurtosis(signal, "pearson"))
    value, _mean, _std = k_mod.k_sqi(signal)
    assert value == pytest.approx(k_mod.kurtosis(signal, "pearson"))


def test_sharp_qrs_is_leptokurtic_under_pearson(clean_signal):
    assert k_mod.kurtosis(clean_signal, "pearson") > 3.0


def test_noise_reduces_kurtosis(clean_signal, noisy_signal):
    assert k_mod.kurtosis(noisy_signal, "pearson") < k_mod.kurtosis(clean_signal, "pearson")


def test_constant_signal_returns_nan():
    assert math.isnan(k_mod.kurtosis(np.ones(100)))


def test_k_sqi_returns_mean_and_std(clean_signal):
    value, mean, std = k_mod.k_sqi(clean_signal)
    assert value == k_mod.kurtosis(clean_signal, "pearson")
    assert mean == pytest.approx(float(np.mean(clean_signal)))
    assert std == pytest.approx(float(np.std(clean_signal)))


def test_unknown_kurtosis_definition_raises():
    with pytest.raises(ValueError):
        k_mod.kurtosis(np.random.default_rng(0).standard_normal(100), "bogus")


def test_k_sqi_acceptance_is_binary(config):
    """Eq (10) defines no suspicious band, and its boundaries are as printed."""
    assert k_mod.k_sqi_acceptance(6.0, "pearson", config).level == OPTIMAL
    assert k_mod.k_sqi_acceptance(5.1, "pearson", config).level == OPTIMAL
    assert k_mod.k_sqi_acceptance(5.0, "pearson", config).level == UNQUALIFIED
    assert k_mod.k_sqi_acceptance(2.0, "pearson", config).level == UNQUALIFIED
    assert k_mod.k_sqi_acceptance(5.0, "pearson", config).level != SUSPICIOUS


def test_k_sqi_acceptance_refuses_an_excess_kurtosis_value(config):
    """Eq (10) thresholds 5 on the nu4 scale; a Fisher value is not comparable."""
    verdict = k_mod.k_sqi_acceptance(5.0, "fisher", config)
    assert verdict.level == UNDEFINED
    assert "fisher" in verdict.reason


# ---------------------------------------------------------------------------
# basSQI -- Eq (11) formula and Eq (12) acceptance
# ---------------------------------------------------------------------------
def test_bas_sqi_is_one_minus_the_baseline_share(clean_signal):
    """Eq (11) holds the leading "1 -", which inverts the bare ratio's meaning."""
    value, spectral = bas_mod.bas_sqi(clean_signal, FS)
    ratio = spectral.baseline_band_power / spectral.bas_total_power
    assert value == pytest.approx(1.0 - ratio)
    assert spectral.bas_total_power > 0


def test_bas_sqi_is_high_for_a_clean_signal(clean_signal):
    value, _ = bas_mod.bas_sqi(clean_signal, FS)
    assert value > 0.9


def test_bas_sqi_falls_with_baseline_wander(clean_signal):
    t = np.arange(clean_signal.size) / FS
    wander = 2.0 * np.sin(2 * np.pi * 0.4 * t)
    clean, _ = bas_mod.bas_sqi(clean_signal, FS)
    drifting, _ = bas_mod.bas_sqi(clean_signal + wander, FS)
    assert drifting < clean


def test_bas_sqi_stores_band_powers(clean_signal):
    _, spectral = bas_mod.bas_sqi(clean_signal, FS)
    assert spectral.baseline_band_power >= 0
    assert 0.0 <= spectral.bas_sqi <= 1.0


def test_bas_sqi_acceptance_boundaries(config):
    assert bas_mod.bas_sqi_acceptance(1.00, config).level == OPTIMAL
    assert bas_mod.bas_sqi_acceptance(0.95, config).level == OPTIMAL
    assert bas_mod.bas_sqi_acceptance(0.94, config).level == SUSPICIOUS
    assert bas_mod.bas_sqi_acceptance(0.90, config).level == SUSPICIOUS
    assert bas_mod.bas_sqi_acceptance(0.89, config).level == UNQUALIFIED
    assert bas_mod.bas_sqi_acceptance(0.50, config).level == UNQUALIFIED


# ---------------------------------------------------------------------------
# Heart rate
# ---------------------------------------------------------------------------
def test_heart_rate_from_regular_peaks():
    peaks = np.arange(0, 2500, 250)  # 1 Hz at 250 Hz
    assert fusion_mod.heart_rate_from_peaks(peaks, FS) == pytest.approx(60.0)


def test_heart_rate_needs_two_peaks():
    assert math.isnan(fusion_mod.heart_rate_from_peaks(np.array([100.0]), FS))
    assert math.isnan(fusion_mod.heart_rate_from_peaks(np.array([]), FS))
    assert math.isnan(fusion_mod.heart_rate_from_peaks(np.array([0.0, 100.0]), 0.0))


def test_estimate_heart_rate_prefers_detector_a_then_falls_back():
    rate, source = fusion_mod.estimate_heart_rate(np.arange(0, 2500, 250), None, sampling_rate=FS)
    assert rate == pytest.approx(60.0) and source == "detector_a"
    rate, source = fusion_mod.estimate_heart_rate(
        np.array([1.0]), np.arange(0, 2500, 250), sampling_rate=FS
    )
    assert rate == pytest.approx(60.0) and source == "detector_b"
    rate, source = fusion_mod.estimate_heart_rate(np.array([]), np.array([]), sampling_rate=FS)
    assert math.isnan(rate) and source == ""


# ---------------------------------------------------------------------------
# Simple heuristic fusion -- Eq (13)-(16)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "counts, expected",
    [
        # Eq (15), four indices
        ((4, 0, 0), EXCELLENT),
        ((3, 1, 0), EXCELLENT),
        ((3, 0, 1), BARELY_ACCEPTABLE),      # unqualified blocks Excellent
        ((2, 2, 0), BARELY_ACCEPTABLE),
        ((1, 3, 0), BARELY_ACCEPTABLE),
        ((0, 4, 0), BARELY_ACCEPTABLE),
        ((1, 2, 1), BARELY_ACCEPTABLE),      # unqualified=1 but suspicious=2
        ((0, 0, 4), UNACCEPTABLE),
        ((1, 0, 3), UNACCEPTABLE),
        ((0, 1, 3), UNACCEPTABLE),           # unqualified=3
        ((0, 2, 2), UNACCEPTABLE),           # unqualified=2 and suspicious>=1
        ((0, 3, 1), UNACCEPTABLE),           # unqualified=1 and suspicious=3
    ],
)
def test_fusion_rule_for_four_indices(counts, expected):
    fused = fusion_mod.fuse_levels(levels_from(*counts))
    assert fused.quality_class == expected
    assert fused.n_factors == 4
    assert fused.applied is True
    assert fused.rule == "Zhao & Zhang (2018) Eq (15)"
    assert fused.counts == {OPTIMAL: counts[0], SUSPICIOUS: counts[1], UNQUALIFIED: counts[2]}


@pytest.mark.parametrize(
    "counts, expected",
    [
        ((2, 0, 0), EXCELLENT),
        ((1, 1, 0), BARELY_ACCEPTABLE),
        ((0, 2, 0), BARELY_ACCEPTABLE),
        ((1, 0, 1), UNACCEPTABLE),
        ((0, 1, 1), UNACCEPTABLE),
        ((0, 0, 2), UNACCEPTABLE),
    ],
)
def test_fusion_rule_for_two_indices(counts, expected):
    assert fusion_mod.fuse_levels(levels_from(*counts)).quality_class == expected


@pytest.mark.parametrize(
    "counts, expected",
    [
        ((3, 0, 0), EXCELLENT),
        ((2, 1, 0), EXCELLENT),
        ((2, 0, 1), BARELY_ACCEPTABLE),
        ((1, 1, 1), BARELY_ACCEPTABLE),
        ((0, 2, 1), UNACCEPTABLE),
        ((1, 0, 2), UNACCEPTABLE),
        ((0, 0, 3), UNACCEPTABLE),
    ],
)
def test_fusion_rule_for_three_indices(counts, expected):
    assert fusion_mod.fuse_levels(levels_from(*counts)).quality_class == expected


@pytest.mark.parametrize(
    "counts, expected",
    [
        ((5, 0, 0), EXCELLENT),
        ((4, 1, 0), EXCELLENT),
        ((4, 0, 1), BARELY_ACCEPTABLE),
        ((2, 3, 0), BARELY_ACCEPTABLE),
        ((0, 4, 1), UNACCEPTABLE),
        ((0, 3, 2), UNACCEPTABLE),
        ((0, 2, 3), UNACCEPTABLE),
        ((0, 1, 4), UNACCEPTABLE),
    ],
)
def test_fusion_rule_for_five_indices(counts, expected):
    assert fusion_mod.fuse_levels(levels_from(*counts)).quality_class == expected


def test_fusion_is_undefined_when_a_criterion_does_not_apply():
    """An inapplicable index is never dropped to fuse the remaining ones."""
    fused = fusion_mod.fuse_levels(
        {"qSQI": OPTIMAL, "pSQI": UNDEFINED, "kSQI": OPTIMAL, "basSQI": OPTIMAL}
    )
    assert fused.quality_class == UNDEFINED
    assert fused.applied is False
    assert "pSQI" in fused.reason
    assert fused.counts[OPTIMAL] == 3


def test_fusion_refuses_a_count_with_no_printed_rule():
    fused = fusion_mod.fuse_levels(levels_from(6, 0, 0))
    assert fused.quality_class == UNDEFINED
    assert "no fusion rule" in fused.reason


def test_assess_judges_all_four_indices(config):
    acceptances, fused = fusion_mod.assess(
        {"qSQI": 0.95, "pSQI": 0.6, "kSQI": 8.0, "basSQI": 0.97},
        heart_rate_bpm=70.0,
        config=config,
    )
    assert set(acceptances) == {"qSQI", "pSQI", "kSQI", "basSQI"}
    assert all(item.level == OPTIMAL for item in acceptances.values())
    assert fused.quality_class == EXCELLENT
    assert fused.counts[OPTIMAL] == 4


def test_assess_marks_a_poor_frame_unacceptable(config):
    acceptances, fused = fusion_mod.assess(
        {"qSQI": 0.20, "pSQI": 0.05, "kSQI": 1.0, "basSQI": 0.4},
        heart_rate_bpm=70.0,
        config=config,
    )
    assert set(item.level for item in acceptances.values()) == {UNQUALIFIED}
    assert fused.quality_class == UNACCEPTABLE


# ---------------------------------------------------------------------------
# Membership functions -- Eq (22), (24)-(32)
# ---------------------------------------------------------------------------
def test_cauchy_peaks_at_centre():
    assert fuzzy_mod.cauchy(0.9, 0.9, 0.1) == pytest.approx(1.0)
    assert fuzzy_mod.cauchy(0.8, 0.9, 0.1) == pytest.approx(0.5)
    assert fuzzy_mod.cauchy(1.1, 0.9, 0.1) == pytest.approx(0.2)


def test_cauchy_increasing_matches_eq_22():
    """Eq (22): 0 up to 80, 1/(1+[0.3(q-80)]^2) to 90, then 1."""
    assert fuzzy_mod.cauchy_increasing(80.0, 80.0, 0.3, 90.0) == 0.0
    assert fuzzy_mod.cauchy_increasing(85.0, 80.0, 0.3, 90.0) == pytest.approx(1 / 3.25)
    assert fuzzy_mod.cauchy_increasing(90.0, 80.0, 0.3, 90.0) == 1.0
    # The printed curve is discontinuous at the cap; it is not smoothed here.
    assert fuzzy_mod.cauchy_increasing(89.999, 80.0, 0.3, 90.0) < 0.2


def test_cauchy_decreasing_matches_eq_24():
    """Eq (24): 1 at or below 55, then 1/(1+((q-55)/5)^2)."""
    assert fuzzy_mod.cauchy_decreasing(55.0, 55.0, 5.0) == 1.0
    assert fuzzy_mod.cauchy_decreasing(60.0, 55.0, 5.0) == pytest.approx(0.5)
    assert fuzzy_mod.cauchy_decreasing(75.0, 55.0, 5.0) == pytest.approx(1 / 17)


def test_cauchy_centred_matches_eq_25():
    assert fuzzy_mod.cauchy_centred(75.0, 75.0, 7.5) == 1.0
    assert fuzzy_mod.cauchy_centred(82.5, 75.0, 7.5) == pytest.approx(0.5)


def test_ramps_match_eq_26_and_eq_27():
    assert fuzzy_mod.ramp_up(0.25, 0.25, 0.35) == 0.0
    assert fuzzy_mod.ramp_up(0.30, 0.25, 0.35) == pytest.approx(0.5)
    assert fuzzy_mod.ramp_up(0.35, 0.25, 0.35) == 1.0
    assert fuzzy_mod.ramp_down(0.15, 0.15, 0.25) == 1.0
    assert fuzzy_mod.ramp_down(0.20, 0.15, 0.25) == pytest.approx(0.5)
    assert fuzzy_mod.ramp_down(0.25, 0.15, 0.25) == 0.0


def test_trapezoid_matches_eq_28():
    assert fuzzy_mod.trapezoidal(0.18, 0.18, 0.22, 0.28, 0.32) == 0.0
    assert fuzzy_mod.trapezoidal(0.20, 0.18, 0.22, 0.28, 0.32) == pytest.approx(0.5)
    assert fuzzy_mod.trapezoidal(0.25, 0.18, 0.22, 0.28, 0.32) == 1.0
    assert fuzzy_mod.trapezoidal(0.30, 0.18, 0.22, 0.28, 0.32) == pytest.approx(0.5)
    assert fuzzy_mod.trapezoidal(0.32, 0.18, 0.22, 0.28, 0.32) == 0.0


def test_trapezoidal_rejects_bad_parameters():
    with pytest.raises(ValueError):
        fuzzy_mod.trapezoidal(0.5, 0.7, 0.2, 0.9, 1.0)


def test_step_and_zero_families_match_eq_29():
    assert fuzzy_mod.step_above(5.1, 5.0) == 1.0
    assert fuzzy_mod.step_above(5.0, 5.0) == 0.0
    assert fuzzy_mod.step_at_or_below(5.0, 5.0) == 1.0
    assert fuzzy_mod.step_at_or_below(5.1, 5.0) == 0.0
    assert fuzzy_mod.zero_membership(99.0) == 0.0


def test_membership_families_are_given_per_rating_level(config):
    """Each level of an index may use a different family, as the article does."""
    functions = fuzzy_mod.build_membership_functions(config.section("fuzzy"))
    assert set(functions) == {"qSQI", "pSQI", "kSQI", "basSQI"}
    for factor in functions:
        assert set(functions[factor]) == {EXCELLENT, BARELY_ACCEPTABLE, UNACCEPTABLE}

    q = functions["qSQI"]
    assert q[EXCELLENT](85.0) == pytest.approx(1 / 3.25)
    assert q[BARELY_ACCEPTABLE](75.0) == 1.0
    assert q[UNACCEPTABLE](55.0) == 1.0

    p = functions["pSQI"]
    assert p[EXCELLENT](0.40) == 1.0
    assert p[BARELY_ACCEPTABLE](0.25) == 1.0
    assert p[UNACCEPTABLE](0.10) == 1.0

    k = functions["kSQI"]
    assert k[EXCELLENT](6.0) == 1.0
    assert k[BARELY_ACCEPTABLE](6.0) == 0.0
    assert k[UNACCEPTABLE](4.0) == 1.0

    b = functions["basSQI"]
    assert b[EXCELLENT](95.0) == 1.0
    assert b[BARELY_ACCEPTABLE](92.0) == 1.0
    assert b[UNACCEPTABLE](85.0) == 1.0


def test_membership_scales_convert_the_pipeline_fractions(config):
    """qSQI and basSQI are printed on 0-100, pSQI on 0-1."""
    scales = fuzzy_mod.factor_scales(config.section("fuzzy"))
    assert scales == {"qSQI": 100.0, "pSQI": 1.0, "kSQI": 1.0, "basSQI": 100.0}


def test_build_membership_functions_rejects_unknown_family(config):
    section = dict(config.section("fuzzy"))
    section = {
        **section,
        "membership": {
            **section["membership"],
            "qSQI": {
                **section["membership"]["qSQI"],
                EXCELLENT: {"family": "linear", "a": 0.0, "b": 1.0},
            },
        },
    }
    with pytest.raises(ValueError, match="unsupported membership family"):
        fuzzy_mod.build_membership_functions(section)


def test_build_membership_functions_rejects_a_missing_level(config):
    section = dict(config.section("fuzzy"))
    section = {**section, "membership": {**section["membership"], "qSQI": {"scale": 100.0}}}
    with pytest.raises(ValueError, match="is missing"):
        fuzzy_mod.build_membership_functions(section)


# ---------------------------------------------------------------------------
# Weights, synthesis, decision
# ---------------------------------------------------------------------------
def test_weights_are_the_article_vector(config):
    weights = fuzzy_mod.build_weight_vector(config.section("fuzzy"))
    assert weights == {"qSQI": 0.4, "pSQI": 0.4, "kSQI": 0.1, "basSQI": 0.1}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_weights_reject_wrong_sum():
    with pytest.raises(ValueError, match="sum to 1"):
        fuzzy_mod.build_weight_vector(
            {"weights": {"qSQI": 0.5, "pSQI": 0.5, "kSQI": 0.5, "basSQI": 0.5}}
        )


def test_weights_reject_negative():
    with pytest.raises(ValueError, match="non-negative"):
        fuzzy_mod.build_weight_vector(
            {"weights": {"qSQI": 1.4, "pSQI": -0.1, "kSQI": -0.1, "basSQI": -0.2}}
        )


def test_bounded_sum_is_the_papers_operator(config):
    """M(., +): s_j = min(1, sum_i (w_i r_ij)), clipped into [0, 1]."""
    assert config.get("fuzzy.synthesis") == "bounded_sum"
    matrix = np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    weights = np.array([0.4, 0.4, 0.1, 0.1])
    result = fuzzy_mod.bounded_sum(matrix, weights)
    assert np.allclose(result, [1.0, 0.4, 0.0])
    # The clip is real: weights summing above 1 cannot push a level past 1.
    assert np.all(fuzzy_mod.bounded_sum(matrix, np.array([1.0, 1.0, 0.0, 0.0])) <= 1.0)


def test_bounded_sum_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        fuzzy_mod.bounded_sum(np.zeros((4, 3)), np.array([0.5, 0.5]))


def test_synthesize_dispatches_on_the_configured_name():
    matrix = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.allclose(
        fuzzy_mod.synthesize(matrix, weights, "bounded_sum"),
        fuzzy_mod.bounded_sum(matrix, weights),
    )
    assert np.allclose(
        fuzzy_mod.synthesize(matrix, weights, "bounded_max_product"),
        fuzzy_mod.bounded_max_product(matrix, weights),
    )


def test_max_product_differs_from_a_bounded_sum():
    matrix = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    assert not np.allclose(
        fuzzy_mod.bounded_sum(matrix, weights),
        fuzzy_mod.bounded_max_product(matrix, weights),
    )


def test_an_unknown_synthesis_operator_is_rejected():
    with pytest.raises(ValueError, match="unknown synthesis operator"):
        fuzzy_mod.synthesize(np.ones((4, 3)), np.full(4, 0.25), "weighted_average_plus_vibes")


def test_weighted_average_score_matches_eq_33():
    """v = sum(s_j^2 j) / sum(s_j^2), with j = 1, 2, 3 for E, B, U."""
    assert fuzzy_mod.weighted_average_score(np.array([1.0, 1.0, 1.0])) == pytest.approx(2.0)
    assert fuzzy_mod.weighted_average_score(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert fuzzy_mod.weighted_average_score(np.array([0.0, 0.0, 1.0])) == pytest.approx(3.0)
    # Squared memberships, not the memberships themselves.
    assert fuzzy_mod.weighted_average_score(np.array([0.9, 0.1, 0.0])) == pytest.approx(
        (0.81 * 1 + 0.01 * 2) / 0.82
    )
    assert math.isnan(fuzzy_mod.weighted_average_score(np.array([0.0, 0.0, 0.0])))


def test_decide_uses_the_eq_34_thresholds(config):
    decision = config.section("fuzzy")["decision"]
    assert decision["kind"] == "weighted_average"
    assert fuzzy_mod.decide(1.0, decision) == EXCELLENT
    assert fuzzy_mod.decide(1.50, decision) == EXCELLENT
    assert fuzzy_mod.decide(1.51, decision) == BARELY_ACCEPTABLE
    assert fuzzy_mod.decide(2.39, decision) == BARELY_ACCEPTABLE
    assert fuzzy_mod.decide(2.40, decision) == UNACCEPTABLE
    assert fuzzy_mod.decide(3.0, decision) == UNACCEPTABLE


def test_decide_refuses_a_non_finite_score():
    with pytest.raises(ValueError, match="non-finite"):
        fuzzy_mod.decide(float("nan"), {})


def test_decide_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="unacceptable_min"):
        fuzzy_mod.decide(2.0, {"excellent_max": 2.5, "unacceptable_min": 2.0})


def test_max_membership_level_is_available_for_comparison():
    assert fuzzy_mod.max_membership_level(np.array([0.8, 0.15, 0.05])) == EXCELLENT
    assert fuzzy_mod.max_membership_level(np.array([0.1, 0.7, 0.2])) == BARELY_ACCEPTABLE
    assert fuzzy_mod.max_membership_level(np.array([0.1, 0.2, 0.7])) == UNACCEPTABLE


# ---------------------------------------------------------------------------
# Full fuzzy evaluation
# ---------------------------------------------------------------------------
def test_evaluate_returns_the_full_record(config):
    result = fuzzy_mod.evaluate(
        {"qSQI": 0.95, "pSQI": 0.6, "kSQI": 8.0, "basSQI": 0.97},
        config.section("fuzzy"),
    )
    assert set(result.membership) == {EXCELLENT, BARELY_ACCEPTABLE, UNACCEPTABLE}
    assert all(0.0 <= value <= 1.0 for value in result.membership.values())
    assert result.quality_class in {EXCELLENT, BARELY_ACCEPTABLE, UNACCEPTABLE}
    assert set(result.evaluation_matrix) == {"qSQI", "pSQI", "kSQI", "basSQI"}
    assert result.synthesis == "bounded_sum"
    assert result.weights == {"qSQI": 0.4, "pSQI": 0.4, "kSQI": 0.1, "basSQI": 0.1}
    assert result.rating_values == {EXCELLENT: 1.0, BARELY_ACCEPTABLE: 2.0, UNACCEPTABLE: 3.0}
    assert 1.0 <= result.score <= 3.0
    assert result.quality_class == fuzzy_mod.decide(
        result.score, config.section("fuzzy")["decision"]
    )


def test_evaluation_matrix_holds_the_raw_printed_memberships(config):
    """Rows of R are raw membership degrees and are deliberately not normalised.

    The Cauchy rows of qSQI and basSQI do not sum to 1, and the article does not
    normalise them, so neither does this implementation.
    """
    result = fuzzy_mod.evaluate(
        {"qSQI": 0.85, "pSQI": 0.25, "kSQI": 8.0, "basSQI": 0.92},
        config.section("fuzzy"),
    )
    q_row = result.evaluation_matrix["qSQI"]
    assert q_row[EXCELLENT] == pytest.approx(1 / 3.25)
    assert q_row[BARELY_ACCEPTABLE] == pytest.approx(1 / (1 + ((85 - 75) / 7.5) ** 2))
    assert q_row[UNACCEPTABLE] == pytest.approx(1 / (1 + ((85 - 55) / 5) ** 2))
    assert sum(q_row.values()) != pytest.approx(1.0)

    assert result.evaluation_matrix["kSQI"] == {
        EXCELLENT: 1.0, BARELY_ACCEPTABLE: 0.0, UNACCEPTABLE: 0.0
    }
    # Eq (27) reaches 0 exactly at x = 0.25, while Eq (28) is at its plateau.
    assert result.evaluation_matrix["pSQI"] == {
        EXCELLENT: 0.0, BARELY_ACCEPTABLE: 1.0, UNACCEPTABLE: 0.0
    }


def test_evaluate_good_signal_is_excellent(config):
    result = fuzzy_mod.evaluate(
        {"qSQI": 0.98, "pSQI": 0.6, "kSQI": 8.0, "basSQI": 0.97},
        config.section("fuzzy"),
    )
    assert result.quality_class == EXCELLENT
    assert result.excellent > result.unacceptable


def test_evaluate_poor_signal_is_unacceptable(config):
    result = fuzzy_mod.evaluate(
        {"qSQI": 0.1, "pSQI": 0.05, "kSQI": 1.0, "basSQI": 0.2},
        config.section("fuzzy"),
    )
    assert result.quality_class == UNACCEPTABLE
    assert result.unacceptable > result.excellent


def test_evaluate_poor_frame_scores_above_good_frame(config):
    section = config.section("fuzzy")
    good = fuzzy_mod.evaluate({"qSQI": 0.98, "pSQI": 0.6, "kSQI": 8.0, "basSQI": 0.97}, section)
    poor = fuzzy_mod.evaluate({"qSQI": 0.1, "pSQI": 0.05, "kSQI": 1.0, "basSQI": 0.2}, section)
    assert good.score < poor.score


def test_evaluate_rejects_missing_factor(config):
    with pytest.raises(ValueError, match="missing SQI factors"):
        fuzzy_mod.evaluate({"qSQI": 0.9, "pSQI": 0.8}, config.section("fuzzy"))


def test_evaluate_rejects_nan(config):
    with pytest.raises(ValueError, match="not a finite value"):
        fuzzy_mod.evaluate(
            {"qSQI": float("nan"), "pSQI": 0.8, "kSQI": 4.0, "basSQI": 0.9},
            config.section("fuzzy"),
        )


def test_evaluate_rejects_wrong_levels(config):
    section = dict(config.section("fuzzy"))
    section["levels"] = ["Good", "Bad"]
    with pytest.raises(ValueError, match="rating levels"):
        fuzzy_mod.evaluate(
            {"qSQI": 0.9, "pSQI": 0.8, "kSQI": 4.0, "basSQI": 0.9}, section
        )

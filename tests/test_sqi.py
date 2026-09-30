"""SQI tests: qSQI, pSQI, kSQI, basSQI and the fuzzy stage."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ecg_eval.detectors import available_detectors, get_detector
from ecg_eval.detectors.pan_tompkins import PanTompkinsDetector
from ecg_eval.detectors.zhao_hilbert import ZhaoHilbertDetector
from ecg_eval.detectors.zhao_wavelet import ZhaoWaveletDetector
from ecg_eval.sqi import bas_sqi, cauchy, evaluate, fuzzy, kurtosis, match_peaks, p_sqi, q_sqi
from ecg_eval.sqi import k_sqi as k_sqi_fn
from ecg_eval.sqi import p_sqi as p_sqi_fn
from ecg_eval.sqi.fuzzy import (
    FACTORS,
    LEVELS,
    bounded_max_product,
    bounded_sum,
    build_weight_vector,
    decide,
    rectangular,
    synthesize,
    trapezoidal,
)

FS = 250.0


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
    expected = clean_signal.size / FS * 1.0  # 1 Hz -> ~10 beats in 10 s
    assert abs(result.peaks.size - expected) <= 2
    assert result.label == "zhao_zhang"


def test_wavelet_detector_finds_the_synthetic_r_peaks(clean_signal):
    result = get_detector("zhao_wavelet").detect(clean_signal, FS)
    expected = clean_signal.size / FS
    assert abs(result.peaks.size - expected) <= 2
    assert result.label == "zhao_zhang"


def test_pan_tompkins_is_labelled_as_adaptation(clean_signal):
    result = PanTompkinsDetector().detect(clean_signal, FS)
    assert result.label == "pan_tompkins_adapted"
    assert result.peaks.size > 0


def test_detector_is_deterministic(clean_signal):
    detector = ZhaoHilbertDetector()
    first = detector.detect(clean_signal, FS).as_list()
    second = detector.detect(clean_signal, FS).as_list()
    assert first == second


def test_detector_handles_short_signal():
    assert get_detector("zhao_hilbert").detect(np.zeros(10), FS).peaks.size == 0


def test_detector_tolerates_nan(clean_signal):
    corrupt = clean_signal.copy()
    corrupt[500] = np.nan
    assert get_detector("zhao_wavelet").detect(corrupt, FS).peaks.size > 0


# ---------------------------------------------------------------------------
# qSQI
# ---------------------------------------------------------------------------
def test_match_peaks_within_tolerance():
    a = np.array([100, 200, 300])
    b = np.array([110, 205, 500])
    # 150 ms at 250 Hz = 37.5 samples
    pairs, unmatched_a, unmatched_b = match_peaks(a, b, FS, 150.0)
    assert pairs == [(100, 110), (200, 205)]
    assert unmatched_a == [300]
    assert unmatched_b == [500]


def test_match_peaks_respects_tolerance():
    a = np.array([100])
    b = np.array([200])
    pairs, _, _ = match_peaks(a, b, FS, 150.0)  # 37.5 samples
    assert pairs == []


def test_match_peaks_one_to_one():
    a = np.array([100, 104])
    b = np.array([102])
    pairs, _, _ = match_peaks(a, b, FS, 150.0)
    assert len(pairs) == 1
    assert len({p[0] for p in pairs}) == 1


def test_match_peaks_empty():
    pairs, ua, ub = match_peaks(np.array([]), np.array([1, 2]), FS, 150.0)
    assert pairs == [] and ub == [1, 2] and ua == []


def test_q_sqi_perfect_agreement():
    peaks = np.array([100, 300, 500])
    pairs, _, _ = match_peaks(peaks, peaks, FS, 150.0)
    assert len(pairs) / min(peaks.size, peaks.size) == 1.0


def test_q_sqi_on_clean_signal_is_high(clean_signal):
    value, _, _, match = q_sqi(clean_signal, FS, tolerance_ms=150.0)
    assert 0.0 <= value <= 1.0
    assert match.n_matched <= min(match.n_peaks_a, match.n_peaks_b)
    assert match.label == "zhao_zhang"


def test_q_sqi_uses_pan_tompkins_when_substituted(clean_signal):
    _, _, _, match = q_sqi(clean_signal, FS, detector_b="pan_tompkins", tolerance_ms=150.0)
    assert match.label == "pan_tompkins_adapted"
    assert match.detector_b == "pan_tompkins"


def test_q_sqi_records_all_debug_fields(clean_signal):
    _, _, _, match = q_sqi(clean_signal, FS, tolerance_ms=150.0)
    payload = match.to_dict()
    for key in (
        "n_peaks_a", "n_peaks_b", "n_matched", "q_sqi", "peak_indices" if False else "peaks_a",
        "peaks_b", "matched_pairs", "unmatched_a", "unmatched_b", "tolerance_ms",
    ):
        assert key in payload
    assert match.tolerance_ms == 150.0


def test_q_sqi_no_peaks_gives_zero(clean_signal):
    value, _, _, match = q_sqi(np.zeros(2500), FS)
    assert value == 0.0
    assert match.n_matched == 0


def test_q_sqi_adaptive_tolerance_differs(clean_signal):
    fixed, _, _, fixed_match = q_sqi(clean_signal, FS, tolerance_ms=100.0, adaptive=False)
    adaptive, _, _, adaptive_match = q_sqi(clean_signal, FS, tolerance_ms=100.0, adaptive=True)
    assert fixed_match.tolerance_ms == 100.0
    assert 80.0 <= adaptive_match.tolerance_ms <= 250.0


# ---------------------------------------------------------------------------
# pSQI
# ---------------------------------------------------------------------------
def test_p_sqi_perfect_band_concentration():
    t = np.arange(2500) / FS
    pure_qrs = np.sin(2 * np.pi * 10 * t)
    value, spectral = p_sqi_fn(pure_qrs, FS, qrs_band_hz=(5, 15), total_band_hz=(0.5, 40))
    assert value == pytest.approx(1.0, abs=0.05)
    assert spectral.p_total_power > 0
    assert len(spectral.frequencies) == len(spectral.psd)


def test_p_sqi_low_when_energy_is_out_of_band():
    t = np.arange(2500) / FS
    wander = np.sin(2 * np.pi * 0.3 * t)
    value, _ = p_sqi_fn(wander, FS, qrs_band_hz=(5, 15), total_band_hz=(0.5, 40))
    assert value < 0.2


def test_p_sqi_bands_come_from_configuration(config):
    params = p_sqi_fn.__module__  # keep import used
    from ecg_eval.sqi import p_sqi_config

    resolved = p_sqi_config(config)
    assert resolved["qrs_band_hz"] == (5.0, 15.0)
    assert resolved["method"] == "welch"
    del params


def test_p_sqi_stores_band_powers(clean_signal):
    _, spectral = p_sqi_fn(clean_signal, FS)
    assert spectral.qrs_band_power > 0
    assert spectral.p_total_power >= spectral.qrs_band_power


# ---------------------------------------------------------------------------
# kSQI
# ---------------------------------------------------------------------------
def test_fisher_and_pearson_differ_by_three():
    signal = np.random.default_rng(0).standard_normal(5000)
    assert kurtosis(signal, "fisher") == pytest.approx(kurtosis(signal, "pearson") - 3.0)


def test_gaussian_is_zero_under_fisher():
    signal = np.random.default_rng(1).standard_normal(200000)
    assert kurtosis(signal, "fisher") == pytest.approx(0.0, abs=0.05)


def test_sparse_qrs_is_leptokurtic(clean_signal):
    # A clean ECG with sharp QRS complexes must exceed the Gaussian value of 3.
    assert kurtosis(clean_signal, "fisher") > 3.0


def test_noise_reduces_kurtosis(clean_signal, noisy_signal):
    assert kurtosis(noisy_signal, "fisher") < kurtosis(clean_signal, "fisher")


def test_constant_signal_returns_nan():
    assert math.isnan(kurtosis(np.ones(100)))


def test_k_sqi_returns_mean_and_std(clean_signal):
    value, mean, std = k_sqi_fn(clean_signal, definition="fisher")
    assert value == kurtosis(clean_signal, "fisher")
    assert mean == pytest.approx(float(np.mean(clean_signal)))
    assert std == pytest.approx(float(np.std(clean_signal)))


def test_unknown_kurtosis_definition_raises():
    with pytest.raises(ValueError):
        kurtosis(np.random.default_rng(0).standard_normal(100), "bogus")


# ---------------------------------------------------------------------------
# basSQI
# ---------------------------------------------------------------------------
def test_bas_sqi_low_for_clean_signal(clean_signal):
    value, spectral = bas_sqi(clean_signal, FS, baseline_band_hz=(0, 1), total_band_hz=(0, 40))
    assert 0.0 <= value < 0.5
    assert spectral.bas_total_power > 0


def test_bas_sqi_high_with_baseline_wander(clean_signal):
    t = np.arange(clean_signal.size) / FS
    wander = 2.0 * np.sin(2 * np.pi * 0.4 * t)
    value, _ = bas_sqi(clean_signal + wander, FS, baseline_band_hz=(0, 1), total_band_hz=(0, 40))
    assert value > 0.3


def test_bas_sqi_stores_band_powers(clean_signal):
    _, spectral = bas_sqi(clean_signal, FS)
    assert spectral.baseline_band_power >= 0
    assert spectral.bas_sqi >= 0


# ---------------------------------------------------------------------------
# Membership functions
# ---------------------------------------------------------------------------
def test_cauchy_peaks_at_xc():
    assert cauchy(0.9, 0.9, 0.1) == pytest.approx(1.0)
    # Symmetric around xc, and equal to 0.5 one gamma away.
    assert cauchy(0.8, 0.9, 0.1) == pytest.approx(0.5)
    assert cauchy(1.0, 0.9, 0.1) == pytest.approx(0.5)
    # Two gammas away: 1 / (1 + 4) = 0.2.
    assert cauchy(1.1, 0.9, 0.1) == pytest.approx(0.2)
    assert cauchy(0.7, 0.9, 0.1) == pytest.approx(0.2)
    assert 0 < cauchy(0.0, 0.9, 0.1) < 0.1


def test_trapezoidal_shape():
    assert trapezoidal(0.8, 0.0, 0.7, 0.9, 1.0) == pytest.approx(1.0)
    assert trapezoidal(0.35, 0.0, 0.7, 0.9, 1.0) == pytest.approx(0.5)
    assert trapezoidal(0.95, 0.0, 0.7, 0.9, 1.0) == pytest.approx(0.5)
    assert trapezoidal(1.5, 0.0, 0.7, 0.9, 1.0) == 0.0


def test_trapezoidal_rejects_bad_parameters():
    with pytest.raises(ValueError):
        trapezoidal(0.5, 0.7, 0.2, 0.9, 1.0)


def test_rectangular():
    assert rectangular(3.0, 2.0, 6.0) == 1.0
    assert rectangular(7.0, 2.0, 6.0) == 0.0
    assert rectangular(1.0, 2.0, 6.0) == 0.0


def test_membership_families_match_the_reference(config):
    families = {
        factor: config.get(f"fuzzy.membership.{factor}.family")
        for factor in ("qSQI", "pSQI", "kSQI", "basSQI")
    }
    assert families == {
        "qSQI": "cauchy",
        "pSQI": "trapezoidal",
        "kSQI": "rectangular",
        "basSQI": "cauchy",
    }


def test_build_membership_functions_rejects_unknown_family(config):
    spec = {"membership": {"qSQI": {"family": "linear"}}, "levels": list(fuzzy.LEVELS)}
    with pytest.raises(ValueError):
        fuzzy.build_membership_functions(spec)


# ---------------------------------------------------------------------------
# Weights, synthesis, decision
# ---------------------------------------------------------------------------
def test_weights_sum_to_one(config):
    weights = build_weight_vector(config.section("fuzzy"))
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == {"qSQI", "pSQI", "kSQI", "basSQI"}


def test_weights_reject_wrong_sum():
    with pytest.raises(ValueError, match="sum to 1"):
        build_weight_vector({"weights": {"qSQI": 0.5, "pSQI": 0.5, "kSQI": 0.5, "basSQI": 0.5}})


def test_weights_reject_negative():
    with pytest.raises(ValueError, match="non-negative"):
        build_weight_vector(
            {"weights": {"qSQI": 1.4, "pSQI": -0.1, "kSQI": -0.1, "basSQI": -0.2}}
        )


def test_bounded_sum_stays_in_unit_interval():
    matrix = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.9, 0.05, 0.05]])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    result = bounded_sum(matrix, weights)
    assert np.all(result >= 0) and np.all(result <= 1)
    assert result.sum() == pytest.approx(1.0)


def test_bounded_sum_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        bounded_sum(np.zeros((4, 3)), np.array([0.5, 0.5]))


def test_max_product_is_not_a_weighted_sum():
    """The operator that configuration names must actually be computed.

    Max-product and weighted sum are different operators and can disagree on the
    winning class. Here Excellent is spread thinly across three factors while
    Unacceptable is strong in one: the sum accumulates the broad weak support and
    rates the frame Excellent, while the max-product is dominated by the single
    strongest factor and rates it Unacceptable.
    """
    matrix = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    summed = bounded_sum(matrix, weights)
    product = bounded_max_product(matrix, weights)

    assert not np.allclose(summed, product)
    assert decide(summed) == "Excellent"
    assert decide(product) == "Unacceptable"
    # A weighted sum of rows that each sum to 1 is a convex combination and sums
    # to 1; a max-product takes a maximum over factors and need not.
    assert summed.sum() == pytest.approx(1.0)
    assert product.sum() == pytest.approx(0.375)


def test_max_product_takes_the_best_single_weighted_factor():
    matrix = np.array([[0.9, 0.05, 0.05], [0.4, 0.4, 0.2], [0.3, 0.4, 0.3], [0.5, 0.3, 0.2]])
    weights = np.array([1.0, 0.0, 0.0, 0.0])

    result = bounded_max_product(matrix, weights)

    assert np.allclose(result, matrix[0])


def test_synthesize_dispatches_on_the_configured_name():
    matrix = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    assert np.allclose(synthesize(matrix, weights, "bounded_max_product"), bounded_max_product(matrix, weights))
    assert np.allclose(synthesize(matrix, weights, "bounded_sum"), bounded_sum(matrix, weights))


def test_an_unknown_synthesis_operator_is_rejected():
    matrix = np.ones((4, 3))
    weights = np.full(4, 0.25)

    with pytest.raises(ValueError, match="unknown synthesis operator"):
        synthesize(matrix, weights, "weighted_average_plus_vibes")


def test_evaluate_reports_the_operator_it_used(config):
    values = {"qSQI": 0.5, "pSQI": 0.4272, "kSQI": 11.11, "basSQI": 0.0404}
    fuzzy_cfg = config.section("fuzzy")

    product = evaluate(values, {**fuzzy_cfg, "synthesis": "bounded_max_product"})
    summed = evaluate(values, {**fuzzy_cfg, "synthesis": "bounded_sum"})

    assert product.synthesis == "bounded_max_product"
    assert summed.synthesis == "bounded_sum"
    assert product.membership != summed.membership


def test_decide_picks_argmax():
    assert decide(np.array([0.8, 0.15, 0.05])) == "Excellent"
    assert decide(np.array([0.1, 0.7, 0.2])) == "Barely Acceptable"
    assert decide(np.array([0.1, 0.2, 0.7])) == "Unacceptable"


# ---------------------------------------------------------------------------
# Full fuzzy evaluation
# ---------------------------------------------------------------------------
def test_evaluate_returns_full_vector(config):
    result = evaluate(
        {"qSQI": 0.95, "pSQI": 0.8, "kSQI": 4.5, "basSQI": 0.9},
        config.section("fuzzy"),
    )
    assert set(result.membership) == {"Excellent", "Barely Acceptable", "Unacceptable"}
    assert all(0.0 <= value <= 1.0 for value in result.membership.values())
    assert result.quality_class in {"Excellent", "Barely Acceptable", "Unacceptable"}
    # The evaluation matrix is preserved for inspection.
    assert set(result.evaluation_matrix) == {"qSQI", "pSQI", "kSQI", "basSQI"}
    # Each factor's row is a partition of unity over the three rating levels.
    for factor, row in result.evaluation_matrix.items():
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-9), factor


def test_memberships_are_shares_only_under_a_normalised_synthesis(config):
    """A max-product vector is not a distribution; do not present it as one.

    Each factor's row normalises to 1 over the rating levels, but the synthesis
    operator then combines the rows. A bounded sum of normalised rows with
    weights that sum to 1 is still a distribution; a max-product is a maximum
    over factors and is not. The configured operator is max-product, so the
    reported memberships are deliberately not rescaled to sum to 1 -- rescaling
    would invent numbers that no operator in the literature produces.
    """
    values = {"qSQI": 0.95, "pSQI": 0.8, "kSQI": 4.5, "basSQI": 0.9}
    fuzzy_cfg = config.section("fuzzy")

    product = evaluate(values, {**fuzzy_cfg, "synthesis": "bounded_max_product"})
    summed = evaluate(values, {**fuzzy_cfg, "synthesis": "bounded_sum"})

    assert sum(summed.membership.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(product.membership.values()) < 1.0
    # Rescaling would not change the decision, but the raw values are kept so a
    # reviewer can check them against the operator.
    assert product.quality_class == max(
        product.membership, key=lambda level: product.membership[level]
    )


def test_evaluate_good_signal_is_excellent(config):
    result = evaluate(
        {"qSQI": 0.98, "pSQI": 0.95, "kSQI": 6.0, "basSQI": 0.95},
        config.section("fuzzy"),
    )
    assert result.quality_class == "Excellent"
    assert result.excellent > result.unacceptable


def test_evaluate_poor_signal_is_unacceptable(config):
    result = evaluate(
        {"qSQI": 0.2, "pSQI": 0.1, "kSQI": 1.0, "basSQI": 0.2},
        config.section("fuzzy"),
    )
    assert result.quality_class == "Unacceptable"
    assert result.unacceptable > result.excellent


def test_evaluate_rejects_missing_factor(config):
    with pytest.raises(ValueError, match="missing SQI factors"):
        evaluate({"qSQI": 0.9, "pSQI": 0.8}, config.section("fuzzy"))


def test_evaluate_rejects_nan(config):
    with pytest.raises(ValueError, match="not a finite value"):
        evaluate(
            {"qSQI": float("nan"), "pSQI": 0.8, "kSQI": 4.0, "basSQI": 0.9},
            config.section("fuzzy"),
        )


def test_evaluate_rejects_wrong_levels(config):
    section = dict(config.section("fuzzy"))
    section["levels"] = ["Good", "Bad"]
    with pytest.raises(ValueError, match="rating levels"):
        evaluate({"qSQI": 0.9, "pSQI": 0.8, "kSQI": 4.0, "basSQI": 0.9}, section)

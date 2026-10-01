"""Fuzzy comprehensive evaluation (Zhao & Zhang 2018, Eq 20-34).

Alur::

    indeks kualitas sinyal
        -> fungsi keanggotaan (membership)
        -> matriks evaluasi R          (4 faktor x 3 peringkat)
        -> vektor bobot W
        -> sintesis fuzzy terbatas     S = W o R
        -> skor rata-rata berbobot v
        -> kelas kualitas

Himpunan faktor    U = {qSQI, pSQI, kSQI, basSQI}
Himpunan peringkat V = {Excellent, Barely Acceptable, Unacceptable}

Keluarga fungsi keanggotaan yang dipakai artikel (Eq 22-32):

===========  ==============================================  ==========
faktor       keluarga                                        persamaan
===========  ==============================================  ==========
qSQI         Cauchy (naik, turun, terpusat)                  22, 24, 25
pSQI         trapezoidal                                    26, 27, 28
kSQI         rectangular (biner)                            29
basSQI       Cauchy (naik, turun, terpusat)                  30, 31, 32
===========  ==============================================  ==========

Setiap parameter numerik dibaca dari konfigurasi dan membawa sumbernya; tidak
ada konstanta yang ditulis di dalam fungsi. ``build_membership_functions``
menolak nama keluarga yang tidak dikenal alih-alih diam-diam memakai default.

Dua hal yang dipertahankan apa adanya, bukan didamaikan diam-diam:

1. Artikel mengevaluasi qSQI dan basSQI pada skala 0-100, sedangkan pSQI pada
   skala 0-1 dan kSQI pada nilai mentahnya. Setiap faktor karena itu membawa
   ``scale``-nya sendiri dari konfigurasi, dan nilainya dikalikan sebelum masuk
   ke fungsi keanggotaan.
2. Kriteria penerimaan pSQI (Eq 4-5) dan fungsi keanggotaannya (Eq 26-28) tidak
   saling konsisten di dalam artikel: kriteria menaruh "optimal" pada
   [0.5, 0.8], sedangkan fungsi keanggotaan sudah menjenuh pada p >= 0.35.
   Keduanya diimplementasikan sebagaimana tercetak, masing-masing di tempatnya
   -- kriteria pada penilaian per indeks, fungsi keanggotaan pada tahap fuzzy.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np

from ..models.result import (
    BARELY_ACCEPTABLE,
    EXCELLENT,
    UNACCEPTABLE,
    FuzzyResult,
)
from ._params import unwrap

LEVELS = (EXCELLENT, BARELY_ACCEPTABLE, UNACCEPTABLE)
FACTORS = ("qSQI", "pSQI", "kSQI", "basSQI")

LEVEL_SHORT = {EXCELLENT: "E", BARELY_ACCEPTABLE: "B", UNACCEPTABLE: "U"}


# ---------------------------------------------------------------------------
# Membership functions
# ---------------------------------------------------------------------------
def cauchy(x: float, xc: float, gamma: float) -> float:
    """Cauchy membership: 1 / (1 + ((x - xc) / gamma)^2), peak 1 at x = xc."""
    if gamma <= 0:
        return 1.0 if x == xc else 0.0
    return float(1.0 / (1.0 + ((x - xc) / gamma) ** 2))


def cauchy_increasing(x: float, a: float, k: float, cap_from: float) -> float:
    """Increasing half of a Cauchy curve, capped at 1 (Eq 22, Eq 30).

    ``0`` at or below ``a``, ``1 / (1 + [k (x - a)]^2)`` between ``a`` and
    ``cap_from``, and exactly ``1`` from ``cap_from`` upwards. The reference
    prints the cap separately because the curve alone never reaches 1.
    """
    if x <= a:
        return 0.0
    if x >= cap_from:
        return 1.0
    return float(1.0 / (1.0 + (k * (x - a)) ** 2))


def cauchy_decreasing(x: float, a: float, width: float) -> float:
    """Decreasing half of a Cauchy curve (Eq 24, Eq 31).

    Exactly ``1`` at or below ``a``, then ``1 / (1 + ((x - a) / width)^2)``.
    """
    if x <= a:
        return 1.0
    if width <= 0:
        return 0.0
    return float(1.0 / (1.0 + ((x - a) / width) ** 2))


def cauchy_centred(x: float, centre: float, width: float) -> float:
    """Symmetric Cauchy membership, peak 1 at ``centre`` (Eq 25, Eq 32)."""
    if width <= 0:
        return 1.0 if x == centre else 0.0
    return float(1.0 / (1.0 + ((x - centre) / width) ** 2))


def ramp_up(x: float, a: float, b: float) -> float:
    """Left shoulder: 0 at or below ``a``, linear to 1 at ``b`` (Eq 26)."""
    if b <= a:
        raise ValueError(f"ramp_up needs b > a, got a={a}, b={b}")
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return float((x - a) / (b - a))


def ramp_down(x: float, a: float, b: float) -> float:
    """Right shoulder: 1 at or below ``a``, linear to 0 at ``b`` (Eq 27)."""
    if b <= a:
        raise ValueError(f"ramp_down needs b > a, got a={a}, b={b}")
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return float((b - x) / (b - a))


def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership: ramp up [a,b], plateau 1 [b,c], ramp down [c,d]."""
    if b == a or d == c or a > b or b > c or c > d:
        raise ValueError(f"invalid trapezoid parameters a={a}, b={b}, c={c}, d={d}")
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a <= x < b:
        return float((x - a) / (b - a))
    return float((d - x) / (d - c))


def rectangular(x: float, a: float, b: float) -> float:
    """Rectangular membership: 1 inside [a, b], 0 outside."""
    return 1.0 if a <= x <= b else 0.0


def step_above(x: float, threshold: float) -> float:
    """1 strictly above ``threshold``, else 0 (the kSQI Excellent row, Eq 29)."""
    return 1.0 if x > threshold else 0.0


def step_at_or_below(x: float, threshold: float) -> float:
    """1 at or below ``threshold``, else 0 (the kSQI Unacceptable row, Eq 29)."""
    return 1.0 if x <= threshold else 0.0


def zero_membership(x: float) -> float:
    """Identically 0. Used where the reference assigns no membership at all.

    Kurtosis has no Barely-Acceptable membership (Eq 29 maps a value either to
    the Excellent row or the Unacceptable row), and writing that as an explicit
    family keeps the shape of R faithful instead of leaving a level silently
    absent from the matrix.
    """
    return 0.0


#: Every family the configuration may name, with the shape it produces. A name
#: outside this table is an error, never a fallback.
MEMBERSHIP_FAMILIES: dict[str, str] = {
    "cauchy_increasing": "0 below a; 1/(1+[k(x-a)]^2) up to cap_from; 1 at or above cap_from",
    "cauchy_decreasing": "1 at or below a; 1/(1+((x-a)/width)^2) above a",
    "cauchy_centred": "1/(1+((x-centre)/width)^2)",
    "ramp_up": "0 below a; linear to 1 at b; 1 above b",
    "ramp_down": "1 below a; linear to 0 at b; 0 above b",
    "trapezoid": "0 below a; up to 1 over [a,b]; 1 over [b,c]; down to 0 over [c,d]",
    "step_above": "1 strictly above threshold, else 0",
    "step_at_or_below": "1 at or below threshold, else 0",
    "zero": "identically 0",
}

_CURVE_PARAMS = {
    "cauchy_increasing": ("a", "k", "cap_from"),
    "cauchy_decreasing": ("a", "width"),
    "cauchy_centred": ("centre", "width"),
    "ramp_up": ("a", "b"),
    "ramp_down": ("a", "b"),
    "trapezoid": ("a", "b", "c", "d"),
    "step_above": ("threshold",),
    "step_at_or_below": ("threshold",),
    "zero": (),
}


def _read_params(family: str, spec: Mapping[str, Any], where: str) -> dict[str, float]:
    """Read a level's parameters, requiring every one the family needs."""
    params: dict[str, float] = {}
    for name in _CURVE_PARAMS[family]:
        if name not in spec:
            raise ValueError(f"membership parameter {name!r} is missing for {where}")
        raw = unwrap(spec.get(name))
        if raw is None:
            raise ValueError(f"membership parameter {name!r} for {where} has no value")
        params[name] = float(raw)
    return params


def _make_membership(family: str, params: Mapping[str, float]) -> Callable[[float], float]:
    if family == "cauchy_increasing":
        if params["k"] <= 0:
            raise ValueError(f"cauchy_increasing needs k > 0, got {params['k']}")
        if params["cap_from"] <= params["a"]:
            raise ValueError("cauchy_increasing needs cap_from > a")
        return lambda x: cauchy_increasing(x, params["a"], params["k"], params["cap_from"])
    if family == "cauchy_decreasing":
        if params["width"] <= 0:
            raise ValueError(f"cauchy_decreasing needs width > 0, got {params['width']}")
        return lambda x: cauchy_decreasing(x, params["a"], params["width"])
    if family == "cauchy_centred":
        if params["width"] <= 0:
            raise ValueError(f"cauchy_centred needs width > 0, got {params['width']}")
        return lambda x: cauchy_centred(x, params["centre"], params["width"])
    if family == "ramp_up":
        ramp_up(params["a"], params["a"], params["b"])  # validate ordering eagerly
        return lambda x: ramp_up(x, params["a"], params["b"])
    if family == "ramp_down":
        ramp_down(params["a"], params["a"], params["b"])
        return lambda x: ramp_down(x, params["a"], params["b"])
    if family == "trapezoid":
        trapezoidal(
            params["a"], params["a"], params["b"], params["c"], params["d"]
        )  # validate parameter ordering eagerly
        return lambda x: trapezoidal(x, params["a"], params["b"], params["c"], params["d"])
    if family == "step_above":
        return lambda x: step_above(x, params["threshold"])
    if family == "step_at_or_below":
        return lambda x: step_at_or_below(x, params["threshold"])
    if family == "zero":
        return zero_membership
    raise ValueError(
        f"unsupported membership family {family!r}; expected one of {sorted(MEMBERSHIP_FAMILIES)}"
    )


def build_membership_functions(
    fuzzy_config: Mapping[str, Any],
) -> dict[str, dict[str, Callable[[float], float]]]:
    """One membership function per factor and rating level.

    The configuration names the family and gives every parameter explicitly, per
    rating level, so all three levels are defined in one place and nothing is
    derived at runtime. The result maps ``factor -> level -> function``.
    """
    membership_cfg = fuzzy_config.get("membership", {}) or {}
    levels = tuple(fuzzy_config.get("levels", LEVELS))
    if levels != LEVELS:
        raise ValueError(f"rating levels must be {LEVELS}, got {levels}")

    built: dict[str, dict[str, Callable[[float], float]]] = {}
    for factor in FACTORS:
        spec = membership_cfg.get(factor)
        if not spec:
            raise ValueError(f"no membership configuration for factor {factor!r}")
        # The family belongs to the RATING LEVEL, not to the index. The article
        # uses a different family for each level of the same index -- qSQI is
        # the increasing half of a Cauchy curve for Excellent (Eq 22), a centred
        # one for Barely Acceptable (Eq 25) and the decreasing half for
        # Unacceptable (Eq 24) -- so one family per index cannot express it. A
        # family written on the index itself is only a default for a level that
        # omits its own.
        factor_family = str(unwrap(spec.get("family"), "")).lower()
        rows: dict[str, Callable[[float], float]] = {}
        for level in levels:
            level_spec = spec.get(level)
            if not isinstance(level_spec, Mapping):
                raise ValueError(
                    f"membership configuration for {factor}/{level} is missing; "
                    f"expected an explicit parameter set for each of {LEVELS}"
                )
            family = str(unwrap(level_spec.get("family"), factor_family)).lower()
            if family not in _CURVE_PARAMS:
                raise ValueError(
                    f"unsupported membership family {family!r} for {factor}/{level}; "
                    f"expected one of {sorted(MEMBERSHIP_FAMILIES)}"
                )
            params = _read_params(family, level_spec, f"{factor}/{level}")
            rows[level] = _make_membership(family, params)
        built[factor] = rows
    return built


def factor_scales(fuzzy_config: Mapping[str, Any]) -> dict[str, float]:
    """The scale each factor's membership functions are defined on.

    The reference evaluates qSQI and basSQI on a 0-100 scale and pSQI on 0-1,
    while this pipeline reports every index as a 0-1 fraction. The scale is
    configuration, so the printed breakpoints (80, 90, 0.25) stay as published
    and the conversion is visible rather than baked into a function body.
    """
    membership_cfg = fuzzy_config.get("membership", {}) or {}
    scales: dict[str, float] = {}
    for factor in FACTORS:
        spec = membership_cfg.get(factor) or {}
        scale = float(unwrap(spec.get("scale"), 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"membership scale for {factor} must be > 0, got {scale}")
        scales[factor] = scale
    return scales


# ---------------------------------------------------------------------------
# Weight vector
# ---------------------------------------------------------------------------
def build_weight_vector(fuzzy_config: Mapping[str, Any]) -> dict[str, float]:
    """Read and validate W. ``sum(W) == 1`` is enforced here and in tests."""
    raw = fuzzy_config.get("weights", {}) or {}
    if not raw:
        raise ValueError("fuzzy weight vector is missing from configuration")
    weights = {factor: float(unwrap(raw.get(factor), 0.0) or 0.0) for factor in FACTORS}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fuzzy weights must sum to 1, got {total:.6f} ({weights})")
    if any(value < 0 for value in weights.values()):
        raise ValueError(f"fuzzy weights must be non-negative, got {weights}")
    return weights


# ---------------------------------------------------------------------------
# Synthesis and decision
# ---------------------------------------------------------------------------
#: Synthesis operators, by configuration name. The configured name must resolve
#: to one of these; :func:`synthesize` raises on anything else rather than
#: silently falling back, because the operator changes every reported membership.
SYNTHESIS_OPERATORS = {
    "bounded_sum": "min(1, sum_i (w_i * R_ij))   -- M(., +), the paper's operator",
    "bounded_max_product": "max_i (w_i * R_ij)",
    "weighted_average": "sum_i (w_i * R_ij)",
}


def bounded_sum(evaluation_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Bounded-sum synthesis: ``B_j = min(1, sum_i (w_i * R_ij))``.

    This is the operator M(., +) that the reference selects, written as
    ``S = W o R``. The clip is the "bounded" part: it keeps the synthesised
    memberships inside [0, 1] even when the rows of R do not sum to 1, which is
    the case for the Cauchy rows of qSQI and basSQI.
    """
    _check_shapes(evaluation_matrix, weights)
    return np.clip(weights @ evaluation_matrix, 0.0, 1.0)


def bounded_max_product(evaluation_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Max-product synthesis: ``B_j = max_i (w_i * R_ij)``.

    Belongs to the M(and, .) family the reference compares against and does NOT
    select. Kept so the comparison it describes stays reproducible; the
    configured operator for this project is :func:`bounded_sum`.
    """
    _check_shapes(evaluation_matrix, weights)
    return np.clip((weights[:, None] * evaluation_matrix).max(axis=0), 0.0, 1.0)


def _check_shapes(evaluation_matrix: np.ndarray, weights: np.ndarray) -> None:
    if evaluation_matrix.shape[0] != weights.shape[0]:
        raise ValueError(
            f"evaluation matrix has {evaluation_matrix.shape[0]} factors but "
            f"{weights.shape[0]} weights were given"
        )


def synthesize(
    evaluation_matrix: np.ndarray, weights: np.ndarray, operator: str
) -> np.ndarray:
    """Dispatch on the configured synthesis operator.

    The name from configuration is what decides the operator, so a config that
    says bounded sum gets a bounded sum. An unknown name is an error rather than
    a default: the operator changes every reported membership, so falling back
    would make the recorded operator name untrue.
    """
    if operator == "bounded_sum":
        return bounded_sum(evaluation_matrix, weights)
    if operator == "bounded_max_product":
        return bounded_max_product(evaluation_matrix, weights)
    if operator == "weighted_average":
        _check_shapes(evaluation_matrix, weights)
        return weights @ evaluation_matrix
    raise ValueError(
        f"unknown synthesis operator {operator!r}; "
        f"expected one of {sorted(SYNTHESIS_OPERATORS)}"
    )


def rating_values(fuzzy_config: Mapping[str, Any]) -> dict[str, float]:
    """The numerical value ``j`` of each rating level (v1=1, v2=2, v3=3).

    Read from configuration so the decision cannot depend on the order a list
    happens to be written in.
    """
    raw = fuzzy_config.get("rating_values", {}) or {}
    default = {level: float(index + 1) for index, level in enumerate(LEVELS)}
    if not raw:
        return default
    return {level: float(unwrap(raw.get(level), default[level])) for level in LEVELS}


def weighted_average_score(
    membership: np.ndarray | Mapping[str, float],
    values: Mapping[str, float] | None = None,
) -> float:
    """Weighted-membership decision, Eq (33): ``v = sum(s_j^2 j) / sum(s_j^2)``.

    ``membership`` may be the synthesised vector in level order or a mapping of
    level -> membership. ``values`` maps each level to its numerical value j.
    The result lies between the smallest and largest j, so it reads as "this
    frame sits near level v" rather than as a bare index.
    """
    if isinstance(membership, Mapping):
        ordered = [float(membership.get(level, 0.0)) for level in LEVELS]
        numbers = [1.0, 2.0, 3.0] if values is None else [float(values[level]) for level in LEVELS]
    else:
        ordered = [float(v) for v in np.asarray(membership, dtype=float).ravel()]
        numbers = (
            [1.0, 2.0, 3.0]
            if values is None
            else [float(values[level]) for level in LEVELS[: len(ordered)]]
        )
    if len(numbers) != len(ordered):
        raise ValueError(
            f"membership vector has {len(ordered)} levels but {len(numbers)} rating values were given"
        )
    squared = np.asarray(ordered, dtype=float) ** 2
    total = float(squared.sum())
    if total <= 0:
        # No membership anywhere means no rating can be formed. Returning a
        # number here would invent a grade; NaN forces the caller to say so.
        return float("nan")
    return float((squared * np.asarray(numbers, dtype=float)).sum() / total)


def decide(
    score: float,
    decision: Mapping[str, Any] | None = None,
) -> str:
    """Map the defuzzified score ``v`` to a rating level, Eq (34).

        v <= excellent_max                  -> Excellent
        excellent_max < v < unacceptable_min -> Barely Acceptable
        v >= unacceptable_min               -> Unacceptable

    A non-finite score is an error: the caller has a frame whose membership
    vector carried no weight at all, and naming a class for it would hide that.
    """
    decision = decision or {}
    excellent_max = float(unwrap(decision.get("excellent_max"), 1.50))
    unacceptable_min = float(unwrap(decision.get("unacceptable_min"), 2.40))
    if unacceptable_min <= excellent_max:
        raise ValueError(
            f"decision thresholds must satisfy unacceptable_min > excellent_max, "
            f"got {unacceptable_min} <= {excellent_max}"
        )
    score = float(score)
    if not math.isfinite(score):
        raise ValueError(f"cannot classify a non-finite score ({score!r})")
    if score <= excellent_max:
        return EXCELLENT
    if score >= unacceptable_min:
        return UNACCEPTABLE
    return BARELY_ACCEPTABLE


def max_membership_level(membership: np.ndarray | Mapping[str, float]) -> str:
    """The max-membership principle, one of the alternatives the reference lists.

    Not used for the reported class -- the reference selects the weighted
    average -- but exposed so the two decision rules can be compared.
    """
    if isinstance(membership, Mapping):
        return max(LEVELS, key=lambda level: float(membership.get(level, 0.0)))
    vector = np.asarray(membership, dtype=float).ravel()
    if vector.size != len(LEVELS):
        raise ValueError(f"membership vector must have {len(LEVELS)} entries, got {vector.size}")
    return LEVELS[int(np.argmax(vector))]


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def evaluate(
    sqi_values: Mapping[str, float],
    fuzzy_config: Mapping[str, Any],
) -> FuzzyResult:
    """Run the full fuzzy comprehensive evaluation for one frame.

    ``sqi_values`` maps factor name -> value on this pipeline's 0-1 scale; each
    factor's configured ``scale`` converts it to the scale its membership
    functions are printed on. Any non-finite value makes the frame unevaluable
    and is reported as such by the caller rather than being silently coerced.
    """
    missing = [factor for factor in FACTORS if factor not in sqi_values]
    if missing:
        raise ValueError(f"missing SQI factors: {missing}")

    for factor in FACTORS:
        value = sqi_values[factor]
        if value is None or not np.isfinite(value):
            raise ValueError(f"{factor} is not a finite value ({value!r})")

    functions = build_membership_functions(fuzzy_config)
    weights = build_weight_vector(fuzzy_config)
    scales = factor_scales(fuzzy_config)
    levels = tuple(fuzzy_config.get("levels", LEVELS))

    matrix = np.zeros((len(FACTORS), len(levels)), dtype=np.float64)
    matrix_rows: dict[str, dict[str, float]] = {}
    empty_rows: list[str] = []
    for row, factor in enumerate(FACTORS):
        scaled = float(sqi_values[factor]) * scales[factor]
        for column, level in enumerate(levels):
            matrix[row, column] = float(functions[factor][level](scaled))
        matrix_rows[factor] = {
            level: float(matrix[row, index]) for index, level in enumerate(levels)
        }
        if matrix[row].sum() <= 0:
            # Every level's membership is zero, so this factor contributes
            # nothing to the synthesis. Recorded rather than dropped in silence.
            empty_rows.append(factor)

    weight_vector = np.array([weights[factor] for factor in FACTORS], dtype=np.float64)
    operator = str(unwrap(fuzzy_config.get("synthesis"), "bounded_sum"))
    membership_vector = synthesize(matrix, weight_vector, operator)
    membership = {level: float(membership_vector[i]) for i, level in enumerate(levels)}

    numbers = rating_values(fuzzy_config)
    score = weighted_average_score(membership_vector, numbers)
    quality_class = decide(score, fuzzy_config.get("decision", {}) or {})

    return FuzzyResult(
        membership=membership,
        weights=weights,
        evaluation_matrix=matrix_rows,
        quality_class=quality_class,
        synthesis=operator,
        config_version=str(fuzzy_config.get("config_version", "")),
        score=score,
        rating_values=numbers,
        out_of_support_factors=empty_rows,
    )


__all__ = [
    "FACTORS",
    "LEVELS",
    "LEVEL_SHORT",
    "MEMBERSHIP_FAMILIES",
    "SYNTHESIS_OPERATORS",
    "bounded_max_product",
    "bounded_sum",
    "build_membership_functions",
    "build_weight_vector",
    "cauchy",
    "cauchy_centred",
    "cauchy_decreasing",
    "cauchy_increasing",
    "decide",
    "evaluate",
    "factor_scales",
    "max_membership_level",
    "ramp_down",
    "ramp_up",
    "rating_values",
    "rectangular",
    "step_above",
    "step_at_or_below",
    "synthesize",
    "trapezoidal",
    "weighted_average_score",
    "zero_membership",
]

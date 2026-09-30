"""Fuzzy comprehensive evaluation.

    SQI values
        -> membership functions
        -> evaluation matrix R
        -> weight vector W
        -> fuzzy synthesis
        -> decision

Rating levels: V = {Excellent, Barely Acceptable, Unacceptable}
Factors:        U = {qSQI, pSQI, kSQI, basSQI}

Membership families required by IDEA.md section 27:

===========  =============
factor       family
===========  =============
qSQI         Cauchy
pSQI         trapezoidal
kSQI         rectangular
basSQI       Cauchy
===========  =============

These are NOT replaced with simple linear thresholds. Every parameter comes
from configuration, the weight vector is validated to sum to 1, and the
resulting membership vector is preserved alongside the final class.

The synthesis operator is selected by ``fuzzy.synthesis`` and dispatched for
real: ``bounded_max_product`` and ``bounded_sum`` are different computations and
give different answers, so an unrecognised name raises instead of falling back.
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


def _unwrap(param: Any, default: Any = None) -> Any:
    """Accept either a bare value or a ``{value, source}`` provenance dict."""
    if isinstance(param, Mapping):
        if "value" in param:
            return param["value"]
        return default
    return param if param is not None else default


def build_membership_functions(fuzzy_config: Mapping[str, Any]) -> dict[str, Callable[[float], dict[str, float]]]:
    """Construct one membership evaluator per SQI factor.

    Configuration gives the family per factor and explicit parameters per
    rating level, so all three levels are defined in one place and nothing is
    derived at runtime. Each evaluator returns the membership of one SQI value
    in the three levels -- the row of the evaluation matrix R for that factor.
    """
    membership_cfg = fuzzy_config.get("membership", {}) or {}
    levels = tuple(fuzzy_config.get("levels", LEVELS))
    if levels != LEVELS:
        raise ValueError(f"rating levels must be {LEVELS}, got {levels}")

    built: dict[str, Callable[[float], dict[str, float]]] = {}

    for factor in FACTORS:
        spec = membership_cfg.get(factor)
        if not spec:
            raise ValueError(f"no membership configuration for factor {factor!r}")
        family = str(spec.get("family", "")).lower()

        rows: dict[str, Callable[[float], float]] = {}
        for level in levels:
            level_spec = spec.get(level)
            if not isinstance(level_spec, Mapping):
                raise ValueError(
                    f"membership configuration for {factor}/{level} is missing; "
                    f"expected an explicit parameter set for each of {LEVELS}"
                )
            if family == "cauchy":
                xc = float(_unwrap(level_spec.get("xc")))
                gamma = float(_unwrap(level_spec.get("gamma")))
                if gamma <= 0:
                    raise ValueError(f"Cauchy gamma for {factor}/{level} must be > 0, got {gamma}")
                rows[level] = lambda x, xc=xc, gamma=gamma: cauchy(x, xc, gamma)
            elif family == "trapezoidal":
                a = float(_unwrap(level_spec.get("a")))
                b = float(_unwrap(level_spec.get("b")))
                c = float(_unwrap(level_spec.get("c")))
                d = float(_unwrap(level_spec.get("d")))
                trapezoidal(0.0, a, b, c, d)  # validate parameter ordering eagerly
                rows[level] = lambda x, a=a, b=b, c=c, d=d: trapezoidal(x, a, b, c, d)
            elif family == "rectangular":
                a = float(_unwrap(level_spec.get("a")))
                b = float(_unwrap(level_spec.get("b")))
                if b < a:
                    raise ValueError(f"rectangular bounds for {factor}/{level} are inverted: a={a}, b={b}")
                rows[level] = lambda x, a=a, b=b: rectangular(x, a, b)
            else:
                raise ValueError(
                    f"unsupported membership family {family!r} for {factor}; "
                    "expected cauchy, trapezoidal or rectangular"
                )

        built[factor] = lambda x, rows=rows: {level: rows[level](x) for level in levels}

    return built


# ---------------------------------------------------------------------------
# Weight vector
# ---------------------------------------------------------------------------
def build_weight_vector(fuzzy_config: Mapping[str, Any]) -> dict[str, float]:
    """Read and validate W. ``sum(W) == 1`` is enforced here and in tests."""
    raw = fuzzy_config.get("weights", {}) or {}
    if not raw:
        raise ValueError("fuzzy weight vector is missing from configuration")
    weights = {factor: float(_unwrap(raw.get(factor), 0.0)) for factor in FACTORS}
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
    "bounded_max_product": "max_i (w_i * R_ij)",
    "bounded_sum": "min(1, sum_i (w_i * R_ij))",
    "weighted_average": "sum_i (w_i * R_ij)",
}


def bounded_max_product(evaluation_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Max-product synthesis: ``B_j = max_i (w_i * R_ij)``.

    This is the operator named in the configuration. It is not the same as a
    weighted sum: the max-product is dominated by a single strong factor instead
    of accumulating weak support from all four, so it rates a frame whose best
    factor is clearly good higher than a sum would.
    """
    _check_shapes(evaluation_matrix, weights)
    return np.clip((weights[:, None] * evaluation_matrix).max(axis=0), 0.0, 1.0)


def bounded_sum(evaluation_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Bounded sum: ``B_j = min(1, sum_i (w_i * R_ij))``.

    The clip is a guard, not the point: with normalised rows and weights that sum
    to 1 the result already lies in [0, 1].
    """
    _check_shapes(evaluation_matrix, weights)
    return np.clip(weights @ evaluation_matrix, 0.0, 1.0)


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
    says max-product gets max-product. An unknown name is an error rather than a
    default: previously the configured operator was copied onto the result as a
    label without ever selecting a computation, so a config could claim any
    operator it liked while the numbers came from a weighted sum.
    """
    try:
        if operator == "bounded_max_product":
            return bounded_max_product(evaluation_matrix, weights)
        if operator == "bounded_sum":
            return bounded_sum(evaluation_matrix, weights)
        if operator == "weighted_average":
            _check_shapes(evaluation_matrix, weights)
            return weights @ evaluation_matrix
    except KeyError:  # pragma: no cover - unreachable, dict lookup above
        pass
    raise ValueError(
        f"unknown synthesis operator {operator!r}; "
        f"expected one of {sorted(SYNTHESIS_OPERATORS)}"
    )


def _nearest_level_weights(
    value: float,
    spec: Mapping[str, Any],
    levels: tuple[str, ...],
) -> np.ndarray:
    """Membership row for a value that lies outside every level's support.

    Picks the level whose support centre is closest to the observed value and
    assigns it full membership. This keeps the factor in the synthesis instead
    of letting an out-of-range value silently drop out of the decision, and it
    is recorded on the result so the fallback is never invisible.
    """
    best_level = levels[0]
    best_distance = math.inf
    for level in levels:
        params = spec.get(level, {}) or {}
        anchors = [
            float(_unwrap(item)) for item in params.values()
            if isinstance(item, (int, float, Mapping))
        ]
        if not anchors:
            continue
        centre = float(np.mean(anchors))
        distance = abs(value - centre)
        if distance < best_distance:
            best_distance, best_level = distance, level
    row = np.zeros(len(levels), dtype=np.float64)
    row[levels.index(best_level)] = 1.0
    return row


def decide(membership: np.ndarray, levels: tuple[str, ...] = LEVELS) -> str:
    """argmax over the membership vector. Ties resolve to the lower level."""
    return levels[int(np.argmax(membership))]


def evaluate(
    sqi_values: Mapping[str, float],
    fuzzy_config: Mapping[str, Any],
) -> FuzzyResult:
    """Run the full fuzzy comprehensive evaluation for one frame.

    ``sqi_values`` maps factor name -> value. Any non-finite value makes the
    frame unevaluable and is reported as such by the caller rather than being
    silently coerced to zero.
    """
    missing = [f for f in FACTORS if f not in sqi_values]
    if missing:
        raise ValueError(f"missing SQI factors: {missing}")

    for factor in FACTORS:
        value = sqi_values[factor]
        if value is None or not np.isfinite(value):
            raise ValueError(f"{factor} is not a finite value ({value!r})")

    functions = build_membership_functions(fuzzy_config)
    weights = build_weight_vector(fuzzy_config)
    levels = tuple(fuzzy_config.get("levels", LEVELS))
    membership_cfg = fuzzy_config.get("membership", {}) or {}
    fallback_used: list[str] = []

    matrix = np.zeros((len(FACTORS), len(levels)), dtype=np.float64)
    matrix_rows: dict[str, dict[str, float]] = {}
    for row, factor in enumerate(FACTORS):
        value = float(sqi_values[factor])
        row_membership = functions[factor](value)
        for col, level in enumerate(levels):
            matrix[row, col] = float(row_membership.get(level, 0.0))
        # Normalise the row so the three levels form a partition of unity.
        # Without this the membership functions are not mutually exclusive --
        # three independent Cauchy curves at different centres all contribute
        # to every level -- and the synthesised vector would not sum to 1, so
        # the reported memberships would not be interpretable as shares.
        total = matrix[row].sum()
        if total <= 0:
            # The value lies outside the support of all three membership
            # functions (for example a kurtosis far above every rectangular
            # band). Assigning all-zero membership would silently remove this
            # factor from the synthesis, so fall back to the level whose
            # support is nearest and record that the fallback was used.
            matrix[row] = _nearest_level_weights(value, membership_cfg.get(factor, {}), levels)
            fallback_used.append(factor)
        else:
            matrix[row] = matrix[row] / total
        matrix_rows[factor] = {level: matrix[row, i] for i, level in enumerate(levels)}

    weight_vector = np.array([weights[f] for f in FACTORS], dtype=np.float64)
    operator = str(fuzzy_config.get("synthesis", "bounded_max_product"))
    membership_vector = synthesize(matrix, weight_vector, operator)
    membership = {level: float(membership_vector[i]) for i, level in enumerate(levels)}

    return FuzzyResult(
        membership=membership,
        weights=weights,
        evaluation_matrix=matrix_rows,
        quality_class=decide(membership_vector, levels),
        synthesis=operator,
        config_version=str(fuzzy_config.get("config_version", "")),
        out_of_support_factors=fallback_used,
    )


__all__ = [
    "EXCELLENT",
    "FACTORS",
    "LEVELS",
    "SYNTHESIS_OPERATORS",
    "bounded_max_product",
    "bounded_sum",
    "build_membership_functions",
    "build_weight_vector",
    "cauchy",
    "decide",
    "evaluate",
    "rectangular",
    "synthesize",
    "trapezoidal",
]

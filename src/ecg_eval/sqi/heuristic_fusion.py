"""Simple heuristic fusion of the signal-quality indices.

Rumus (Zhao & Zhang 2018, Eq 13-16). Setiap indeks dinilai sendiri lebih dulu
menjadi salah satu dari tiga tingkat -- optimal, suspicious, unqualified --
lalu kelas kualitas sinyal ditentukan dari *hitungan* tingkat tersebut, bukan
dari nilai indeksnya.

Untuk empat indeks, yang dipakai di sini (Eq 15)::

    Excellent (E)          #optimal >= 3 dan #unqualified = 0
    Unacceptable (U)       #unqualified >= 3
                           atau (#unqualified = 2 dan #suspicious >= 1)
                           atau (#unqualified = 1 dan #suspicious = 3)
    Barely acceptable (B)  selain kondisi di atas

E diuji lebih dulu, lalu U, lalu B. Aturan untuk 2, 3 dan 5 indeks (Eq 13, 14
dan 16) ikut diimplementasikan karena artikel mendefinisikannya, tetapi jalur
produksi memakai empat indeks: artikel sendiri memilih

    U = {qSQI, pSQI, kSQI, basSQI}

sebagai kombinasi terbaik untuk tahap fuzzy, karena menambahkan indeks kelima
justru menurunkan akurasi pada kedua basis datanya.

Koefisien pada Eq (13)-(16) dipilih artikel secara ad hoc dan tidak dioptimasi,
sehingga aturannya ditulis apa adanya di sini alih-alih digeneralisasi menjadi
ambang yang tidak ada di artikel.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..models.result import (
    ACCEPTANCE_LEVELS,
    BARELY_ACCEPTABLE,
    EXCELLENT,
    OPTIMAL,
    SUSPICIOUS,
    UNACCEPTABLE,
    UNDEFINED,
    UNQUALIFIED,
    Acceptance,
    HeuristicFusionResult,
)

#: The four indices of the reference's best combination, in the paper's order.
FUSION_FACTORS = ("qSQI", "pSQI", "kSQI", "basSQI")

#: Which printed equation each index count belongs to.
FUSION_RULE_REFS: dict[int, str] = {
    2: "Zhao & Zhang (2018) Eq (13)",
    3: "Zhao & Zhang (2018) Eq (14)",
    4: "Zhao & Zhang (2018) Eq (15)",
    5: "Zhao & Zhang (2018) Eq (16)",
}


# ---------------------------------------------------------------------------
# The fusion rules, one function per index count (Eq 13-16)
# ---------------------------------------------------------------------------
def _fuse_two(optimal: int, suspicious: int, unqualified: int) -> str:
    """Eq (13)."""
    if optimal == 2:
        return EXCELLENT
    if suspicious == 2 or (optimal == 1 and suspicious == 1):
        return BARELY_ACCEPTABLE
    return UNACCEPTABLE


def _fuse_three(optimal: int, suspicious: int, unqualified: int) -> str:
    """Eq (14)."""
    if optimal >= 2 and unqualified == 0:
        return EXCELLENT
    if unqualified >= 2 or (suspicious == 2 and unqualified == 1):
        return UNACCEPTABLE
    return BARELY_ACCEPTABLE


def _fuse_four(optimal: int, suspicious: int, unqualified: int) -> str:
    """Eq (15) -- the rule this project uses."""
    if optimal >= 3 and unqualified == 0:
        return EXCELLENT
    if (
        unqualified >= 3
        or (unqualified == 2 and suspicious >= 1)
        or (unqualified == 1 and suspicious == 3)
    ):
        return UNACCEPTABLE
    return BARELY_ACCEPTABLE


def _fuse_five(optimal: int, suspicious: int, unqualified: int) -> str:
    """Eq (16)."""
    if optimal >= 4 and unqualified == 0:
        return EXCELLENT
    if (
        unqualified >= 4
        or (unqualified == 3 and suspicious >= 1)
        or (unqualified == 2 and suspicious >= 2)
        or (unqualified == 1 and suspicious == 4)
    ):
        return UNACCEPTABLE
    return BARELY_ACCEPTABLE


FUSION_RULES = {
    2: _fuse_two,
    3: _fuse_three,
    4: _fuse_four,
    5: _fuse_five,
}


def fuse_levels(
    levels: Mapping[str, str],
    *,
    n_factors: int | None = None,
) -> HeuristicFusionResult:
    """Fuse per-index acceptance levels into one quality class.

    ``levels`` maps index name -> acceptance level. An index whose level is
    :data:`~ecg_eval.models.result.UNDEFINED` has no applicable criterion, so no
    fused class exists and the result says so. Dropping that index and fusing
    the rest would be a different computation: Eq (15) counts four indices, and
    silently fusing three answers a question the reference never asked.
    """
    undefined = sorted(name for name, level in levels.items() if level not in ACCEPTANCE_LEVELS)
    n = int(len(levels) if n_factors is None else n_factors)
    counts = {
        OPTIMAL: sum(1 for level in levels.values() if level == OPTIMAL),
        SUSPICIOUS: sum(1 for level in levels.values() if level == SUSPICIOUS),
        UNQUALIFIED: sum(1 for level in levels.values() if level == UNQUALIFIED),
    }
    rule = FUSION_RULE_REFS.get(n, f"Zhao & Zhang (2018) Eq (13)-(16), n={n}")

    if undefined:
        return HeuristicFusionResult(
            quality_class=UNDEFINED,
            n_factors=n,
            counts=counts,
            levels=dict(levels),
            rule=rule,
            applied=False,
            reason=(
                "no acceptance criterion applies to: " + ", ".join(undefined)
                + "; the fusion counts four indices and none may be dropped"
            ),
        )

    if n not in FUSION_RULES:
        return HeuristicFusionResult(
            quality_class=UNDEFINED,
            n_factors=n,
            counts=counts,
            levels=dict(levels),
            rule=rule,
            applied=False,
            reason=f"the reference defines no fusion rule for {n} indices",
        )

    quality_class = FUSION_RULES[n](
        counts[OPTIMAL], counts[SUSPICIOUS], counts[UNQUALIFIED]
    )
    return HeuristicFusionResult(
        quality_class=quality_class,
        n_factors=n,
        counts=counts,
        levels=dict(levels),
        rule=rule,
        applied=True,
    )


# ---------------------------------------------------------------------------
# Heart rate, which the pSQI criterion depends on
# ---------------------------------------------------------------------------
def heart_rate_from_peaks(peaks: Sequence[float] | np.ndarray, sampling_rate: float) -> float:
    """Heart rate in bpm as ``60 / mean(R-R)``, or NaN when it cannot be found.

    The mean R-R interval is the standard definition; it is used rather than the
    median because it is what the cardiac-cycle literature the reference cites
    uses. Fewer than two peaks leaves the interval undefined and yields NaN
    rather than a fabricated rate.
    """
    values = np.asarray(peaks, dtype=float)
    if values.size < 2:
        return float("nan")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        return float("nan")
    intervals = np.diff(np.sort(values))
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        return float("nan")
    mean_rr_s = float(np.mean(intervals)) / float(sampling_rate)
    if mean_rr_s <= 0:
        return float("nan")
    return 60.0 / mean_rr_s


def estimate_heart_rate(
    peaks_a: Sequence[float] | np.ndarray,
    peaks_b: Sequence[float] | np.ndarray | None,
    *,
    sampling_rate: float,
) -> tuple[float, str]:
    """Heart rate plus which detector it came from.

    Detector A is preferred because the reference's Algorithm 1 (Hilbert with a
    dynamic adaptive threshold) is the detector its pSQI limits were calibrated
    around. Detector B is used only when A yields fewer than two peaks, and the
    returned source says so, so a substituted rate can never be silent.
    """
    rate = heart_rate_from_peaks(peaks_a, sampling_rate)
    if np.isfinite(rate):
        return rate, "detector_a"
    if peaks_b is not None:
        rate = heart_rate_from_peaks(peaks_b, sampling_rate)
        if np.isfinite(rate):
            return rate, "detector_b"
    return float("nan"), ""


# ---------------------------------------------------------------------------
# One call that judges the four indices and fuses them
# ---------------------------------------------------------------------------
def assess(
    values: Mapping[str, float],
    *,
    heart_rate_bpm: float,
    config: Any = None,
    kurtosis_definition: str | None = None,
) -> tuple[dict[str, Acceptance], HeuristicFusionResult]:
    """Judge every index against its own criterion, then fuse the levels.

    This is the reference's step 1 in one place, so the pipeline, the report and
    the tests all reach the same verdicts. ``config`` is the whole
    :class:`~ecg_eval.config.Config`; each criterion reads its own section.
    """
    from .bas_sqi import bas_sqi_acceptance
    from .k_sqi import PEARSON, k_sqi_acceptance
    from .p_sqi import p_sqi_acceptance
    from .q_sqi import q_sqi_acceptance

    definition = PEARSON if kurtosis_definition is None else str(kurtosis_definition)
    acceptances = {
        "qSQI": q_sqi_acceptance(values["qSQI"], config),
        "pSQI": p_sqi_acceptance(values["pSQI"], heart_rate_bpm, config),
        "kSQI": k_sqi_acceptance(values["kSQI"], definition, config),
        "basSQI": bas_sqi_acceptance(values["basSQI"], config),
    }
    fusion = fuse_levels(
        {name: acceptance.level for name, acceptance in acceptances.items()},
        n_factors=len(FUSION_FACTORS),
    )
    return acceptances, fusion


__all__ = [
    "FUSION_FACTORS",
    "FUSION_RULES",
    "FUSION_RULE_REFS",
    "assess",
    "estimate_heart_rate",
    "fuse_levels",
    "heart_rate_from_peaks",
]

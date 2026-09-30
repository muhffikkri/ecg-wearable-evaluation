"""Shared test fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_eval.config import load_config  # noqa: E402
from ecg_eval.models.frame import ECGFrame  # noqa: E402

#: The recording tree is deliberately untracked, so it is absent from a fresh
#: clone. Tests that exercise the real data depend on it and skip with a reason
#: rather than failing, otherwise a fresh checkout looks broken.
DATA_DIR = REPO_ROOT / "data"
RAW_RECORDING = DATA_DIR / "29-09-2026" / "Bryan"

requires_real_data = pytest.mark.skipif(
    not RAW_RECORDING.is_dir(),
    reason=(
        "the recorded Raspberry Pi session is not present; data/ is untracked by "
        "design, so a fresh clone has no recording to exercise"
    ),
)


@pytest.fixture(scope="session")
def config():
    return load_config("default")


@pytest.fixture(scope="session")
def zhao_config():
    return load_config("zhao_zhang")


def synthetic_ecg(
    n_samples: int = 2500,
    sampling_rate: float = 250.0,
    *,
    heart_rate: float = 60.0,
    noise: float = 0.05,
    baseline: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """A clean single-lead ECG-like signal for deterministic SQI tests.

    PQRST-shaped complexes on a regular RR interval plus optional noise. Not
    used to validate clinical accuracy, only to exercise the pipeline.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sampling_rate
    rr = 60.0 / heart_rate
    signal = np.zeros_like(t)

    for beat in np.arange(0, t[-1], rr):
        centre = beat + 0.1 * rr
        signal += 1.0 * np.exp(-((t - centre) ** 2) / (2 * 0.008**2))        # R
        signal += -0.15 * np.exp(-((t - centre + 0.03) ** 2) / (2 * 0.010**2))  # Q
        signal += 0.25 * np.exp(-((t - centre - 0.03) ** 2) / (2 * 0.012**2))   # S
        signal += 0.12 * np.exp(-((t - centre - 0.20) ** 2) / (2 * 0.030**2))   # T

    signal = signal + noise * rng.standard_normal(t.size) + baseline
    return signal.astype(np.float32)


@pytest.fixture
def clean_signal() -> np.ndarray:
    return synthetic_ecg(noise=0.02)


@pytest.fixture
def noisy_signal() -> np.ndarray:
    return synthetic_ecg(noise=1.2, seed=3)


@pytest.fixture
def make_frame():
    def _make(
        signal: np.ndarray | None = None,
        *,
        sampling_rate: float = 250.0,
        subject_id: str = "S01",
        session_id: str = "ses_test",
        frame_id: str = "000001",
        timestamp: str = "2026-09-29T20:00:00+07:00",
    ) -> ECGFrame:
        data = synthetic_ecg() if signal is None else np.asarray(signal, dtype=np.float32)
        if data.ndim == 1:
            data = np.column_stack([data, data * 0.5, data * 0.25])
        return ECGFrame(
            subject_id=subject_id,
            session_id=session_id,
            frame_id=frame_id,
            source_file="tests/fixture.jsonl",
            source_format="jsonl",
            timestamp=timestamp,
            sampling_rate=sampling_rate,
            duration_s=data.shape[0] / sampling_rate,
            signal=data,
            metadata={"validation": {"status": "PASS", "warnings": []}},
        )

    return _make


@pytest.fixture
def jsonl_file(tmp_path: Path) -> Path:
    """Two valid frames plus one malformed line."""
    path = tmp_path / "ses000000000001.jsonl"
    lines = []
    for index, signal in enumerate([synthetic_ecg(seed=1), synthetic_ecg(seed=2)], start=1):
        record = {
            "message_id": f"device01-ses000000000001-frame_{index:06d}",
            "device_id": "device01",
            "session_id": "ses000000000001",
            "frame_id": f"{index:06d}",
            "created_at": f"2026-09-29T20:0{index}:00+07:00",
            "sampling_rate_hz": 250.0,
            "duration_s": 10.0,
            "validation": {"status": "PASS", "warnings": []},
            "ecg": {
                "format": "samples_by_time",
                "samples": [[float(v), float(v) * 0.5, float(v) * 0.25] for v in signal],
            },
            "prediction": {"status": "PASS", "label": "Normal"},
            "system": {"cpu_usage_percent": 18.2},
        }
        lines.append(json.dumps(record))
    lines.insert(1, "{not valid json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

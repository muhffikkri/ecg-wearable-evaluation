"""Shared test fixtures."""

from __future__ import annotations

import json
import sys
import uuid
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


def build_test_recording(tmp_path: Path, *, frames: int = 2) -> Path:
    """Build a minimal recording laid out like a real Raspberry Pi session.

    Shared by the CLI tests and the ingestion tests: the derived-copy check
    needs the same recording shape the converter is pointed at.
    """
    root = tmp_path / "data" / "29-09-2026" / "S01"
    root.mkdir(parents=True)
    session_id = "session_test_000000"
    (root / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "started_at": "2026-09-28T20:26:45+07:00",
                "ended_at": "2026-09-28T20:26:55+07:00",
                "sampling_rate_hz": 250,
                "duration_per_frame_s": 1.0,
                "frame_count": frames,
                "channel_order": ["Lead I", "Lead II", "Lead III"],
                "unit": "mV",
                "status": "STOPPED",
            }
        ),
        encoding="utf-8",
    )

    rng = np.random.default_rng(7)
    for folder, suffix in (("calibrated", "_mv"), ("filtered", "_mv")):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "model_ready").mkdir(parents=True, exist_ok=True)
    (root / "predictions").mkdir(parents=True, exist_ok=True)

    for index in range(1, frames + 1):
        name = f"frame_{index:06d}"
        measurement_id = str(uuid.uuid4())
        signal = rng.normal(0, 0.1, size=(250, 3)).astype(np.float32)

        shared = {
            "source_frame": name,
            "dtype": "float32",
            "shape": [250, 3],
            "sample_count": 250,
            "sample_rate_hz": 250,
            "duration_seconds": 1.0,
            "unit": "mV",
            "channel_order": ["Lead I", "Lead II", "Lead III"],
            "created_at_utc": f"2026-09-28T13:27:0{index}.000000Z",
            "source_metadata": {"measurement_id": measurement_id, "device_id": "device01"},
        }
        np.save(root / "calibrated" / f"{name}{suffix}.npy", signal)
        (root / "calibrated" / f"{name}{suffix}.json").write_text(
            json.dumps(shared), encoding="utf-8"
        )
        np.save(root / "filtered" / f"{name}{suffix}.npy", signal)
        (root / "filtered" / f"{name}{suffix}.json").write_text(
            json.dumps({**shared, "source_metadata": {**shared["source_metadata"], "measurement_id": measurement_id}}),
            encoding="utf-8",
        )

        np.save(root / "model_ready" / f"{name}_input.npy", signal)
        (root / "model_ready" / f"{name}_input.json").write_text(
            json.dumps(
                {
                    "frame_id": name,
                    "channel_order": ["Lead I", "Lead II", "Lead III"],
                    "sample_rate_hz": 250,
                    "samples_per_channel": 250,
                    "shape": [250, 3],
                    "dtype": "float32",
                    "unit": "mV",
                    "source_file": f"/device/{session_id}/filtered/{name}_mv.npy",
                    "validation": {"status": "PASS", "warnings": []},
                    "signal_quality": {"status": "PASS", "reasons": []},
                }
            ),
            encoding="utf-8",
        )
        (root / "predictions" / f"{name}_prediction.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "frame_id": f"{index:06d}",
                    "source_file": f"/device/{session_id}/model_ready/{name}_input.npy",
                    "prediction": "Normal",
                    "confidence_percent": 99.7,
                    "probabilities": {"Normal": 99.7, "AF": 0.1},
                    "threshold": 0.5,
                    "latency_ms": 250.0,
                    "runtime": "ai-edge-litert",
                }
            ),
            encoding="utf-8",
        )
    return root

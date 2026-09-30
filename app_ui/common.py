"""Shared Streamlit session state, caching and page scaffolding.

The app is a presentation and orchestration layer only. Every scientific
calculation lives in ``src/ecg_eval/`` (IDEA.md section 64).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from . import REPO_ROOT

from ecg_eval import REFERENCE, __version__
from ecg_eval.analysis import FrameAnalyzer, ResultCache, run_analysis
from ecg_eval.annotation import storage as annotation_storage
from ecg_eval.config import available_configs, load_config
from ecg_eval.ingestion import dataset_summary, ingest
from ecg_eval.visualization import sqi_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("ecg_eval.app")

SCOPE_NOTE = (
    "**Scope.** This application evaluates **ECG signal quality** acquired by the "
    "ECGRHYTHMIA wearable prototype under three static body positions. It is **not** a "
    "diagnostic tool. No result here indicates arrhythmia, cardiac disease or any other "
    "clinical condition."
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Ingesting dataset...")
def get_dataset(data_dir: str, config_name: str, fingerprint: str) -> Any:
    """Ingest once per session; Streamlit keeps the object alive across reruns."""
    config = load_config(config_name)
    logger.info("ingesting %s (config %s)", data_dir, config_name)
    return ingest(Path(data_dir), config)


@st.cache_data(show_spinner="Loading annotations...")
def get_annotations(directory: str, mtime: float) -> list[Any]:
    return annotation_storage.load_annotations(Path(directory))


@st.cache_resource(show_spinner="Preparing result cache...")
def get_cache(directory: str, enabled: bool) -> ResultCache:
    return ResultCache(Path(directory) if directory else None, enabled=enabled)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "config_name": "default",
        "config": None,
        "dataset": None,
        "annotations": None,
        "annotations_dirty": False,
        "analysis": None,
        "selected_subject": None,
        "selected_session": None,
        "selected_frame_index": 0,
        "preprocessing_enabled": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


STAGE_LABELS = {
    "wavelet": "Wavelet denoising (db4, L4)",
    "baseline": "Median baseline removal (kernel 51)",
    "bandpass": "Butter bandpass (0.5-45 Hz, order 4)",
    "resample": "Polyphase resample",
    "notch": "Notch 50 Hz",
    "normalize": "Z-score + clip (normalisation)",
}


def _stage_order() -> tuple[str, ...]:
    """Canonical stage order, so execution is reproducible."""
    from ecg_eval.preprocessing import STAGE_ORDER

    return STAGE_ORDER


def defaults_chain(config: dict) -> dict:
    """The chain as configured in YAML, with the master switch attached."""
    chain = dict(config)
    chain["applied"] = bool(config.get("applied", False))
    return chain


def _stage_panel(config: dict) -> dict:
    """Render the filter toggles and edit the PENDING selection.

    Returns the pending chain. Nothing here changes what the analysis uses;
    only the Activate button commits it (see :func:`activate_chain`).
    """
    pending = st.session_state.get("pp_pending") or defaults_chain(config)

    for stage in _stage_order():
        spec = config.get(stage)
        spec = spec if isinstance(spec, dict) else {}
        current = bool(pending.get(stage, {}).get("enabled", False))
        enabled = st.checkbox(
            STAGE_LABELS.get(stage, stage),
            value=current,
            key=f"pp_stage_{stage}",
        )
        pending[stage] = {**spec, "enabled": enabled}

    pending["applied"] = True
    st.session_state.pp_pending = pending
    return pending


def active_chain() -> dict:
    """The chain the analysis actually runs, i.e. the ACTIVATED one."""
    return st.session_state.get("pp_active") or {}


def activate_chain(pending: dict) -> dict:
    """Commit the pending selection and drop results computed without it."""
    chain = dict(pending)
    chain["applied"] = True
    st.session_state.pp_active = chain
    # Results computed on the previous chain are no longer valid.
    st.session_state.analysis = None
    return chain


def reset_chain(config: dict) -> dict:
    """Restore the configured defaults into both pending and active."""
    chain = defaults_chain(config)
    st.session_state.pp_pending = dict(chain)
    st.session_state.pp_active = dict(chain)
    for stage in _stage_order():
        spec = config.get(stage)
        spec = spec if isinstance(spec, dict) else {}
        st.session_state[f"pp_stage_{stage}"] = bool(spec.get("enabled", False))
    st.session_state.analysis = None
    return chain

def sidebar() -> dict[str, Any]:
    """Sidebar: dataset, subject, session and analysis configuration."""
    with st.sidebar:
        st.markdown("### ECGRHYTHMIA ECG Evaluation")
        st.caption(f"v{__version__}")

        config_names = available_configs()
        default_index = config_names.index("default") if "default" in config_names else 0
        chosen = st.selectbox(
            "Analysis configuration", config_names, index=default_index,
            help="Configuration selects the R-peak detector pairing and every SQI parameter.",
        )
        if chosen != st.session_state.config_name:
            st.session_state.config_name = chosen
            st.session_state.analysis = None
        config = load_config(chosen)
        st.session_state.config = config

        st.divider()
        st.markdown("#### Dataset")
        data_dir = REPO_ROOT / str(config.get("paths.data_dir", "data"))
        annotations_dir = REPO_ROOT / str(config.get("paths.annotations_dir", "annotations"))
        results_dir = REPO_ROOT / str(config.get("paths.results_dir", "results"))
        processed_dir = REPO_ROOT / str(config.get("paths.processed_dir", "processed"))
        st.caption(f"`{data_dir.relative_to(REPO_ROOT) if data_dir.is_relative_to(REPO_ROOT) else data_dir}`")

        dataset = get_dataset(str(data_dir), chosen, config.scientific_fingerprint())
        st.session_state.dataset = dataset

        annotations_file = annotations_dir / annotation_storage.ANNOTATIONS_FILE
        mtime = annotations_file.stat().st_mtime if annotations_file.exists() else 0.0
        # Reload from disk ONLY when the file actually changed. Reloading on
        # every rerun silently discarded unsaved assignments the moment the user
        # navigated to another tab, because Streamlit reruns this script on
        # every interaction.
        if st.session_state.get("annotations_source_mtime") != mtime:
            st.session_state.annotations = get_annotations(str(annotations_dir), mtime)
            st.session_state.annotations_source_mtime = mtime

        summary = dataset_summary(dataset)
        st.metric("Subjects", summary["subjects"].__len__())
        st.metric("Frames", summary["frames"])
        if summary["malformed"]:
            st.warning(f"{summary['malformed']} malformed record(s)")
        if summary.get("derived_datasets"):
            # A dataset generated from a recording already in the tree would
            # otherwise count the same frames twice, so it is skipped. Say so
            # rather than letting the count look unexplained.
            st.info(
                f"{len(summary['derived_datasets'])} generated dataset(s) skipped: "
                "already ingested from their raw recordings."
            )

        st.divider()
        st.markdown("#### Method")
        method = "Zhao-Zhang reproduction" if config.get("rpeak.primary_detector_b") != "pan_tompkins" else "Pan-Tompkins adaptation"
        st.caption(f"{sqi_label('qSQI')}: {method}")
        st.caption(
            f"Detectors: `{config.get('rpeak.primary_detector_a')}` vs "
            f"`{config.get('rpeak.primary_detector_b')}`"
        )
        pending = config.pending_parameters()
        if pending:
            st.caption(f"⚠️ {len(pending)} parameter(s) pending verification against the paper")
        if config.adaptation_note:
            st.info("Adaptation: " + config.adaptation_note)

        cache_enabled = bool(config.get("cache.enabled", True))
        cache = get_cache(str(processed_dir / "cache"), cache_enabled)
        st.session_state.cache = cache

        st.divider()
        st.markdown("#### Preprocessing")
        st.caption(
            "Tick the filters you want, then press Activate filters. The analysis runs "
            "on the ACTIVATED selection and re-runs with it once activated. Stage "
            "order and the DSP implementations match the ECG dashboard "
            "(`templates/preprocessing.py`)."
        )
        configured = config.section("preprocessing")

        # First visit: start from the configured defaults.
        if "pp_active" not in st.session_state:
            defaults = defaults_chain(configured)
            st.session_state.pp_pending = dict(defaults)
            st.session_state.pp_active = dict(defaults)
            for stage_name in _stage_order():
                stage_spec = configured.get(stage_name)
                stage_spec = stage_spec if isinstance(stage_spec, dict) else {}
                st.session_state[f"pp_stage_{stage_name}"] = bool(
                    stage_spec.get("enabled", False)
                )

        pending = _stage_panel(configured)

        pending_on = [s for s in _stage_order() if pending.get(s, {}).get("enabled")]
        active = active_chain()
        active_on = [s for s in _stage_order() if active.get(s, {}).get("enabled")]
        dirty = pending_on != active_on

        if active_on:
            st.caption("Active filters: " + ", ".join(active_on))
        else:
            st.caption("No filter active. The analysis measures the raw signal.")
        if dirty:
            st.warning(
                "Selection changed but not activated. Press **Activate filters** so "
                "the analysis uses it."
            )

        b1, b2 = st.columns([2, 1])
        if b1.button(
            "Activate filters",
            type="primary",
            disabled=not dirty,
            help="Apply the ticked filters to the analysis.",
            key="pp_activate",
        ):
            activate_chain(pending)
            st.rerun()
        if b2.button("Reset", key="pp_reset", help="Restore the configured defaults."):
            reset_chain(configured)
            st.rerun()

    # The analysis consumes the ACTIVATED chain only. Un-ticking a box changes
    # the pending selection and nothing else, so a run can never silently use a
    # filter the user has not activated.
    chain = active_chain()
    chain_on = bool(chain) and any(chain.get(s, {}).get("enabled") for s in _stage_order())
    st.session_state.preprocessing_enabled = chain_on

    # A change of the activated chain invalidates any previous run, so drop it
    # here before the page renders.
    fingerprint = json.dumps(chain, sort_keys=True, default=str)
    if st.session_state.analysis is not None:
        if st.session_state.get("preprocessing_run_switch") != fingerprint:
            st.session_state.analysis = None
    st.session_state.preprocessing_run_switch = fingerprint

    return {
        "config": config,
        "dataset": dataset,
        "annotations": st.session_state.annotations,
        "data_dir": data_dir,
        "annotations_dir": annotations_dir,
        "results_dir": results_dir,
        "processed_dir": processed_dir,
        "cache": cache,
        "summary": summary,
        "preprocessing_enabled": chain_on,
        # activated per-stage chain, passed to the analyzer so the SQI
        # computation uses exactly the filters shown as active.
        "preprocessing_stages": chain or None,
        "preprocessing_active": [s for s in _stage_order() if chain.get(s, {}).get("enabled")],
    }


def header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown(SCOPE_NOTE)
    st.markdown("")


def results_frame(analysis: dict[str, Any]):
    """The frame-level DataFrame, built once per analysis run."""
    import pandas as pd

    if analysis is None:
        return pd.DataFrame()
    from ecg_eval.analysis import results_to_frame

    return results_to_frame(analysis["results"])


def run_or_get_analysis(
    dataset: Any, annotations: list[Any], config: Any, cache: ResultCache
) -> dict[str, Any]:
    """Execute the analysis, reporting progress (IDEA.md section 49)."""
    if st.session_state.analysis is not None:
        return st.session_state.analysis

    progress_bar = st.progress(0.0, text="Preparing analysis...")
    status = st.empty()

    def on_progress(done: int, total: int, result: Any) -> None:
        progress_bar.progress(min(1.0, done / max(1, total)), text=f"Analysed {done} / {total} frames")
        status.caption(
            f"Latest: {result.subject_id} · {result.position} · frame {result.frame_id} → "
            f"{sqi_label('qSQI')} {result.q_sqi:.3f}, {sqi_label('pSQI')} {result.p_sqi:.3f}, "
            f"{sqi_label('kSQI')} {result.k_sqi:.3f}, {sqi_label('basSQI')} {result.bas_sqi:.3f} "
            f"→ {result.quality_class}"
        )

    analysis = run_analysis(
        dataset, annotations, config, cache=cache, progress=on_progress,
        preprocessing_enabled=st.session_state.preprocessing_enabled,
        preprocessing_stages=st.session_state.get("pp_active") or None,
    )
    cache.flush()
    progress_bar.empty()
    status.empty()
    st.session_state.analysis = analysis
    return analysis


__all__ = [
    "REFERENCE",
    "REPO_ROOT",
    "SCOPE_NOTE",
    "FrameAnalyzer",
    "get_annotations",
    "get_cache",
    "get_dataset",
    "header",
    "init_state",
    "load_config",
    "results_frame",
    "run_or_get_analysis",
    "sidebar",
]

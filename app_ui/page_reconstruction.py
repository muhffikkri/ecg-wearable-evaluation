"""Reconstruction page: raw Raspberry Pi recording -> canonical frames -> JSONL.

This replaces the old Conversion tab. The three raw folders are not two
representations of one recording to be converted back and forth; they are
complementary evidence about the same frames, and this page shows how each
canonical frame was assembled from them before anything is written out.

The recording under ``data/`` is only ever read. Generated JSONL goes to
``processed/``, or is offered as a download (IDEA-REVISED.md sections 8 and 10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ecg_eval.jsonl import UNAVAILABLE_IN_RAW, build_record, template_fields, write_jsonl
from ecg_eval.reconstruction import reconstruct_directory

from .common import header

STATUS_ORDER = ("VERIFIED", "WARNING", "UNRESOLVED", "ERROR")


def _recording_dirs(data_dir: Path) -> list[Path]:
    """Find recording folders under the raw tree.

    Accepts both the documented ``data/<date>/raw/<subject>`` layout and the flat
    ``data/<date>/<subject>`` layout the recorded data actually uses, by looking
    for the folders a Raspberry Pi recording actually contains.
    """
    if not data_dir.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(data_dir.rglob("*")):
        if not child.is_dir():
            continue
        names = {entry.name for entry in child.iterdir() if entry.is_dir()}
        if {"calibrated", "filtered", "model_ready"} & names:
            found.append(child)
    return found


def _mappings_table(mappings: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame_id": m.frame_id,
                "status": m.status,
                "signal_source": m.signal_source,
                "method": m.method,
                "sources": ", ".join(
                    f"{folder}#{ref.frame_number}" for folder, ref in m.sources.items()
                ),
                "warnings": "; ".join(m.warnings),
                "reasons": "; ".join(m.reasons),
            }
            for m in mappings
        ]
    )


def _provenance_table(frame: Any) -> pd.DataFrame:
    provenance = dict(frame.provenance)
    sources = provenance.pop("sources", []) or []
    rows = [{"key": "frame_id", "value": frame.frame_id}]
    rows.append({"key": "frame_id_raw", "value": provenance.pop("frame_id_raw", "")})
    for key, value in provenance.items():
        rows.append({"key": key, "value": json.dumps(value, default=str) if isinstance(value, (dict, list)) else value})
    for ref in sources:
        rows.append(
            {
                "key": f"source:{ref.folder}",
                "value": f"frame {ref.frame_number} from {', '.join(ref.files) or 'n/a'}",
            }
        )
    return pd.DataFrame(rows)


def render(ctx: dict[str, Any]) -> None:
    header(
        "Reconstruction",
        "Assemble canonical ECG frames from a raw Raspberry Pi recording, then generate the website dataset.",
    )
    data_dir = Path(ctx["data_dir"])
    processed_dir = Path(ctx["processed_dir"])

    st.info(
        "`calibrated/`, `filtered/` and `model_ready/` hold complementary evidence about the "
        "same frames, not two formats to convert between. Each frame is rebuilt by matching the "
        "pointers the recording actually stores. A frame is only marked VERIFIED when that "
        "evidence resolves; nothing is matched on frame numbering alone, and an unresolved frame "
        "is reported rather than guessed."
    )

    candidates = _recording_dirs(data_dir)
    if not candidates:
        st.warning(
            f"No Raspberry Pi recording found under `{data_dir}`. A recording is a folder "
            "containing at least one of `calibrated/`, `filtered/` or `model_ready/`."
        )
        return

    labels = [str(path.relative_to(data_dir)) for path in candidates]
    selected = st.selectbox("Recording", labels)
    recording = candidates[labels.index(selected)]

    @st.cache_resource(show_spinner="Reconstructing frames...")
    def _reconstruct(path_str: str, fingerprint: str) -> Any:
        return reconstruct_directory(Path(path_str))

    result = _reconstruct(str(recording), ctx.get("config_fingerprint", ""))

    st.subheader("Reconstruction outcome")
    counts = pd.Series([m.status for m in result.mappings]).value_counts()
    metric_columns = st.columns(len(STATUS_ORDER))
    for column, status in zip(metric_columns, STATUS_ORDER):
        column.metric(status, int(counts.get(status, 0)))
    extra = st.columns(3)
    extra[0].metric("Frames", len(result.frames))
    extra[1].metric("Dropped", len(result.dropped))
    extra[2].metric("Errors", len(result.errors))

    if result.errors:
        st.error(f"{len(result.errors)} error(s) during reconstruction:")
        st.dataframe(pd.DataFrame(result.errors), use_container_width=True, hide_index=True)
    if result.warnings:
        with st.expander(f"{len(result.warnings)} warning(s)"):
            st.dataframe(pd.DataFrame(result.warnings), use_container_width=True, hide_index=True)

    st.subheader("Frame mapping")
    mappings = _mappings_table(result.mappings)
    st.dataframe(mappings, use_container_width=True, hide_index=True)

    unresolved = mappings[mappings["status"] == "UNRESOLVED"] if not mappings.empty else mappings
    if not unresolved.empty:
        st.warning(
            "Some frames could not be matched from recorded evidence. They are excluded from "
            "generation rather than paired by frame number."
        )

    if not result.frames:
        st.error("No canonical frames were produced, so there is nothing to generate.")
        return

    frame_ids = [f.frame_id for f in result.frames]
    chosen_id = st.selectbox("Inspect frame", frame_ids, key="reconstruction_frame")
    frame = result.frames[frame_ids.index(chosen_id)]

    left, right = st.columns(2)
    with left:
        st.markdown("**Provenance**")
        st.dataframe(_provenance_table(frame), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Merged metadata**")
        st.dataframe(
            pd.DataFrame(
                [{"key": k, "value": json.dumps(v, default=str) if isinstance(v, (dict, list)) else v}
                 for k, v in frame.metadata.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -- JSONL generation ------------------------------------------------
    st.subheader("Generate website dataset")

    built = build_record(frame)
    with st.expander("Record schema and field origins"):
        st.write("Fields in `templates/json-web.jsonl`:")
        st.code(", ".join(template_fields()), language="text")
        st.dataframe(
            pd.DataFrame([{"field": k, "origin": v} for k, v in built.origins.items()]),
            use_container_width=True,
            hide_index=True,
        )
        missing = [name for name in UNAVAILABLE_IN_RAW]
        st.write("Present in the template but absent from the raw recording:")
        st.code(", ".join(missing), language="text")
        st.caption(
            "These are omitted rather than filled with invented values. A generated "
            "`system.cpu_usage_percent` would be indistinguishable from a measurement the "
            "device never made."
        )

    filename = st.text_input(
        "Output filename",
        value=f"{result.subject_id}_{result.session_id}.jsonl",
        key="reconstruction_filename",
    )
    write_manifest = st.checkbox("Write manifest sidecar", value=True, key="reconstruction_manifest")
    allow_invalid = st.checkbox(
        "Include frames that fail validation",
        value=False,
        key="reconstruction_allow_invalid",
        help="Off by default, so a dataset is never published with known-bad records in it.",
    )

    destination = st.radio(
        "Destination",
        ["Preview only (no files)", "Write to processed/"],
        horizontal=True,
        key="reconstruction_destination",
        help="The recording under data/ is never written to.",
    )
    write_files = destination == "Write to processed/"

    if st.button("Generate", type="primary", key="reconstruction_generate"):
        try:
            write_result = write_jsonl(
                result.frames,
                processed_dir,
                source_root=recording,
                filename=filename,
                write_manifest=write_manifest,
                allow_invalid=allow_invalid,
                dry_run=not write_files,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            status_column, message = st.columns(2)
            status_column.success(f"Status: {write_result.status}")
            message.caption(
                f"{write_result.written} of {len(result.frames)} frame(s) written, "
                f"{len(write_result.skipped)} skipped."
            )
            if write_result.dry_run:
                st.info(
                    "Preview only, so nothing was written. Choose \"Write to processed/\" "
                    "to create the file."
                )
            if write_result.output_path:
                output_path = Path(write_result.output_path)
                # In a dry run nothing is on disk yet, so offer the payload
                # that WAS produced rather than reading a file that does not
                # exist. Reading it unconditionally crashed the page.
                if output_path.exists():
                    payload = output_path.read_text(encoding="utf-8")
                else:
                    payload = write_result.preview_text()
                st.download_button(
                    "Download JSONL",
                    payload,
                    file_name=output_path.name,
                    mime="application/x-ndjson",
                )
                if write_result.manifest_path:
                    manifest_path = Path(write_result.manifest_path)
                    if manifest_path.exists():
                        st.download_button(
                            "Download manifest",
                            manifest_path.read_text(encoding="utf-8"),
                            file_name=manifest_path.name,
                            mime="application/json",
                        )
            if write_result.skipped:
                st.warning("Skipped records:")
                st.dataframe(
                    pd.DataFrame(write_result.skipped), use_container_width=True, hide_index=True
                )
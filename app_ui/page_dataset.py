"""Dataset page: inventory and waveform inspection (IDEA.md sections 12, 13, 47)."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ecg_eval.visualization import ecg_figure, multilead_figure

from .common import header


def render(ctx: dict) -> None:
    header("Dataset", "What data do I have?")
    dataset = ctx["dataset"]
    config = ctx["config"]

    if dataset is None or not dataset.frames_by_session:
        st.error("No recordings were ingested. Check the data directory in the sidebar configuration.")
        return

    inventory = dataset.inventory
    coverage = inventory.coverage()

    # -- coverage ------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Subjects", f"{coverage['actual_subjects']} / {coverage['expected_subjects']}")
    col2.metric("Sessions", len(dataset.sessions()))
    col3.metric("Frames", f"{coverage['actual_frames']} / {coverage['expected_frames']}")
    col4.metric("Malformed records", coverage["malformed_records"])

    st.caption(
        f"Sampling rate(s): {', '.join(str(r) for r in dataset.sampling_rates())} Hz · "
        f"source format(s): {', '.join(dataset.source_formats())} · "
        f"expected {coverage['expected_frames']} frames is {coverage['expected_design']} "
        f"of {config.get('signal.frame_duration_s', 10.0):.0f} s"
    )

    st.subheader("Inventory")
    st.dataframe(
        pd.DataFrame(inventory.rows()),
        use_container_width=True,
        hide_index=True,
    )

    sessions_with_issues = [s for s in inventory.sessions if s.status != "OK"]
    if sessions_with_issues:
        with st.expander(f"Validation report ({len(sessions_with_issues)} session(s) need attention)", expanded=True):
            for session in sessions_with_issues:
                st.markdown(f"**{session.subject_id} / {session.session_id}** — {session.status}")
                for issue in session.issues:
                    st.markdown(f"- {issue}")
            if inventory.malformed:
                st.markdown("**Malformed records**")
                st.dataframe(
                    pd.DataFrame([m.to_dict() for m in inventory.malformed]),
                    use_container_width=True, hide_index=True,
                )
    if dataset.warnings:
        with st.expander("Ingestion warnings"):
            for warning in dataset.warnings:
                st.markdown(f"- {warning}")

    # -- frame selection ----------------------------------------------
    st.divider()
    st.subheader("Waveform inspection")
    subject = st.selectbox("Subject", dataset.subjects())
    subject_sessions = sorted({s for (s, _) in dataset.sessions() if s == subject})
    session = st.selectbox("Session", subject_sessions)
    frames = dataset.session_frames(subject, session)
    if not frames:
        st.warning("This session has no readable frames.")
        return

    frame = st.selectbox(
        "Frame", list(range(len(frames))),
        format_func=lambda i: frames[i].label(),
    )
    chosen = frames[frame]

    info = chosen.summary()
    cols = st.columns(6)
    cols[0].metric("Sampling rate", f"{info['sampling_rate']:.0f} Hz")
    cols[1].metric("Duration", f"{info['duration_s']:.2f} s")
    cols[2].metric("Samples", f"{info['n_samples']}")
    cols[3].metric("Channels", f"{info['n_channels']}")
    cols[4].metric("Source", info["source_format"])
    cols[5].metric("Device validation", info["device_validation"] or "n/a")

    issues = chosen.signal_issues()
    if issues:
        st.warning("Signal issues: " + "; ".join(issues))

    lead_index = config.get("signal.channel_order", ["Lead I", "Lead II", "Lead III"])
    analysis_lead = str(config.get("signal.analysis_lead", "Lead II"))
    highlight = lead_index.index(analysis_lead) if analysis_lead in lead_index else min(1, chosen.n_channels - 1)

    st.plotly_chart(
        ecg_figure(
            chosen.signal, chosen.sampling_rate,
            lead_names=chosen.channel_order(), highlight_lead=highlight,
            title=f"{chosen.label()} — RAW signal (x: seconds, y: mV)",
        ),
        use_container_width=True,
    )
    st.caption(
        "Trace label **RAW** means the signal exactly as acquired. A **PREPROCESSED** "
        "trace only appears after preprocessing has been explicitly executed."
    )

    with st.expander("All leads"):
        st.plotly_chart(
            multilead_figure(
                chosen.signal, chosen.sampling_rate,
                lead_names=chosen.channel_order(),
                title=f"{chosen.label()} — per-lead view",
            ),
            use_container_width=True,
        )

    with st.expander("Frame metadata"):
        st.json(
            {
                "internal_id": chosen.internal_id,
                "source_file": chosen.source_file,
                "timestamp": chosen.timestamp,
                "date_folder": chosen.date_folder,
                "record_index": chosen.record_index,
                "channel_order": list(chosen.channel_order()),
                "provenance": chosen.provenance,
                "metadata": {
                    k: v for k, v in chosen.metadata.items()
                    if k not in {"system", "stress_test", "network", "prediction"}
                },
            },
            expanded=False,
        )
        st.caption(
            "Device-side prediction and telemetry are retained for traceability but are "
            "never used by the SQI evaluation."
        )
        st.download_button(
            "Download frame metadata (JSON)",
            json.dumps(chosen.summary(), indent=2, default=str),
            file_name=f"{chosen.subject_id}_{chosen.session_id}_{chosen.frame_id}_meta.json",
            mime="application/json",
        )

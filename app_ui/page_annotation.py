"""Annotation page: assign body positions to frames (IDEA.md sections 14-17, 48)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ecg_eval.annotation import manager, storage
from ecg_eval.models.annotation import ASSIGNABLE_LABELS, STATIC_POSITIONS, TRANSITION, UNLABELED
from ecg_eval.visualization import ecg_figure

from .common import header

LABEL_HELP = {
    "SUPINE": "Lying down",
    "SITTING": "Seated, upright",
    "STANDING": "Upright on feet",
    TRANSITION: "Movement between positions — excluded from the position comparison",
}


def _timeline(labels: list[str], current: str) -> None:
    """Colour strip showing every frame's label."""
    colors = {
        "SUPINE": "#1565c0", "SITTING": "#6a1b9a", "STANDING": "#ef6c00",
        TRANSITION: "#9e9e9e", UNLABELED: "#eceff1",
    }
    st.markdown(
        "<div style='display:flex;width:100%;height:26px;border-radius:4px;overflow:hidden;'>"
        + "".join(
            f"<div title='frame {i}: {label}' style='flex:1;background:{colors.get(label, '#eceff1')};"
            "border-right:1px solid #ffffff'></div>"
            for i, label in enumerate(labels)
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    legend = " · ".join(
        f"<span style='color:{color}'>■</span> {label}" for label, color in colors.items()
    )
    st.markdown(
        f"<div style='font-size:0.75rem;color:#78909c'>{legend} · "
        f"highlighted: {current}</div>",
        unsafe_allow_html=True,
    )


def render(ctx: dict) -> None:
    header("Annotation", "Mark which frames were recorded supine, sitting or standing.")
    dataset = ctx["dataset"]
    annotations = ctx["annotations"]
    config = ctx["config"]

    if dataset is None or not dataset.frames_by_session:
        st.error("No recordings were ingested.")
        return

    # -- selectors -----------------------------------------------------
    subject = st.selectbox("Subject", dataset.subjects(), key="ann_subject")
    subject_sessions = sorted({s for (s, _) in dataset.sessions() if s == subject})
    session = st.selectbox("Session", subject_sessions, key="ann_session")
    frames = dataset.session_frames(subject, session)
    n_frames = len(frames)

    labels = manager.labels_for_frames(annotations, subject, session, n_frames)
    counts = manager.per_position_counts(annotations, subject, session, n_frames)

    st.subheader(f"{subject} · {session} — {n_frames} frames")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SUPINE", counts["SUPINE"])
    c2.metric("SITTING", counts["SITTING"])
    c3.metric("STANDING", counts["STANDING"])
    c4.metric("TRANSITION", counts["TRANSITION"])
    c5.metric("Unlabeled", counts["UNLABELED"])

    st.caption(
        f"{counts['SUPINE'] + counts['SITTING'] + counts['STANDING']} frames are eligible for "
        "the position comparison. TRANSITION frames are stored but never counted as a "
        "static condition, so movement artifacts during posture changes are excluded."
    )

    _timeline(labels, "hover a segment for its frame index")

    # -- range assignment ----------------------------------------------
    st.divider()
    st.subheader("Assign a range")
    mode = st.radio("Address frames by", ["Frame index", "Time"], horizontal=True)

    left, right = st.columns(2)
    with left:
        start_frame = st.number_input(
            "Start frame", min_value=0, max_value=max(0, n_frames - 1), value=0, step=1,
        )
    with right:
        end_frame = st.number_input(
            "End frame (inclusive)", min_value=0, max_value=max(0, n_frames - 1),
            value=min(n_frames - 1, 5), step=1,
        )
    if end_frame < start_frame:
        st.info("End frame is before start frame; the range will be normalised on save.")

    start_time = end_time = None
    if mode == "Time":
        start_time = st.text_input(
            "Start timestamp (ISO-8601, optional)",
            value=frames[int(start_frame)].timestamp or "",
            help="Frames carry their own created_at timestamp. Fill this in when your notes "
                 "refer to a wall-clock time rather than a frame index.",
        ) or None
        end_time = st.text_input(
            "End timestamp (ISO-8601, optional)",
            value=frames[int(end_frame)].timestamp or "",
        ) or None

    label = st.selectbox(
        "Label", ASSIGNABLE_LABELS,
        format_func=lambda value: f"{value} — {LABEL_HELP[value]}",
    )
    note = st.text_input("Note (optional)", value="")

    b1, b2, b3 = st.columns([1, 1, 3])
    assign_clicked = b1.button("Assign range", type="primary")
    clear_clicked = b2.button("Clear range")

    if assign_clicked:
        manager.assign(
            annotations, subject, session, label,
            int(start_frame), int(end_frame),
            start_time=start_time, end_time=end_time, note=note,
        )
        st.session_state.annotations = annotations
        st.success(
            f"Assigned {label} to frames {int(min(start_frame, end_frame))}–{int(max(start_frame, end_frame))}. "
            "Remember to save."
        )
    if clear_clicked:
        removed = manager.clear_range(annotations, subject, session, int(start_frame), int(end_frame))
        st.session_state.annotations = annotations
        st.info(f"Cleared {removed} segment(s) in that range. Remember to save.")

    # -- save ----------------------------------------------------------
    st.divider()
    left, right = st.columns([1, 2])
    if left.button("Save annotations", type="primary"):
        path = storage.save_annotations(ctx["annotations_dir"], annotations)
        st.success(f"Saved to `{path.relative_to(ctx['annotations_dir'].parent.parent)}`")
        st.rerun()
    with right:
        st.caption(
            f"Stored separately from the raw data in `annotations/annotations.json` plus a "
            f"flat `annotation_manifest.csv`. {storage.summary(annotations)['segments']} segment(s) "
            "currently held in session state."
        )

    problems = manager.validate(annotations, dataset.frames_by_session)
    if problems:
        st.warning("Annotation validation issues:")
        for problem in problems:
            st.markdown(f"- {problem}")

    # -- current segments ----------------------------------------------
    st.divider()
    st.subheader("Current segments")
    record = storage.annotation_index(annotations).get((subject, session))
    if record is None or not record.segments:
        st.info("No segments annotated for this session yet.")
    else:
        rows = []
        for seg in record.segments:
            rows.append(
                {
                    "label": seg.label,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "n_frames": seg.n_frames,
                    "start_time": seg.start_time or "",
                    "end_time": seg.end_time or "",
                    "note": seg.note,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # -- preview -------------------------------------------------------
    st.divider()
    st.subheader("Preview a frame")
    index = st.slider("Frame", 0, max(0, n_frames - 1), value=0)
    frame = frames[index]
    st.caption(f"Current label: **{labels[index]}**")

    lead_order = config.get("signal.channel_order", ["Lead I", "Lead II", "Lead III"])
    analysis_lead = str(config.get("signal.analysis_lead", "Lead II"))
    highlight = lead_order.index(analysis_lead) if analysis_lead in lead_order else min(1, frame.n_channels - 1)
    st.plotly_chart(
        ecg_figure(
            frame.signal, frame.sampling_rate,
            lead_names=frame.channel_order(), highlight_lead=highlight,
            title=f"{frame.label()} · {labels[index]} (RAW)",
        ),
        use_container_width=True,
    )

    st.caption(
        f"Coverage across all sessions: "
        f"{sum(v for k, v in counts.items() if k in STATIC_POSITIONS)} statically-labelled "
        "frame(s) in this session."
    )

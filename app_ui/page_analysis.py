"""Analysis page: run the SQI pipeline and inspect results (IDEA.md sections 30, 49, 50)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ecg_eval.analysis import position_comparison, position_summary, subject_summary
from ecg_eval.ingestion import annotation_coverage, dataset_summary
from ecg_eval.models.result import BARELY_ACCEPTABLE, EXCELLENT, UNACCEPTABLE
from ecg_eval.visualization import (
    ecg_figure,
    fuzzy_matrix_heatmap,
    heatmap_figure,
    membership_figure,
    psd_figure,
    quality_distribution_figure,
    sqi_box_figure,
    sqi_trend_figure,
)

from .common import header, results_frame, run_or_get_analysis


def render(ctx: dict) -> None:
    header("Analysis", "Four SQIs per annotated 10-second frame, then Zhao-Zhang fuzzy evaluation.")
    dataset = ctx["dataset"]
    annotations = ctx["annotations"]
    config = ctx["config"]
    cache = ctx["cache"]

    coverage = annotation_coverage(
        dataset.frames_by_session, annotations,
        expected_total_frames=int(config.get("dataset.expected_total_frames", 216)),
        expected_subjects=int(config.get("dataset.expected_subjects", 12)),
        expected_positions=int(config.get("dataset.expected_positions", 3)),
        expected_frames_per_position=int(config.get("dataset.expected_frames_per_position", 6)),
    )

    st.subheader("Frame accounting")
    cols = st.columns(6)
    cols[0].metric("Expected frames", coverage["expected_frames"])
    cols[1].metric("Ingested frames", coverage["total_frames"])
    cols[2].metric("Annotated", coverage["annotated_frames"])
    cols[3].metric("Analyzable", coverage["analyzable_frames"])
    cols[4].metric("Excluded transitions", coverage["excluded_transitions"])
    cols[5].metric("Unlabeled", coverage["unlabeled_frames"])
    st.caption(
        "Only frames explicitly labelled SUPINE, SITTING or STANDING enter the evaluation. "
        "TRANSITION and unlabeled frames are excluded and counted above."
    )

    if coverage["analyzable_frames"] == 0:
        st.warning(
            "No frames are annotated with a static body position yet. Go to the Annotation "
            "tab and assign ranges first."
        )
        return

    # -- run -----------------------------------------------------------
    st.divider()
    left, middle, right = st.columns([1, 1, 3])
    rerun = left.button("Run analysis", type="primary")
    if st.session_state.analysis is not None:
        middle.button("Clear cached results", on_click=cache.clear)
    with right:
        cache_stats = cache.stats()
        st.caption(
            f"Cache: {cache_stats['entries']} entries, {cache_stats['hits']} hit(s), "
            f"{cache_stats['misses']} miss(es). Results are reused when the frame content, "
            "configuration and annotation are all unchanged."
        )
    if rerun:
        st.session_state.analysis = None

    if st.session_state.analysis is None:
        with st.spinner("Running the SQI pipeline..."):
            run_or_get_analysis(dataset, annotations, config, cache)

    analysis = st.session_state.analysis
    provenance = analysis["provenance"]
    counts = analysis["counts"]

    st.success(
        f"Analysed {counts['valid']} valid frame(s); {counts['invalid']} invalid. "
        f"Method: {provenance['sqi_method']}."
    )

    with st.expander("Run provenance (reproducibility record)", expanded=False):
        st.json(provenance, expanded=False)
        pending = config.pending_parameters()
        if pending:
            st.warning(
                f"{len(pending)} parameter(s) in `{config.name}` are still pending verification "
                "against Zhao & Zhang (2018). This run is a structured reproduction, not a "
                "verified one."
            )
            st.dataframe(
                pd.DataFrame(pending, columns=["parameter", "value"]),
                use_container_width=True, hide_index=True,
            )

    # -- level 1: individual frame -------------------------------------
    st.divider()
    st.subheader("Level 1 — individual frame")
    df = results_frame(analysis)
    if df.empty:
        st.info("No results to display.")
        return

    subject = st.selectbox("Subject", sorted(df["subject_id"].unique()))
    subject_rows = df[df["subject_id"] == subject]
    position_options = ["(all)"] + sorted(subject_rows["position"].dropna().unique())
    position = st.selectbox("Position", position_options)
    if position != "(all)":
        subject_rows = subject_rows[subject_rows["position"] == position]

    candidates = subject_rows.reset_index(drop=True)
    labels = [
        f"{row.position} · frame {row.frame_id} · {row.quality_class}" for row in candidates.itertuples()
    ]
    pick = st.selectbox("Frame", list(range(len(candidates))), format_func=lambda i: labels[i])
    row = candidates.iloc[pick]
    result = next(
        r for r in analysis["results"] if r.internal_id == row["internal_id"] and r.position == row["position"]
    )

    metrics = st.columns(4)
    metrics[0].metric("qSQI", f"{result.q_sqi:.3f}" if result.valid else "n/a")
    metrics[1].metric("pSQI", f"{result.p_sqi:.3f}" if result.valid else "n/a")
    metrics[2].metric("kSQI", f"{result.k_sqi:.3f}" if result.valid else "n/a")
    metrics[3].metric("basSQI", f"{result.bas_sqi:.3f}" if result.valid else "n/a")

    fuzzy_cols = st.columns(4)
    fuzzy_cols[0].metric(f"Fuzzy · {EXCELLENT}", f"{result.fuzzy_excellent:.3f}" if result.valid else "n/a")
    fuzzy_cols[1].metric(f"Fuzzy · {BARELY_ACCEPTABLE}", f"{result.fuzzy_barely_acceptable:.3f}" if result.valid else "n/a")
    fuzzy_cols[2].metric(f"Fuzzy · {UNACCEPTABLE}", f"{result.fuzzy_unacceptable:.3f}" if result.valid else "n/a")
    fuzzy_cols[3].metric("Final class", result.quality_class or "n/a")

    frame = dataset.frame_by_id(result.internal_id)
    if frame is not None:
        detail = result.run_provenance
        processed = None
        if result.preprocessing_applied:
            from ecg_eval.preprocessing import preprocess

            processed = preprocess(
                frame.lead(str(config.get("signal.analysis_lead", "Lead II"))),
                frame.sampling_rate,
                config.section("preprocessing"),
                enabled=ctx.get("preprocessing_enabled"),
            ).signal

        lead_order = config.get("signal.channel_order", ["Lead I", "Lead II", "Lead III"])
        analysis_lead = str(config.get("signal.analysis_lead", "Lead II"))
        highlight = lead_order.index(analysis_lead) if analysis_lead in lead_order else 1
        st.plotly_chart(
            ecg_figure(
                frame.signal, frame.sampling_rate,
                lead_names=frame.channel_order(), highlight_lead=highlight,
                processed=processed,
                peaks_a=result.peaks_a, peaks_b=result.peaks_b,
                label_a=f"Detector A · {detail.get('detector_a', '')}",
                label_b=f"Detector B · {detail.get('detector_b', '')}",
                title=f"{result.subject_id} · {result.position} · frame {result.frame_id}",
            ),
            use_container_width=True,
        )
        st.caption(
            f"qSQI = {result.n_matched} matched peaks / min({result.n_peaks_a}, {result.n_peaks_b}) "
            f"= {result.q_sqi:.3f} at a tolerance of "
            f"{detail.get('match_tolerance_ms', float('nan')):.0f} ms. "
            f"Method: {detail.get('q_sqi_method', '')}."
        )

    with st.expander("Intermediates: PSD, kurtosis, fuzzy matrix"):
        spectral = result.run_provenance.get("spectral", {})
        from ecg_eval.sqi import compute_psd

        if frame is not None:
            signal = frame.lead(str(config.get("signal.analysis_lead", "Lead II")))
            if result.preprocessing_applied and processed is not None:
                signal = processed
            frequencies, psd = compute_psd(
                signal, frame.sampling_rate,
                nperseg=int(config.get("p_sqi.psd_nperseg", 500)),
                method=str(config.get("p_sqi.psd_method", "welch")),
            )
            st.plotly_chart(
                psd_figure(
                    frequencies, psd,
                    qrs_band=tuple(config.get("p_sqi.qrs_band_hz", (5, 15))),
                    baseline_band=tuple(config.get("bas_sqi.baseline_band_hz", (0, 1))),
                ),
                use_container_width=True,
            )

        a, b, c = st.columns(3)
        a.metric("QRS band power", f"{spectral.get('qrs_band_power', float('nan')):.4g}")
        b.metric("Baseline band power", f"{spectral.get('baseline_band_power', float('nan')):.4g}")
        c.metric("Kurtosis (mean / SD)",
                 f"{result.kurtosis_mean:.3f} / {result.kurtosis_std:.3f} "
                 f"({result.kurtosis_definition})")

        fuzzy = result.run_provenance.get("fuzzy", {})
        if fuzzy:
            left, right = st.columns([1, 1])
            with left:
                st.plotly_chart(
                    membership_figure(fuzzy.get("membership", {})),
                    use_container_width=True,
                )
            with right:
                st.plotly_chart(
                    fuzzy_matrix_heatmap(fuzzy.get("evaluation_matrix", {})),
                    use_container_width=True,
                )
            st.caption(
                f"Weights: {fuzzy.get('weights', {})} · synthesis: {fuzzy.get('synthesis', '')}. "
                "The membership vector is preserved, not just the final class."
            )

    # -- level 2: subject + position -----------------------------------
    st.divider()
    st.subheader("Level 2 — subject × position")
    st.dataframe(
        subject_summary(df[df["valid"]]), use_container_width=True, hide_index=True,
    )

    # -- level 3: overall ----------------------------------------------
    st.divider()
    st.subheader("Level 3 — overall and position comparison")
    position_df = position_summary(df[df["valid"]])
    st.dataframe(position_df, use_container_width=True, hide_index=True)

    if not position_df.empty:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(quality_distribution_figure(df[df["valid"]]), use_container_width=True)
        with right:
            st.plotly_chart(sqi_box_figure(df), use_container_width=True)

        st.plotly_chart(sqi_trend_figure(df), use_container_width=True)

        matrix = df[df["valid"]].pivot_table(
            index="subject_id", columns="position", values="qSQI", aggfunc="mean"
        )
        order = [p for p in ("SUPINE", "SITTING", "STANDING") if p in matrix.columns]
        st.plotly_chart(
            heatmap_figure(matrix[order] if order else matrix, value_label="mean qSQI"),
            use_container_width=True,
        )

        st.subheader("Statistical analysis")
        comparison = position_comparison(
            df,
            alpha=float(config.get("statistics.alpha", 0.05)),
            run_friedman=bool(config.get("statistics.run_friedman", True)),
        )
        rows = []
        for column, outcome in comparison["tests"].items():
            rows.append(
                {
                    "SQI": column,
                    "test": outcome.get("test"),
                    "status": outcome.get("status"),
                    "n_subjects": outcome.get("n_subjects"),
                    "statistic": outcome.get("statistic"),
                    "p_value": outcome.get("p_value"),
                    "Kendall's W": outcome.get("kendall_w"),
                    "detectable difference": comparison["detectable_difference"].get(column),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(comparison["disclaimer"])
        st.session_state["comparison"] = comparison

    # -- frame-level table ---------------------------------------------
    st.divider()
    st.subheader("Frame-level results")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download frame results (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="frame_results.csv",
        mime="text/csv",
    )

    st.session_state["frame_df"] = df
    st.session_state["position_df"] = position_df
    st.session_state["dataset_summary"] = dataset_summary(dataset)
    st.session_state["coverage"] = coverage

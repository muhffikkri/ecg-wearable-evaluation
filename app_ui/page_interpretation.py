"""Interpretation page: structured report and frame-level diagnostics.

Every statement is generated from calculated results; no conclusion is
hard-coded (IDEA.md sections 36-39, 51, 52).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ecg_eval.analysis import export_results, subject_summary, write_markdown
from ecg_eval.interpretation import build_report, describe_frame, problematic_frames
from ecg_eval.models.result import BARELY_ACCEPTABLE, EXCELLENT, UNACCEPTABLE
from ecg_eval.visualization import (
    ecg_figure,
    membership_figure,
    psd_figure,
    quality_distribution_figure,
)

from .common import header, results_frame


def _guard(text: str) -> None:
    """Withhold generated text that asserts a claim the engine may not make.

    The block list lives in the interpretation engine so the scientific layer
    and the UI cannot drift apart (IDEA.md sections 52 and 56).
    """
    from ecg_eval.interpretation.engine import _asserted_claims

    asserted = _asserted_claims(text)
    if asserted:
        st.error(
            f"Generated interpretation contained a disallowed claim ({asserted[0]!r}). "
            "This is a bug in the interpretation engine and the text is withheld."
        )
        st.stop()


def render(ctx: dict) -> None:
    header("Interpretation", "Human-readable summary generated from the calculated results.")
    analysis = st.session_state.analysis
    if analysis is None:
        st.info("Run the analysis on the Analysis tab first.")
        return

    df = results_frame(analysis)
    if df.empty:
        st.info("No results available.")
        return
    valid = df[df["valid"]]
    if valid.empty:
        st.warning("No valid frames, so no interpretation can be generated.")
        return

    config = ctx["config"]
    position_df = st.session_state.get("position_df")
    if position_df is None or position_df.empty:
        from ecg_eval.analysis import position_summary

        position_df = position_summary(valid)
    subject_df = subject_summary(valid)
    comparison = st.session_state.get("comparison")

    # -- overall -------------------------------------------------------
    st.subheader("Overall")
    a, b, c, d = st.columns(4)
    a.metric("Analysed frames", len(valid))
    counts = valid["quality_class"].value_counts()
    total = max(1, len(valid))
    b.metric(f"% {EXCELLENT}", f"{100 * counts.get(EXCELLENT, 0) / total:.1f}%")
    c.metric(f"% {BARELY_ACCEPTABLE}", f"{100 * counts.get(BARELY_ACCEPTABLE, 0) / total:.1f}%")
    d.metric(f"% {UNACCEPTABLE}", f"{100 * counts.get(UNACCEPTABLE, 0) / total:.1f}%")

    st.plotly_chart(quality_distribution_figure(valid), use_container_width=True)

    # -- by position ---------------------------------------------------
    st.subheader("By position")
    st.dataframe(
        position_df[[
            "position", "n_frames",
            "qSQI_mean", "qSQI_median", "qSQI_sd",
            "pSQI_mean", "pSQI_median", "pSQI_sd",
            "kSQI_mean", "kSQI_median", "kSQI_sd",
            "basSQI_mean", "basSQI_median", "basSQI_sd",
        ]] if not position_df.empty else position_df,
        use_container_width=True, hide_index=True,
    )

    # -- problematic frames --------------------------------------------
    st.subheader("Problematic frames")
    problems = problematic_frames(df, limit=50)
    st.caption(
        "Frames ranked by Unacceptable membership. Click one to inspect its waveform, "
        "R-peaks, PSD, SQIs and fuzzy membership."
    )
    if problems.empty:
        st.info("No problematic frames to inspect.")
    else:
        table = problems[[
            "subject_id", "position", "frame_id",
            "qSQI", "pSQI", "kSQI", "basSQI",
            "fuzzy_excellent", "fuzzy_barely_acceptable", "fuzzy_unacceptable", "quality_class",
        ]]
        st.dataframe(table, use_container_width=True, hide_index=True)

        pick = st.selectbox(
            "Inspect frame", list(range(len(problems))),
            format_func=lambda i: (
                f"{problems.iloc[i]['subject_id']} · {problems.iloc[i]['position']} · "
                f"frame {problems.iloc[i]['frame_id']} · {problems.iloc[i]['quality_class']} · "
                f"U={problems.iloc[i]['fuzzy_unacceptable']:.3f}"
            ),
        )
        target = problems.iloc[pick]
        result = next(
            r for r in analysis["results"]
            if r.internal_id == target["internal_id"] and r.position == target["position"]
        )
        st.markdown("**Diagnostic explanation**")
        st.markdown(describe_frame(result))

        frame = ctx["dataset"].frame_by_id(result.internal_id)
        if frame is not None:
            processed = None
            if result.preprocessing_applied:
                from ecg_eval.preprocessing import preprocess

                stages = ctx.get("preprocessing_stages") or config.section("preprocessing")
                processed = preprocess(
                    frame.lead(str(config.get("signal.analysis_lead", "Lead II")))[:, None],
                    frame.sampling_rate, stages,
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
                    label_a=f"Detector A · {result.run_provenance.get('detector_a', '')}",
                    label_b=f"Detector B · {result.run_provenance.get('detector_b', '')}",
                    title=f"{result.subject_id} · {result.position} · frame {result.frame_id} (RAW)",
                ),
                use_container_width=True,
            )

            from ecg_eval.sqi import compute_psd

            signal = processed if processed is not None else frame.lead(analysis_lead)
            frequencies, psd = compute_psd(
                signal, frame.sampling_rate,
                nperseg=int(config.get("p_sqi.psd_nperseg", 500)),
            )
            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    psd_figure(
                        frequencies, psd,
                        qrs_band=tuple(config.get("p_sqi.qrs_band_hz", (5, 15))),
                        baseline_band=tuple(config.get("bas_sqi.baseline_band_hz", (0, 1))),
                    ),
                    use_container_width=True,
                )
            with right:
                fuzzy = result.run_provenance.get("fuzzy", {})
                if fuzzy:
                    st.plotly_chart(
                        membership_figure(fuzzy.get("membership", {})),
                        use_container_width=True,
                    )

    # -- full report ----------------------------------------------------
    st.divider()
    st.subheader("Generated interpretation")
    report = build_report(
        frame_df=df,
        position_df=position_df,
        subject_df=subject_df,
        comparison=comparison,
        dataset_summary=st.session_state.get("dataset_summary"),
        coverage=st.session_state.get("coverage"),
        config_pending=config.pending_parameters(),
        config_name=config.name,
        provenance=analysis["provenance"],
        problematic_threshold=float(
            config.get("interpretation.problematic_unacceptable_threshold", 0.5)
        ),
    )
    _guard(report)
    st.markdown(report)

    st.download_button(
        "Download interpretation (Markdown)",
        report.encode("utf-8"),
        file_name="interpretation.md",
        mime="text/markdown",
    )

    # -- export ---------------------------------------------------------
    st.divider()
    st.subheader("Export")
    st.caption(
        "Results are written to `results/`, separate from the raw data. Nothing under "
        "`data/` is modified."
    )
    if st.button("Export all results"):
        written = export_results(
            ctx["results_dir"], analysis["results"],
            provenance=analysis["provenance"],
            dataset_summary=st.session_state.get("dataset_summary"),
            annotation_summary={"note": "see annotations/annotation_manifest.csv"},
            coverage=st.session_state.get("coverage"),
            comparison=comparison,
        )
        write_markdown(ctx["results_dir"] / "interpretation.md", report)
        for name, path in written.items():
            st.success(f"{name}: `{path}`")
        st.success(f"interpretation: `{ctx['results_dir'] / 'interpretation.md'}`")

    with st.expander("Export the Markdown report on its own"):
        st.download_button(
            "interpretation.md", report.encode("utf-8"),
            file_name="interpretation.md", mime="text/markdown",
        )
        st.download_button(
            "position_results.csv",
            (position_df.to_csv(index=False) if not position_df.empty else "").encode("utf-8"),
            file_name="position_results.csv", mime="text/csv",
        )
        st.download_button(
            "subject_results.csv", subject_df.to_csv(index=False).encode("utf-8"),
            file_name="subject_results.csv", mime="text/csv",
        )

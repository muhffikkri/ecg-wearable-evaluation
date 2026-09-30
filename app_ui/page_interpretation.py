"""Interpretation page: structured report and frame-level diagnostics.

Every statement is generated from calculated results; no conclusion is
hard-coded (IDEA.md sections 36-39, 51, 52).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ecg_eval.analysis import export_results, subject_summary, write_markdown
from ecg_eval.analysis.statistics import acceptable_share, activity_recap, class_table
from ecg_eval.interpretation import build_report, describe_frame, problematic_frames
from ecg_eval.models.result import BARELY_ACCEPTABLE, EXCELLENT, UNACCEPTABLE
from ecg_eval.visualization import (
    POSITION_SUMMARY_LABELS,
    SQI_ORDER,
    acceptance_table,
    acceptance_table_interpretation,
    acceptance_table_legend,
    class_label,
    activity_recap_distribution_interpretation,
    activity_recap_distribution_legend,
    activity_recap_figure,
    activity_recap_interpretation,
    activity_recap_legend,
    activity_recap_table,
    ecg_figure,
    ecg_legend,
    membership_figure,
    membership_interpretation,
    membership_legend,
    position_label,
    psd_figure,
    psd_interpretation,
    psd_legend,
    quality_by_activity_figure,
    quality_by_activity_interpretation,
    quality_by_activity_legend,
    quality_distribution_figure,
    quality_distribution_interpretation,
    quality_distribution_legend,
    sqi_box_figure,
    sqi_box_interpretation,
    sqi_box_legend,
    sqi_label,
    sqi_stat_columns,
    sqi_stat_display_names,
)

from .common import header, results_frame

#: The analysis frames keep machine names (``qSQI_mean``, ``position``) so the
#: exported CSV and the report still line up with the analysis code; only the
#: header a reader sees changes. The mapping lives in ``labels_id`` so every
#: table on this page spells its columns the same way.
_DISPLAY_NAMES = {**POSITION_SUMMARY_LABELS, **sqi_stat_display_names()}


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
    st.caption(quality_distribution_legend())
    st.caption(quality_distribution_interpretation(valid))

    # -- by position ---------------------------------------------------
    st.subheader("By position")
    st.dataframe(
        position_df[["position", "n_frames", *sqi_stat_columns()]].rename(
            columns=_DISPLAY_NAMES
        ) if not position_df.empty else position_df,
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "**Legenda tabel.** Baris adalah aktivitas rekaman. Tiga ukuran "
        "setiap indeks kualitas sinyal dihitung dari frame yang dianalisis di "
        "aktivitas tersebut: `rata-rata`, `tengah` (median), dan `simpangan "
        "baku` (rumus contoh, ddof=1). Nama indeks memakai Bahasa Indonesia di "
        "header, sedangkan file ekspor tetap memakai kode mesin seperti "
        "`qSQI_mean`. Tabel ini merangkum frame yang sama dengan tabel "
        "rekapitulasi di bawah, tetapi menyimpan nilai indeks secara langsung, "
        "bukan kelas penerimaannya."
    )
    if not position_df.empty:
        # The box plot needs the frame-level rows: it plots every analysed
        # frame, grouped by activity, so the summary table cannot feed it.
        st.plotly_chart(sqi_box_figure(valid), use_container_width=True)
        st.caption(sqi_box_legend())
        st.caption(sqi_box_interpretation(valid))

    # -- klasifikasi kualitas per aktivitas ------------------------------
    st.divider()
    st.subheader("Kualitas penerimaan per aktivitas")
    classes = class_table(valid)
    share = acceptable_share(valid)

    st.markdown("**Tabel total klasifikasi kualitas per aktivitas**")
    table = acceptance_table(classes)
    if table.empty:
        st.info("Belum ada frame yang dapat dikelompokkan per aktivitas.")
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(acceptance_table_legend())
        st.caption(acceptance_table_interpretation(classes))

    st.markdown("**Kualitas sinyal per peserta dan aktivitas**")
    if share.empty:
        st.info("Belum ada data per peserta untuk digambar.")
    else:
        st.plotly_chart(quality_by_activity_figure(share), use_container_width=True)
        st.caption(quality_by_activity_legend())
        st.caption(quality_by_activity_interpretation(share))
        with st.expander("Data nilai yang digambar pada figure"):
            detail = share.copy()
            detail["Aktivitas"] = detail["position"].map(position_label)
            detail = detail.rename(
                columns={
                    "subject_id": "Peserta",
                    "n_frames": "Total Frame",
                    "n_accepted": "Diterima (n)",
                    "pct_accepted": "Diterima (%)",
                }
            )
            st.dataframe(
                detail[["Peserta", "Aktivitas", "Total Frame",
                        "Diterima (n)", "Diterima (%)"]],
                use_container_width=True, hide_index=True,
            )

    # -- rekapitulasi per aktivitas --------------------------------------
    st.divider()
    st.subheader("Rekapitulasi per aktivitas")
    recap = activity_recap(valid)
    st.caption(
        "Bagian ini menggabungkan seluruh frame satu aktivitas menjadi satu "
        "baris. Tabel di bawah merangkum kelas penerimaan sekaligus sebaran "
        "empat indeks kualitas sinyal, sedangkan figure menunjukkan sebaran "
        "frame yang mendasarinya. Perbandingan dengan figure per peserta "
        "diatas: yang satu menggabungkan frame per aktivitas, yang satu lagi "
        "memberi satu nilai per peserta."
    )

    st.markdown("**Tabel rekapitulasi per aktivitas**")
    recap_display = activity_recap_table(recap)
    if recap_display.empty:
        st.info("Belum ada frame yang dapat direkap per aktivitas.")
    else:
        st.dataframe(recap_display, use_container_width=True, hide_index=True)
        st.caption(activity_recap_legend())
        st.caption(activity_recap_interpretation(recap))

    st.markdown("**Sebaran indeks kualitas sinyal per aktivitas**")
    if recap_display.empty:
        st.info("Belum ada data untuk digambar.")
    else:
        st.plotly_chart(activity_recap_figure(valid), use_container_width=True)
        st.caption(activity_recap_distribution_legend())
        st.caption(activity_recap_distribution_interpretation(valid))

    # -- problematic frames --------------------------------------------
    st.subheader("Problematic frames")
    problems = problematic_frames(df, limit=50)
    st.caption(
        "**Legenda tabel.** Frame diurutkan berdasarkan nilai keanggotaan "
        "*Tidak Diterima* (`U`) dari yang tertinggi. Kolom aktivitas, identitas "
        "frame, kelas kualitas, dan indeks kualitas sudah memakai Bahasa "
        "Indonesia; nilai keanggotaan ditampilkan apa adanya sebagai angka "
        "keanggotaan 0-1 agar bisa dibandingkan langsung dengan `U` pada "
        "penyaring dan file ekspor. Pilih satu baris untuk melihat gelombang, "
        "puncak R, spektral, indeks kualitas, dan keanggotaan fuzzy frame "
        "tersebut."
    )
    if problems.empty:
        st.info("No problematic frames to inspect.")
    else:
        table = problems[[
            "subject_id", "position", "frame_id",
            "qSQI", "pSQI", "kSQI", "basSQI",
            "fuzzy_excellent", "fuzzy_unacceptable", "quality_class",
        ]].rename(columns={**POSITION_SUMMARY_LABELS,
                           "subject_id": "Peserta",
                           "frame_id": "Frame",
                           "fuzzy_excellent": "Keanggotaan Sangat Baik",
                           "fuzzy_unacceptable": "Keanggotaan Tidak Diterima (U)",
                           "quality_class": "Kelas Kualitas",
                           **{column: sqi_label(column) for column in SQI_ORDER}})
        st.dataframe(table, use_container_width=True, hide_index=True)

        pick = st.selectbox(
            "Inspect frame", list(range(len(problems))),
            format_func=lambda i: (
                f"{position_label(problems.iloc[i]['position'])} · "
                f"frame {problems.iloc[i]['frame_id']} · "
                f"{class_label(problems.iloc[i]['quality_class'])} · "
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
            st.caption(ecg_legend())

            from ecg_eval.sqi import compute_psd

            signal = processed if processed is not None else frame.lead(analysis_lead)
            frequencies, psd = compute_psd(
                signal, frame.sampling_rate,
                nperseg=int(config.get("p_sqi.psd_nperseg", 500)),
            )
            qrs_band = tuple(config.get("p_sqi.qrs_band_hz", (5, 15)))
            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    psd_figure(
                        frequencies, psd,
                        qrs_band=qrs_band,
                        baseline_band=tuple(config.get("bas_sqi.baseline_band_hz", (0, 1))),
                    ),
                    use_container_width=True,
                )
                st.caption(psd_legend())
                st.caption(
                    psd_interpretation(
                        frequencies, psd, qrs_band=qrs_band,
                        analysis_band=(float(frequencies.min()), float(frequencies.max()))
                        if len(frequencies) else None,
                    )
                )
            with right:
                fuzzy = result.run_provenance.get("fuzzy", {})
                if fuzzy:
                    st.plotly_chart(
                        membership_figure(fuzzy.get("membership", {})),
                        use_container_width=True,
                    )
                    st.caption(membership_legend())
                    st.caption(
                        membership_interpretation(fuzzy.get("membership", {}))
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

"""Plotly figures for the Streamlit UI.

Kept separate from the app so figures can also be generated headlessly for
export (IDEA.md section 53). Raw and processed signals are always plotted as
separate, explicitly labelled traces; a processed trace never replaces the raw
one (IDEA.md section 13).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..models.result import BARELY_ACCEPTABLE, EXCELLENT, UNACCEPTABLE

CLASS_COLORS = {
    EXCELLENT: "#2e7d32",
    BARELY_ACCEPTABLE: "#f9a825",
    UNACCEPTABLE: "#c62828",
    "INVALID": "#6a1b9a",
}
POSITION_COLORS = {"SUPINE": "#1565c0", "SITTING": "#6a1b9a", "STANDING": "#ef6c00"}
SQI_COLORS = {"qSQI": "#1565c0", "pSQI": "#2e7d32", "kSQI": "#6a1b9a", "basSQI": "#ef6c00"}


def _time_axis(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    return np.arange(signal.shape[0]) / float(sampling_rate)


def ecg_figure(
    raw: np.ndarray,
    sampling_rate: float,
    *,
    lead_names: Sequence[str] = ("Lead I", "Lead II", "Lead III"),
    highlight_lead: int = 1,
    processed: np.ndarray | None = None,
    peaks_a: Sequence[int] | None = None,
    peaks_b: Sequence[int] | None = None,
    label_a: str = "Detector A",
    label_b: str = "Detector B",
    title: str = "",
    height: int = 520,
) -> go.Figure:
    """Multi-lead ECG with optional processed overlay and R-peak markers.

    RAW is always drawn. PREPROCESSED is only drawn when preprocessing was
    actually executed, and carries its own trace name and legend entry.
    """
    raw = np.asarray(raw, dtype=np.float64)
    n_leads = raw.shape[1] if raw.ndim > 1 else 1
    t_raw = _time_axis(raw, sampling_rate)
    names = list(lead_names)[:n_leads] or [f"ch{i}" for i in range(n_leads)]

    figure = go.Figure()
    for index in range(n_leads):
        emphasis = index == highlight_lead
        figure.add_trace(
            go.Scatter(
                x=t_raw,
                y=raw[:, index],
                name=f"RAW · {names[index]}",
                mode="lines",
                line=dict(
                    width=1.6 if emphasis else 1.0,
                    color="#1565c0" if emphasis else "#90a4ae",
                ),
                opacity=1.0 if emphasis else 0.55,
                hovertemplate=(
                    f"{names[index]}<br>t=%{{x:.3f}} s<br>%{{y:.3f}} mV<extra>RAW</extra>"
                ),
            )
        )

    if processed is not None and np.asarray(processed).size:
        processed = np.asarray(processed, dtype=np.float64)
        if processed.ndim == 1:
            processed = processed[:, None]
        t_proc = _time_axis(processed, sampling_rate)
        for index in range(min(processed.shape[1], n_leads)):
            figure.add_trace(
                go.Scatter(
                    x=t_proc,
                    y=processed[:, index],
                    name=f"PREPROCESSED · {names[index]}",
                    mode="lines",
                    line=dict(width=1.4, color="#2e7d32", dash="dot"),
                    hovertemplate=(
                        f"{names[index]}<br>t=%{{x:.3f}} s<br>%{{y:.3f}} mV"
                        "<extra>PREPROCESSED</extra>"
                    ),
                )
            )

    lead_t = t_raw
    lead_series = raw[:, highlight_lead] if n_leads > 1 else raw[:, 0]

    for peaks, label, color, symbol in (
        (peaks_a, label_a, "#d81b60", "triangle-up"),
        (peaks_b, label_b, "#00acc1", "diamond"),
    ):
        if peaks is None or len(peaks) == 0:
            continue
        indices = [int(p) for p in peaks if 0 <= int(p) < lead_t.size]
        if not indices:
            continue
        figure.add_trace(
            go.Scatter(
                x=lead_t[indices],
                y=lead_series[indices],
                name=label,
                mode="markers",
                marker=dict(symbol=symbol, size=9, color=color, line=dict(width=1, color="white")),
                hovertemplate=f"{label}<br>t=%{{x:.3f}} s<extra></extra>",
            )
        )

    figure.update_layout(
        title=title or None,
        height=height,
        margin=dict(l=60, r=20, t=40 if title else 12, b=40),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (mV)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
        uirevision=title or "ecg",
    )
    figure.update_xaxes(showgrid=True, zeroline=False)
    figure.update_yaxes(showgrid=True, zeroline=False)
    return figure


def multilead_figure(
    signal: np.ndarray,
    sampling_rate: float,
    lead_names: Sequence[str] = ("Lead I", "Lead II", "Lead III"),
    *,
    title: str = "ECG leads",
    height: int = 460,
) -> go.Figure:
    """Stacked per-lead view, useful for spotting which lead carries motion."""
    signal = np.asarray(signal, dtype=np.float64)
    n_leads = signal.shape[1] if signal.ndim > 1 else 1
    names = list(lead_names)[:n_leads] or [f"ch{i}" for i in range(n_leads)]
    t = _time_axis(signal, sampling_rate)

    figure = make_subplots(
        rows=n_leads, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, subplot_titles=[f"{name} (mV)" for name in names],
    )
    for index in range(n_leads):
        figure.add_trace(
            go.Scatter(x=t, y=signal[:, index], mode="lines", name=names[index],
                       line=dict(width=1.2, color="#1565c0"), showlegend=False),
            row=index + 1, col=1,
        )
    figure.update_layout(
        title=title, height=height, template="plotly_white",
        margin=dict(l=60, r=20, t=40, b=40), hovermode="closest",
    )
    figure.update_xaxes(title_text="Time (s)", row=n_leads, col=1)
    return figure


def psd_figure(
    frequencies: Sequence[float],
    psd: Sequence[float],
    *,
    qrs_band: tuple[float, float] | None = None,
    baseline_band: tuple[float, float] | None = None,
    height: int = 380,
    title: str = "Power spectral density",
) -> go.Figure:
    """PSD with the configured QRS and baseline bands shaded."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=list(frequencies), y=list(psd), mode="lines", name="PSD",
                   line=dict(width=1.4, color="#37474f"))
    )
    for band, color, label in (
        (qrs_band, "#2e7d32", "QRS band (pSQI)"),
        (baseline_band, "#ef6c00", "Baseline band (basSQI)"),
    ):
        if not band:
            continue
        figure.add_vrect(
            x0=band[0], x1=band[1], fillcolor=color, opacity=0.15,
            line_width=0, annotation_text=label, annotation_position="top",
        )
    figure.update_layout(
        title=title, height=height, template="plotly_white",
        xaxis_title="Frequency (Hz)", yaxis_title="Power",
        margin=dict(l=60, r=20, t=40, b=40), hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figure


def membership_figure(
    membership: dict[str, float], *, height: int = 260, title: str = "Fuzzy membership"
) -> go.Figure:
    levels = [EXCELLENT, BARELY_ACCEPTABLE, UNACCEPTABLE]
    values = [float(membership.get(level, 0.0)) for level in levels]
    figure = go.Figure(
        go.Bar(
            x=levels, y=values,
            marker_color=[CLASS_COLORS[level] for level in levels],
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title, height=height, template="plotly_white",
        yaxis=dict(title="Membership", range=[0, 1.15]),
        margin=dict(l=60, r=20, t=40, b=40), showlegend=False,
    )
    return figure


def quality_distribution_figure(
    df: pd.DataFrame, *, height: int = 380
) -> go.Figure:
    """Stacked bar chart of quality class share per body position."""
    if df.empty:
        return go.Figure()
    counts = (
        df.groupby(["position", "quality_class"]).size().unstack(fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1), axis=0) * 100.0
    order = [p for p in ("SUPINE", "SITTING", "STANDING") if p in shares.index]
    shares = shares.loc[order] if order else shares

    figure = go.Figure()
    for level in (EXCELLENT, BARELY_ACCEPTABLE, UNACCEPTABLE):
        if level not in shares.columns:
            continue
        figure.add_bar(
            name=level, x=list(shares.index), y=list(shares[level]),
            marker_color=CLASS_COLORS[level],
            text=[f"{v:.0f}%" if v >= 3 else "" for v in shares[level]],
            textposition="inside",
        )
    figure.update_layout(
        barmode="stack", height=height, template="plotly_white",
        yaxis=dict(title="Share of frames (%)", range=[0, 100]),
        xaxis_title="Body position", margin=dict(l=60, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figure


def sqi_box_figure(
    df: pd.DataFrame, columns: Sequence[str] = ("qSQI", "pSQI", "kSQI", "basSQI"),
    *, height: int = 420,
) -> go.Figure:
    """Box plots of each SQI across the three body positions."""
    if df.empty:
        return go.Figure()
    valid = df[df["valid"]] if "valid" in df else df
    figure = go.Figure()
    for column in columns:
        if column not in valid:
            continue
        positions = [p for p in ("SUPINE", "SITTING", "STANDING") if p in set(valid["position"])]
        for position in positions:
            values = valid[valid["position"] == position][column].dropna().tolist()
            if not values:
                continue
            figure.add_trace(
                go.Box(
                    y=values, name=f"{position}", legendgroup=column,
                    legendgrouptitle_text=column if position == positions[0] else None,
                    marker_color=POSITION_COLORS.get(position, "#546e7a"),
                    boxmean=True, points="outliers", pointpos=0, jitter=0.3,
                    line=dict(width=1.2),
                )
            )
    figure.update_layout(
        title="SQI distribution by body position", height=height, template="plotly_white",
        yaxis_title="SQI value", xaxis_title="", margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figure


def heatmap_figure(
    matrix: pd.DataFrame, *, value_label: str = "mean SQI", height: int = 420
) -> go.Figure:
    """Per-subject x position heatmap."""
    if matrix.empty:
        return go.Figure()
    z = matrix.to_numpy(dtype=float)
    figure = go.Figure(
        go.Heatmap(
            z=z, x=list(matrix.columns), y=list(matrix.index),
            colorscale="RdYlGn", zmid=float(np.nanmean(z)) if np.isfinite(z).any() else None,
            colorbar=dict(title=value_label),
            hovertemplate="Subject %{y}<br>%{x}: %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Per-subject {value_label} by body position",
        height=height, template="plotly_white",
        xaxis_title="Body position", yaxis_title="Subject",
        margin=dict(l=80, r=20, t=50, b=40),
    )
    return figure


def sqi_trend_figure(df: pd.DataFrame, *, height: int = 380) -> go.Figure:
    """The four SQIs over frame order, coloured by position."""
    if df.empty:
        return go.Figure()
    valid = df[df["valid"]] if "valid" in df else df
    figure = go.Figure()
    for column in ("qSQI", "pSQI", "kSQI", "basSQI"):
        if column not in valid:
            continue
        figure.add_trace(
            go.Scatter(
                y=valid[column], mode="markers", name=column,
                marker=dict(color=[POSITION_COLORS.get(p, "#546e7a") for p in valid["position"]], size=7),
                hovertemplate=(
                    f"{column}<br>subject %{{customdata[0]}}<br>position %{{customdata[1]}}"
                    "<br>value %{y:.3f}<extra></extra>"
                ),
                customdata=np.column_stack([valid["subject_id"], valid["position"], valid["frame_id"]]),
            )
        )
    figure.update_layout(
        title="SQI per analysed frame", height=height, template="plotly_white",
        yaxis_title="SQI value", xaxis_title="Frame index", margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figure


def fuzzy_matrix_heatmap(
    matrix: dict[str, dict[str, float]], *, height: int = 300
) -> go.Figure:
    """The evaluation matrix R, shown so the fuzzy step is inspectable."""
    if not matrix:
        return go.Figure()
    factors = list(matrix.keys())
    levels = list(next(iter(matrix.values())).keys())
    z = [[float(matrix[f][l]) for l in levels] for f in factors]
    figure = go.Figure(
        go.Heatmap(
            z=z, x=levels, y=factors, colorscale="Blues", zmin=0, zmax=1,
            colorbar=dict(title="membership"),
            hovertemplate="%{y} → %{x}: %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Fuzzy evaluation matrix R", height=height, template="plotly_white",
        margin=dict(l=90, r=20, t=50, b=40),
    )
    return figure


__all__ = [
    "CLASS_COLORS",
    "POSITION_COLORS",
    "SQI_COLORS",
    "ecg_figure",
    "fuzzy_matrix_heatmap",
    "heatmap_figure",
    "membership_figure",
    "multilead_figure",
    "psd_figure",
    "quality_distribution_figure",
    "sqi_box_figure",
    "sqi_trend_figure",
]

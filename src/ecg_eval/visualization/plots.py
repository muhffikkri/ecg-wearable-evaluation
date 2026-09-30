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

from ..models.result import (
    BARELY_ACCEPTABLE,
    EXCELLENT,
    SQI_KEYS,
    SQI_LABELS as _SQI_LABELS,
    UNACCEPTABLE,
    sqi_label,
)

CLASS_COLORS = {
    EXCELLENT: "#2e7d32",
    BARELY_ACCEPTABLE: "#f9a825",
    UNACCEPTABLE: "#c62828",
    "INVALID": "#6a1b9a",
}
POSITION_COLORS = {"SUPINE": "#1565c0", "SITTING": "#6a1b9a", "STANDING": "#ef6c00"}
SQI_COLORS = {"qSQI": "#1565c0", "pSQI": "#2e7d32", "kSQI": "#6a1b9a", "basSQI": "#ef6c00"}

#: Display names for the signal-quality indices.
#:
#: The short codes (``qSQI`` and friends) stay as the column names in the result
#: table, the config keys and the exported CSV, because those are data and are
#: referenced everywhere downstream. Only the *label a reader sees* changes, so
#: a figure or a report never has to explain what "pSQI" was supposed to mean.
SQI_LABELS = _SQI_LABELS
SQI_ORDER = SQI_KEYS

#: Light-mode palette for print/report use. These figures are meant to sit in a
#: document, so the background and text colours are pinned explicitly instead of
#: inheriting whatever theme the surrounding app happens to be running.
PAPER_COLOR = "#ffffff"
PLOT_COLOR = "#ffffff"
INK = "#102a43"
INK_SOFT = "#486581"
GRID = "#d9e2ec"


def _style(
    figure: go.Figure,
    *,
    title: str = "",
    height: int = 460,
    left: int = 76,
    right: int = 24,
    top: int = 84,
    bottom: int = 108,
    showlegend: bool = True,
) -> go.Figure:
    """Apply the shared light-mode report style.

    The title is pinned to the *container* (the whole figure) and the legend is
    pushed *below the plotting area*. Both used to be anchored near the top of
    the plot, so any figure with a legend drawn a long detector or band name
    printed straight through its own title. Anchoring them to opposite ends
    makes overlap structurally impossible rather than a matter of margins.
    """
    # NOTE: ``update_layout(legend=None)`` is a no-op, not a reset, so the
    # no-legend case has to go through the top-level ``showlegend`` flag.
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=PLOT_COLOR,
        font=dict(family="Arial, Helvetica, sans-serif", size=13, color=INK),
        colorway=list(CLASS_COLORS.values()),
        height=height,
        margin=dict(l=left, r=right, t=top, b=bottom),
        showlegend=showlegend,
        legend=(
            dict(
                orientation="h",
                yanchor="top",
                y=-0.17,
                xanchor="left",
                x=0,
                font=dict(size=11, color=INK_SOFT),
                title_text="",
            )
            if showlegend
            else dict()
        ),
        hovermode="closest",
    )
    if title:
        figure.update_layout(
            title=dict(
                text=title,
                x=0.012,
                xref="container",
                xanchor="left",
                y=0.985,
                yref="container",
                yanchor="top",
                font=dict(size=16, color=INK),
            )
        )
    else:
        figure.update_layout(title_text=None)
    figure.update_xaxes(
        gridcolor=GRID, zeroline=False, linecolor=GRID,
        tickfont=dict(size=11, color=INK_SOFT),
        title_font=dict(size=12, color=INK),
    )
    figure.update_yaxes(
        gridcolor=GRID, zeroline=False, linecolor=GRID,
        tickfont=dict(size=11, color=INK_SOFT),
        title_font=dict(size=12, color=INK),
    )
    return figure


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

    _style(
        figure,
        title=title,
        height=height,
        left=76,
        right=24,
        top=84 if title else 40,
        bottom=112,
        showlegend=True,
    )
    figure.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (mV)",
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
    height: int = 620,
) -> go.Figure:
    """Stacked per-lead view, useful for spotting which lead carries motion."""
    signal = np.asarray(signal, dtype=np.float64)
    n_leads = signal.shape[1] if signal.ndim > 1 else 1
    names = list(lead_names)[:n_leads] or [f"ch{i}" for i in range(n_leads)]
    t = _time_axis(signal, sampling_rate)

    # Each subplot carries its own header. They used to be spaced by 4% of the
    # plot height, which is less than the text needs, so a lead name printed over
    # the trace below it. The spacing is now generous and the headers are styled
    # explicitly rather than inheriting the figure font.
    figure = make_subplots(
        rows=n_leads, cols=1, shared_xaxes=True,
        vertical_spacing=0.11,
        row_heights=[1.0] * n_leads,
        subplot_titles=[f"{name} (mV)" for name in names],
    )
    for index in range(n_leads):
        figure.add_trace(
            go.Scatter(x=t, y=signal[:, index], mode="lines", name=names[index],
                       line=dict(width=1.2, color="#1565c0"), showlegend=False),
            row=index + 1, col=1,
        )
    figure.update_annotations(
        font=dict(size=12, color=INK),
        xanchor="left",
        x=0,
    )
    _style(
        figure,
        title=title,
        height=height,
        left=76,
        right=24,
        top=84,
        bottom=84,
        showlegend=False,
    )
    figure.update_xaxes(title_text="Time (s)", row=n_leads, col=1)
    return figure


def psd_figure(
    frequencies: Sequence[float],
    psd: Sequence[float],
    *,
    qrs_band: tuple[float, float] | None = None,
    baseline_band: tuple[float, float] | None = None,
    height: int = 620,
    title: str = "Power spectral density",
) -> go.Figure:
    """PSD with the configured QRS and baseline bands shaded.

    The band captions are anchored *inside* the plotting area rather than above
    it. ``add_vrect``'s default ``"top"`` position puts them in the same band as
    the figure title, which is where they previously collided; and the full
    descriptive band names are longer than the codes they replaced, so they are
    staggered top/bottom to stay clear of each other and of the curve.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=list(frequencies), y=list(psd), mode="lines", name="Power spectral density",
                   line=dict(width=1.4, color="#37474f"))
    )
    for band, color, label, where in (
        (qrs_band, "#2e7d32", sqi_label("pSQI"), "inside top"),
        (baseline_band, "#ef6c00", sqi_label("basSQI"), "inside bottom"),
    ):
        if not band:
            continue
        figure.add_vrect(
            x0=band[0], x1=band[1], fillcolor=color, opacity=0.13,
            line_width=0, annotation_text=label, annotation_position=where,
            annotation_font=dict(size=11, color=INK),
        )
    _style(
        figure,
        title=title,
        height=height,
        left=88,
        right=24,
        top=84,
        bottom=96,
        showlegend=True,
    )
    figure.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Power",
    )
    return figure


def membership_figure(
    membership: dict[str, float], *, height: int = 320, title: str = "Fuzzy membership"
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
    _style(
        figure,
        title=title,
        height=height,
        left=76,
        right=24,
        top=84,
        bottom=72,
        showlegend=False,
    )
    figure.update_layout(yaxis=dict(title="Membership", range=[0, 1.15]))
    return figure


def quality_distribution_figure(
    df: pd.DataFrame, *, height: int = 460
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
    _style(
        figure,
        height=height,
        left=76,
        right=24,
        top=48,
        bottom=108,
        showlegend=True,
    )
    figure.update_layout(
        barmode="stack",
        yaxis=dict(title="Share of frames (%)", range=[0, 100]),
        xaxis_title="Body position",
    )
    return figure


def sqi_box_figure(
    df: pd.DataFrame, columns: Sequence[str] = SQI_ORDER,
    *, height: int = 560,
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
                    legendgrouptitle_text=sqi_label(column) if position == positions[0] else None,
                    marker_color=POSITION_COLORS.get(position, "#546e7a"),
                    boxmean=True, boxpoints="outliers", pointpos=0, jitter=0.3,
                    line=dict(width=1.2),
                )
            )
    # The legend now carries four full descriptive names instead of four codes,
    # so it wraps onto several rows and needs a taller reserved area than the
    # old one-line legend did.
    _style(
        figure,
        title="Signal quality index distribution by body position",
        height=height,
        left=76,
        right=24,
        top=84,
        bottom=180,
        showlegend=True,
    )
    figure.update_layout(yaxis_title="Index value", xaxis_title="")
    return figure


def heatmap_figure(
    matrix: pd.DataFrame, *, value_label: str = "mean index value", height: int = 460
) -> go.Figure:
    """Per-subject x position heatmap."""
    if matrix.empty:
        return go.Figure()
    z = matrix.to_numpy(dtype=float)
    figure = go.Figure(
        go.Heatmap(
            z=z, x=list(matrix.columns), y=list(matrix.index),
            colorscale="RdYlGn", zmid=float(np.nanmean(z)) if np.isfinite(z).any() else None,
            colorbar=dict(title=dict(text=value_label, font=dict(size=11, color=INK_SOFT))),
            hovertemplate="Subject %{y}<br>%{x}: %{z:.3f}<extra></extra>",
        )
    )
    _style(
        figure,
        title=f"{value_label} per subject and body position",
        height=height,
        left=88,
        right=24,
        top=84,
        bottom=72,
        showlegend=False,
    )
    figure.update_layout(xaxis_title="Body position", yaxis_title="Subject")
    return figure


def sqi_trend_figure(df: pd.DataFrame, *, height: int = 460) -> go.Figure:
    """The four SQIs over frame order, coloured by position."""
    if df.empty:
        return go.Figure()
    valid = df[df["valid"]] if "valid" in df else df
    figure = go.Figure()
    for column in SQI_ORDER:
        if column not in valid:
            continue
        name = sqi_label(column)
        figure.add_trace(
            go.Scatter(
                y=valid[column], mode="markers", name=name,
                marker=dict(color=[POSITION_COLORS.get(p, "#546e7a") for p in valid["position"]], size=7),
                hovertemplate=(
                    f"{name}<br>subject %{{customdata[0]}}<br>position %{{customdata[1]}}"
                    "<br>value %{y:.3f}<extra></extra>"
                ),
                customdata=np.column_stack([valid["subject_id"], valid["position"], valid["frame_id"]]),
            )
        )
    _style(
        figure,
        title="Signal quality index per analysed frame",
        height=height,
        left=76,
        right=24,
        top=84,
        bottom=150,
        showlegend=True,
    )
    figure.update_layout(yaxis_title="Index value", xaxis_title="Frame index")
    return figure


def fuzzy_matrix_heatmap(
    matrix: dict[str, dict[str, float]], *, height: int = 380
) -> go.Figure:
    """The evaluation matrix R, shown so the fuzzy step is inspectable."""
    if not matrix:
        return go.Figure()
    factors = [sqi_label(f) if f in SQI_LABELS else f for f in matrix]
    raw_factors = list(matrix.keys())
    levels = list(next(iter(matrix.values())).keys())
    z = [[float(matrix[f][l]) for l in levels] for f in raw_factors]
    figure = go.Figure(
        go.Heatmap(
            z=z, x=levels, y=factors, colorscale="Blues", zmin=0, zmax=1,
            colorbar=dict(title=dict(text="membership", font=dict(size=11, color=INK_SOFT))),
            hovertemplate="%{y} → %{x}: %{z:.3f}<extra></extra>",
        )
    )
    _style(
        figure,
        title="Fuzzy evaluation matrix R",
        height=height,
        left=180,
        right=24,
        top=84,
        bottom=72,
        showlegend=False,
    )
    return figure


__all__ = [
    "CLASS_COLORS",
    "POSITION_COLORS",
    "SQI_COLORS",
    "SQI_LABELS",
    "SQI_ORDER",
    "ecg_figure",
    "fuzzy_matrix_heatmap",
    "heatmap_figure",
    "membership_figure",
    "multilead_figure",
    "psd_figure",
    "quality_distribution_figure",
    "sqi_box_figure",
    "sqi_label",
    "sqi_trend_figure",
]

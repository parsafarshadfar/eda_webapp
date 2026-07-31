"""Interactive Plotly figures built from pre-aggregated summaries.

Nothing in this module ever receives a raw column. A histogram is sent to the
browser as a handful of bar heights and a box plot as five numbers plus a
capped list of outliers, so chart payloads stay in the kilobyte range no matter
how many rows the dataset has.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .summaries import (
    KIND_CATEGORICAL,
    QUANTILE,
    BoxSummary,
    CategoricalDistribution,
    NumericDistribution,
)

__all__ = [
    "PALETTE",
    "color_for",
    "histogram_figure",
    "box_figure",
    "correlation_heatmap",
    "missing_values_figure",
    "empty_figure",
]

#: Qualitative palette with enough contrast to stay readable in both themes.
PALETTE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#EECA3B",
    "#FF9DA6",
    "#9D755D",
    "#8C8C8C",
)

_BASE_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=60, r=25, t=55, b=50),
    hoverlabel=dict(font_size=12),
    separators=".,",
)

#: Above this many features the heatmap cells become too small for numbers.
_HEATMAP_ANNOTATION_LIMIT = 20


def color_for(index: int) -> str:
    """Stable colour for the feature at position ``index``."""
    return PALETTE[index % len(PALETTE)]


def empty_figure(message: str) -> go.Figure:
    """Placeholder used when a chart has nothing to show."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font=dict(size=13, color="#666"))
    figure.update_layout(
        **_BASE_LAYOUT,
        height=180,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return figure


def histogram_figure(
    summary: NumericDistribution | CategoricalDistribution,
    *,
    color: str = PALETTE[0],
    height: int = 320,
    show_percentage: bool = False,
) -> go.Figure:
    """Bar chart of a pre-binned distribution."""
    if summary.kind == KIND_CATEGORICAL:
        return _categorical_figure(summary, color=color, height=height, show_percentage=show_percentage)
    return _numeric_figure(summary, color=color, height=height, show_percentage=show_percentage)


def _axis_values(counts: np.ndarray, show_percentage: bool) -> tuple[np.ndarray, str, str]:
    total = int(counts.sum())
    if show_percentage and total:
        return counts / total * 100.0, "Percentage of valid values", ":.3f"
    return counts.astype(float), "Count", ":,d"


def _numeric_figure(
    summary: NumericDistribution,
    *,
    color: str,
    height: int,
    show_percentage: bool,
) -> go.Figure:
    if summary.n_valid == 0:
        return empty_figure(f"{summary.feature}: no non-missing values")

    values, y_title, _ = _axis_values(summary.counts, show_percentage)
    value_format = ".3f" if show_percentage else ",d"
    suffix = "%" if show_percentage else ""

    # Quantile bins have wildly different widths, so they read better on a
    # categorical axis; equal-width bins keep their true numeric positions.
    categorical_axis = summary.binning == QUANTILE

    figure = go.Figure()
    if categorical_axis:
        figure.add_bar(
            x=summary.labels,
            y=values,
            marker_color=color,
            hovertemplate=f"%{{x}}<br>%{{y:{value_format}}}{suffix}<extra></extra>",
        )
        figure.update_xaxes(type="category", title_text=f"{summary.feature} ({summary.binning} bins)")
    else:
        figure.add_bar(
            x=summary.centers,
            y=values,
            width=summary.widths * 0.98,
            marker_color=color,
            customdata=np.asarray(summary.labels, dtype=object),
            hovertemplate=f"%{{customdata}}<br>%{{y:{value_format}}}{suffix}<extra></extra>",
        )
        figure.update_xaxes(title_text=summary.feature)

    non_finite = getattr(summary, "n_non_finite", 0)
    extra = f", {non_finite:,} non-finite" if non_finite else ""
    subtitle = (
        f"{summary.n_valid:,} values, {summary.n_missing:,} missing{extra}, "
        f"{summary.n_bins} bins"
    )
    figure.update_yaxes(title_text=y_title, rangemode="tozero")
    figure.update_layout(
        **_BASE_LAYOUT,
        height=height,
        bargap=0.05,
        showlegend=False,
        title=dict(text=f"{summary.feature}<br><sup>{subtitle}</sup>", font=dict(size=14)),
    )
    return figure


def _categorical_figure(
    summary: CategoricalDistribution,
    *,
    color: str,
    height: int,
    show_percentage: bool,
) -> go.Figure:
    if summary.n_valid == 0:
        return empty_figure(f"{summary.feature}: no non-missing values")

    values, y_title, _ = _axis_values(summary.counts, show_percentage)
    value_format = ".3f" if show_percentage else ",d"
    suffix = "%" if show_percentage else ""

    figure = go.Figure()
    figure.add_bar(
        x=summary.categories,
        y=values,
        marker_color=color,
        hovertemplate=f"%{{x}}<br>%{{y:{value_format}}}{suffix}<extra></extra>",
    )
    subtitle = (
        f"{summary.n_valid:,} values, {summary.n_missing:,} missing, "
        f"{summary.n_categories:,} distinct categories"
    )
    figure.update_xaxes(type="category", title_text=summary.feature)
    figure.update_yaxes(title_text=y_title, rangemode="tozero")
    figure.update_layout(
        **_BASE_LAYOUT,
        height=height,
        bargap=0.2,
        showlegend=False,
        title=dict(text=f"{summary.feature}<br><sup>{subtitle}</sup>", font=dict(size=14)),
    )
    return figure


def box_figure(
    summary: BoxSummary,
    *,
    color: str = PALETTE[0],
    height: int = 260,
) -> go.Figure:
    """Horizontal box plot drawn from precomputed quartiles and fences."""
    if summary.n_valid == 0:
        return empty_figure(f"{summary.feature}: no non-missing values")
    if not np.isfinite([summary.q1, summary.median, summary.q3]).all():
        return empty_figure(f"{summary.feature}: no finite values")

    figure = go.Figure()
    figure.add_trace(
        go.Box(
            y=[summary.feature],
            q1=[summary.q1],
            median=[summary.median],
            q3=[summary.q3],
            lowerfence=[summary.lower_fence],
            upperfence=[summary.upper_fence],
            mean=[summary.mean],
            sd=[summary.std],
            orientation="h",
            boxmean=True,
            marker_color=color,
            line_width=1.4,
            name=summary.feature,
            hovertemplate=(
                f"min {summary.minimum:,.4g}<br>"
                f"lower fence {summary.lower_fence:,.4g}<br>"
                f"Q1 {summary.q1:,.4g}<br>"
                f"median {summary.median:,.4g}<br>"
                f"Q3 {summary.q3:,.4g}<br>"
                f"upper fence {summary.upper_fence:,.4g}<br>"
                f"max {summary.maximum:,.4g}<br>"
                f"mean {summary.mean:,.4g}<extra></extra>"
            ),
        )
    )

    if summary.outliers.size:
        # SVG, not Scattergl. A WebGL canvas created inside a container that is
        # hidden or zero-width at mount — a collapsed section, a column that has
        # not been laid out yet — comes up blank and only paints once something
        # forces a resize, which is why these plots used to appear only after
        # the full-screen button. Outliers are capped at _MAX_OUTLIER_POINTS
        # (1500), well inside what the SVG renderer handles comfortably.
        figure.add_trace(
            go.Scatter(
                x=summary.outliers,
                y=[summary.feature] * summary.outliers.size,
                mode="markers",
                marker=dict(color="#E45756", size=5, symbol="diamond", opacity=0.65),
                name="outliers",
                hovertemplate="outlier: %{x:,.6g}<extra></extra>",
            )
        )

    shown = "" if not summary.outliers_truncated else f" ({summary.outliers.size:,} shown)"
    non_finite = getattr(summary, "n_non_finite", 0)
    extra = f", {non_finite:,} non-finite" if non_finite else ""
    subtitle = (
        f"{summary.n_valid:,} values, {summary.n_missing:,} missing, "
        f"{summary.n_outliers:,} outside 1.5xIQR{shown}{extra}"
    )
    figure.update_xaxes(title_text=summary.feature)
    figure.update_yaxes(showticklabels=False)
    figure.update_layout(
        **_BASE_LAYOUT,
        height=height,
        showlegend=False,
        title=dict(text=f"{summary.feature}<br><sup>{subtitle}</sup>", font=dict(size=14)),
    )
    return figure


def correlation_heatmap(
    corr: pd.DataFrame,
    method: str,
    *,
    threshold: float | None = None,
    height: int | None = None,
    show_values: bool | None = None,
) -> go.Figure:
    """Correlation heatmap.

    Cell labels are drawn only for small matrices; annotating a wide matrix
    produces tens of thousands of SVG text nodes and stalls the browser.
    """
    labels = [str(c) for c in corr.columns]
    n = len(labels)
    if n == 0:
        return empty_figure("No numeric features available for correlation analysis")

    values = corr.to_numpy(dtype=np.float64, copy=False)
    if show_values is None:
        show_values = n <= _HEATMAP_ANNOTATION_LIMIT

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            colorscale="RdBu_r",
            texttemplate="%{z:.2f}" if show_values else None,
            textfont=dict(size=max(7, min(12, int(220 / max(n, 1))))),
            hovertemplate="%{y} vs %{x}<br>r = %{z:.4f}<extra></extra>",
            colorbar=dict(title="r", thickness=14),
        )
    )

    if threshold is not None:
        # Ring the pairs that clear the reporting threshold so they are visible
        # even when the matrix is too large for numeric labels.
        rows, cols = np.triu_indices(n, k=1)
        flagged = np.abs(values[rows, cols]) > float(threshold)
        for row, col in zip(rows[flagged], cols[flagged]):
            for x_index, y_index in ((col, row), (row, col)):
                figure.add_shape(
                    type="rect",
                    x0=x_index - 0.5,
                    x1=x_index + 0.5,
                    y0=y_index - 0.5,
                    y1=y_index + 0.5,
                    line=dict(color="#111111", width=1.4),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                )

    size = height or int(np.clip(90 + 26 * n, 320, 900))
    figure.update_layout(
        **{**_BASE_LAYOUT, "margin": dict(l=90, r=25, t=60, b=100)},
        height=size,
        title=dict(text=f"{method.capitalize()} correlation", font=dict(size=15)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
    )
    return figure


def missing_values_figure(missing: pd.DataFrame, *, height: int | None = None) -> go.Figure:
    """Horizontal bar chart of the percentage of missing values per column."""
    if missing.empty:
        return empty_figure("No missing values detected")

    ordered = missing.sort_values("percentage_of_missing", ascending=True, kind="stable")
    labels = [str(index) for index in ordered.index]

    figure = go.Figure(
        go.Bar(
            x=ordered["percentage_of_missing"].to_numpy(dtype=float),
            y=labels,
            orientation="h",
            marker_color=PALETTE[3],
            customdata=ordered["number_of_missing"].to_numpy(dtype=np.int64),
            hovertemplate="%{y}<br>%{x:.3f}% missing (%{customdata:,d} rows)<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Missing (%)", rangemode="tozero")
    figure.update_layout(
        **{**_BASE_LAYOUT, "margin": dict(l=160, r=25, t=50, b=45)},
        height=height or int(np.clip(90 + 26 * len(labels), 220, 900)),
        showlegend=False,
        title=dict(text="Missing values by column", font=dict(size=15)),
    )
    return figure

"""Static Matplotlib figures for the PDF and ZIP exports.

The object-oriented Matplotlib API is used throughout: figures are created as
``Figure`` instances rather than through ``pyplot``. ``pyplot`` keeps a global
registry of every figure it creates, which leaks memory in a long-running
Streamlit process unless each figure is explicitly closed. Working with
``Figure`` directly means a figure is freed as soon as it goes out of scope.

The figures here consume exactly the same summaries as the interactive charts,
so an exported page can never disagree with what was shown on screen.
"""

from __future__ import annotations

import inspect
import io
import math
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

# Type 42 keeps the text in exported PDFs as real, selectable and searchable
# text rather than outlines, which matters when a report is used as evidence.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .summaries import (
    KIND_CATEGORICAL,
    QUANTILE,
    BoxSummary,
    CategoricalDistribution,
    NumericDistribution,
)

__all__ = [
    "A4_LANDSCAPE",
    "PALETTE",
    "figure_to_png",
    "save_figure",
    "save_table_png",
    "table_figures",
    "heatmap_figure",
    "missing_bar_figure",
    "histogram_figure",
    "box_figure",
    "histogram_grid_figures",
    "box_grid_figures",
]

#: Page size shared by every exported page, in inches.
A4_LANDSCAPE = (11.69, 8.27)

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

_HEADER_COLOR = "#2F4B63"
_ROW_COLORS = ("#EEF3F7", "#FFFFFF")
_GRID_COLOR = "#D6DEE5"
_TEXT_COLOR = "#22303C"

_DEFAULT_DPI = 160

# Matplotlib 3.10 replaced ``bxp(vert=False)`` with ``bxp(orientation=...)``
# and deprecated the old spelling. Resolve it once so both are supported.
_HORIZONTAL_BOX = (
    {"orientation": "horizontal"}
    if "orientation" in inspect.signature(Axes.bxp).parameters
    else {"vert": False}
)


def _color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _format_cell(value: object, max_length: int = 22) -> str:
    """Compact, human-readable rendering of a single table cell."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,d}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ""
        if not math.isfinite(number):
            return "inf" if number > 0 else "-inf"
        magnitude = abs(number)
        if magnitude != 0 and (magnitude >= 1e7 or magnitude < 1e-3):
            return f"{number:.3e}"
        text = f"{number:,.4f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    text = str(value)
    if len(text) > max_length:
        return text[: max_length - 1] + "…"
    return text


def figure_to_png(figure: Figure, dpi: int = _DEFAULT_DPI, *, crop: bool = False) -> bytes:
    """Render a figure to PNG bytes without touching the pyplot registry.

    ``crop`` enables ``bbox_inches="tight"``. It roughly doubles the cost
    because the figure has to be drawn twice, so it is off by default and used
    only for tables, whose axes are placed manually.
    """
    FigureCanvasAgg(figure)
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=dpi,
        facecolor="white",
        **({"bbox_inches": "tight"} if crop else {}),
    )
    return buffer.getvalue()


def save_figure(figure: Figure, path: str, dpi: int = _DEFAULT_DPI, *, crop: bool = True) -> None:
    """Write a figure to disk as PNG."""
    FigureCanvasAgg(figure)
    figure.savefig(
        path, dpi=dpi, facecolor="white", **({"bbox_inches": "tight"} if crop else {})
    )


def save_table_png(frame: pd.DataFrame, path: str, title: str | None = None, dpi: int = _DEFAULT_DPI) -> None:
    """Render a whole DataFrame as a single PNG table."""
    figures = table_figures(frame, title=title, rows_per_page=len(frame) or 1, fit_page=False)
    if figures:
        save_figure(figures[0], path, dpi=dpi)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def table_figures(
    frame: pd.DataFrame,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    rows_per_page: int = 22,
    show_index: bool = True,
    index_label: str = "",
    fit_page: bool = True,
    highlight: Iterable[tuple[int, int]] | None = None,
) -> list[Figure]:
    """Render a DataFrame as one or more table pages.

    Parameters
    ----------
    rows_per_page
        Rows drawn on a single page before starting a new one.
    fit_page
        Use the shared A4 landscape page size. ``False`` sizes the figure to
        its content instead, which is what standalone PNGs want.
    highlight
        ``(row, column)`` positions, relative to the whole frame, to shade.
    """
    if frame is None or frame.shape[1] == 0:
        return [_message_figure(title or "Table", "No data available")]
    if frame.shape[0] == 0:
        return [_message_figure(title or "Table", "No rows to display")]

    columns = ([index_label] if show_index else []) + [str(c) for c in frame.columns]
    body = [[_format_cell(v) for v in row] for row in frame.to_numpy(dtype=object)]
    if show_index:
        for row, index_value in zip(body, frame.index):
            row.insert(0, _format_cell(index_value, max_length=28))

    highlight_set = set(highlight or ())
    rows_per_page = max(1, int(rows_per_page))
    n_pages = math.ceil(len(body) / rows_per_page)

    widths = _column_widths(columns, body)
    figures: list[Figure] = []

    for page in range(n_pages):
        start = page * rows_per_page
        chunk = body[start : start + rows_per_page]
        page_title = title if title is None or n_pages == 1 else f"{title} ({page + 1}/{n_pages})"
        page_highlight = {
            (r - start, c) for (r, c) in highlight_set if start <= r < start + rows_per_page
        }
        figures.append(
            _table_figure(
                columns,
                chunk,
                widths,
                title=page_title,
                subtitle=subtitle if page == 0 else None,
                fit_page=fit_page,
                highlight=page_highlight,
                index_column=show_index,
            )
        )
    return figures


def _column_widths(columns: Sequence[str], body: Sequence[Sequence[str]]) -> np.ndarray:
    lengths = np.array([len(c) for c in columns], dtype=float)
    for row in body:
        lengths = np.maximum(lengths, [len(cell) for cell in row])
    lengths = np.clip(lengths, 4.0, 30.0)
    return lengths / lengths.sum()


def _table_figure(
    columns: Sequence[str],
    body: Sequence[Sequence[str]],
    widths: np.ndarray,
    *,
    title: str | None,
    subtitle: str | None,
    fit_page: bool,
    highlight: set[tuple[int, int]],
    index_column: bool,
) -> Figure:
    n_rows = len(body)
    n_cols = len(columns)

    if fit_page:
        figsize = A4_LANDSCAPE
    else:
        width = float(np.clip(1.4 + 0.13 * sum(len(c) for c in columns) + 0.9 * n_cols, 6.0, 26.0))
        height = float(np.clip(1.3 + 0.32 * n_rows, 2.0, 40.0))
        figsize = (width, height)

    figure = Figure(figsize=figsize, dpi=_DEFAULT_DPI, facecolor="white")
    axes = figure.add_axes((0.02, 0.03, 0.96, 0.86 if title else 0.94))
    axes.axis("off")

    font_size = float(np.clip(150.0 / max(n_cols, 1), 5.5, 10.0))
    if fit_page:
        font_size = min(font_size, float(np.clip(360.0 / max(n_rows, 1), 5.5, 10.0)))

    table = axes.table(
        cellText=[list(row) for row in body],
        colLabels=list(columns),
        colWidths=list(widths),
        cellLoc="center",
        loc="upper center",
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_edgecolor("#FFFFFF")
        if row == 0:
            cell.set_facecolor(_HEADER_COLOR)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(cell.get_height() * 1.25)
        else:
            data_row = row - 1
            if (data_row, col) in highlight:
                cell.set_facecolor("#FFE9A8")
            else:
                cell.set_facecolor(_ROW_COLORS[row % len(_ROW_COLORS)])
            cell.set_text_props(color=_TEXT_COLOR)
            if index_column and col == 0:
                cell.set_text_props(color=_TEXT_COLOR, weight="bold")
                cell.get_text().set_horizontalalignment("left")

    if title:
        figure.suptitle(title, fontsize=13, fontweight="bold", color=_TEXT_COLOR, y=0.975)
    if subtitle:
        figure.text(0.5, 0.935, subtitle, ha="center", fontsize=9, color="#5A6B7A")
    return figure


def _message_figure(title: str, message: str) -> Figure:
    figure = Figure(figsize=A4_LANDSCAPE, dpi=_DEFAULT_DPI, facecolor="white")
    axes = figure.add_subplot(111)
    axes.axis("off")
    axes.text(0.5, 0.55, message, ha="center", va="center", fontsize=13, color="#5A6B7A")
    figure.suptitle(title, fontsize=13, fontweight="bold", color=_TEXT_COLOR, y=0.95)
    return figure


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def heatmap_figure(
    corr: pd.DataFrame,
    method: str,
    *,
    threshold: float | None = None,
    fit_page: bool = True,
) -> Figure:
    """Correlation heatmap rendered with Matplotlib's image backend."""
    labels = [str(c) for c in corr.columns]
    n = len(labels)
    if n == 0:
        return _message_figure(f"{method.capitalize()} correlation", "No numeric features available")

    values = corr.to_numpy(dtype=np.float64, copy=False)
    figsize = A4_LANDSCAPE if fit_page else (float(np.clip(2.5 + 0.42 * n, 5, 22)),) * 2

    figure = Figure(figsize=figsize, dpi=_DEFAULT_DPI, facecolor="white")
    axes = figure.add_subplot(111)
    image = axes.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")

    tick_size = float(np.clip(150.0 / max(n, 1), 4.0, 9.0))
    axes.set_xticks(np.arange(n))
    axes.set_yticks(np.arange(n))
    axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_size)
    axes.set_yticklabels(labels, fontsize=tick_size)
    axes.set_xticks(np.arange(-0.5, n, 1), minor=True)
    axes.set_yticks(np.arange(-0.5, n, 1), minor=True)
    axes.grid(which="minor", color="white", linewidth=0.4)
    axes.tick_params(which="minor", length=0)

    if n <= 20:
        for i in range(n):
            for j in range(n):
                value = values[i, j]
                if not np.isfinite(value):
                    continue
                axes.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=max(4.5, tick_size - 1.5),
                    color="white" if abs(value) > 0.55 else _TEXT_COLOR,
                )

    if threshold is not None:
        rows, cols = np.triu_indices(n, k=1)
        flagged = np.abs(values[rows, cols]) > float(threshold)
        for row, col in zip(rows[flagged], cols[flagged]):
            for x_index, y_index in ((col, row), (row, col)):
                axes.add_patch(
                    Rectangle(
                        (x_index - 0.5, y_index - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=1.1,
                    )
                )

    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    colorbar.set_label("correlation coefficient", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    subtitle = f"{n} numeric features"
    if threshold is not None:
        subtitle += f"  |  outlined cells exceed |r| > {float(threshold):g}"
    axes.set_title(f"{method.capitalize()} correlation\n{subtitle}", fontsize=12, color=_TEXT_COLOR, pad=12)
    figure.tight_layout()
    return figure


def missing_bar_figure(missing: pd.DataFrame, *, fit_page: bool = True) -> Figure:
    """Horizontal bar chart of missing-value percentages."""
    if missing.empty:
        return _message_figure("Missing values", "No missing values detected")

    ordered = missing.sort_values("percentage_of_missing", ascending=True, kind="stable")
    labels = [str(index) for index in ordered.index]
    percentages = ordered["percentage_of_missing"].to_numpy(dtype=float)
    counts = ordered["number_of_missing"].to_numpy(dtype=np.int64)

    figsize = A4_LANDSCAPE if fit_page else (10.0, float(np.clip(1.4 + 0.32 * len(labels), 2.5, 30)))
    figure = Figure(figsize=figsize, dpi=_DEFAULT_DPI, facecolor="white")
    axes = figure.add_subplot(111)

    positions = np.arange(len(labels))
    axes.barh(positions, percentages, color=PALETTE[3], height=0.72)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=float(np.clip(150.0 / max(len(labels), 1), 5.0, 9.5)))
    axes.set_xlabel("Missing (%)", fontsize=9)
    axes.set_title("Missing values by column", fontsize=12, color=_TEXT_COLOR, pad=10)
    axes.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)

    span = percentages.max() if percentages.size else 0.0
    for position, percentage, count in zip(positions, percentages, counts):
        axes.text(
            percentage + span * 0.01,
            position,
            f"{percentage:.2f}%  ({count:,})",
            va="center",
            fontsize=7,
            color="#5A6B7A",
        )
    axes.set_xlim(0, max(span * 1.22, 1e-9))
    figure.tight_layout()
    return figure


def _draw_histogram(axes, summary, color: str, show_percentage: bool) -> None:
    if summary.kind == KIND_CATEGORICAL:
        counts = summary.counts.astype(float)
        total = counts.sum()
        heights = counts / total * 100.0 if (show_percentage and total) else counts
        positions = np.arange(len(summary.categories))
        axes.bar(positions, heights, color=color, width=0.78)
        axes.set_xticks(positions)
        axes.set_xticklabels(
            summary.categories,
            rotation=45,
            ha="right",
            fontsize=max(5.0, 9.0 - 0.15 * len(summary.categories)),
        )
        axes.set_xlabel(summary.feature, fontsize=9)
    elif summary.n_valid == 0:
        axes.text(0.5, 0.5, "no non-missing values", ha="center", va="center", fontsize=8, color="#8899A6")
        axes.set_xlabel(summary.feature, fontsize=9)
        axes.set_xticks([])
        axes.set_yticks([])
        return
    else:
        counts = summary.counts.astype(float)
        total = counts.sum()
        heights = counts / total * 100.0 if (show_percentage and total) else counts
        if summary.binning == QUANTILE:
            positions = np.arange(summary.n_bins)
            axes.bar(positions, heights, color=color, width=0.86)
            axes.set_xticks(positions)
            axes.set_xticklabels(
                summary.labels,
                rotation=45,
                ha="right",
                fontsize=max(4.5, 8.0 - 0.18 * summary.n_bins),
            )
            axes.set_xlabel(f"{summary.feature} ({summary.binning} bins)", fontsize=9)
        else:
            axes.bar(
                summary.centers,
                heights,
                width=summary.widths * 0.98,
                color=color,
                align="center",
            )
            axes.set_xlabel(summary.feature, fontsize=9)
            axes.tick_params(axis="x", labelsize=7)

    axes.set_ylabel("Percentage" if show_percentage else "Count", fontsize=8)
    axes.tick_params(axis="y", labelsize=7)
    axes.grid(axis="y", color=_GRID_COLOR, linewidth=0.5)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    axes.set_title(
        f"{summary.feature}  ({summary.n_valid:,} values, {summary.n_missing:,} missing)",
        fontsize=8.5,
        color=_TEXT_COLOR,
    )


def histogram_figure(
    summary: NumericDistribution | CategoricalDistribution,
    *,
    color: str = PALETTE[0],
    show_percentage: bool = False,
    figsize: tuple[float, float] = (7.2, 4.0),
) -> Figure:
    """Standalone histogram/count plot for one feature."""
    figure = Figure(figsize=figsize, dpi=_DEFAULT_DPI, facecolor="white")
    axes = figure.add_subplot(111)
    _draw_histogram(axes, summary, color, show_percentage)
    figure.tight_layout()
    return figure


def _draw_box(axes, summary: BoxSummary, color: str) -> None:
    if summary.n_valid == 0:
        axes.text(0.5, 0.5, "no non-missing values", ha="center", va="center", fontsize=8, color="#8899A6")
        axes.set_xlabel(summary.feature, fontsize=9)
        axes.set_xticks([])
        axes.set_yticks([])
        return

    stats = [
        {
            "label": summary.feature,
            "med": summary.median,
            "q1": summary.q1,
            "q3": summary.q3,
            "whislo": summary.lower_fence,
            "whishi": summary.upper_fence,
            "mean": summary.mean,
            "fliers": summary.outliers,
        }
    ]
    axes.bxp(
        stats,
        **_HORIZONTAL_BOX,
        showmeans=True,
        meanline=False,
        patch_artist=True,
        widths=0.55,
        boxprops=dict(facecolor=color, alpha=0.55, edgecolor=color, linewidth=1.1),
        medianprops=dict(color="#22303C", linewidth=1.6),
        whiskerprops=dict(color=color, linewidth=1.1),
        capprops=dict(color=color, linewidth=1.1),
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#22303C", markersize=4),
        flierprops=dict(
            marker="d", markerfacecolor="#E45756", markeredgecolor="none", markersize=3, alpha=0.6
        ),
    )
    axes.set_yticklabels([])
    axes.set_xlabel(summary.feature, fontsize=9)
    axes.tick_params(axis="x", labelsize=7)
    axes.grid(axis="x", color=_GRID_COLOR, linewidth=0.5)
    axes.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axes.spines[spine].set_visible(False)
    shown = "" if not summary.outliers_truncated else f", {summary.outliers.size:,} drawn"
    axes.set_title(
        f"{summary.feature}  ({summary.n_valid:,} values, {summary.n_outliers:,} outside 1.5xIQR{shown})",
        fontsize=8.5,
        color=_TEXT_COLOR,
    )


def box_figure(
    summary: BoxSummary,
    *,
    color: str = PALETTE[0],
    figsize: tuple[float, float] = (7.2, 2.4),
) -> Figure:
    """Standalone box plot for one feature."""
    figure = Figure(figsize=figsize, dpi=_DEFAULT_DPI, facecolor="white")
    axes = figure.add_subplot(111)
    _draw_box(axes, summary, color)
    figure.tight_layout()
    return figure


def _grid_pages(
    n_items: int, n_cols: int, n_rows: int
) -> list[tuple[int, int]]:
    per_page = n_cols * n_rows
    return [(start, min(start + per_page, n_items)) for start in range(0, n_items, per_page)]


def histogram_grid_figures(
    summaries: Sequence[NumericDistribution | CategoricalDistribution],
    *,
    n_cols: int = 3,
    n_rows: int = 3,
    show_percentage: bool = False,
    title: str = "Distributions",
    color_offset: int = 0,
) -> list[Figure]:
    """Paginated grid of histograms sized for the shared page format."""
    if not summaries:
        return [_message_figure(title, "No features selected")]

    figures: list[Figure] = []
    pages = _grid_pages(len(summaries), n_cols, n_rows)
    for page_index, (start, stop) in enumerate(pages, start=1):
        figure = Figure(figsize=A4_LANDSCAPE, dpi=_DEFAULT_DPI, facecolor="white")
        axes_grid = figure.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
        flat = axes_grid.ravel()
        for offset, summary in enumerate(summaries[start:stop]):
            _draw_histogram(flat[offset], summary, _color(start + offset + color_offset), show_percentage)
        for axis in flat[stop - start :]:
            axis.axis("off")
        heading = title if len(pages) == 1 else f"{title} ({page_index}/{len(pages)})"
        figure.suptitle(heading, fontsize=13, fontweight="bold", color=_TEXT_COLOR)
        figure.tight_layout(rect=(0, 0, 1, 0.955))
        figures.append(figure)
    return figures


def box_grid_figures(
    summaries: Sequence[BoxSummary],
    *,
    n_rows: int = 6,
    title: str = "Outliers",
    color_offset: int = 0,
) -> list[Figure]:
    """Paginated stack of box plots sized for the shared page format."""
    if not summaries:
        return [_message_figure(title, "No numeric features selected")]

    figures: list[Figure] = []
    pages = _grid_pages(len(summaries), 1, n_rows)
    for page_index, (start, stop) in enumerate(pages, start=1):
        figure = Figure(figsize=A4_LANDSCAPE, dpi=_DEFAULT_DPI, facecolor="white")
        axes_grid = figure.subplots(nrows=n_rows, ncols=1, squeeze=False)
        flat = axes_grid.ravel()
        for offset, summary in enumerate(summaries[start:stop]):
            _draw_box(flat[offset], summary, _color(start + offset + color_offset))
        for axis in flat[stop - start :]:
            axis.axis("off")
        heading = title if len(pages) == 1 else f"{title} ({page_index}/{len(pages)})"
        figure.suptitle(heading, fontsize=13, fontweight="bold", color=_TEXT_COLOR)
        figure.tight_layout(rect=(0, 0, 1, 0.955))
        figures.append(figure)
    return figures

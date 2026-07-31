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
import os
import re
from pathlib import Path
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

from . import storage
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
    "dataframe_figure",
    "df_to_img",
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

# Table and chart images are read on screen and pasted into documents, so
# they are rendered well above screen resolution.
_DEFAULT_DPI = 240

SaveTarget = bool | str | os.PathLike[str]

_HEADER_ALIASES = {
    "DataType": "Data type",
    "n_uniques (excl. Nulls)": "Unique values\n(excl. nulls)",
    "n_Missing": "Missing\ncount",
    "Perc. of Missing": "Missing\n(%)",
    "n_Zeros": "Zero\ncount",
    "Perc. of Zeros": "Zeros\n(%)",
    "1st Most Freq": "Most frequent\nvalue",
    "Perc. of 1st Most Freq": "Most frequent\n(%)",
    "number_of_missing": "Missing\ncount",
    "percentage_of_missing": "Missing\n(%)",
    "n_outliers": "Outlier\ncount",
    "perc_outliers": "Outliers\n(%)",
    "share_of_rows_%": "Share of rows\n(%)",
}

# Matplotlib 3.10 replaced ``bxp(vert=False)`` with ``bxp(orientation=...)``
# and deprecated the old spelling. Resolve it once so both are supported.
_HORIZONTAL_BOX = (
    {"orientation": "horizontal"}
    if "orientation" in inspect.signature(Axes.bxp).parameters
    else {"vert": False}
)


def _color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _format_cell(value: object, max_length: int = 22, *, percentage: bool = False) -> str:
    """Compact, human-readable rendering of a single table cell."""
    if value is None:
        return "-"
    try:
        if bool(pd.isna(value)):
            return "-"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        text = f"{int(value):,d}"
        return f"{text}.000%" if percentage else text
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "-"
        if not math.isfinite(number):
            return "inf" if number > 0 else "-inf"
        if percentage:
            return f"{number:,.3f}%"
        magnitude = abs(number)
        if magnitude != 0 and (magnitude >= 1e7 or magnitude < 1e-3):
            return f"{number:.3e}"
        text = f"{number:,.3f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    text = str(value)
    if len(text) > max_length:
        return text[: max_length - 1] + "…"
    return text


def _wrap_two_lines(value: object, max_line_length: int) -> str:
    """Wrap text at a natural boundary, never using more than two lines."""
    text = str(value)
    if "\n" in text or len(text) <= max_line_length:
        return text

    # Prefer a break close to the middle. Spaces and punctuation boundaries
    # keep labels such as ``percentage_of_missing`` understandable.
    candidates: set[int] = set()
    for match in re.finditer(r"\s+|(?<=[_/\-])|(?=[([])", text):
        position = match.end() if match.group(0) else match.start()
        if 2 <= position <= len(text) - 2:
            candidates.add(position)

    if candidates:
        midpoint = len(text) / 2.0
        position = min(
            candidates,
            key=lambda point: (
                max(len(text[:point].rstrip()), len(text[point:].lstrip()))
                > max(22, max_line_length),
                abs(point - midpoint),
            ),
        )
    else:
        position = min(max_line_length, len(text) - 1)

    first = text[:position].rstrip()
    second = text[position:].lstrip()
    limit = max(22, max_line_length)
    if len(first) > limit:
        first = first[: limit - 1] + "…"
    if len(second) > limit:
        second = second[: limit - 1] + "…"
    return f"{first}\n{second}"


def _display_header(value: object, max_line_length: int = 15) -> str:
    """Return a readable header with no more than two balanced lines."""
    raw = str(value)
    return _wrap_two_lines(_HEADER_ALIASES.get(raw, raw), max_line_length)


def _visible_length(value: str) -> int:
    return max((len(line) for line in value.splitlines()), default=0)


def _is_percentage_column(value: object) -> bool:
    label = str(value).strip().casefold()
    return (
        "percentage" in label
        or "perc." in label
        or label.startswith("perc_")
        or label.startswith("perc ")
        or label.endswith("_%")
        or label.endswith("(%)")
    )


def _safe_image_stem(value: object) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return stem or "figure"


def _resolve_save_address(save: SaveTarget | None, default_name: str) -> str | None:
    """Turn a ``save`` argument into one output address, or ``None``.

    ``True`` means "the standard place with a sensible name" -- the folder
    reported by :func:`profiling.output_dir`. A folder address saves under that
    folder with the same sensible name; a file address is used verbatim, and a
    missing extension becomes ``.png``.
    """
    if save is False or save is None:
        return None
    if save is True:
        return storage.join_address(storage.default_output_dir(), default_name)
    if not isinstance(save, (str, os.PathLike)):
        raise TypeError("save must be False, True, or a filesystem/cloud address")

    address = str(save)
    looks_like_folder = address.rstrip().endswith(("/", "\\")) or (
        not storage.is_remote(address) and Path(address).is_dir()
    )
    if looks_like_folder:
        return storage.join_address(address, default_name)
    return storage.with_suffix(address, ".png")


def _save_requested(
    figure: Figure,
    save: SaveTarget | None,
    default_name: str,
    *,
    dpi: int = _DEFAULT_DPI,
    crop: bool = True,
) -> str | None:
    address = _resolve_save_address(save, default_name)
    if address is not None:
        save_figure(figure, address, dpi=dpi, crop=crop)
    return address


def _save_pages(
    figures: Sequence[Figure],
    save: SaveTarget | None,
    default_name: str,
    *,
    dpi: int = _DEFAULT_DPI,
    crop: bool = True,
) -> None:
    address = _resolve_save_address(save, default_name)
    if address is None:
        return

    if len(figures) == 1:
        save_figure(figures[0], address, dpi=dpi, crop=crop)
        return

    for page, figure in enumerate(figures, start=1):
        save_figure(figure, storage.numbered_address(address, page), dpi=dpi, crop=crop)


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


def save_figure(
    figure: Figure,
    path: str | os.PathLike[str],
    dpi: int = _DEFAULT_DPI,
    *,
    crop: bool = True,
) -> str:
    """Write a figure to ``path``, creating its parent folder when needed.

    The PNG is rendered in memory and then written in one operation, so a DBFS
    or ADLS address works exactly like a local one. The written address is
    returned, which is handy for printing it.
    """
    return storage.write_figure(figure, path, dpi=dpi, crop=crop)


def save_table_png(
    frame: pd.DataFrame,
    path: str | os.PathLike[str],
    title: str | None = None,
    dpi: int = _DEFAULT_DPI,
) -> str | None:
    """Render a whole DataFrame as a single PNG table."""
    figures = table_figures(frame, title=title, rows_per_page=len(frame) or 1, fit_page=False)
    return save_figure(figures[0], path, dpi=dpi) if figures else None


def dataframe_figure(
    frame: pd.DataFrame,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    decimals: int | None = 3,
    show_index: bool = False,
    index_label: str = "Feature",
    fit_page: bool = False,
    font_size: float | None = None,
    index_font_size: float | None = None,
    header_color: str = _HEADER_COLOR,
    row_colors: Sequence[str] = _ROW_COLORS,
    bbox: tuple[float, float, float, float] = (0, 0, 1, 1),
    wrap: int = 15,
    index_wrap: int = 28,
    save: SaveTarget = False,
    save_as: str | os.PathLike[str] | None = None,
    dpi: int = _DEFAULT_DPI,
) -> Figure:
    """Render any DataFrame as one table image, sized to its content.

    The one-liner for putting a table into a slide or a notebook cell as a
    picture rather than as text. Everything lands on a single page, however many
    rows there are; use :func:`table_figures` when pagination is wanted.

    Parameters
    ----------
    frame
        Table to draw. An empty frame produces a figure saying so rather than
        raising.
    subtitle
        Optional explanatory line displayed below the title.
    decimals
        Rounding applied to numeric cells before rendering. ``None`` leaves the
        values untouched.
    fit_page
        Use the shared A4 landscape canvas. The default sizes the image to its
        content, while ``True`` is convenient for a slide or report page.
    save
        ``False`` renders only. ``True`` also writes a named PNG into the
        standard output folder (:func:`profiling.output_dir`); a path or cloud
        address writes there instead. Parent folders are created automatically.
    save_as
        Backward-compatible alias for a custom ``save`` address. Do not pass both.
    """
    if frame is None or frame.empty:
        figure = _message_figure(title or "Table", "No data to display")
    else:
        rounded = frame.round(decimals) if decimals is not None else frame
        figure = table_figures(
            rounded,
            title=title,
            subtitle=subtitle,
            rows_per_page=max(len(rounded), 1),
            show_index=show_index,
            index_label=index_label,
            fit_page=fit_page,
            font_size=font_size,
            index_font_size=index_font_size,
            header_color=header_color,
            row_colors=row_colors,
            bbox=bbox,
            wrap=wrap,
            index_wrap=index_wrap,
        )[0]

    if save_as is not None:
        if save is not False and save is not None:
            raise ValueError("Pass either save or save_as, not both")
        save = save_as

    default_name = f"{_safe_image_stem(title or 'dataframe')}.png"
    _save_requested(figure, save, default_name, dpi=dpi, crop=not fit_page)
    return figure


def df_to_img(
    data,
    font_size: float = 10,
    index_font_size: float = 9,
    header_color: str = _HEADER_COLOR,
    row_colors: Sequence[str] = _ROW_COLORS,
    bbox: tuple[float, float, float, float] = (0, 0, 1, 1),
    title: str | None = None,
    wrap: int = 10,
    index_wrap: int = 10,
    save: SaveTarget = False,
    save_name: str | os.PathLike[str] = "Dataframe.png",
    show_index: bool = False,
    round: int | None = 3,
    **kwargs,
) -> Figure:
    """Render a dataframe as a blue, publication-ready table image.

    The familiar ``df_to_img`` interface, delegating to
    :func:`dataframe_figure`. ``save=True`` writes into the standard output
    folder (:func:`profiling.output_dir`); ``save=<address>`` -- or the legacy
    ``save=True, save_name=<address>`` -- writes to that address instead, local
    or cloud.
    """
    frame = data.to_pandas() if hasattr(data, "to_pandas") else data
    save_target: SaveTarget = save
    if save is True and Path(save_name) != Path("Dataframe.png"):
        save_target = save_name

    return dataframe_figure(
        frame,
        title=title,
        decimals=round,
        show_index=show_index,
        font_size=font_size,
        index_font_size=index_font_size,
        header_color=header_color,
        row_colors=row_colors,
        bbox=bbox,
        wrap=wrap,
        index_wrap=index_wrap,
        save=save_target,
        **kwargs,
    )


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
    font_size: float | None = None,
    index_font_size: float | None = None,
    header_color: str = _HEADER_COLOR,
    row_colors: Sequence[str] = _ROW_COLORS,
    bbox: tuple[float, float, float, float] = (0, 0, 1, 1),
    wrap: int = 15,
    index_wrap: int = 28,
    save: SaveTarget = False,
    dpi: int = _DEFAULT_DPI,
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
    save
        ``False`` renders only. ``True`` writes into the standard output folder
        (:func:`profiling.output_dir`); an address selects where instead.
        Multiple pages receive ``_01``, ``_02``, ... suffixes.
    """
    if frame is None or frame.shape[1] == 0:
        figures = [_message_figure(title or "Table", "No data available")]
        _save_pages(
            figures,
            save,
            f"{_safe_image_stem(title or 'table')}.png",
            dpi=dpi,
            crop=not fit_page,
        )
        return figures
    if frame.shape[0] == 0:
        figures = [_message_figure(title or "Table", "No rows to display")]
        _save_pages(
            figures,
            save,
            f"{_safe_image_stem(title or 'table')}.png",
            dpi=dpi,
            crop=not fit_page,
        )
        return figures

    raw_columns = ([index_label] if show_index else []) + [str(c) for c in frame.columns]
    columns = [_display_header(column, max(4, int(wrap))) for column in raw_columns]
    data_percentage_columns = [_is_percentage_column(column) for column in frame.columns]
    numeric_columns = ([False] if show_index else []) + [
        bool(pd.api.types.is_numeric_dtype(dtype))
        and not bool(pd.api.types.is_bool_dtype(dtype))
        for dtype in frame.dtypes.to_list()
    ]
    body = [
        [
            _format_cell(value, percentage=data_percentage_columns[column])
            for column, value in enumerate(row)
        ]
        for row in frame.to_numpy(dtype=object)
    ]
    if show_index:
        for row, index_value in zip(body, frame.index):
            formatted_index = _format_cell(
                index_value,
                max_length=max(28, 2 * max(4, int(index_wrap))),
            )
            row.insert(0, _wrap_two_lines(formatted_index, max(4, int(index_wrap))))

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
                numeric_columns=numeric_columns,
                requested_font_size=font_size,
                index_font_size=index_font_size,
                header_color=header_color,
                row_colors=row_colors,
                bbox=bbox,
            )
        )
    default_name = f"{_safe_image_stem(title or 'table')}.png"
    _save_pages(figures, save, default_name, dpi=dpi, crop=not fit_page)
    return figures


def _column_widths(columns: Sequence[str], body: Sequence[Sequence[str]]) -> np.ndarray:
    lengths = np.array([_visible_length(c) for c in columns], dtype=float)
    for row in body:
        lengths = np.maximum(lengths, [_visible_length(cell) for cell in row])
    lengths = np.clip(lengths + 1.5, 5.5, 24.0)
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
    numeric_columns: Sequence[bool],
    requested_font_size: float | None,
    index_font_size: float | None,
    header_color: str,
    row_colors: Sequence[str],
    bbox: tuple[float, float, float, float],
) -> Figure:
    n_rows = len(body)
    n_cols = len(columns)

    if fit_page:
        figsize = A4_LANDSCAPE
    else:
        width = float(
            np.clip(
                1.4 + 0.13 * sum(_visible_length(c) for c in columns) + 0.65 * n_cols,
                6.0,
                26.0,
            )
        )
        height = float(np.clip(1.3 + 0.32 * n_rows, 2.0, 40.0))
        figsize = (width, height)

    figure = Figure(figsize=figsize, dpi=_DEFAULT_DPI, facecolor="white")
    axes = figure.add_axes((0.02, 0.03, 0.96, 0.86 if title else 0.94))
    axes.axis("off")

    if requested_font_size is None:
        font_size = float(np.clip(128.0 / max(n_cols, 1), 5.5, 9.5))
        if fit_page:
            font_size = min(font_size, float(np.clip(330.0 / max(n_rows, 1), 5.5, 9.5)))
    else:
        font_size = max(1.0, float(requested_font_size))

    table = axes.table(
        cellText=[list(row) for row in body],
        colLabels=list(columns),
        colWidths=list(widths),
        cellLoc="center",
        loc="upper center",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    header_lines = any("\n" in column for column in columns)
    body_lines = any("\n" in value for row in body for value in row)
    header_weight = 1.75 if header_lines else 1.3
    body_weight = 1.45 if body_lines else 1.0
    row_unit = 1.0 / max(n_rows * body_weight + header_weight, 1.0)
    resolved_row_colors = tuple(row_colors) or _ROW_COLORS

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor(_GRID_COLOR)
        cell.PAD = 0.08
        cell.set_height(row_unit * (header_weight if row == 0 else body_weight))
        visible_chars = max(
            _visible_length(columns[col]),
            max((_visible_length(values[col]) for values in body), default=1),
            1,
        )
        available_points = (
            figsize[0] * 72.0 * 0.96 * max(float(bbox[2]), 0.01) * float(widths[col])
        )
        fitted_size = 0.76 * available_points / (0.58 * visible_chars)
        minimum_font_size = 1.0 if requested_font_size is not None else 5.0
        resolved_font_size = max(minimum_font_size, min(font_size, fitted_size))
        if row > 0 and index_column and col == 0 and index_font_size is not None:
            resolved_font_size = min(resolved_font_size, max(1.0, float(index_font_size)))
        cell.get_text().set_fontsize(resolved_font_size)
        cell.get_text().set_clip_on(True)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold", ha="center", va="center")
        else:
            data_row = row - 1
            if (data_row, col) in highlight:
                cell.set_facecolor("#FFE9A8")
            else:
                cell.set_facecolor(resolved_row_colors[row % len(resolved_row_colors)])
            cell.set_text_props(color=_TEXT_COLOR)
            if index_column and col == 0:
                cell.set_text_props(color=_TEXT_COLOR, weight="bold", ha="left")
            elif numeric_columns[col]:
                cell.get_text().set_horizontalalignment("right")
            else:
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
    save: SaveTarget = False,
    dpi: int = _DEFAULT_DPI,
) -> Figure:
    """Correlation heatmap rendered with Matplotlib's image backend.

    Pass ``save=True`` for a default PNG or ``save=path`` for a custom address.
    """
    labels = [str(c) for c in corr.columns]
    n = len(labels)
    if n == 0:
        figure = _message_figure(
            f"{method.capitalize()} correlation", "No numeric features available"
        )
        _save_requested(
            figure,
            save,
            f"{_safe_image_stem(method)}_correlation.png",
            dpi=dpi,
            crop=not fit_page,
        )
        return figure

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
    _save_requested(
        figure,
        save,
        f"{_safe_image_stem(method)}_correlation.png",
        dpi=dpi,
        crop=not fit_page,
    )
    return figure


def missing_bar_figure(
    missing: pd.DataFrame,
    *,
    fit_page: bool = True,
    save: SaveTarget = False,
    dpi: int = _DEFAULT_DPI,
) -> Figure:
    """Horizontal bar chart of missing-value percentages.

    Pass ``save=True`` for a default PNG or ``save=path`` for a custom address.
    """
    if missing.empty:
        figure = _message_figure("Missing values", "No missing values detected")
        _save_requested(
            figure,
            save,
            "missing_values.png",
            dpi=dpi,
            crop=not fit_page,
        )
        return figure

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
            f"{percentage:.3f}%  ({count:,})",
            va="center",
            fontsize=7,
            color="#5A6B7A",
        )
    axes.set_xlim(0, max(span * 1.22, 1e-9))
    figure.tight_layout()
    _save_requested(
        figure,
        save,
        "missing_values.png",
        dpi=dpi,
        crop=not fit_page,
    )
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
    non_finite = getattr(summary, "n_non_finite", 0)
    extra = f", {non_finite:,} non-finite" if non_finite else ""
    axes.set_title(
        f"{summary.feature}  ({summary.n_valid:,} values, "
        f"{summary.n_missing:,} missing{extra})",
        fontsize=8.5,
        color=_TEXT_COLOR,
    )


def histogram_figure(
    summary: NumericDistribution | CategoricalDistribution,
    *,
    color: str = PALETTE[0],
    show_percentage: bool = False,
    figsize: tuple[float, float] = (7.2, 4.0),
    save: SaveTarget = False,
    dpi: int = _DEFAULT_DPI,
) -> Figure:
    """Standalone histogram/count plot for one feature.

    Pass ``save=True`` for a default PNG or ``save=path`` for a custom address.
    """
    figure = Figure(figsize=figsize, dpi=_DEFAULT_DPI, facecolor="white")
    axes = figure.add_subplot(111)
    _draw_histogram(axes, summary, color, show_percentage)
    figure.tight_layout()
    _save_requested(
        figure,
        save,
        f"{_safe_image_stem(summary.feature)}_histogram.png",
        dpi=dpi,
    )
    return figure


def _draw_box(axes, summary: BoxSummary, color: str) -> None:
    if summary.n_valid == 0:
        axes.text(0.5, 0.5, "no non-missing values", ha="center", va="center", fontsize=8, color="#8899A6")
        axes.set_xlabel(summary.feature, fontsize=9)
        axes.set_xticks([])
        axes.set_yticks([])
        return
    if not np.isfinite([summary.q1, summary.median, summary.q3]).all():
        axes.text(
            0.5,
            0.5,
            "no finite values",
            ha="center",
            va="center",
            fontsize=8,
            color="#8899A6",
        )
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
    non_finite = getattr(summary, "n_non_finite", 0)
    extra = f", {non_finite:,} non-finite" if non_finite else ""
    axes.set_title(
        f"{summary.feature}  ({summary.n_valid:,} values, "
        f"{summary.n_outliers:,} outside 1.5xIQR{shown}{extra})",
        fontsize=8.5,
        color=_TEXT_COLOR,
    )


def box_figure(
    summary: BoxSummary,
    *,
    color: str = PALETTE[0],
    figsize: tuple[float, float] = (7.2, 2.4),
    save: SaveTarget = False,
    dpi: int = _DEFAULT_DPI,
) -> Figure:
    """Standalone box plot for one feature.

    Pass ``save=True`` for a default PNG or ``save=path`` for a custom address.
    """
    figure = Figure(figsize=figsize, dpi=_DEFAULT_DPI, facecolor="white")
    axes = figure.add_subplot(111)
    _draw_box(axes, summary, color)
    figure.tight_layout()
    _save_requested(
        figure,
        save,
        f"{_safe_image_stem(summary.feature)}_box_plot.png",
        dpi=dpi,
    )
    return figure


def _grid_pages(
    n_items: int, n_cols: int, n_rows: int
) -> list[tuple[int, int]]:
    if n_cols < 1 or n_rows < 1:
        raise ValueError("n_cols and n_rows must be positive integers")
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
    save: SaveTarget = False,
    dpi: int = _DEFAULT_DPI,
) -> list[Figure]:
    """Paginated grid of histograms sized for the shared page format.

    Saved multipage output receives deterministic ``_01``, ``_02``, ... names.
    """
    if isinstance(n_cols, bool) or isinstance(n_rows, bool):
        raise TypeError("n_cols and n_rows must be positive integers")
    try:
        n_cols, n_rows = int(n_cols), int(n_rows)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("n_cols and n_rows must be positive integers") from exc
    if n_cols < 1 or n_rows < 1:
        raise ValueError("n_cols and n_rows must be positive integers")

    if not summaries:
        figures = [_message_figure(title, "No features selected")]
        _save_pages(
            figures,
            save,
            f"{_safe_image_stem(title)}.png",
            dpi=dpi,
            crop=False,
        )
        return figures

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
    _save_pages(
        figures,
        save,
        f"{_safe_image_stem(title)}.png",
        dpi=dpi,
        crop=False,
    )
    return figures


def box_grid_figures(
    summaries: Sequence[BoxSummary],
    *,
    n_rows: int = 6,
    title: str = "Outliers",
    color_offset: int = 0,
    save: SaveTarget = False,
    dpi: int = _DEFAULT_DPI,
) -> list[Figure]:
    """Paginated stack of box plots sized for the shared page format.

    Saved multipage output receives deterministic ``_01``, ``_02``, ... names.
    """
    if isinstance(n_rows, bool):
        raise TypeError("n_rows must be a positive integer")
    try:
        n_rows = int(n_rows)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("n_rows must be a positive integer") from exc
    if n_rows < 1:
        raise ValueError("n_rows must be a positive integer")

    if not summaries:
        figures = [_message_figure(title, "No numeric features selected")]
        _save_pages(
            figures,
            save,
            f"{_safe_image_stem(title)}.png",
            dpi=dpi,
            crop=False,
        )
        return figures

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
    _save_pages(
        figures,
        save,
        f"{_safe_image_stem(title)}.png",
        dpi=dpi,
        crop=False,
    )
    return figures

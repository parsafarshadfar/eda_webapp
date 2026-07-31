"""Small rendering helpers shared by drift reports and the dashboard.

The drift engine intentionally keeps table rendering here rather than relying on
``dataframe_image``.  That keeps the dependency footprint small and, more
importantly, gives callers ownership of the returned Matplotlib figure so it can
be closed promptly in long-running applications.

Tables are drawn in the same blue as the profiling package, so a report that
mixes drift and profiling images looks like one document. Risky values can be
shaded amber and red in place of a separate status column -- see
``color_thresholds`` on :func:`df_to_img`.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import storage

__all__ = [
    "ALERT_COLOR",
    "DEFAULT_DPI",
    "HEADER_COLOR",
    "PSI_THRESHOLDS",
    "ROW_COLORS",
    "WARN_COLOR",
    "df_to_img",
]

#: Table header blue, shared with ``profiling.df_to_img``.
HEADER_COLOR = "#2F4B63"

#: Alternating body rows: a pale tint of the header, then white.
ROW_COLORS = ("#EEF3F7", "#FFFFFF")

#: Body text colour, dark enough to stay readable on both row colours.
TEXT_COLOR = "#22303C"

#: "Worth a look" and "act on this" cell fills. Both are pale enough to keep
#: dark body text readable, and they read the same way on a projector.
WARN_COLOR = "#FFF0BF"
ALERT_COLOR = "#F7C9C2"

#: The usual PSI triage bands: amber from 0.10, red from 0.25.
PSI_THRESHOLDS = (0.10, 0.25)

#: Resolution of a saved table image. Well above screen resolution, because
#: these end up in slides and documents where they get scaled up.
DEFAULT_DPI = 330


def _threshold_color(
    value: Any,
    thresholds: Sequence[float],
    *,
    higher_is_worse: bool,
    warn_color: str,
    alert_color: str,
) -> str | None:
    """Fill colour for one cell, or ``None`` to leave the row colour alone."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    warn, alert = float(thresholds[0]), float(thresholds[1])
    if higher_is_worse:
        if number >= alert:
            return alert_color
        if number >= warn:
            return warn_color
    else:  # small values are the bad ones, as with a p-value
        if number <= alert:
            return alert_color
        if number <= warn:
            return warn_color
    return None


def _threshold_legend(
    thresholds: Sequence[float],
    *,
    higher_is_worse: bool,
) -> str:
    """One line explaining what the shading means, drawn under the table."""
    warn, alert = float(thresholds[0]), float(thresholds[1])
    comparison = "≥" if higher_is_worse else "≤"
    return f"shaded amber {comparison} {warn:g}   red {comparison} {alert:g}"


def _wrapped_text(value: Any, width: int) -> str:
    """Wrap a label or cell at word boundaries, without changing the value.

    Breaking mid-word ("substantial sh / ift") makes a table image hard to read,
    so words are kept whole; only a single word longer than ``width`` is split.
    """
    text = str(value)
    if width <= 0 or len(text) <= width:
        return text

    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        while len(word) > width:  # one word too long to fit on any line
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:width])
            word = word[width:]
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _normalise_col_widths(widths: Iterable[float]) -> list[float]:
    """Convert display-character widths to the relative widths ax.table expects."""
    values = np.asarray(list(widths), dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return [1.0 / max(len(values), 1)] * len(values)
    return (values / total).tolist()


def df_to_img(
    data,
    font_size=10,
    index_font_size=9,
    header_color=HEADER_COLOR,
    row_colors=ROW_COLORS,
    bbox=(0, 0, 1, 1),
    title=None,
    wrap=10,
    index_wrap=10,
    save=False,
    save_name="Dataframe.png",
    show_index=False,
    round=3,
    color_thresholds=None,
    higher_is_worse=True,
    warn_color=WARN_COLOR,
    alert_color=ALERT_COLOR,
    **kwargs,
):
    """Render a dataframe as a blue Matplotlib table.

    The public signature is kept compatible with the original helper.  The
    implementation avoids a deep copy of the source frame, handles empty tables,
    produces bounded figure dimensions, and accepts ``save_name`` both with and
    without a file extension.

    Parameters
    ----------
    data
        A pandas DataFrame or an object accepted by ``pd.DataFrame``.
    wrap, index_wrap
        Maximum characters per line for cells and column labels.
    save
        ``False`` renders only. ``True`` writes ``save_name`` into the standard
        results folder (:func:`drift.output_dir`) when ``save_name`` is a bare
        file name, or to ``save_name`` itself when it is a full address. Passing
        an address as ``save`` directly works too. Local, DBFS and ADLS
        addresses are all written in one operation.
    round
        Number of decimal places used for display only.
    color_thresholds
        ``(warn, alert)`` pair that shades risky numeric cells amber and red --
        for example :data:`PSI_THRESHOLDS`, ``(0.10, 0.25)``. A legend line is
        drawn under the table so the image explains itself. ``None`` (default)
        keeps the plain alternating rows.
    higher_is_worse
        ``True`` (default) treats large values as the risky ones, which is right
        for PSI or a KS statistic. ``False`` shades small values instead, which
        is what a p-value needs.
    warn_color, alert_color
        Fill colours for the two bands.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure. Closing it stays the caller's responsibility.
    """
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except (TypeError, ValueError) as exc:
            raise TypeError("data must be convertible to a pandas DataFrame") from exc

    try:
        wrap = int(wrap)
        index_wrap = int(index_wrap)
        decimals = int(round)
    except (TypeError, ValueError) as exc:
        raise TypeError("wrap, index_wrap and round must be integers") from exc

    if wrap <= 0:
        warnings.warn("wrap must be positive; using 10.", RuntimeWarning, stacklevel=2)
        wrap = 10
    if index_wrap <= 0:
        warnings.warn("index_wrap must be positive; using 10.", RuntimeWarning, stacklevel=2)
        index_wrap = 10
    if not row_colors:
        raise ValueError("row_colors must contain at least one colour")

    # ``round`` already creates the display-only frame we need.  A shallow copy
    # is sufficient for non-numeric frames and avoids duplicating source blocks.
    try:
        display = data.round(decimals)
    except (TypeError, ValueError):
        display = data.copy(deep=False)

    if show_index:
        # Build the index column explicitly so an existing column named
        # ``"index"`` cannot collide with ``reset_index``.
        index_values = pd.Series(data.index.to_list(), name=" ")
        display = pd.concat(
            [index_values, display.reset_index(drop=True)],
            axis=1,
        )

    column_labels = [_wrapped_text(column, index_wrap) for column in display.columns]
    n_rows, n_cols = display.shape

    # Convert one column at a time.  Peak transient memory is therefore one
    # string Series instead of a second full object-typed dataframe.
    cell_columns: list[list[str]] = []
    cell_fills: list[list[str | None]] = []
    for column_index in range(n_cols):
        # Position-based access also supports duplicate user column labels.
        column = display.iloc[:, column_index]
        values = column.astype("string").fillna("<NA>")
        cell_columns.append([_wrapped_text(value, wrap) for value in values.array])
        if color_thresholds is None or (show_index and column_index == 0):
            cell_fills.append([None] * len(values))
        else:
            # Risky cells are shaded instead of being called out in a separate
            # status column; the numbers stay the numbers.
            cell_fills.append(
                [
                    _threshold_color(
                        value,
                        color_thresholds,
                        higher_is_worse=higher_is_worse,
                        warn_color=warn_color,
                        alert_color=alert_color,
                    )
                    for value in column.array
                ]
            )

    if n_rows:
        cell_text = [list(row) for row in zip(*cell_columns)]
    else:
        cell_text = []

    if n_cols:
        widths = []
        for index, label in enumerate(column_labels):
            body_width = max(
                (max((len(line) for line in value.splitlines()), default=0) for value in cell_columns[index]),
                default=0,
            )
            header_width = max((len(line) for line in label.splitlines()), default=0)
            widths.append(max(3, min(max(body_width, header_width), wrap) + 2))
        col_widths = _normalise_col_widths(widths)
    else:
        col_widths = []

    row_line_counts = [
        max((value.count("\n") + 1 for value in row), default=1) for row in cell_text
    ]
    header_lines = max((label.count("\n") + 1 for label in column_labels), default=1)

    # Bound canvas dimensions.  Extremely wide result tables remain renderable
    # instead of exceeding Matplotlib's pixel limit.
    width_inches = min(100.0, max(4.0, sum(widths) * 0.12 if n_cols else 4.0))
    height_inches = min(
        100.0,
        max(1.8, (sum(row_line_counts) + header_lines + (1 if title else 0)) * 0.32),
    )
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))
    ax.axis("off")

    if n_cols:
        table = ax.table(
            cellText=cell_text,
            bbox=bbox,
            colLabels=column_labels,
            colWidths=col_widths,
            rowLoc="center",
            cellLoc="center",
            colLoc="center",
            **kwargs,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)

        # Heights are relative to the table bounding box and sum to one.
        raw_heights = np.asarray([header_lines, *row_line_counts], dtype=np.float64)
        normalised_heights = raw_heights / max(float(raw_heights.sum()), 1.0)
        cells = table.get_celld()
        for (row_index, column_index), cell in cells.items():
            cell.set_linewidth(0.0)
            if 0 <= row_index < len(normalised_heights):
                cell.set_height(float(normalised_heights[row_index]))
            if row_index == 0 or column_index < 0:
                cell.set_text_props(color="white", weight="bold")
                cell.set_facecolor(header_color)
                cell.set_fontsize(max(1, font_size - 2))
            else:
                cell.set_text_props(color=TEXT_COLOR)
                if show_index and column_index == 0:
                    cell.set_text_props(color=TEXT_COLOR, weight="bold")
                    cell.set_fontsize(index_font_size)
                fill = cell_fills[column_index][row_index - 1]
                cell.set_facecolor(
                    fill or row_colors[(row_index - 1) % len(row_colors)]
                )
    else:
        ax.text(0.5, 0.5, "Empty dataframe", ha="center", va="center", fontsize=font_size)

    if title:
        ax.set_title(str(title), fontsize=max(10, font_size), color=TEXT_COLOR)

    if color_thresholds is not None and n_cols:
        # Draw the legend just below the table so the image explains its colours
        # without depending on a caption somewhere else.
        ax.text(
            0.5,
            -0.015,
            _threshold_legend(color_thresholds, higher_is_worse=higher_is_worse),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=max(6.0, font_size - 3),
            color="#5A6B7A",
        )

    if save is not False and save is not None:
        # ``save`` may itself be the address; otherwise ``save_name`` is one, and
        # a bare file name lands in the standard output folder.
        address = save if isinstance(save, (str, bytes)) or hasattr(save, "__fspath__") else save_name
        address = storage.with_suffix(address, ".png")
        if storage.parent_address(address) in {".", ""}:
            address = storage.join_address(storage.default_output_dir(), address)
        # Rendered to bytes first, then written once: see drift.storage.
        storage.write_figure(fig, address, dpi=DEFAULT_DPI)

    return fig

"""Data-profiling helper functions.

This module is the stable, public surface of the profiling toolkit: the same
function names and signatures the notebook workflow has always used. The
implementations now delegate to the vectorised engine in the :mod:`profiling`
package, so behaviour is unchanged while the heavy work runs far faster and
with a much smaller memory footprint.

Figures are built through Matplotlib's object-oriented API rather than
``pyplot``. ``pyplot`` keeps a global reference to every figure it creates,
which slowly leaks memory in a long-running process unless each one is closed
by hand.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from profiling import static as _static
from profiling.stats import correlation_matrix as _correlation_matrix
from profiling.stats import describe_data as _describe_data
from profiling.stats import missing_count as _missing_count
from profiling.stats import top_absolute_correlations as _top_absolute_correlations
from profiling.summaries import (
    build_box_summaries as _build_box_summaries,
)
from profiling.summaries import (
    build_distributions as _build_distributions,
)

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is not None:
    import pyspark.pandas as ps

__all__ = [
    "df_to_img",
    "describe_data",
    "missing_count",
    "perform_correlation_analysis",
    "plot_histograms_of_data_v3",
    "plot_outliers_v3",
    "merge_one_hot_encoded_columns",
]

#: Resolution used for saved figures. The previous value of 500 produced files
#: of tens of megabytes and dominated the runtime; 200 is still well beyond
#: print quality for these charts.
DEFAULT_DPI = 200


def _to_pandas(data):
    """Materialise a Spark frame as pandas; pass anything else through."""
    if "spark" in str(type(data)).lower() and hasattr(data, "to_pandas"):
        return data.to_pandas()
    return data


def df_to_img(
    data,
    font_size=10,
    index_font_size=9,
    header_color="forestgreen",
    row_colors=("#E5EEE4", "w"),
    bbox=(0, 0, 1, 1),
    title=None,
    wrap=10,
    index_wrap=10,
    save=False,
    save_name="Dataframe.png",
    show_index=False,
    round=3,
    **kwargs,
):
    """Render a dataframe as a table image.

    Parameters
    ----------
    data : DataFrame
        Table to draw.
    font_size, index_font_size : int
        Font sizes for the body and for the index column.
    header_color, row_colors : str, tuple
        Header fill and the alternating row fills.
    title : str, optional
        Title drawn above the table.
    show_index : bool
        Include the index as the first column.
    round : int
        Decimal places applied to numeric cells before rendering.
    save : bool
        Also write the image to ``save_name``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    frame = _to_pandas(data)
    if frame is None or frame.empty:
        figure = Figure(figsize=(6, 2), dpi=DEFAULT_DPI, facecolor="white")
        axes = figure.add_subplot(111)
        axes.axis("off")
        axes.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=12)
        return figure

    rounded = frame.round(round) if round is not None else frame
    figures = _static.table_figures(
        rounded,
        title=title,
        rows_per_page=max(len(rounded), 1),
        show_index=show_index,
        fit_page=False,
    )
    figure = figures[0]

    if save:
        _static.save_figure(figure, f"{save_name}.png", dpi=DEFAULT_DPI)
    return figure


def describe_data(
    data,
    numeric_only=True,
    round_columns=None,
    drop_columns=None,
    saving_path=None,
):
    """Per-column profile of a dataframe.

    Reports dtype, distinct-value count, missing and zero counts with their
    percentages, the most frequent value with its share, and the minimum, 1st,
    50th and 99th percentiles and maximum.

    Parameters
    ----------
    data : DataFrame
        Input data.
    numeric_only : bool
        Profile only the numeric columns (default). When ``False``, text and
        categorical columns are included and their numeric fields are null.
    round_columns : dict, optional
        ``{column: decimals}``. When given, those output columns are rendered
        as formatted strings. Left as ``None`` the values stay numeric, which
        is what CSV and Excel exports need.
    drop_columns : list, optional
        Output columns to omit.
    saving_path : str, optional
        Folder to write ``Data_Description.csv`` and ``Data_Description.png``
        into.

    Returns
    -------
    DataFrame
        One row per profiled column.
    """
    frame = _to_pandas(data)
    for name in frame.columns:
        if str(frame[name].dtype) in ("object", "category"):
            warnings.warn(
                f"Warning:{name} is not a numerical feature and the execution time may increase significantly",
                stacklevel=2,
            )
    return _describe_data(
        frame,
        numeric_only=numeric_only,
        round_columns=round_columns,
        drop_columns=drop_columns,
        saving_path=saving_path,
    )


def missing_count(data, sort=True, saving_path=None):
    """Counts and percentages of missing values, for the affected columns.

    Parameters
    ----------
    data : DataFrame
        Input data.
    sort : bool
        Sort descending by number of missing values.
    saving_path : str, optional
        Folder to write ``Missing_analysis.csv`` and ``Missing_analysis.png``
        into. The CSV lists every column; the returned frame lists only the
        columns that actually have missing values.

    Returns
    -------
    DataFrame
    """
    return _missing_count(_to_pandas(data), sort=sort, saving_path=saving_path)


def perform_correlation_analysis(data, methods=("Pearson", "Spearman"), top_corr_thr=0.7, saving_path=None):
    """Correlation analysis across one or more methods.

    Missing values are handled pairwise-complete, matching
    ``DataFrame.corr``. Spearman and Pearson are solved with a single matrix
    product when the data has no gaps; Kendall is spread over a thread pool.

    Parameters
    ----------
    data : DataFrame
        Input data. Non-numeric columns are ignored.
    methods : tuple
        Any of ``'pearson'``, ``'spearman'``, ``'kendall'`` (case-insensitive).
    top_corr_thr : float
        Pairs with an absolute coefficient above this are reported.
    saving_path : str, optional
        Folder to write the heatmaps, the ranked pair tables and
        ``absolute_correlations.xlsx`` into.

    Returns
    -------
    DataFrame
        Ranked feature pairs above the threshold for the last method computed.
    """
    frame = _to_pandas(data)
    top_abs_correlations = None
    workbook_sheets: dict[str, pd.DataFrame] = {}

    for method in methods:
        key = method.lower()
        corr_vals = _correlation_matrix(frame, key)
        if corr_vals.empty:
            continue

        abs_correlations = _top_absolute_correlations(corr_vals, None, method_label=key)
        top_abs_correlations = abs_correlations[
            abs_correlations[f"absolute {key} correlation values"] > top_corr_thr
        ].reset_index(drop=True)
        workbook_sheets[key] = abs_correlations

        if saving_path is not None:
            figure = _static.heatmap_figure(corr_vals, key, threshold=top_corr_thr, fit_page=False)
            _static.save_figure(figure, f"{saving_path}/Correlation_matrix_{key}.jpg", dpi=DEFAULT_DPI)
            if not top_abs_correlations.empty:
                _static.save_table_png(
                    top_abs_correlations.round(4),
                    f"{saving_path}/top_abs_correlations_{key}.png",
                    title=f"Top {key} correlations",
                    dpi=DEFAULT_DPI,
                )

    if saving_path is not None and workbook_sheets:
        with pd.ExcelWriter(f"{saving_path}/absolute_correlations.xlsx", engine="openpyxl", mode="w") as writer:
            for key, sheet in workbook_sheets.items():
                sheet.to_excel(writer, sheet_name=key[:31], index=False)

    return top_abs_correlations


def plot_histograms_of_data_v3(data, selected_features, saving_path, group_name, plot_kwargs):
    """Grid of histograms and count plots for the columns of ``data``.

    Parameters
    ----------
    data : DataFrame
        Input data.
    selected_features : list, optional
        Columns to plot. ``None`` plots every column.
    saving_path : str, optional
        Folder to write ``DataHistograms_<n>bins.jpg`` into.
    group_name : str
        Label used in the figure title.
    plot_kwargs : dict
        ``n_bins`` (default 40), ``n_cols`` (default 3) and ``stat``
        (``'count'`` or ``'probability'``, default ``'probability'``).

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    frame = _to_pandas(data)
    n_bins = plot_kwargs.get("n_bins", 40)
    n_cols = plot_kwargs.get("n_cols", 3)
    stat = plot_kwargs.get("stat", "probability")

    if selected_features is None:
        selected_features = list(frame.columns)
    selected_features = [f for f in selected_features if f in frame.columns]

    summaries = _build_distributions(frame, selected_features, n_bins=n_bins)
    ordered = [summaries[str(f)] for f in selected_features if str(f) in summaries]

    n_rows = max(1, int(np.ceil(len(ordered) / max(n_cols, 1))))
    figures = _static.histogram_grid_figures(
        ordered,
        n_cols=n_cols,
        n_rows=n_rows,
        show_percentage=stat != "count",
        title=f"Histogram plot: {group_name}",
    )
    figure = figures[0]

    if saving_path is not None:
        _static.save_figure(figure, f"{saving_path}/DataHistograms_{n_bins}bins.jpg", dpi=DEFAULT_DPI)

    return figure, np.asarray(figure.axes)


def plot_outliers_v3(data, selected_features, saving_path, group_name, plot_kwargs):
    """Stack of box plots for the numeric columns of ``data``.

    Parameters
    ----------
    data : DataFrame
        Input data.
    selected_features : list, optional
        Columns to plot. ``None`` plots every numeric column.
    saving_path : str, optional
        Folder to write ``Outliers.jpg`` into.
    group_name : str
        Label used in the figure title.
    plot_kwargs : dict
        Accepted for signature compatibility.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    frame = _to_pandas(data)
    if selected_features is None:
        selected_features = list(frame.columns)
    selected_features = [f for f in selected_features if f in frame.columns]

    summaries = _build_box_summaries(frame, selected_features)
    ordered = [summaries[str(f)] for f in selected_features if str(f) in summaries]

    figures = _static.box_grid_figures(
        ordered, n_rows=max(1, len(ordered)), title=f"Outliers plot: {group_name}"
    )
    figure = figures[0]

    if saving_path is not None:
        _static.save_figure(figure, f"{saving_path}/Outliers.jpg", dpi=DEFAULT_DPI)

    return figure, np.asarray(figure.axes)


def merge_one_hot_encoded_columns(df, OHE_features="Auto", dummy_separator="__"):
    """Collapse one-hot-encoded columns back into single categorical columns.

    Parameters
    ----------
    df : DataFrame
        Input data.
    OHE_features : dict, list or ``"Auto"``
        A mapping of original feature to its dummy columns, a list of original
        feature names, or ``"Auto"`` to infer groups from ``dummy_separator``.
    dummy_separator : str
        Separator used in the dummy column names.

    Returns
    -------
    DataFrame
        A copy with each dummy group replaced by one column holding the name of
        the winning dummy.
    """

    def _get_ohe_dict(features):
        columns = list(df.columns)
        return {feature: [c for c in columns if feature in c] for feature in features}

    if isinstance(OHE_features, list):
        OHE_features = _get_ohe_dict(OHE_features)

    if str(OHE_features).lower() == "auto":
        OHE_features = {}
        for column in df.columns:
            position = str(column).rfind(dummy_separator)
            if position != -1:
                original = str(column)[:position]
                OHE_features.setdefault(original, []).append(column)

    for merged_feature, dummy_features in OHE_features.items():
        if len(dummy_features) > 1:
            position = df.columns.get_loc(dummy_features[0])
            winner = df[dummy_features].idxmax(axis=1, skipna=True)
            df = df.drop(dummy_features, axis=1)
            df.insert(position, merged_feature, winner)

    return df

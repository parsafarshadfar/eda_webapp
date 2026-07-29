"""PDF and ZIP export of a completed profiling run.

The PDF is produced with Matplotlib's own ``PdfPages`` writer, so no additional
document library is needed: every page is a figure, drawn from the same
summaries the dashboard renders interactively.

Both bundles are self-describing. A cover page and a ``metadata.json`` record
the dataset fingerprint, the exact settings, the library versions and the
methodology, which is what makes an exported report usable as audit evidence.
"""

from __future__ import annotations

import io
import json
import re
import textwrap
import zipfile
from typing import Iterable, Sequence

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .pipeline import ProfilingResult
from .static import (
    A4_LANDSCAPE,
    box_figure,
    box_grid_figures,
    figure_to_png,
    heatmap_figure,
    histogram_figure,
    histogram_grid_figures,
    missing_bar_figure,
    table_figures,
)
from .summaries import (
    NumericDistribution,
    box_summaries_to_frame,
    distributions_to_frame,
)

__all__ = ["build_pdf", "build_zip", "safe_filename", "suggested_basename"]

_TEXT_COLOR = "#22303C"
_ACCENT_COLOR = "#2F4B63"
_MUTED_COLOR = "#5A6B7A"

#: Beyond this many features the bundle keeps the grid pages only, so a wide
#: dataset cannot produce a ZIP with thousands of images in it.
_MAX_INDIVIDUAL_CHARTS = 200

_METHODOLOGY = [
    (
        "Descriptive statistics",
        "Counts, distinct values, zeros and quantiles are read from a single sorted copy of each "
        "column. Quantiles (1%, 50%, 99%) use linear interpolation between order statistics, the "
        "same convention as numpy.quantile and pandas.Series.quantile. The most frequent value "
        "counts nulls as a candidate; ties resolve to the smallest value so repeated runs agree.",
    ),
    (
        "Missing values",
        "A value is missing when pandas reports it as null (NaN, NaT or None). Percentages are "
        "relative to the total row count of the dataset, not to the non-null count.",
    ),
    (
        "Correlation",
        "Pearson is the linear correlation of the raw values; Spearman is Pearson applied to "
        "average ranks; Kendall is the tau-b coefficient with tie correction. Pairs are formed "
        "from rows where both features are present (pairwise-complete observations), matching "
        "pandas.DataFrame.corr. Constant features yield an undefined coefficient and are left "
        "blank rather than reported as zero. When row subsampling is enabled it applies to every "
        "method, all of which are computed from the same seeded sample, so the coefficients stay "
        "directly comparable; the sample size and seed are stated on the cover page.",
    ),
    (
        "Distributions",
        "Equal-width binning splits the observed range, padded by 0.1% at each end so the maximum "
        "falls inside the last bin, into equally sized intervals. Quantile binning splits on "
        "empirical quantiles, drops duplicate edges and uses right-closed intervals with the "
        "lowest edge included. Bins count non-missing values only.",
    ),
    (
        "Outliers",
        "Boxes span the first quartile (Q1) to the third quartile (Q3), with the median marked "
        "and the mean shown as a diamond. The interquartile range (IQR) is the distance from Q1 "
        "to Q3. Whiskers extend to the most extreme observation still within 1.5 x IQR of the "
        "nearer quartile; anything beyond that is plotted as an outlier point. Outlier counts are "
        "exact; when a feature has more outliers than can be drawn legibly, an evenly spaced "
        "subset is plotted and the page says so.",
    ),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def safe_filename(name: str, fallback: str = "feature") -> str:
    """Turn an arbitrary column name into a portable file name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._-")
    return (cleaned or fallback)[:80]


def suggested_basename(result: ProfilingResult) -> str:
    """``<dataset>_profiling_<timestamp>`` with no unsafe characters."""
    stem = safe_filename(result.dataset_name.rsplit(".", 1)[0], "dataset")
    return f"{stem}_profiling_{result.generated_at:%Y%m%d_%H%M%S}"


def _safe_sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\[\]:*?/\\]", "_", str(name))[:31] or "Sheet"
    candidate = cleaned
    suffix = 1
    while candidate.lower() in used:
        candidate = f"{cleaned[: 31 - len(str(suffix)) - 1]}_{suffix}"
        suffix += 1
    used.add(candidate.lower())
    return candidate


def _text_page(
    title: str,
    lines: Sequence[tuple[str, str]],
    *,
    subtitle: str | None = None,
    wrap_at: int = 110,
) -> Figure:
    """A page of label/value or heading/paragraph pairs."""
    figure = Figure(figsize=A4_LANDSCAPE, dpi=160, facecolor="white")
    figure.text(0.06, 0.93, title, fontsize=16, fontweight="bold", color=_ACCENT_COLOR)
    if subtitle:
        figure.text(0.06, 0.895, subtitle, fontsize=10, color=_MUTED_COLOR)

    y = 0.845 if subtitle else 0.87
    for label, value in lines:
        if not label and not value:
            y -= 0.022
            continue
        figure.text(0.06, y, label, fontsize=9.5, fontweight="bold", color=_TEXT_COLOR)
        wrapped = _wrap(value, wrap_at)
        figure.text(0.28, y, wrapped, fontsize=9, color=_TEXT_COLOR, va="top")
        y -= 0.030 + 0.018 * (wrapped.count("\n"))
        if y < 0.06:
            break
    return figure


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width)) or ""


def _cover_page(result: ProfilingResult) -> Figure:
    figure = Figure(figsize=A4_LANDSCAPE, dpi=160, facecolor="white")
    figure.add_artist(
        Rectangle(
            (0, 0.78), 1, 0.22, transform=figure.transFigure, facecolor=_ACCENT_COLOR, zorder=0
        )
    )
    figure.text(0.06, 0.885, "Data Profiling Report", fontsize=30, fontweight="bold", color="white")
    figure.text(0.06, 0.835, result.dataset_name, fontsize=14, color="#D7E3ED")

    settings = result.settings
    included = [
        name
        for flag, name in (
            (settings.include_describe, "Descriptive statistics"),
            (settings.include_missing, "Missing values"),
            (settings.include_correlation, "Correlation"),
            (settings.include_histograms, "Distributions"),
            (settings.include_box_plots, "Outliers"),
        )
        if flag
    ]

    rows = [
        ("Generated", result.generated_at.strftime("%Y-%m-%d %H:%M:%S %Z").strip()),
        ("Rows", f"{result.n_rows:,}"),
        ("Columns in file", f"{result.n_cols_total:,}"),
        ("Columns analysed", f"{len(settings.selected_features):,}"),
        ("Numeric features", f"{len(result.numeric_features):,}"),
        ("Sections", ", ".join(included) or "none"),
        ("Binning", f"{settings.binning}, {settings.n_bins} bins"),
        (
            "Correlation",
            ", ".join(m.capitalize() for m in settings.correlation_methods) or "not run",
        ),
        ("Correlation threshold", f"|r| > {settings.correlation_threshold:g}"),
        ("Compute time", f"{result.total_seconds:.2f} s"),
    ]

    y = 0.66
    for label, value in rows:
        figure.text(0.06, y, label, fontsize=10, fontweight="bold", color=_TEXT_COLOR)
        figure.text(0.30, y, value, fontsize=10, color=_TEXT_COLOR)
        y -= 0.045

    if result.notes:
        figure.text(0.06, y - 0.015, "Notes", fontsize=10, fontweight="bold", color=_TEXT_COLOR)
        y -= 0.055
        for note in result.notes[:3]:
            figure.text(0.06, y, _wrap(f"- {note}", 120), fontsize=8.5, color=_MUTED_COLOR, va="top")
            y -= 0.038

    figure.text(
        0.06,
        0.04,
        "Every figure in this report is derived from the same pre-aggregated summaries as the "
        "interactive dashboard. See the methodology page for the exact definitions.",
        fontsize=8,
        color=_MUTED_COLOR,
    )
    return figure


class _PdfWriter:
    """Writes figures into a PDF, stamping a footer on each page."""

    def __init__(self, pdf: PdfPages, footer: str) -> None:
        self._pdf = pdf
        self._footer = footer
        self._page = 0

    def add(self, figure: Figure) -> None:
        self._page += 1
        figure.text(0.06, 0.015, self._footer, fontsize=7, color=_MUTED_COLOR)
        figure.text(0.94, 0.015, f"Page {self._page}", fontsize=7, color=_MUTED_COLOR, ha="right")
        self._pdf.savefig(figure)
        figure.clear()

    def add_all(self, figures: Iterable[Figure]) -> None:
        for figure in figures:
            self.add(figure)

    @property
    def page_count(self) -> int:
        return self._page


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def build_pdf(result: ProfilingResult) -> bytes:
    """Render a complete, paginated PDF report and return its bytes."""
    buffer = io.BytesIO()
    settings = result.settings
    footer = f"{result.dataset_name} - generated {result.generated_at:%Y-%m-%d %H:%M}"

    with PdfPages(buffer) as pdf:
        writer = _PdfWriter(pdf, footer)
        writer.add(_cover_page(result))
        writer.add(_overview_page(result))

        if settings.include_describe and result.describe is not None:
            writer.add_all(
                table_figures(
                    result.describe,
                    title="1. Descriptive statistics",
                    subtitle=f"{len(result.describe):,} numeric features",
                    rows_per_page=20,
                    show_index=True,
                    index_label="Feature",
                )
            )

        if settings.include_missing and result.missing_all is not None:
            writer.add(missing_bar_figure(result.missing, fit_page=True))
            table = result.missing if not result.missing.empty else result.missing_all
            writer.add_all(
                table_figures(
                    table.rename(
                        columns={
                            "number_of_missing": "Missing rows",
                            "percentage_of_missing": "Missing (%)",
                        }
                    ),
                    title="2. Missing values",
                    subtitle=(
                        "No missing values were found; all columns are listed for completeness."
                        if result.missing.empty
                        else f"{len(result.missing):,} of {len(result.missing_all):,} columns contain missing values"
                    ),
                    rows_per_page=22,
                    show_index=True,
                    index_label="Feature",
                )
            )

        if settings.include_correlation and result.correlations:
            for method, matrix in result.correlations.items():
                writer.add(
                    heatmap_figure(matrix, method, threshold=settings.correlation_threshold, fit_page=True)
                )
                top = result.top_correlations.get(method)
                if top is not None:
                    writer.add_all(
                        table_figures(
                            top.head(120).round(4),
                            title=f"3. Correlation - {method.capitalize()} pairs above threshold",
                            subtitle=(
                                f"{len(top):,} feature pairs exceed |r| > {settings.correlation_threshold:g}"
                                if len(top)
                                else f"No feature pair exceeds |r| > {settings.correlation_threshold:g}"
                            ),
                            rows_per_page=22,
                            show_index=False,
                        )
                    )

        if settings.include_histograms and result.distributions:
            summaries = list(result.distributions.values())
            writer.add_all(
                histogram_grid_figures(
                    summaries,
                    n_cols=3,
                    n_rows=3,
                    show_percentage=settings.show_percentage,
                    title=f"4. Distributions ({settings.binning} binning, {settings.n_bins} bins)",
                )
            )

        if settings.include_box_plots and result.box_summaries:
            writer.add_all(
                box_grid_figures(list(result.box_summaries.values()), n_rows=6, title="5. Outliers")
            )

        writer.add(_methodology_page())
        writer.add(_environment_page(result))

        info = pdf.infodict()
        info["Title"] = f"Data profiling report - {result.dataset_name}"
        info["Subject"] = "Automated exploratory data analysis and data quality profile"
        info["Keywords"] = "data profiling, EDA, data quality, model validation"
        info["Creator"] = "Data Profiler Dashboard"
        info["CreationDate"] = result.generated_at
        info["ModDate"] = result.generated_at

    return buffer.getvalue()


def _overview_page(result: ProfilingResult) -> Figure:
    dtype_counts = pd.Series(list(result.dtypes.values())).value_counts()
    dtype_text = ", ".join(f"{dtype} x{count}" for dtype, count in dtype_counts.items())
    source = result.source_info

    lines: list[tuple[str, str]] = [
        ("Dataset", result.dataset_name),
        ("Shape", f"{result.n_rows:,} rows x {result.n_cols_total:,} columns"),
        ("Analysed columns", f"{len(result.settings.selected_features):,}"),
        ("Column types", dtype_text or "n/a"),
        ("In-memory size", f"{result.memory_mb:,.2f} MB"),
    ]
    if source:
        if "parser" in source:
            lines.append(("CSV parser", str(source["parser"])))
        if "delimiter" in source:
            lines.append(("Delimiter", repr(source["delimiter"])))
        if "token" in source:
            lines.append(("Content fingerprint", f"blake2b-128 {source['token']}"))
        if "memory_saved_mb" in source:
            lines.append(("Memory reclaimed", f"{float(source['memory_saved_mb']):,.2f} MB by lossless dtype tuning"))

    lines.append(("", ""))
    lines.append(("Step timings", ""))
    for key, seconds in result.timings.items():
        lines.append((f"   {key.replace('_', ' ')}", f"{seconds:.3f} s"))
    lines.append(("   total", f"{result.total_seconds:.3f} s"))

    if result.notes:
        lines.append(("", ""))
        lines.append(("Notes", ""))
        for note in result.notes:
            lines.append(("", f"- {note}"))

    return _text_page("Dataset overview", lines, subtitle="Provenance and run summary")


def _methodology_page() -> Figure:
    figure = Figure(figsize=A4_LANDSCAPE, dpi=160, facecolor="white")
    figure.text(0.06, 0.93, "Methodology", fontsize=16, fontweight="bold", color=_ACCENT_COLOR)
    figure.text(
        0.06,
        0.895,
        "Definitions behind every number in this report.",
        fontsize=10,
        color=_MUTED_COLOR,
    )

    y = 0.835
    for heading, body in _METHODOLOGY:
        figure.text(0.06, y, heading, fontsize=10.5, fontweight="bold", color=_TEXT_COLOR)
        wrapped = _wrap(body, 128)
        figure.text(0.06, y - 0.026, wrapped, fontsize=8.6, color=_TEXT_COLOR, va="top")
        y -= 0.052 + 0.0175 * (wrapped.count("\n") + 1)
    return figure


def _environment_page(result: ProfilingResult) -> Figure:
    metadata = result.metadata()
    environment = metadata["environment"]
    lines = [(key, str(value)) for key, value in environment.items()]
    lines.append(("", ""))
    lines.append(("Settings", ""))
    for key, value in result.settings.as_dict().items():
        if key == "selected_features":
            value = f"{len(value)} columns"
        lines.append((f"   {key}", str(value)))
    return _text_page(
        "Reproducibility",
        lines,
        subtitle="Library versions and the exact settings used for this run",
    )


# --------------------------------------------------------------------------- #
# ZIP
# --------------------------------------------------------------------------- #
def build_zip(
    result: ProfilingResult,
    *,
    include_pdf: bool = True,
    include_individual_charts: bool = True,
    max_individual_charts: int = _MAX_INDIVIDUAL_CHARTS,
    pdf_bytes: bytes | None = None,
) -> bytes:
    """Bundle every table, chart and the PDF into one organised ZIP archive.

    Pass ``pdf_bytes`` when the PDF has already been rendered; rebuilding it is
    the single most expensive part of the export.
    """
    buffer = io.BytesIO()
    root = suggested_basename(result)
    settings = result.settings

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:

        def write_text(path: str, text: str) -> None:
            archive.writestr(f"{root}/{path}", text)

        def write_bytes(path: str, payload: bytes) -> None:
            archive.writestr(f"{root}/{path}", payload)

        def write_csv(path: str, frame: pd.DataFrame, index: bool = True) -> None:
            write_text(path, frame.to_csv(index=index, lineterminator="\n"))

        def write_png(path: str, figure: Figure, dpi: int = 160, crop: bool = False) -> None:
            write_bytes(path, figure_to_png(figure, dpi=dpi, crop=crop))

        write_text("metadata.json", json.dumps(result.metadata(), indent=2, default=str))

        if settings.include_describe and result.describe is not None:
            folder = "01_descriptive_statistics"
            write_csv(f"{folder}/descriptive_statistics.csv", result.describe)
            write_png(
                f"{folder}/descriptive_statistics.png",
                table_figures(
                    result.describe,
                    title="Descriptive statistics",
                    rows_per_page=max(len(result.describe), 1),
                    show_index=True,
                    index_label="Feature",
                    fit_page=False,
                )[0],
                crop=True,
            )

        if settings.include_missing and result.missing_all is not None:
            folder = "02_missing_values"
            write_csv(f"{folder}/missing_values_all_columns.csv", result.missing_all)
            write_csv(f"{folder}/missing_values_affected_columns.csv", result.missing)
            write_png(f"{folder}/missing_values.png", missing_bar_figure(result.missing, fit_page=False))

        if settings.include_correlation and result.correlations:
            folder = "03_correlation"
            for method, matrix in result.correlations.items():
                write_csv(f"{folder}/correlation_matrix_{safe_filename(method)}.csv", matrix.round(6))
                top = result.top_correlations.get(method)
                if top is not None:
                    write_csv(
                        f"{folder}/top_correlations_{safe_filename(method)}.csv",
                        top.round(6),
                        index=False,
                    )
                write_png(
                    f"{folder}/correlation_heatmap_{safe_filename(method)}.png",
                    heatmap_figure(
                        matrix, method, threshold=settings.correlation_threshold, fit_page=False
                    ),
                )
            write_bytes(f"{folder}/correlations.xlsx", _correlations_workbook(result))

        if settings.include_histograms and result.distributions:
            folder = "04_distributions"
            write_csv(f"{folder}/histogram_bins.csv", distributions_to_frame(result.distributions), index=False)
            summaries = list(result.distributions.values())
            for page, figure in enumerate(
                histogram_grid_figures(
                    summaries, n_cols=3, n_rows=3, show_percentage=settings.show_percentage
                ),
                start=1,
            ):
                write_png(f"{folder}/overview_page_{page:02d}.png", figure)
            if include_individual_charts and len(summaries) <= max_individual_charts:
                for index, summary in enumerate(summaries):
                    write_png(
                        f"{folder}/by_feature/{index + 1:03d}_{safe_filename(summary.feature)}.png",
                        histogram_figure(
                            summary,
                            color=_palette_color(index),
                            show_percentage=settings.show_percentage,
                        ),
                    )

        if settings.include_box_plots and result.box_summaries:
            folder = "05_outliers"
            write_csv(
                f"{folder}/box_plot_statistics.csv",
                box_summaries_to_frame(result.box_summaries).round(6),
                index=False,
            )
            summaries = list(result.box_summaries.values())
            for page, figure in enumerate(box_grid_figures(summaries, n_rows=6), start=1):
                write_png(f"{folder}/overview_page_{page:02d}.png", figure)
            if include_individual_charts and len(summaries) <= max_individual_charts:
                for index, summary in enumerate(summaries):
                    write_png(
                        f"{folder}/by_feature/{index + 1:03d}_{safe_filename(summary.feature)}.png",
                        box_figure(summary, color=_palette_color(index)),
                    )

        if include_pdf:
            write_bytes(
                "report.pdf",
                pdf_bytes if pdf_bytes is not None else build_pdf(result),
            )

        write_text("README.txt", _zip_readme(result, include_pdf=include_pdf))

    return buffer.getvalue()


def _palette_color(index: int) -> str:
    from .static import PALETTE

    return PALETTE[index % len(PALETTE)]


def _correlations_workbook(result: ProfilingResult) -> bytes:
    """One Excel workbook holding every matrix and every ranked pair list."""
    buffer = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for method, matrix in result.correlations.items():
            matrix.round(6).to_excel(writer, sheet_name=_safe_sheet_name(f"matrix_{method}", used))
        for method, top in result.top_correlations.items():
            top.round(6).to_excel(
                writer, sheet_name=_safe_sheet_name(f"pairs_{method}", used), index=False
            )
    return buffer.getvalue()


def _zip_readme(result: ProfilingResult, *, include_pdf: bool) -> str:
    settings = result.settings
    sections = []
    if settings.include_describe:
        sections.append(
            "01_descriptive_statistics/  Per-feature profile: dtype, distinct values, missing and\n"
            "                            zero counts, most frequent value, min / 1% / 50% / 99% / max.\n"
            "                            CSV holds full precision; the PNG is a formatted rendering."
        )
    if settings.include_missing:
        sections.append(
            "02_missing_values/          Missing counts and percentages. The '_all_columns' file lists\n"
            "                            every analysed column; the '_affected_columns' file lists only\n"
            "                            those with at least one missing value."
        )
    if settings.include_correlation:
        sections.append(
            "03_correlation/             Full correlation matrix per method, the ranked list of feature\n"
            "                            pairs above the threshold, a heatmap per method, and an Excel\n"
            "                            workbook containing all of the above on separate sheets."
        )
    if settings.include_histograms:
        sections.append(
            "04_distributions/           histogram_bins.csv contains every bin edge and count, so the\n"
            "                            charts can be reproduced exactly. overview_page_*.png are grid\n"
            "                            views; by_feature/ holds one chart per feature."
        )
    if settings.include_box_plots:
        sections.append(
            "05_outliers/                box_plot_statistics.csv contains quartiles, Tukey fences and\n"
            "                            exact outlier counts per feature, plus matching charts."
        )

    pdf_line = "report.pdf                  Full paginated report, including the methodology page.\n" if include_pdf else ""

    return f"""DATA PROFILING REPORT
=====================

Dataset      : {result.dataset_name}
Shape        : {result.n_rows:,} rows x {result.n_cols_total:,} columns
Analysed     : {len(settings.selected_features):,} columns ({len(result.numeric_features):,} numeric)
Generated    : {result.generated_at:%Y-%m-%d %H:%M:%S %Z}
Compute time : {result.total_seconds:.2f} s

CONTENTS
--------
{pdf_line}metadata.json               Machine-readable record of the dataset fingerprint, the exact
                            settings, step timings and library versions used for this run.

{chr(10).join(sections)}

REPRODUCIBILITY
---------------
Loading the dataset again and re-running with the settings recorded in
metadata.json reproduces every number in this bundle. Correlation subsampling,
when enabled, is seeded and the seed is recorded.

NOTES
-----
{chr(10).join(f'- {note}' for note in result.notes) if result.notes else '- none'}
"""

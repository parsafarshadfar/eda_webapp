"""Orchestration: turn a DataFrame plus a settings object into a result object.

Everything the dashboard shows and everything the exports contain comes from a
single :class:`ProfilingResult`. Running the analysis once and then rendering
it many times is what keeps the app responsive, and it is also what guarantees
the PDF, the ZIP and the screen agree.

:func:`run_segmented_profiling` is the same pipeline applied once per segment of
a split dataset. It returns a :class:`SegmentedResult`, which holds one ordinary
:class:`ProfilingResult` per segment plus the cross-segment comparison tables --
so every renderer and exporter that understands a single run also works on one
segment of a segmented run, with no second code path.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import __version__, _spark
from .dataio import (
    DEFAULT_TAG_COLUMN,
    combine_datasets,
    is_numeric_series,
    numeric_columns,
    to_pandas,
)
from .grouping import BY_VALUE, Grouping, Segment, make_grouping
from .stats import (
    correlation_matrices,
    describe_data,
    missing_count,
    top_absolute_correlations,
)
from .summaries import (
    EQUAL_WIDTH,
    BoxSummary,
    CategoricalDistribution,
    NumericDistribution,
    box_summaries_to_frame,
    build_box_summaries,
    build_distributions,
    distributions_to_frame,
)

__all__ = [
    "ProfilingSettings",
    "ProfilingResult",
    "SegmentedResult",
    "run_profiling",
    "run_segmented_profiling",
    "environment_report",
]

ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class ProfilingSettings:
    """Everything that influences the numbers in a profiling run.

    The whole object is written into the exported metadata, so a report can
    always be traced back to the configuration that produced it.
    """

    selected_features: tuple[str, ...] = ()
    include_describe: bool = True
    include_missing: bool = True
    include_correlation: bool = True
    include_histograms: bool = True
    include_box_plots: bool = True
    correlation_methods: tuple[str, ...] = ("kendall",)
    correlation_threshold: float = 0.65
    correlation_max_rows: int | None = None
    binning: str = EQUAL_WIDTH
    n_bins: int = 8
    max_categories: int = 30
    show_percentage: bool = False
    random_state: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_features": list(self.selected_features),
            "n_selected_features": len(self.selected_features),
            "include_describe": self.include_describe,
            "include_missing": self.include_missing,
            "include_correlation": self.include_correlation,
            "include_histograms": self.include_histograms,
            "include_box_plots": self.include_box_plots,
            "correlation_methods": list(self.correlation_methods),
            "correlation_threshold": self.correlation_threshold,
            "correlation_max_rows": self.correlation_max_rows,
            "binning": self.binning,
            "n_bins": self.n_bins,
            "max_categories": self.max_categories,
            "histogram_units": "percentage" if self.show_percentage else "count",
            "random_state": self.random_state,
        }


@dataclass
class ProfilingResult:
    """The complete output of one profiling run."""

    dataset_name: str
    settings: ProfilingSettings
    n_rows: int
    n_cols_total: int
    generated_at: datetime

    describe: pd.DataFrame | None = None
    missing: pd.DataFrame | None = None
    missing_all: pd.DataFrame | None = None
    correlations: dict[str, pd.DataFrame] = field(default_factory=dict)
    top_correlations: dict[str, pd.DataFrame] = field(default_factory=dict)
    distributions: dict[str, NumericDistribution | CategoricalDistribution] = field(default_factory=dict)
    box_summaries: dict[str, BoxSummary] = field(default_factory=dict)

    numeric_features: tuple[str, ...] = ()
    dtypes: dict[str, str] = field(default_factory=dict)
    memory_mb: float = 0.0
    source_info: dict[str, object] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return float(sum(self.timings.values()))

    @property
    def correlation_sampled(self) -> bool:
        limit = self.settings.correlation_max_rows
        return limit is not None and self.n_rows > limit

    def metadata(self) -> dict[str, object]:
        """JSON-serialisable description of the run, for the export bundle."""
        return {
            "report_format_version": __version__,
            "dataset": {
                "name": self.dataset_name,
                "n_rows": self.n_rows,
                "n_columns_total": self.n_cols_total,
                "n_columns_selected": len(self.settings.selected_features),
                "numeric_features": list(self.numeric_features),
                "dtypes": self.dtypes,
                "in_memory_size_mb": round(self.memory_mb, 4),
                **{k: v for k, v in self.source_info.items()},
            },
            "settings": self.settings.as_dict(),
            "correlation_rows_used": (
                min(self.n_rows, self.settings.correlation_max_rows)
                if self.correlation_sampled
                else self.n_rows
            ),
            "correlation_subsampled": self.correlation_sampled,
            "generated_at_utc": self.generated_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "timings_seconds": {k: round(v, 4) for k, v in self.timings.items()},
            "environment": environment_report(),
            "notes": list(self.notes),
        }


def environment_report() -> dict[str, str]:
    """Library versions, recorded so a report can be reproduced later."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(terse=True),
    }
    for module_name in ("numpy", "pandas", "scipy", "matplotlib", "plotly", "streamlit", "seaborn"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return versions


def run_profiling(
    data: pd.DataFrame,
    settings: ProfilingSettings,
    *,
    dataset_name: str = "dataset",
    dtype_labels: dict[str, str] | None = None,
    source_info: dict[str, object] | None = None,
    progress: ProgressCallback | None = None,
) -> ProfilingResult:
    """Run every enabled analysis once and collect the results.

    Parameters
    ----------
    data
        The full frame. Only ``settings.selected_features`` are analysed.
    settings
        Analysis configuration.
    dataset_name
        Used in report headers and file names.
    dtype_labels
        Original dtypes, so the report shows the types as they were in the
        source file even after lossless downcasting.
    source_info
        Extra provenance (parser used, delimiter, file fingerprint) copied into
        the exported metadata.
    progress
        Optional ``callback(label, fraction)`` used to drive a progress bar.
    """
    features = [f for f in settings.selected_features if f in data.columns]
    if not features:
        features = list(data.columns)
        settings = replace(settings, selected_features=tuple(features))

    frame = data[features]
    numeric = numeric_columns(frame)

    result = ProfilingResult(
        dataset_name=dataset_name,
        settings=settings,
        n_rows=int(data.shape[0]),
        n_cols_total=int(data.shape[1]),
        generated_at=datetime.now().astimezone(),
        numeric_features=tuple(numeric),
        dtypes={str(name): str(frame[name].dtype) for name in features},
        memory_mb=float(frame.memory_usage(deep=True).sum()) / (1024.0 * 1024.0),
        source_info=dict(source_info or {}),
    )

    if dtype_labels:
        result.dtypes = {name: dtype_labels.get(name, result.dtypes[name]) for name in result.dtypes}

    steps = _enabled_steps(settings)
    done = 0

    def advance(label: str) -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(label, min(1.0, done / max(len(steps), 1)))

    if settings.include_describe:
        with _timer(result, "descriptive_statistics"):
            result.describe = describe_data(frame, numeric_only=True, dtype_labels=dtype_labels)
        advance("Descriptive statistics")

    if settings.include_missing:
        with _timer(result, "missing_values"):
            result.missing_all = missing_count(frame, sort=True, non_zero_only=False)
            result.missing = result.missing_all[result.missing_all["number_of_missing"] > 0].copy()
        advance("Missing values")

    if settings.include_correlation and settings.correlation_methods:
        with _timer(result, "correlation"):
            if len(numeric) < 2:
                result.notes.append(
                    "Correlation analysis skipped: fewer than two numeric features were selected."
                )
            else:
                result.correlations = correlation_matrices(
                    frame,
                    settings.correlation_methods,
                    columns=numeric,
                    max_rows=settings.correlation_max_rows,
                    random_state=settings.random_state,
                )
                for method, matrix in result.correlations.items():
                    result.top_correlations[method] = top_absolute_correlations(
                        matrix, settings.correlation_threshold, method_label=method
                    )
                if result.correlation_sampled:
                    result.notes.append(
                        f"Correlations computed on a deterministic subsample of "
                        f"{settings.correlation_max_rows:,} rows "
                        f"(seed {settings.random_state}) out of {result.n_rows:,}."
                    )
        advance("Correlation analysis")

    if settings.include_histograms:
        with _timer(result, "distributions"):
            result.distributions = build_distributions(
                frame,
                features,
                binning=settings.binning,
                n_bins=settings.n_bins,
                max_categories=settings.max_categories,
            )
        advance("Distributions")

    if settings.include_box_plots:
        with _timer(result, "box_plots"):
            result.box_summaries = build_box_summaries(frame, numeric)
        advance("Box plots")

    skipped = [f for f in features if not is_numeric_series(frame[f])]
    if skipped and settings.include_box_plots:
        result.notes.append(
            f"{len(skipped)} non-numeric feature(s) are excluded from correlation and box-plot "
            "analysis: " + ", ".join(map(str, skipped[:12])) + ("..." if len(skipped) > 12 else "")
        )

    if progress is not None:
        progress("Done", 1.0)
    return result


def _enabled_steps(settings: ProfilingSettings) -> list[str]:
    flags = [
        (settings.include_describe, "describe"),
        (settings.include_missing, "missing"),
        (settings.include_correlation and bool(settings.correlation_methods), "correlation"),
        (settings.include_histograms, "histograms"),
        (settings.include_box_plots, "box"),
    ]
    return [name for enabled, name in flags if enabled]


class _timer:
    """Context manager that records the wall time of one pipeline step."""

    def __init__(self, result: ProfilingResult, key: str) -> None:
        self._result = result
        self._key = key
        self._start = 0.0

    def __enter__(self) -> "_timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._result.timings[self._key] = time.perf_counter() - self._start


# --------------------------------------------------------------------------- #
# segmented profiling
# --------------------------------------------------------------------------- #
@dataclass
class SegmentedResult:
    """One :class:`ProfilingResult` per segment, plus the comparisons across them.

    Indexing is by segment key -- ``segmented["Train"]`` or
    ``segmented["(10.0, 26.5]"]`` -- and each value is an ordinary
    :class:`ProfilingResult`, so anything that renders or exports a single run
    also handles one segment of a segmented run.

    The ``*_comparison`` methods return long-format frames with a ``segment``
    column in front, which is the shape that pivots, plots and diffs cleanly.
    """

    dataset_name: str
    settings: ProfilingSettings
    grouping: Grouping
    results: dict[str, ProfilingResult]
    generated_at: datetime
    source_info: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results.values())

    def __getitem__(self, key: str | int) -> ProfilingResult:
        if isinstance(key, int):
            return list(self.results.values())[key]
        if key in self.results:
            return self.results[key]
        for segment in self.grouping.segments:
            if key == segment.label:
                return self.results[segment.key]
        raise KeyError(key)

    def items(self):
        return self.results.items()

    @property
    def grouping_var(self) -> str:
        return self.grouping.grouping_var

    @property
    def method(self) -> str:
        return self.grouping.method

    @property
    def keys(self) -> list[str]:
        return list(self.results.keys())

    @property
    def total_seconds(self) -> float:
        return float(sum(result.total_seconds for result in self.results.values()))

    # -- comparisons -------------------------------------------------------- #
    def segment_overview(self) -> pd.DataFrame:
        """Rows, features and compute time per segment."""
        rows = [
            {
                "segment": key,
                "n_rows": result.n_rows,
                "share_of_rows_%": round(100.0 * result.n_rows / self.grouping.n_rows_total, 4)
                if self.grouping.n_rows_total
                else 0.0,
                "n_features": len(result.settings.selected_features),
                "n_numeric_features": len(result.numeric_features),
                "memory_mb": round(result.memory_mb, 4),
                "seconds": round(result.total_seconds, 4),
            }
            for key, result in self.results.items()
        ]
        return pd.DataFrame(rows)

    def describe_comparison(self) -> pd.DataFrame:
        """Descriptive statistics of every segment, stacked."""
        return self._stack(
            lambda result: None
            if result.describe is None
            else result.describe.rename_axis("feature").reset_index()
        )

    def missing_comparison(self, *, affected_only: bool = False) -> pd.DataFrame:
        """Missing-value counts of every segment, stacked."""

        def extract(result: ProfilingResult) -> pd.DataFrame | None:
            table = result.missing if affected_only else result.missing_all
            if table is None:
                return None
            return table.rename_axis("feature").reset_index()

        return self._stack(extract)

    def box_comparison(self) -> pd.DataFrame:
        """Quartiles, fences and outlier counts of every segment, stacked."""
        return self._stack(
            lambda result: box_summaries_to_frame(result.box_summaries) if result.box_summaries else None
        )

    def distribution_comparison(self) -> pd.DataFrame:
        """Every bin of every feature of every segment, stacked.

        Bin edges are chosen per segment, so two segments only share an axis when
        ``binning="equal-width"`` happens to produce the same range. Compare the
        ``percentage`` column rather than ``count`` when the segments differ in
        size.
        """
        return self._stack(
            lambda result: distributions_to_frame(result.distributions) if result.distributions else None
        )

    def top_correlation_comparison(self, method: str | None = None) -> pd.DataFrame:
        """Above-threshold feature pairs of every segment, stacked.

        A pair listed for one segment and not another is the interesting case:
        it exceeded the threshold there and not here.
        """

        def extract(result: ProfilingResult) -> pd.DataFrame | None:
            frames = []
            for name, top in result.top_correlations.items():
                if method is not None and name != method.lower():
                    continue
                if top is None or top.empty:
                    continue
                block = top.copy()
                block.insert(0, "method", name)
                block.columns = [
                    "absolute_correlation" if column.startswith("absolute ") else column
                    for column in block.columns
                ]
                frames.append(block)
            return pd.concat(frames, ignore_index=True) if frames else None

        return self._stack(extract)

    def correlation_comparison(self, method: str, *, min_abs: float | None = None) -> pd.DataFrame:
        """One row per feature pair, one column per segment.

        Built from the full matrices rather than the thresholded lists, so every
        pair is present in every segment and the row can be read across. This is
        the view that shows a relationship appearing, vanishing or flipping sign
        between segments.
        """
        key = method.lower()
        columns: dict[str, pd.Series] = {}
        for segment_key, result in self.results.items():
            matrix = result.correlations.get(key)
            if matrix is None or matrix.empty:
                continue
            values = matrix.to_numpy(dtype=float, copy=False)
            rows, cols = np.triu_indices(len(matrix.columns), k=1)
            names = np.asarray([str(c) for c in matrix.columns], dtype=object)
            columns[segment_key] = pd.Series(
                values[rows, cols],
                index=pd.MultiIndex.from_arrays(
                    [names[rows], names[cols]], names=["Feature 1", "Feature 2"]
                ),
            )

        if not columns:
            return pd.DataFrame(columns=["Feature 1", "Feature 2"])

        table = pd.DataFrame(columns)
        table["max_abs_difference"] = (table.max(axis=1) - table.min(axis=1)).abs()
        table = table.sort_values("max_abs_difference", ascending=False, kind="stable")
        if min_abs is not None:
            keep = table[list(columns)].abs().max(axis=1) > float(min_abs)
            table = table[keep]
        return table.reset_index()

    def _stack(self, extract) -> pd.DataFrame:
        frames = []
        for key, result in self.results.items():
            block = extract(result)
            if block is None or block.empty:
                continue
            block = block.copy()
            block.insert(0, "segment", key)
            frames.append(block)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # -- provenance --------------------------------------------------------- #
    def metadata(self) -> dict[str, object]:
        """JSON-serialisable description of the segmented run."""
        return {
            "report_format_version": __version__,
            "dataset": {"name": self.dataset_name, **{k: v for k, v in self.source_info.items()}},
            "grouping": {
                "variable": self.grouping.grouping_var,
                "method": self.grouping.method,
                "n_segments": len(self.grouping),
                "segments": self.grouping.to_frame().to_dict(orient="records"),
                "n_rows_total": self.grouping.n_rows_total,
                "n_rows_unassigned": self.grouping.n_rows_unassigned,
            },
            "settings": self.settings.as_dict(),
            "generated_at_utc": self.generated_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "total_seconds": round(self.total_seconds, 4),
            "environment": environment_report(),
            "notes": list(self.notes),
        }


def run_segmented_profiling(
    data: Any,
    settings: ProfilingSettings,
    *,
    by: Any = None,
    bins: Sequence[tuple[float, float]] | None = None,
    n_bins: int | None = None,
    n_quantiles: int | None = None,
    dataset_name: str = "dataset",
    tag_column: str = DEFAULT_TAG_COLUMN,
    include_grouping_column: bool = True,
    dtype_labels: dict[str, str] | None = None,
    source_info: dict[str, object] | None = None,
    progress: ProgressCallback | None = None,
) -> SegmentedResult:
    """Profile each segment of a dataset separately.

    Parameters
    ----------
    data
        A DataFrame, or a mapping such as ``{"Train": train, "Test": test}``. A
        mapping is stacked into one frame with a ``tag_column`` recording the
        origin of each row; pass ``by=tag_column`` to then compare the inputs.
        pandas and Spark frames are both accepted -- a Spark frame is segmented
        in Spark and materialised one segment at a time.
    settings
        Analysis configuration, applied identically to every segment so the
        numbers stay comparable. An empty ``selected_features`` means every
        feature column.
    by
        Column name, Series or array to segment on. ``None`` profiles the whole
        dataset as the single segment ``Data=All``.
    bins, n_bins, n_quantiles
        Segmentation controls; see :func:`profiling.grouping.make_grouping`. At
        most one may be given.
    include_grouping_column
        Keep the grouping column among the analysed features. When segmenting on
        distinct values the column is constant inside each segment, so its
        profile is degenerate and its correlations undefined; pass ``False`` to
        leave it out. It is never *added* to the features -- the tag column of a
        combined mapping stays excluded either way.
    dtype_labels, source_info, progress
        As for :func:`run_profiling`. ``progress`` reports overall completion
        across all segments.

    Returns
    -------
    SegmentedResult
    """
    combined = combine_datasets(data, tag_column)
    # A ``pyspark.sql`` frame has no positional indexer, so move to the pandas
    # API before slicing segments out of it. Non-Spark frames pass through.
    frame = _spark.to_pandas_api(combined.frame)

    grouping = make_grouping(frame, by, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles)

    requested = [f for f in settings.selected_features if f in combined.feature_names]
    features = requested or list(combined.feature_names)
    if not include_grouping_column and grouping.is_grouped:
        features = [f for f in features if f != grouping.grouping_var]
    if not features:
        raise ValueError(
            "No feature columns left to analyse. Check selected_features and "
            "include_grouping_column."
        )

    segment_settings = replace(settings, selected_features=tuple(features))
    provenance = dict(source_info or {})
    if combined.is_combined:
        provenance.setdefault("combined_sources", combined.sources)
        provenance.setdefault("tag_column", combined.tag_column)

    n_segments = max(len(grouping), 1)
    results: dict[str, ProfilingResult] = {}

    for position, (segment, subset) in enumerate(grouping.frames(frame)):
        segment_progress = _segment_progress(progress, position, n_segments, segment)
        results[segment.key] = run_profiling(
            to_pandas(subset),
            segment_settings,
            dataset_name=_segment_dataset_name(dataset_name, grouping, segment),
            dtype_labels=dtype_labels,
            source_info={**provenance, "segment": segment.label},
            progress=segment_progress,
        )

    notes: list[str] = []
    if grouping.n_rows_unassigned:
        notes.append(
            f"{grouping.n_rows_unassigned:,} of {grouping.n_rows_total:,} rows have a missing or "
            f"out-of-range {grouping.grouping_var!r} value and belong to no segment."
        )
    if grouping.is_grouped and grouping.method == BY_VALUE and grouping.grouping_var in features:
        notes.append(
            f"{grouping.grouping_var!r} is constant within each segment; its per-segment profile "
            "is degenerate. Pass include_grouping_column=False to omit it."
        )

    if progress is not None:
        progress("Done", 1.0)

    return SegmentedResult(
        dataset_name=dataset_name,
        settings=segment_settings,
        grouping=grouping,
        results=results,
        generated_at=datetime.now().astimezone(),
        source_info=provenance,
        notes=notes,
    )


def _segment_dataset_name(dataset_name: str, grouping: Grouping, segment: Segment) -> str:
    return dataset_name if not grouping.is_grouped else f"{dataset_name} [{segment.label}]"


def _segment_progress(
    progress: ProgressCallback | None, position: int, n_segments: int, segment: Segment
) -> ProgressCallback | None:
    """Rescale one segment's progress into its slice of the overall bar."""
    if progress is None:
        return None

    def report(label: str, fraction: float) -> None:
        overall = (position + max(0.0, min(1.0, fraction))) / n_segments
        progress(f"{segment.label} - {label}", min(1.0, overall))

    return report

"""Orchestration: turn a DataFrame plus a settings object into a result object.

Everything the dashboard shows and everything the exports contain comes from a
single :class:`ProfilingResult`. Running the analysis once and then rendering
it many times is what keeps the app responsive, and it is also what guarantees
the PDF, the ZIP and the screen agree.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Sequence

import pandas as pd

from . import __version__
from .dataio import is_numeric_series, numeric_columns
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
    build_box_summaries,
    build_distributions,
)

__all__ = [
    "ProfilingSettings",
    "ProfilingResult",
    "run_profiling",
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

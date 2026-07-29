"""Pre-aggregated summaries of distributions and outliers.

Every chart in this project -- interactive or exported -- is drawn from one of
the small dataclasses defined here rather than from the raw column. That has
two consequences:

* the browser receives a few hundred bytes per chart instead of one point per
  row, which is what makes the dashboard responsive on large files;
* the Plotly chart on screen and the Matplotlib chart in the PDF are guaranteed
  to show the same numbers, because they consume the same summary.

Each summary is computed from a single sorted copy of the column, so a feature
is never scanned more than once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .dataio import is_categorical_series, is_numeric_series
from .stats import quantiles_of_sorted

__all__ = [
    "EQUAL_WIDTH",
    "QUANTILE",
    "KIND_NUMERIC",
    "KIND_CATEGORICAL",
    "NumericDistribution",
    "CategoricalDistribution",
    "BoxSummary",
    "numeric_distribution",
    "categorical_distribution",
    "distribution_of",
    "box_summary",
    "build_distributions",
    "build_box_summaries",
    "distributions_to_frame",
    "box_summaries_to_frame",
]

EQUAL_WIDTH = "equal-width"
QUANTILE = "quantile"

#: Discriminator carried by every distribution summary.
#:
#: Consumers must branch on this rather than on ``isinstance``. Summaries are
#: cached across reruns while Streamlit may reload this module, which produces
#: a new class object; an instance built before the reload then fails an
#: ``isinstance`` check against the new class even though it is the same type.
#: Comparing a plain string is immune to that.
KIND_NUMERIC = "numeric"
KIND_CATEGORICAL = "categorical"

#: Range padding used so that the maximum value falls inside the last bin.
_RANGE_EPSILON = 1e-3
#: Outlier points kept for plotting. The reported count is always exact.
_MAX_OUTLIER_POINTS = 1500
#: Categories drawn individually before the tail is folded into "Other".
_MAX_CATEGORIES = 30


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e6 or magnitude < 1e-4:
        return f"{value:.3e}"
    return f"{value:.4g}"


def _sorted_values(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    values = values[~np.isnan(values)]
    values.sort()
    return values


@dataclass(frozen=True)
class NumericDistribution:
    """Binned counts for one numeric feature."""

    feature: str
    edges: np.ndarray
    counts: np.ndarray
    labels: list[str]
    binning: str
    n_valid: int
    n_missing: int
    minimum: float
    maximum: float

    kind: str = KIND_NUMERIC

    @property
    def centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def n_bins(self) -> int:
        return int(self.counts.size)


@dataclass(frozen=True)
class CategoricalDistribution:
    """Value counts for one discrete feature."""

    feature: str
    categories: list[str]
    counts: np.ndarray
    n_valid: int
    n_missing: int
    n_categories: int
    n_other: int = 0

    kind: str = KIND_CATEGORICAL

    @property
    def truncated(self) -> bool:
        return self.n_categories > len(self.categories) - (1 if self.n_other else 0)


@dataclass(frozen=True)
class BoxSummary:
    """Five-number summary plus Tukey fences for one numeric feature."""

    feature: str
    n_valid: int
    n_missing: int
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    mean: float
    std: float
    lower_fence: float
    upper_fence: float
    n_outliers_low: int
    n_outliers_high: int
    outliers: np.ndarray = field(default_factory=lambda: np.empty(0))
    outliers_truncated: bool = False

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    @property
    def n_outliers(self) -> int:
        return self.n_outliers_low + self.n_outliers_high


def _equal_width_bins(sorted_values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width histogram over a slightly padded data range."""
    low = float(sorted_values[0])
    high = float(sorted_values[-1])

    span = high - low
    padding = span * _RANGE_EPSILON if span > 0 else max(abs(low) * _RANGE_EPSILON, 0.5)
    low -= padding
    high += padding

    counts, edges = np.histogram(sorted_values, bins=int(n_bins), range=(low, high))
    return edges, counts.astype(np.int64, copy=False)


def _quantile_bins(sorted_values: np.ndarray, n_quantiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-closed quantile bins, matching ``pandas.qcut(duplicates="drop")``."""
    probabilities = np.linspace(0.0, 1.0, int(n_quantiles) + 1)
    edges = np.unique(quantiles_of_sorted(sorted_values, probabilities))

    if edges.size < 2:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5]) if edges.size else np.array([0.0, 1.0])

    # ``searchsorted(..., "left")`` on the interior edges assigns a value that
    # sits exactly on an edge to the bin below it, i.e. intervals are (a, b]
    # with the lowest edge included -- the same convention as ``qcut``.
    interior = edges[1:-1]
    indices = np.searchsorted(interior, sorted_values, side="left")
    counts = np.bincount(indices, minlength=edges.size - 1).astype(np.int64, copy=False)
    return edges, counts


def _distinct_value_bins(sorted_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One bin per distinct value, for low-cardinality numeric columns."""
    values, counts = np.unique(sorted_values, return_counts=True)
    return values, counts.astype(np.int64, copy=False)


def _interval_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for i in range(edges.size - 1):
        left = _format_number(float(edges[i]))
        right = _format_number(float(edges[i + 1]))
        bracket = "[" if i == 0 else "("
        labels.append(f"{bracket}{left}, {right}]")
    return labels


def numeric_distribution(
    series: pd.Series,
    *,
    binning: str = EQUAL_WIDTH,
    n_bins: int = 8,
) -> NumericDistribution:
    """Bin one numeric column.

    ``binning="equal-width"`` splits the padded data range into ``n_bins``
    equal intervals. ``binning="quantile"`` splits on empirical quantiles and
    drops duplicate edges. In either mode a column with no more distinct values
    than requested bins gets one bin per distinct value, which avoids the
    degenerate single-bin result on flag-like columns.
    """
    name = str(series.name)
    n_rows = int(series.size)
    values = _sorted_values(series)
    n_valid = int(values.size)
    n_missing = n_rows - n_valid
    n_bins = max(1, int(n_bins))

    if n_valid == 0:
        return NumericDistribution(
            feature=name,
            edges=np.array([0.0, 1.0]),
            counts=np.zeros(1, dtype=np.int64),
            labels=["no data"],
            binning=binning,
            n_valid=0,
            n_missing=n_missing,
            minimum=np.nan,
            maximum=np.nan,
        )

    n_distinct = int(np.unique(values).size)

    if binning == QUANTILE and n_distinct <= n_bins:
        centers, counts = _distinct_value_bins(values)
        labels = [_format_number(float(v)) for v in centers]
        half = 0.5 if centers.size < 2 else float(np.min(np.diff(centers))) / 2.0
        edges = np.append(centers - half, centers[-1] + half)
    elif binning == QUANTILE:
        edges, counts = _quantile_bins(values, n_bins)
        labels = _interval_labels(edges)
    else:
        edges, counts = _equal_width_bins(values, n_bins)
        labels = _interval_labels(edges)

    return NumericDistribution(
        feature=name,
        edges=np.asarray(edges, dtype=np.float64),
        counts=counts,
        labels=labels,
        binning=binning,
        n_valid=n_valid,
        n_missing=n_missing,
        minimum=float(values[0]),
        maximum=float(values[-1]),
    )


def categorical_distribution(
    series: pd.Series,
    *,
    max_categories: int = _MAX_CATEGORIES,
) -> CategoricalDistribution:
    """Value counts for one discrete column, with a folded tail."""
    name = str(series.name)
    n_rows = int(series.size)
    counts = series.value_counts(dropna=True, sort=True)
    n_valid = int(counts.sum())
    n_categories = int(counts.size)

    n_other = 0
    if n_categories > max_categories:
        tail = counts.iloc[max_categories:]
        n_other = int(tail.sum())
        counts = counts.iloc[:max_categories]

    categories = [str(value) for value in counts.index]
    values = counts.to_numpy(dtype=np.int64, copy=False)

    if n_other:
        categories = [*categories, f"Other ({n_categories - max_categories} categories)"]
        values = np.append(values, n_other)

    return CategoricalDistribution(
        feature=name,
        categories=categories,
        counts=values,
        n_valid=n_valid,
        n_missing=n_rows - n_valid,
        n_categories=n_categories,
        n_other=n_other,
    )


def distribution_of(
    series: pd.Series,
    *,
    binning: str = EQUAL_WIDTH,
    n_bins: int = 8,
    max_categories: int = _MAX_CATEGORIES,
) -> NumericDistribution | CategoricalDistribution | None:
    """Dispatch to the numeric or categorical summary, or ``None`` if neither."""
    if is_numeric_series(series):
        return numeric_distribution(series, binning=binning, n_bins=n_bins)
    if is_categorical_series(series):
        return categorical_distribution(series, max_categories=max_categories)
    return None


def box_summary(
    series: pd.Series,
    *,
    max_outlier_points: int = _MAX_OUTLIER_POINTS,
) -> BoxSummary | None:
    """Quartiles, Tukey fences and outliers for one numeric column.

    Quartiles use linear interpolation, which is the convention Plotly and
    Matplotlib both default to, so the drawn box matches the reported numbers.
    Fences are the standard ``Q1 - 1.5*IQR`` / ``Q3 + 1.5*IQR`` pulled back to
    the most extreme observation still inside them.
    """
    if not is_numeric_series(series):
        return None

    name = str(series.name)
    n_rows = int(series.size)
    values = _sorted_values(series)
    n_valid = int(values.size)

    if n_valid == 0:
        return BoxSummary(
            feature=name,
            n_valid=0,
            n_missing=n_rows,
            minimum=np.nan,
            q1=np.nan,
            median=np.nan,
            q3=np.nan,
            maximum=np.nan,
            mean=np.nan,
            std=np.nan,
            lower_fence=np.nan,
            upper_fence=np.nan,
            n_outliers_low=0,
            n_outliers_high=0,
        )

    q1, median, q3 = quantiles_of_sorted(values, (0.25, 0.50, 0.75))
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    low_cut = int(np.searchsorted(values, lower_limit, side="left"))
    high_cut = int(np.searchsorted(values, upper_limit, side="right"))

    lower_fence = float(values[low_cut]) if low_cut < n_valid else float(values[0])
    upper_fence = float(values[high_cut - 1]) if high_cut > 0 else float(values[-1])

    outliers = np.concatenate((values[:low_cut], values[high_cut:]))
    truncated = outliers.size > max_outlier_points
    if truncated:
        # Keep an evenly spaced, deterministic subset so the tails stay visible.
        keep = np.linspace(0, outliers.size - 1, max_outlier_points).astype(np.intp)
        outliers = outliers[keep]

    return BoxSummary(
        feature=name,
        n_valid=n_valid,
        n_missing=n_rows - n_valid,
        minimum=float(values[0]),
        q1=float(q1),
        median=float(median),
        q3=float(q3),
        maximum=float(values[-1]),
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if n_valid > 1 else 0.0,
        lower_fence=lower_fence,
        upper_fence=upper_fence,
        n_outliers_low=low_cut,
        n_outliers_high=n_valid - high_cut,
        outliers=outliers,
        outliers_truncated=truncated,
    )


def build_distributions(
    data: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    binning: str = EQUAL_WIDTH,
    n_bins: int = 8,
    max_categories: int = _MAX_CATEGORIES,
) -> dict[str, NumericDistribution | CategoricalDistribution]:
    """Summarise the distribution of every requested feature."""
    names = list(data.columns) if features is None else [f for f in features if f in data.columns]
    summaries: dict[str, NumericDistribution | CategoricalDistribution] = {}
    for name in names:
        summary = distribution_of(
            data[name], binning=binning, n_bins=n_bins, max_categories=max_categories
        )
        if summary is not None:
            summaries[str(name)] = summary
    return summaries


def build_box_summaries(
    data: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    max_outlier_points: int = _MAX_OUTLIER_POINTS,
) -> dict[str, BoxSummary]:
    """Summarise the spread of every requested numeric feature."""
    names = list(data.columns) if features is None else [f for f in features if f in data.columns]
    summaries: dict[str, BoxSummary] = {}
    for name in names:
        summary = box_summary(data[name], max_outlier_points=max_outlier_points)
        if summary is not None:
            summaries[str(name)] = summary
    return summaries


def distributions_to_frame(
    summaries: dict[str, NumericDistribution | CategoricalDistribution],
) -> pd.DataFrame:
    """Long-format table of every bin, for CSV export."""
    rows: list[dict[str, object]] = []
    for name, summary in summaries.items():
        if summary.kind == KIND_NUMERIC:
            total = int(summary.counts.sum())
            for index, (label, count) in enumerate(zip(summary.labels, summary.counts)):
                rows.append(
                    {
                        "feature": name,
                        "type": "numeric",
                        "binning": summary.binning,
                        "bin": index + 1,
                        "bin_label": label,
                        "bin_lower": float(summary.edges[index]),
                        "bin_upper": float(summary.edges[index + 1]),
                        "count": int(count),
                        "percentage": round(100.0 * count / total, 6) if total else 0.0,
                        "n_missing": summary.n_missing,
                    }
                )
        else:
            total = int(summary.counts.sum())
            for index, (label, count) in enumerate(zip(summary.categories, summary.counts)):
                rows.append(
                    {
                        "feature": name,
                        "type": "categorical",
                        "binning": "value-counts",
                        "bin": index + 1,
                        "bin_label": label,
                        "bin_lower": np.nan,
                        "bin_upper": np.nan,
                        "count": int(count),
                        "percentage": round(100.0 * count / total, 6) if total else 0.0,
                        "n_missing": summary.n_missing,
                    }
                )
    return pd.DataFrame(rows)


def box_summaries_to_frame(summaries: dict[str, BoxSummary]) -> pd.DataFrame:
    """Wide table of the box-plot statistics, for CSV export."""
    rows = [
        {
            "feature": name,
            "n_valid": summary.n_valid,
            "n_missing": summary.n_missing,
            "min": summary.minimum,
            "lower_fence": summary.lower_fence,
            "q1": summary.q1,
            "median": summary.median,
            "q3": summary.q3,
            "upper_fence": summary.upper_fence,
            "max": summary.maximum,
            "mean": summary.mean,
            "std": summary.std,
            "iqr": summary.iqr,
            "n_outliers_low": summary.n_outliers_low,
            "n_outliers_high": summary.n_outliers_high,
            "n_outliers": summary.n_outliers,
            "perc_outliers": round(100.0 * summary.n_outliers / summary.n_valid, 4) if summary.n_valid else 0.0,
        }
        for name, summary in summaries.items()
    ]
    return pd.DataFrame(rows)

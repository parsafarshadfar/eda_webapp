"""Segmentation of a dataset into groups that are then profiled separately.

A dataset can be split four ways:

``by`` alone            one group per distinct value of a column
``bins``                explicit intervals, e.g. ``[(1, 20), (20, 50)]``
``n_bins``              that many equal-width intervals over the observed range
``n_quantiles``         that many equal-frequency intervals

The result is a :class:`Grouping`: a small, inspectable object holding the row
index of every segment. It carries no data, so building one is cheap and the
same grouping can be reused for several analyses. Slicing the frame happens
lazily, one segment at a time, in :meth:`Grouping.frames`.

Grouping a Spark frame is delegated to :mod:`profiling._spark`, which does the
bucketing in Spark instead of collecting the column onto the driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from . import _spark

__all__ = [
    "NO_GROUPING",
    "BY_VALUE",
    "BY_INTERVALS",
    "BY_EQUAL_WIDTH",
    "BY_QUANTILE",
    "ALL_SEGMENT",
    "UNGROUPED_VAR",
    "Segment",
    "Grouping",
    "make_grouping",
    "segment_frames",
]

#: ``Grouping.method`` values.
NO_GROUPING = "none"
BY_VALUE = "values"
BY_INTERVALS = "intervals"
BY_EQUAL_WIDTH = "equal-width"
BY_QUANTILE = "quantile"

#: Segment key and grouping-variable name used when nothing is grouped on.
ALL_SEGMENT = "All"
UNGROUPED_VAR = "Data"

#: Name given to a grouping series that arrives as a bare array.
_ANONYMOUS_SERIES = "Grouped_variable"


@dataclass(frozen=True)
class Segment:
    """One group of rows: its key, its label and the rows it contains."""

    key: str
    grouping_var: str
    index: pd.Index
    n_rows: int
    positions: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def label(self) -> str:
        """``"<grouping_var>=<key>"``, e.g. ``"Age=(10.0, 26.5]"``."""
        return f"{self.grouping_var}={self.key}"


@dataclass(frozen=True)
class Grouping:
    """The segments a dataset was split into.

    Holds only row indices, never data. ``method`` records how the split was
    made so a report can state it, and ``n_rows_total`` is the row count of the
    frame the grouping was built from -- which can exceed the sum of the
    segments, because rows with a missing grouping value belong to no segment.
    """

    grouping_var: str
    method: str
    segments: tuple[Segment, ...]
    n_rows_total: int

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self) -> Iterator[Segment]:
        return iter(self.segments)

    def __getitem__(self, item: int | str) -> Segment:
        if isinstance(item, int):
            return self.segments[item]
        for segment in self.segments:
            if item in (segment.key, segment.label):
                return segment
        raise KeyError(item)

    @property
    def is_grouped(self) -> bool:
        return self.method != NO_GROUPING

    @property
    def keys(self) -> list[str]:
        return [segment.key for segment in self.segments]

    @property
    def labels(self) -> list[str]:
        return [segment.label for segment in self.segments]

    @property
    def n_rows_assigned(self) -> int:
        return int(sum(segment.n_rows for segment in self.segments))

    @property
    def n_rows_unassigned(self) -> int:
        """Rows with a missing or out-of-range grouping value."""
        return max(0, self.n_rows_total - self.n_rows_assigned)

    def frames(self, data: Any) -> Iterator[tuple[Segment, Any]]:
        """Yield ``(segment, rows_of_that_segment)`` one segment at a time."""
        if not self.is_grouped:
            # Avoid an unnecessary whole-frame copy for the common ungrouped
            # path, and remain correct when the frame index contains duplicates.
            yield self.segments[0], data
            return

        spark = _spark.is_spark(data)
        for segment in self.segments:
            if segment.positions is not None:
                yield segment, data.iloc[segment.positions]
            elif spark:
                yield segment, data.iloc[segment.index.to_numpy()]
            else:
                yield segment, data.loc[segment.index]

    def to_frame(self) -> pd.DataFrame:
        """Segment sizes as a table, for a quick sanity check on the split."""
        rows = [
            {
                "segment": segment.key,
                "label": segment.label,
                "n_rows": segment.n_rows,
                "share_of_rows_%": round(100.0 * segment.n_rows / self.n_rows_total, 4)
                if self.n_rows_total
                else 0.0,
            }
            for segment in self.segments
        ]
        frame = pd.DataFrame(rows, columns=["segment", "label", "n_rows", "share_of_rows_%"])
        frame.attrs["grouping_var"] = self.grouping_var
        frame.attrs["method"] = self.method
        return frame


def _resolve_series(data: Any, by: Any) -> Any:
    """Turn ``by`` into a named series taken from, or aligned with, ``data``."""
    if isinstance(by, str):
        if by not in data.columns:
            raise KeyError(f"Grouping column {by!r} is not in the data.")
        return data[by]

    if isinstance(by, (np.ndarray, np.generic, list, tuple)):
        values = np.asarray(by)
        if values.ndim != 1:
            raise ValueError("Grouping arrays must be one-dimensional.")
        if values.shape[0] != data.shape[0]:
            raise ValueError(
                f"Grouping array has {values.shape[0]} values but the data has {data.shape[0]} rows."
            )
        series = pd.Series(values, name=_ANONYMOUS_SERIES)
        # Align to the frame so ``.loc`` on the resulting index selects rows.
        if not _spark.is_spark(data):
            series.index = data.index
        return series

    if not hasattr(by, "name") or not hasattr(by, "index"):
        raise TypeError("by must be a column name, a Series, or a one-dimensional array")
    if len(by) != data.shape[0]:
        raise ValueError(
            f"Grouping Series has {len(by)} values but the data has {data.shape[0]} rows."
        )
    if not _spark.is_spark(data) and not by.index.equals(data.index):
        if not by.index.is_unique:
            raise ValueError("A grouping Series with a different index must have unique labels.")
        by = by.reindex(data.index)

    if by.name is None:
        by = by.rename(_ANONYMOUS_SERIES)
    return by


def _selected_method(bins: Any, n_bins: Any, n_quantiles: Any) -> str:
    chosen = [
        name
        for name, value in (
            (BY_INTERVALS, bins),
            (BY_EQUAL_WIDTH, n_bins),
            (BY_QUANTILE, n_quantiles),
        )
        if value is not None
    ]
    if len(chosen) > 1:
        raise ValueError(
            "Choose at most one of bins, n_bins or n_quantiles; got " + ", ".join(chosen) + "."
        )
    return chosen[0] if chosen else BY_VALUE


def _pandas_groups(
    series: pd.Series,
    method: str,
    bins: Sequence[tuple[float, float]] | None,
    n_bins: int | None,
    n_quantiles: int | None,
) -> dict[Any, tuple[pd.Index, np.ndarray]]:
    if method == BY_VALUE:
        grouped = series.groupby(series, observed=True)
    elif method == BY_INTERVALS:
        intervals = pd.IntervalIndex.from_tuples([tuple(interval[:2]) for interval in bins])
        grouped = series.groupby(pd.cut(series, bins=intervals), observed=True)
    elif method == BY_EQUAL_WIDTH:
        grouped = series.groupby(pd.cut(series, bins=int(n_bins)), observed=True)
    else:
        grouped = series.groupby(
            pd.qcut(series, q=int(n_quantiles), duplicates="drop"),
            observed=True,
        )

    # Keep the user-visible index labels for inspection, plus positional
    # indices for slicing. Label-based slicing duplicates rows when a DataFrame
    # itself has duplicate index labels.
    return {
        key: (
            pd.Index(index),
            np.asarray(grouped.indices[key], dtype=np.intp),
        )
        for key, index in grouped.groups.items()
    }


def _validate_grouping_request(
    bins: Sequence[tuple[float, float]] | None,
    n_bins: int | None,
    n_quantiles: int | None,
) -> None:
    """Fail early with clear messages for malformed bucket settings."""
    for name, value in (("n_bins", n_bins), ("n_quantiles", n_quantiles)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be a positive integer or None")
        if int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer or None")

    if bins is None:
        return
    if len(bins) == 0:
        raise ValueError("bins must contain at least one (left, right) interval")
    for position, interval in enumerate(bins):
        try:
            has_endpoints = len(interval) >= 2
        except TypeError:
            has_endpoints = False
        if not has_endpoints:
            raise ValueError(f"bins[{position}] must contain a left and right endpoint")
        left, right = interval[:2]
        try:
            valid = np.isfinite(left) and np.isfinite(right) and float(left) < float(right)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(
                f"bins[{position}] must have finite endpoints with left < right; "
                f"got {interval!r}."
            )


def make_grouping(
    data: Any,
    by: Any = None,
    *,
    bins: Sequence[tuple[float, float]] | None = None,
    n_bins: int | None = None,
    n_quantiles: int | None = None,
) -> Grouping:
    """Split ``data`` into segments and record the row index of each one.

    Parameters
    ----------
    data
        The frame to segment. pandas, ``pyspark.pandas`` and ``pyspark.sql``
        frames are all accepted.
    by
        Column name, Series or array to group on. ``None`` produces the single
        segment ``Data=All``, which is what lets every caller take the segmented
        code path unconditionally.
    bins
        Explicit intervals, e.g. ``[(1, 20), (20, 50)]``. Right-closed. Rows
        outside every interval belong to no segment.
    n_bins
        Number of equal-width intervals. Occupancy is not balanced and an
        interval can end up empty.
    n_quantiles
        Number of equal-frequency intervals, so every segment holds a similar
        number of rows.

    Returns
    -------
    Grouping
        Empty segments are dropped. Segments come out in ascending key order
        for values and in interval order for the three binned modes.
    """
    method = _selected_method(bins, n_bins, n_quantiles)
    _validate_grouping_request(bins, n_bins, n_quantiles)

    if _spark.is_spark_sql(data):
        data = _spark.to_pandas_api(data)
    if isinstance(data, pd.DataFrame) and not data.columns.is_unique:
        raise ValueError("make_grouping() requires unique column names.")

    n_rows_total = int(data.shape[0])

    if by is None:
        return Grouping(
            grouping_var=UNGROUPED_VAR,
            method=NO_GROUPING,
            segments=(
                Segment(
                    key=ALL_SEGMENT,
                    grouping_var=UNGROUPED_VAR,
                    index=data.index,
                    n_rows=n_rows_total,
                ),
            ),
            n_rows_total=n_rows_total,
        )

    series = _resolve_series(data, by)
    grouping_var = str(series.name)

    if _spark.is_spark(series):
        groups = _spark.spark_groups(
            series, grouping_var, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles
        )
    else:
        groups = _pandas_groups(series, method, bins, n_bins, n_quantiles)

    segment_list: list[Segment] = []
    segment_keys: set[str] = set()
    spark_series = _spark.is_spark(series)
    for key, group in groups.items():
        index, positions = (group, None) if spark_series else group
        if len(index) == 0:
            continue
        text_key = str(key)
        if text_key in segment_keys:
            raise ValueError(
                "Grouping values must remain unique when converted to text; "
                f"more than one value becomes {text_key!r}."
            )
        segment_keys.add(text_key)
        segment_list.append(
            Segment(
                key=text_key,
                grouping_var=grouping_var,
                index=index,
                n_rows=int(len(index)),
                positions=positions,
            )
        )
    segments = tuple(segment_list)
    return Grouping(
        grouping_var=grouping_var,
        method=method,
        segments=segments,
        n_rows_total=n_rows_total,
    )


def segment_frames(
    data: Any,
    by: Any = None,
    *,
    bins: Sequence[tuple[float, float]] | None = None,
    n_bins: int | None = None,
    n_quantiles: int | None = None,
) -> Iterator[tuple[Segment, Any]]:
    """Build a grouping and immediately iterate over its ``(segment, frame)`` pairs."""
    grouping = make_grouping(data, by, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles)
    yield from grouping.frames(data)

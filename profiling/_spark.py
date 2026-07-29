"""Optional Spark bridge for the Databricks notebook workflow.

The package has no Spark dependency. Nothing in this module is imported until a
Spark object is actually handed to :mod:`profiling.grouping` or
:func:`profiling.dataio.combine_datasets`, and the ``pyspark`` imports live
inside the functions rather than at module scope. On a plain pandas install the
detection helpers are the only code that ever runs, and they only look at type
names.

Segmentation of a Spark frame is done *in Spark* -- with ``Bucketizer`` and
``QuantileDiscretizer`` -- rather than by pulling the grouping column to the
driver, which is the whole reason this path exists.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "is_spark",
    "is_spark_sql",
    "to_pandas_api",
    "to_pandas",
    "spark_groups",
    "spark_concat",
]


def _type_name(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def is_spark(obj: Any) -> bool:
    """True for ``pyspark.sql`` and ``pyspark.pandas`` frames and series."""
    return "spark" in _type_name(obj).lower()


def is_spark_sql(obj: Any) -> bool:
    """True only for the ``pyspark.sql`` (non-pandas-API) objects."""
    return "pyspark.sql" in _type_name(obj)


def to_pandas_api(obj: Any) -> Any:
    """Convert a ``pyspark.sql`` frame to the pandas API; pass anything else through."""
    if is_spark_sql(obj) and hasattr(obj, "pandas_api"):
        return obj.pandas_api()
    return obj


def to_pandas(obj: Any) -> Any:
    """Materialise a Spark frame on the driver as real pandas.

    Anything that is not a Spark object is returned unchanged, so this is safe
    to call unconditionally.
    """
    if not is_spark(obj):
        return obj
    if hasattr(obj, "toPandas"):  # pyspark.sql.DataFrame
        return obj.toPandas()
    if hasattr(obj, "to_pandas"):  # pyspark.pandas.DataFrame / Series
        return obj.to_pandas()
    return obj


def _to_spark(frame: Any) -> Any:
    """Convert a pandas-API frame back to ``pyspark.sql``, which the ML API needs."""
    return frame.to_spark() if hasattr(frame, "to_spark") else frame


def _labelled_buckets(sdf: Any, grouping_var: str, splits: list[float], labels: dict) -> dict[str, Any]:
    """Bucketise ``grouping_var`` and return ``{interval label: row index}``."""
    from pyspark.ml.feature import Bucketizer
    from pyspark.sql.functions import col, udf

    def map_bucket(key, mapping=labels):
        return mapping.get(key, None)

    bucketizer = Bucketizer(
        splits=splits, inputCol=grouping_var, outputCol="bucket", handleInvalid="keep"
    )
    bucketed = bucketizer.transform(sdf).withColumn("bucket", udf(map_bucket)(col("bucket")))
    bucketed = bucketed.pandas_api()

    groups: dict[str, Any] = {}
    for key in bucketed["bucket"].unique().dropna().to_list():
        groups[key] = bucketed.groupby("bucket").get_group(key).index
    return groups


def spark_groups(
    series: Any,
    grouping_var: str,
    *,
    bins: list[tuple[float, float]] | None = None,
    n_bins: int | None = None,
    n_quantiles: int | None = None,
) -> dict[str, Any]:
    """``{group label: row index}`` for a Spark grouping column.

    Exactly one of ``bins``, ``n_bins`` or ``n_quantiles`` may be given; with
    none of them the distinct values of the column become the groups.
    """
    frame = series.to_frame()

    if bins is None and n_bins is None and n_quantiles is None:
        frame = to_pandas_api(frame)
        groups: dict[str, Any] = {}
        for key in frame[grouping_var].unique().dropna().to_list():
            groups[key] = frame.groupby(grouping_var).get_group(key).index
        return groups

    sdf = _to_spark(frame)

    if bins is not None:
        # The tuple ends are flattened into one sorted list of split points, so
        # a gap between two requested intervals becomes a bin of its own rather
        # than being dropped. This differs from ``pandas.IntervalIndex`` and is
        # kept for compatibility with the original Databricks behaviour.
        edges = sorted({float(edge) for interval in bins for edge in interval[:2]})
        labels = {None: None, np.nan: np.nan}
        for i in range(len(edges) - 1):
            labels[i] = str(pd.Interval(left=edges[i], right=edges[i + 1], closed="right"))
        return _labelled_buckets(sdf, grouping_var, edges, labels)

    from pyspark.sql import functions as F

    if n_bins is not None:
        minimum, maximum = sdf.select(F.min(grouping_var), F.max(grouping_var)).first()
        minimum = minimum - 0.001 * abs(minimum)
        width = (maximum - minimum) / int(n_bins)

        edges = [minimum] + [minimum + i * width for i in range(1, int(n_bins))] + [maximum]
        labels = {None: None, np.nan: np.nan}
        for i in range(int(n_bins)):
            labels[i] = str(
                pd.Interval(
                    left=np.round(edges[i], 4), right=np.round(edges[i + 1], 4), closed="right"
                )
            )
        # Infinite outer splits so a value on the boundary cannot fall outside.
        splits = [float("-inf")] + edges[1:-1] + [float("inf")]
        return _labelled_buckets(sdf, grouping_var, splits, labels)

    from pyspark.ml.feature import QuantileDiscretizer
    from pyspark.sql.functions import col, udf

    discretizer = QuantileDiscretizer(
        numBuckets=int(n_quantiles),
        inputCol=grouping_var,
        outputCol="bucket",
        handleInvalid="keep",
        relativeError=0.0001,
    )
    model = discretizer.fit(sdf)
    bucketed = model.transform(sdf)

    # Recover the split points so the buckets can be given interval labels.
    minimum, maximum = bucketed.select(F.min(grouping_var), F.max(grouping_var)).first()
    edges = [minimum] + sorted(model.getSplits()[1:-1]) + [maximum]

    labels = {None: None, np.nan: np.nan}
    for i in range(len(edges) - 1):
        labels[i] = str(pd.Interval(left=edges[i], right=edges[i + 1], closed="left"))
    # ``qcut`` closes the top bin on both sides; mirror that.
    labels[len(edges) - 2] = labels[len(edges) - 2].replace(")", "]")

    def map_bucket(key, mapping=labels):
        return mapping.get(key, None)

    bucketed = bucketed.withColumn("bucket", udf(map_bucket)(col("bucket"))).pandas_api()

    groups = {}
    for key in bucketed["bucket"].unique().dropna().to_list():
        groups[key] = bucketed.groupby("bucket").get_group(key).index
    return groups


def spark_concat(frames: dict[str, Any], tag_column: str) -> Any:
    """Concatenate Spark frames, tagging each row with the key it came from."""
    import pyspark.pandas as ps

    combined = ps.DataFrame({})
    for key, frame in frames.items():
        tagged = to_pandas_api(frame).copy()
        tagged[tag_column] = key
        combined = ps.concat([combined, tagged], ignore_index=True)
    return combined

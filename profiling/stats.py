"""Descriptive statistics, missing-value analysis and correlation analysis.

Performance notes
-----------------
``describe_data`` used to make roughly six passes over every column (a
``value_counts``, a ``nunique``, three separate ``quantile`` calls and a
min/max scan). Every one of those quantities can be read off a single sorted
copy of the column, so this module sorts once and then walks the sorted array.
Peak extra memory is one column, never the whole frame.

Correlations are computed from one shared float64 matrix instead of being
re-extracted per method. When the matrix has no missing values, Spearman and
Pearson are solved with a single BLAS matrix product rather than a Python loop
over feature pairs. The pairwise-complete fallback used when NaNs are present
reproduces :meth:`pandas.DataFrame.corr` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .dataio import is_numeric_series, numeric_columns

__all__ = [
    "DESCRIBE_COLUMNS",
    "describe_data",
    "missing_count",
    "correlation_matrix",
    "correlation_matrices",
    "top_absolute_correlations",
    "quantiles_of_sorted",
]

DESCRIBE_COLUMNS = [
    "DataType",
    "n_uniques (excl. Nulls)",
    "n_Missing",
    "Perc. of Missing",
    "n_Zeros",
    "Perc. of Zeros",
    "1st Most Freq",
    "Perc. of 1st Most Freq",
    "Min",
    "1%",
    "50%",
    "99%",
    "Max",
]

_DESCRIBE_QUANTILES = (0.01, 0.50, 0.99)
_SUPPORTED_CORRELATIONS = ("pearson", "spearman", "kendall")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def quantiles_of_sorted(sorted_values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """Linear-interpolated quantiles read straight off a sorted array.

    Identical to ``numpy.quantile(..., method="linear")`` and therefore to
    ``pandas.Series.quantile``, but without re-sorting for every quantile.
    """
    q = np.asarray(quantiles, dtype=np.float64)
    n = sorted_values.size
    if n == 0:
        return np.full(q.shape, np.nan)
    if n == 1:
        return np.full(q.shape, sorted_values[0], dtype=np.float64)

    position = q * (n - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.ceil(position).astype(np.intp)
    weight = position - lower
    low_values = sorted_values[lower].astype(np.float64, copy=False)
    high_values = sorted_values[upper].astype(np.float64, copy=False)
    result = np.empty(q.shape, dtype=np.float64)

    # Avoid ``inf - inf`` and ``0 * inf`` warnings at exact order statistics.
    exact = (lower == upper) | (low_values == high_values)
    result[exact] = low_values[exact]

    interpolate = ~exact & np.isfinite(low_values) & np.isfinite(high_values)
    result[interpolate] = low_values[interpolate] + (
        high_values[interpolate] - low_values[interpolate]
    ) * weight[interpolate]

    non_finite = ~(exact | interpolate)
    result[non_finite] = np.where(
        np.isneginf(low_values[non_finite])
        & np.isposinf(high_values[non_finite]),
        np.nan,
        np.where(
            np.isneginf(low_values[non_finite]),
            -np.inf,
            np.inf,
        ),
    )
    return result


def _sorted_valid_values(series: pd.Series) -> np.ndarray:
    """Sorted array of the non-null values of a numeric column."""
    dtype = series.dtype
    if pd.api.types.is_integer_dtype(dtype):
        # Keep integers as integers while sorting. Converting int64/uint64 to
        # float64 first merges distinct values above 2**53 and can corrupt
        # cardinality, mode, min and max.
        target = np.uint64 if pd.api.types.is_unsigned_integer_dtype(dtype) else np.int64
        raw = series.to_numpy(dtype=target, na_value=0, copy=False)
        valid = ~series.isna().to_numpy(dtype=bool, copy=False)
        values = raw[valid]
    else:
        raw = series.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        values = raw[~np.isnan(raw)]
    values.sort()
    return values


def _runs_of_sorted(sorted_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (start indices of each distinct run, run lengths).

    Runs a single O(n) scan over an already sorted array, replacing the hash
    table built by ``value_counts``/``nunique``.
    """
    n = sorted_values.size
    if n == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.int64)

    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    if n > 1:
        np.not_equal(sorted_values[1:], sorted_values[:-1], out=is_new[1:])
    starts = np.flatnonzero(is_new)
    lengths = np.empty(starts.size, dtype=np.int64)
    if starts.size > 1:
        np.subtract(starts[1:], starts[:-1], out=lengths[:-1], casting="unsafe")
    lengths[-1] = n - starts[-1]
    return starts, lengths


def _numeric_column_stats(series: pd.Series, n_rows: int) -> dict[str, object]:
    """Every descriptive statistic for one numeric column, from one sort."""
    sorted_values = _sorted_valid_values(series)
    n_valid = int(sorted_values.size)
    n_missing = int(n_rows - n_valid)

    starts, lengths = _runs_of_sorted(sorted_values)
    n_unique = int(starts.size)

    zero_lo = int(np.searchsorted(sorted_values, 0.0, side="left"))
    zero_hi = int(np.searchsorted(sorted_values, 0.0, side="right"))
    n_zeros = zero_hi - zero_lo

    if n_unique:
        # ``argmax`` returns the first maximum, so ties resolve to the smallest
        # value. That makes the reported mode reproducible run after run.
        best = int(np.argmax(lengths))
        mode_value: object = sorted_values[starts[best]].item()
        mode_count = int(lengths[best])
    else:
        mode_value = np.nan
        mode_count = 0

    # ``value_counts(dropna=False)`` treats nulls as a candidate value.
    if n_missing > mode_count:
        mode_value = np.nan
        mode_count = n_missing

    q01, q50, q99 = quantiles_of_sorted(sorted_values, _DESCRIBE_QUANTILES)

    return {
        "n_uniques (excl. Nulls)": n_unique,
        "n_Missing": n_missing,
        "n_Zeros": n_zeros,
        "1st Most Freq": mode_value,
        "Perc. of 1st Most Freq": (100.0 * mode_count / n_rows) if n_rows else np.nan,
        "Min": sorted_values[0].item() if n_valid else np.nan,
        "1%": float(q01),
        "50%": float(q50),
        "99%": float(q99),
        "Max": sorted_values[-1].item() if n_valid else np.nan,
    }


def _discrete_column_stats(series: pd.Series, n_rows: int) -> dict[str, object]:
    """Descriptive statistics for a text/categorical/boolean column."""
    n_valid = int(series.count())
    n_missing = int(n_rows - n_valid)

    # One value-count pass gives both cardinality and the most frequent
    # non-null value. Missing sentinels (None, NaN, pd.NA) are deliberately
    # pooled into one candidate via ``n_missing``.
    counts = series.value_counts(dropna=True, sort=True)
    if len(counts):
        mode_count = int(counts.iloc[0])
        tied = counts.index[counts.to_numpy(copy=False) == mode_count]
        try:
            mode_value = min(tied)
        except TypeError:
            mode_value = min(tied, key=lambda value: (type(value).__name__, str(value)))
    else:
        mode_value = np.nan
        mode_count = 0

    if n_missing > mode_count:
        mode_value = np.nan
        mode_count = n_missing

    try:
        n_zeros = int((series == 0).sum())
    except (TypeError, ValueError):
        n_zeros = 0

    return {
        "n_uniques (excl. Nulls)": int(counts.size),
        "n_Missing": n_missing,
        "n_Zeros": n_zeros,
        "1st Most Freq": mode_value,
        "Perc. of 1st Most Freq": (100.0 * mode_count / n_rows) if n_rows else np.nan,
        "Min": np.nan,
        "1%": np.nan,
        "50%": np.nan,
        "99%": np.nan,
        "Max": np.nan,
    }


# --------------------------------------------------------------------------- #
# descriptive statistics
# --------------------------------------------------------------------------- #
def describe_data(
    data: pd.DataFrame,
    numeric_only: bool = True,
    round_columns: dict[str, int] | None = None,
    drop_columns: Iterable[str] | None = None,
    saving_path: str | None = None,
    dtype_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per-column profile: dtype, cardinality, nulls, zeros, mode and quantiles.

    Parameters
    ----------
    data
        Input frame.
    numeric_only
        Restrict the profile to numeric columns (the default). When ``False``,
        text and categorical columns are profiled too and their numeric fields
        are reported as null.
    round_columns
        Optional ``{column: decimals}`` mapping. When supplied, those output
        columns are rendered as formatted strings. Left as ``None`` the result
        stays numeric, which is what the CSV/Excel exports need.
    drop_columns
        Output columns to omit.
    saving_path
        Directory to write ``Data_Description.csv`` and a PNG rendering into.
    dtype_labels
        Overrides for the reported ``DataType``. Used to show the dtypes as
        they were in the source file even when the frame has since been
        losslessly downcast.

    Returns
    -------
    pandas.DataFrame
        One row per profiled column, indexed by column name.
    """
    if not data.columns.is_unique:
        raise ValueError("describe_data() requires unique column names.")
    if numeric_only:
        selected = numeric_columns(data)
    else:
        selected = list(data.columns)

    n_rows = int(data.shape[0])
    if not selected:
        return pd.DataFrame(columns=DESCRIBE_COLUMNS)

    records: list[dict[str, object]] = []
    for name in selected:
        series = data[name]
        if is_numeric_series(series):
            record = _numeric_column_stats(series, n_rows)
        else:
            record = _discrete_column_stats(series, n_rows)
        record["DataType"] = (dtype_labels or {}).get(str(name), str(series.dtype))
        records.append(record)

    result = pd.DataFrame(records, index=pd.Index(selected, name=None))
    result["Perc. of Missing"] = (result["n_Missing"] / n_rows * 100.0) if n_rows else np.nan
    result["Perc. of Zeros"] = (result["n_Zeros"] / n_rows * 100.0) if n_rows else np.nan
    result = result[DESCRIBE_COLUMNS]

    for column in ("Perc. of Missing", "Perc. of Zeros", "Perc. of 1st Most Freq"):
        result[column] = result[column].astype(float).round(3)

    if round_columns:
        for column, decimals in round_columns.items():
            if column in result.columns:
                result[column] = result[column].map(
                    lambda value, d=decimals: value
                    if not isinstance(value, (int, float, np.number)) or pd.isna(value)
                    else f"{value:,.{d}f}"
                )

    if drop_columns is not None:
        result = result.drop(list(drop_columns), axis=1, errors="ignore")

    if saving_path is not None:
        from .static import df_to_img

        folder = Path(saving_path)
        folder.mkdir(parents=True, exist_ok=True)
        result.to_csv(folder / "Data_Description.csv", index=True)
        figure = df_to_img(
            result,
            title="Descriptive statistics",
            show_index=True,
            save=folder / "Data_Description.png",
        )
        figure.clear()

    return result


# --------------------------------------------------------------------------- #
# missing values
# --------------------------------------------------------------------------- #
def missing_count(
    data: pd.DataFrame,
    sort: bool = True,
    saving_path: str | None = None,
    non_zero_only: bool = True,
) -> pd.DataFrame:
    """Count and percentage of missing values per column.

    Parameters
    ----------
    data
        Input frame.
    sort
        Sort descending by number of missing values.
    saving_path
        Directory to write ``Missing_analysis.csv`` and a PNG rendering into.
        The CSV always contains every column; the returned frame honours
        ``non_zero_only``.
    non_zero_only
        Return only the columns that actually have missing values.
    """
    n_rows = int(data.shape[0])
    # Summing one column at a time caps temporary memory at one boolean Series.
    # ``data.isna()`` materialises an n_rows x n_columns boolean DataFrame.
    counts = np.fromiter(
        (int(series.isna().sum()) for _, series in data.items()),
        dtype=np.int64,
        count=int(data.shape[1]),
    )

    missing = pd.DataFrame(
        {
            "number_of_missing": counts,
            "percentage_of_missing": (counts / n_rows * 100.0) if n_rows else np.zeros_like(counts, dtype=float),
        },
        index=pd.Index(data.columns.to_numpy()),
    )
    missing["percentage_of_missing"] = missing["percentage_of_missing"].round(4)

    result = missing[missing["number_of_missing"] > 0].copy() if non_zero_only else missing.copy()
    if sort:
        result = result.sort_values(by="number_of_missing", ascending=False, kind="stable")

    if saving_path is not None:
        from .static import df_to_img

        folder = Path(saving_path)
        folder.mkdir(parents=True, exist_ok=True)
        missing.to_csv(folder / "Missing_analysis.csv", index=True)
        figure = df_to_img(
            result,
            title="Missing values",
            show_index=True,
            save=folder / "Missing_analysis.png",
        )
        figure.clear()

    return result


# --------------------------------------------------------------------------- #
# correlation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _CorrelationInput:
    """Numeric matrix shared by every correlation method."""

    values: np.ndarray  # (n_rows, n_features) float64
    columns: list[str]
    has_missing: bool
    n_rows_used: int
    n_rows_total: int
    sampled: bool


def _prepare_correlation_input(
    data: pd.DataFrame,
    columns: Sequence[str] | None,
    max_rows: int | None,
    random_state: int,
) -> _CorrelationInput:
    if max_rows is not None:
        if isinstance(max_rows, bool) or not isinstance(max_rows, (int, np.integer)):
            raise TypeError("max_rows must be a positive integer or None")
        if int(max_rows) <= 0:
            raise ValueError("max_rows must be a positive integer or None")
        max_rows = int(max_rows)

    names = numeric_columns(data, list(columns) if columns is not None else None)
    n_rows_total = int(data.shape[0])

    if not names:
        return _CorrelationInput(
            values=np.empty((0, 0), dtype=np.float64),
            columns=[],
            has_missing=False,
            n_rows_used=0,
            n_rows_total=n_rows_total,
            sampled=False,
        )

    block = data[names]
    sampled = False
    if max_rows is not None and n_rows_total > max_rows:
        # Deterministic subsample so a rerun of the audit reproduces the number.
        rng = np.random.default_rng(random_state)
        take = np.sort(rng.choice(n_rows_total, size=max_rows, replace=False))
        block = block.iloc[take]
        sampled = True

    values = block.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    return _CorrelationInput(
        values=values,
        columns=names,
        has_missing=bool(np.isnan(values).any()),
        n_rows_used=int(values.shape[0]),
        n_rows_total=n_rows_total,
        sampled=sampled,
    )


def _pearson_dense(values: np.ndarray, *, overwrite: bool = False) -> np.ndarray:
    """Pearson matrix for a matrix with no missing values, via one BLAS call."""
    n_rows, n_cols = values.shape
    if n_rows < 2 or n_cols == 0:
        return np.full((n_cols, n_cols), np.nan)

    centered = values if overwrite else values.copy()
    centered -= centered.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.einsum("ij,ij->j", centered, centered))
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = (centered.T @ centered) / np.outer(norms, norms)

    constant = norms == 0.0
    if constant.any():
        matrix[constant, :] = np.nan
        matrix[:, constant] = np.nan

    np.clip(matrix, -1.0, 1.0, out=matrix)
    diagonal = np.where(constant, np.nan, 1.0)
    np.fill_diagonal(matrix, diagonal)
    return matrix


def _rank_columns(values: np.ndarray) -> np.ndarray:
    """Average ranks per column, vectorised across the whole matrix."""
    from scipy.stats import rankdata

    return rankdata(values, axis=0).astype(np.float64, copy=False)


def _kendall_pairwise(values: np.ndarray, *, max_workers: int | None = None) -> np.ndarray:
    """Kendall tau-b over feature pairs, matching ``DataFrame.corr``.

    Kendall is the only correlation here that cannot be reduced to a matrix
    product: every pair needs its own O(n log n) inversion count. SciPy's
    implementation releases the GIL, so the pairs are spread over a thread
    pool. Each pair writes to its own cell of the output, so the result does
    not depend on how the work was scheduled.
    """
    from scipy.stats import kendalltau

    n_cols = values.shape[1]
    matrix = np.full((n_cols, n_cols), np.nan)
    # Fortran order keeps each column contiguous, which is what SciPy wants.
    columns = np.asfortranarray(values)
    missing = np.isnan(columns)
    any_missing = bool(missing.any())
    if any_missing:
        valid: np.ndarray | None = np.asfortranarray(~missing)
        n_valid = valid.sum(axis=0)
    else:
        valid = None
        n_valid = np.full(n_cols, values.shape[0], dtype=np.int64)
    del missing

    # A feature with no spread has an undefined tau; detecting that once per
    # column avoids paying for a full O(n log n) pass that can only return NaN.
    constant = np.empty(n_cols, dtype=bool)
    for i in range(n_cols):
        present = columns[:, i][valid[:, i]] if valid is not None else columns[:, i]
        constant[i] = present.size == 0 or bool(np.all(present == present[0]))

    # ``DataFrame.corr`` leaves the diagonal undefined only for a column with
    # no observations at all; a constant column still correlates with itself.
    matrix[np.arange(n_cols), np.arange(n_cols)] = np.where(n_valid > 0, 1.0, np.nan)

    usable = np.flatnonzero(~constant)
    n_pairs = int(usable.size * (usable.size - 1) // 2)
    if n_pairs == 0:
        return matrix

    def iter_pairs():
        for position, i in enumerate(usable):
            for j in usable[position + 1 :]:
                yield int(i), int(j)

    def tau_for(pair: tuple[int, int]) -> tuple[int, int, float]:
        i, j = pair
        if valid is not None:
            mask = valid[:, i] & valid[:, j]
            if int(mask.sum()) < 2:
                return i, j, np.nan
            a = columns[mask, i]
            b = columns[mask, j]
        else:
            a = columns[:, i]
            b = columns[:, j]
        return i, j, float(kendalltau(a, b, variant="b")[0])

    workers = _worker_count(n_pairs, values.shape[0], max_workers)
    if workers > 1:
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = iter(iter_pairs())
            pending = {
                pool.submit(tau_for, pair)
                for _, pair in zip(range(workers * 2), pairs)
            }
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    i, j, tau = future.result()
                    matrix[i, j] = matrix[j, i] = tau
                    try:
                        pending.add(pool.submit(tau_for, next(pairs)))
                    except StopIteration:
                        pass
    else:
        for pair in iter_pairs():
            i, j, tau = tau_for(pair)
            matrix[i, j] = matrix[j, i] = tau

    return matrix


def _worker_count(n_pairs: int, n_rows: int, max_workers: int | None) -> int:
    """Threads worth using for the pairwise Kendall loop.

    Below a few thousand rows the pool costs more than it saves, so the serial
    path is kept for small inputs.
    """
    if max_workers is not None:
        return max(1, int(max_workers))
    if n_pairs < 3 or n_rows < 2_000:
        return 1
    import os

    available = os.cpu_count() or 1
    return int(max(1, min(available, 8, n_pairs)))


def correlation_matrix(
    data: pd.DataFrame,
    method: str = "pearson",
    *,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Correlation matrix over the numeric columns of ``data``.

    Missing values are handled pairwise-complete, matching the default of
    :meth:`pandas.DataFrame.corr`.

    Parameters
    ----------
    method
        ``"pearson"``, ``"spearman"`` or ``"kendall"`` (case-insensitive).
    columns
        Restrict the analysis to these columns; non-numeric ones are dropped.
    max_rows
        Deterministically subsample to at most this many rows before computing.
        ``None`` (the default) always uses every row.
    random_state
        Seed for that subsample, so the result is reproducible.
    """
    key = _normalise_correlation_method(method)
    prepared = _prepare_correlation_input(data, columns, max_rows, random_state)
    return _correlation_from_input(prepared, key)


def _normalise_correlation_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError("correlation method must be a string")
    key = method.casefold()
    if key not in _SUPPORTED_CORRELATIONS:
        raise ValueError(
            f"Unsupported correlation method: {method!r}. "
            f"Expected one of {_SUPPORTED_CORRELATIONS}."
        )
    return key


def _correlation_from_input(prepared: _CorrelationInput, method: str) -> pd.DataFrame:
    key = _normalise_correlation_method(method)

    names = prepared.columns
    if not names:
        return pd.DataFrame(dtype=float)

    values = prepared.values
    index = pd.Index(names)

    if key == "pearson":
        matrix = (
            _pearson_dense(values)
            if not prepared.has_missing
            else pd.DataFrame(values, columns=names, copy=False).corr(method="pearson").to_numpy()
        )
    elif key == "spearman":
        if not prepared.has_missing:
            # Ranking once and reusing the dense Pearson kernel turns an
            # O(k^2) Python loop into a single matrix product.
            matrix = _pearson_dense(_rank_columns(values), overwrite=True)
        else:
            matrix = pd.DataFrame(values, columns=names, copy=False).corr(method="spearman").to_numpy()
    else:
        matrix = _kendall_pairwise(values)

    return pd.DataFrame(matrix, index=index, columns=index)


def correlation_matrices(
    data: pd.DataFrame,
    methods: Sequence[str] = ("pearson", "spearman"),
    *,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
    random_state: int = 0,
) -> dict[str, pd.DataFrame]:
    """Several correlation matrices from a single extraction of the data.

    Preparing the numeric matrix once and reusing it is the reason this is
    cheaper than calling :func:`correlation_matrix` in a loop.
    """
    normalised = list(dict.fromkeys(_normalise_correlation_method(method) for method in methods))
    if not normalised:
        return {}
    prepared = _prepare_correlation_input(data, columns, max_rows, random_state)
    return {method: _correlation_from_input(prepared, method) for method in normalised}


def top_absolute_correlations(
    corr: pd.DataFrame,
    threshold: float | None = None,
    method_label: str = "correlation",
) -> pd.DataFrame:
    """Feature pairs ranked by absolute correlation.

    Only the upper triangle is considered, so each pair appears once. The
    ranking is built with NumPy index arrays instead of ``unstack``, which
    keeps it usable on wide frames.
    """
    columns = [str(c) for c in corr.columns]
    n = len(columns)
    empty = pd.DataFrame(columns=["Feature 1", "Feature 2", f"absolute {method_label} correlation values"])
    if n < 2:
        return empty

    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1, or None")

    values = corr.to_numpy(dtype=np.float64, copy=False)
    rows, cols = np.triu_indices(n, k=1)
    magnitudes = np.abs(values[rows, cols])

    keep = np.isfinite(magnitudes)
    if threshold is not None:
        keep &= magnitudes > threshold
    rows, cols, magnitudes = rows[keep], cols[keep], magnitudes[keep]
    if magnitudes.size == 0:
        return empty

    order = np.argsort(-magnitudes, kind="stable")
    rows, cols, magnitudes = rows[order], cols[order], magnitudes[order]

    names = np.asarray(columns, dtype=object)
    result = pd.DataFrame(
        {
            "Feature 1": names[rows],
            "Feature 2": names[cols],
            f"absolute {method_label} correlation values": magnitudes,
        }
    )
    return result.reset_index(drop=True)

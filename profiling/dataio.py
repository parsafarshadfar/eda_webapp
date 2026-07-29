"""Loading and memory-efficient preparation of tabular data.

Two things matter here for large files:

* the CSV parser -- PyArrow's multi-threaded reader is used when available and
  falls back to the pandas C parser, so behaviour is identical either way;
* the resulting dtypes -- only *lossless* conversions are applied, so the
  numbers used by every downstream analysis are bit-for-bit the values that
  were present in the file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from dataclasses import dataclass, field
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

__all__ = [
    "LoadedData",
    "load_table",
    "optimize_memory",
    "memory_usage_mb",
    "dtype_summary",
    "is_numeric_series",
    "is_categorical_series",
    "numeric_columns",
]

# A string column is only converted to ``category`` when it repeats enough to
# actually save memory, and only when the category index stays small.
_CATEGORY_MAX_CARDINALITY = 10_000
_CATEGORY_MAX_UNIQUE_RATIO = 0.5

_CANDIDATE_DELIMITERS = ",;\t|"
_SNIFF_BYTES = 64 * 1024
_HASH_CHUNK = 4 * 1024 * 1024


@dataclass(frozen=True)
class LoadedData:
    """A loaded dataset together with everything needed to audit the load."""

    frame: pd.DataFrame
    source_name: str
    parser: str
    delimiter: str
    token: str
    original_dtypes: dict[str, str] = field(default_factory=dict)
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    optimizations: dict[str, str] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.frame.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.frame.shape[1])

    @property
    def memory_saved_mb(self) -> float:
        return max(0.0, self.memory_before_mb - self.memory_after_mb)


def _read_bytes(source: Any) -> tuple[bytes, str]:
    """Return the raw bytes of ``source`` plus a human readable name."""
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        with open(path, "rb") as handle:
            return handle.read(), os.path.basename(path)

    name = getattr(source, "name", "uploaded_file")
    if hasattr(source, "seek"):
        source.seek(0)
    payload = source.read()
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return payload, os.path.basename(str(name))


def _token_for(payload: bytes) -> str:
    """Content fingerprint used as a cache key.

    Hashing the raw upload is far cheaper than letting the cache layer hash a
    materialised DataFrame, and it is stable across reruns and sessions.
    """
    digest = hashlib.blake2b(digest_size=16)
    for start in range(0, len(payload), _HASH_CHUNK):
        digest.update(payload[start : start + _HASH_CHUNK])
    digest.update(str(len(payload)).encode("ascii"))
    return digest.hexdigest()


def _detect_delimiter(payload: bytes) -> str:
    """Pick the delimiter from the header line.

    Only the characters in :data:`_CANDIDATE_DELIMITERS` are considered and a
    comma is used whenever the sample is inconclusive, so a normal CSV is never
    parsed in a surprising way.
    """
    sample = payload[:_SNIFF_BYTES].decode("utf-8", errors="replace")
    if not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=_CANDIDATE_DELIMITERS)
        if dialect.delimiter in _CANDIDATE_DELIMITERS:
            return dialect.delimiter
    except csv.Error:
        pass

    header = sample.splitlines()[0]
    counts = {sep: header.count(sep) for sep in _CANDIDATE_DELIMITERS}
    best = max(counts, key=lambda sep: counts[sep])
    return best if counts[best] > 0 else ","


def _read_with_pyarrow(payload: bytes, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload), engine="pyarrow", sep=delimiter)


def _read_with_c_parser(payload: bytes, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload), sep=delimiter, low_memory=False)


def load_table(
    source: Any,
    *,
    delimiter: str | None = None,
    optimize: bool = True,
    prefer_pyarrow: bool = True,
) -> LoadedData:
    """Read a delimited text file into a memory-optimised DataFrame.

    Parameters
    ----------
    source
        A filesystem path or any binary file-like object (including the object
        returned by Streamlit's file uploader).
    delimiter
        Forced field separator. ``None`` detects it from the header line.
    optimize
        Apply the lossless dtype conversions described in
        :func:`optimize_memory`.
    prefer_pyarrow
        Try the multi-threaded PyArrow CSV parser first. The pandas C parser is
        used as a fallback and produces the same frame.
    """
    payload, source_name = _read_bytes(source)
    token = _token_for(payload)
    sep = delimiter or _detect_delimiter(payload)

    frame: pd.DataFrame | None = None
    parser = "pandas-c"
    if prefer_pyarrow:
        try:
            frame = _read_with_pyarrow(payload, sep)
            parser = "pyarrow"
        except Exception:
            frame = None
    if frame is None:
        frame = _read_with_c_parser(payload, sep)
        parser = "pandas-c"

    del payload

    original_dtypes = {str(col): str(dtype) for col, dtype in frame.dtypes.items()}
    memory_before = memory_usage_mb(frame)

    optimizations: dict[str, str] = {}
    if optimize:
        frame, optimizations = optimize_memory(frame)

    return LoadedData(
        frame=frame,
        source_name=source_name,
        parser=parser,
        delimiter=sep,
        token=token,
        original_dtypes=original_dtypes,
        memory_before_mb=memory_before,
        memory_after_mb=memory_usage_mb(frame),
        optimizations=optimizations,
    )


def optimize_memory(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Shrink a DataFrame using conversions that cannot change a value.

    Two transformations are applied:

    * integer columns are downcast to the narrowest integer type that still
      represents every value exactly;
    * low-cardinality text columns become ``category``.

    Floating point columns are deliberately left at their original precision:
    demoting ``float64`` to ``float32`` would change statistics and is not
    acceptable for validation work.
    """
    applied: dict[str, str] = {}
    new_columns: dict[str, pd.Series] = {}

    for name in frame.columns:
        series = frame[name]
        dtype = series.dtype
        kind = getattr(dtype, "kind", "")

        if kind in "iu":
            downcast = "unsigned" if kind == "u" else "integer"
            converted = pd.to_numeric(series, downcast=downcast)
            if converted.dtype != dtype:
                new_columns[name] = converted
                applied[str(name)] = f"{dtype} -> {converted.dtype}"
            continue

        if isinstance(dtype, pd.CategoricalDtype) or kind == "f" or kind == "b":
            continue

        if _is_text_dtype(dtype):
            n_rows = len(series)
            if n_rows == 0:
                continue
            n_unique = int(series.nunique(dropna=True))
            if n_unique <= _CATEGORY_MAX_CARDINALITY and n_unique <= _CATEGORY_MAX_UNIQUE_RATIO * n_rows:
                new_columns[name] = series.astype("category")
                applied[str(name)] = f"{dtype} -> category"

    if new_columns:
        frame = frame.copy(deep=False)
        for name, series in new_columns.items():
            frame[name] = series

    return frame, applied


def _is_text_dtype(dtype: Any) -> bool:
    """True for object/str columns across pandas 2.x and 3.x."""
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    if getattr(dtype, "kind", "") in "OU":
        return True
    return pd.api.types.is_string_dtype(dtype)


def memory_usage_mb(frame: pd.DataFrame) -> float:
    """Deep memory footprint of ``frame`` in megabytes."""
    if frame.empty and frame.shape[1] == 0:
        return 0.0
    return float(frame.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)


def is_numeric_series(series: pd.Series) -> bool:
    """Numeric for profiling purposes: numbers, but not booleans."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    if pd.api.types.is_bool_dtype(dtype):
        return False
    return pd.api.types.is_numeric_dtype(dtype)


def is_categorical_series(series: pd.Series) -> bool:
    """Discrete for profiling purposes: text, category, or boolean."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(dtype):
        return True
    return _is_text_dtype(dtype)


def numeric_columns(frame: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    """Names of the numeric columns, preserving the requested order."""
    candidates = list(frame.columns) if columns is None else [c for c in columns if c in frame.columns]
    return [c for c in candidates if is_numeric_series(frame[c])]


def dtype_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Column-count-per-dtype breakdown used on the report overview page."""
    counts = frame.dtypes.astype(str).value_counts()
    return pd.DataFrame(
        {"dtype": counts.index.to_numpy(), "n_columns": counts.to_numpy(dtype=np.int64)}
    ).reset_index(drop=True)

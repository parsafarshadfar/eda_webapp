"""Loading and memory-efficient preparation of tabular data.

Two things matter here for large files:

* the CSV parser -- PyArrow's multi-threaded reader is used when available and
  falls back to the pandas C parser, so behaviour is identical either way;
* the resulting dtypes -- only *lossless* conversions are applied, so the
  numbers used by every downstream analysis are bit-for-bit the values that
  were present in the file.

The second half of the module is reshaping rather than loading: stacking several
frames into one tagged frame (:func:`combine_datasets`) and collapsing one-hot
encoded columns back into single categorical ones
(:func:`merge_one_hot_encoded_columns`). Both are preparation steps that belong
before a profiling run rather than inside it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import _spark

__all__ = [
    "LoadedData",
    "CombinedData",
    "load_table",
    "optimize_memory",
    "memory_usage_mb",
    "dtype_summary",
    "is_numeric_series",
    "is_categorical_series",
    "numeric_columns",
    "to_pandas",
    "combine_datasets",
    "merge_one_hot_encoded_columns",
]

#: Default name of the column that records which input frame a row came from.
DEFAULT_TAG_COLUMN = "Dataset"

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
    """Return a file-like source as bytes without changing its cursor."""
    name = getattr(source, "name", "uploaded_file")
    original_position: int | None = None
    if hasattr(source, "tell"):
        try:
            original_position = int(source.tell())
        except (OSError, TypeError, ValueError):
            original_position = None

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        payload = source.read()
    finally:
        if original_position is not None and hasattr(source, "seek"):
            try:
                source.seek(original_position)
            except (OSError, TypeError, ValueError):
                pass

    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    elif isinstance(payload, (bytearray, memoryview)):
        payload = bytes(payload)
    elif not isinstance(payload, bytes):
        raise TypeError(
            "load_table() expects a path or a file-like object whose read() "
            "method returns bytes or text."
        )
    return payload, os.path.basename(str(name)) or "uploaded_file"


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


def _inspect_path(path: str | os.PathLike[str]) -> tuple[str, bytes]:
    """Fingerprint a file incrementally and retain only its delimiter sample.

    Reading a filesystem source directly through pandas avoids holding the raw
    file and the parsed DataFrame in memory at the same time.  The digest stays
    byte-for-byte compatible with :func:`_token_for`.
    """
    digest = hashlib.blake2b(digest_size=16)
    sample = bytearray()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
            size += len(chunk)
            if len(sample) < _SNIFF_BYTES:
                needed = _SNIFF_BYTES - len(sample)
                sample.extend(chunk[:needed])
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest(), bytes(sample)


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


def _csv_source(source: Any) -> Any:
    """Wrap byte payloads while leaving filesystem paths untouched."""
    return io.BytesIO(source) if isinstance(source, bytes) else source


def _read_with_pyarrow(source: Any, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(_csv_source(source), engine="pyarrow", sep=delimiter)


def _read_with_c_parser(source: Any, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(_csv_source(source), sep=delimiter, low_memory=False)


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
    if delimiter is not None and (not isinstance(delimiter, str) or len(delimiter) != 1):
        raise ValueError("delimiter must be exactly one character or None")

    payload: bytes | None
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        token, sample = _inspect_path(path)
        parser_source: Any = path
        source_name = os.path.basename(path) or str(path)
        payload = None
    else:
        payload, source_name = _read_bytes(source)
        token = _token_for(payload)
        sample = payload[:_SNIFF_BYTES]
        parser_source = payload

    sep = delimiter if delimiter is not None else _detect_delimiter(sample)

    frame: pd.DataFrame | None = None
    parser = "pandas-c"
    if prefer_pyarrow:
        try:
            frame = _read_with_pyarrow(parser_source, sep)
            parser = "pyarrow"
        except Exception:
            frame = None
    if frame is None:
        frame = _read_with_c_parser(parser_source, sep)
        parser = "pandas-c"

    if payload is not None:
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
    if not frame.columns.is_unique:
        raise ValueError("optimize_memory() requires unique column names.")

    applied: dict[str, str] = {}
    optimized = frame

    for name in frame.columns:
        series = frame[name]
        dtype = series.dtype
        kind = getattr(dtype, "kind", "")
        converted: pd.Series | None = None

        if kind in ("i", "u"):
            downcast = "unsigned" if kind == "u" else "integer"
            converted = pd.to_numeric(series, downcast=downcast)
        elif (
            not isinstance(dtype, pd.CategoricalDtype)
            and kind not in ("f", "b")
            and _is_text_dtype(dtype)
        ):
            n_rows = len(series)
            if n_rows == 0:
                continue
            n_unique = int(series.nunique(dropna=True))
            if n_unique <= _CATEGORY_MAX_CARDINALITY and n_unique <= _CATEGORY_MAX_UNIQUE_RATIO * n_rows:
                converted = series.astype("category")

        if converted is None or converted.dtype == dtype:
            continue

        # Copy the frame lazily, then install each converted column immediately.
        # The previous implementation retained every converted Series in a
        # dictionary until the end, which could temporarily double the memory
        # used by all optimised columns on a wide dataset.
        if optimized is frame:
            optimized = frame.copy(deep=False)
        optimized[name] = converted
        applied[str(name)] = f"{dtype} -> {converted.dtype}"

    return optimized, applied


def _is_text_dtype(dtype: Any) -> bool:
    """True for object/str columns across pandas 2.x and 3.x."""
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    if getattr(dtype, "kind", "") in ("O", "U"):
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
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
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
    if not frame.columns.is_unique:
        raise ValueError("Numeric feature selection requires unique column names.")
    candidates = list(frame.columns) if columns is None else [c for c in columns if c in frame.columns]
    selected: list[str] = []
    seen: set[Any] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if is_numeric_series(frame[candidate]):
            selected.append(candidate)
    return selected


def dtype_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Column-count-per-dtype breakdown used on the report overview page."""
    counts = frame.dtypes.astype(str).value_counts()
    return pd.DataFrame(
        {"dtype": counts.index.to_numpy(), "n_columns": counts.to_numpy(dtype=np.int64)}
    ).reset_index(drop=True)


def to_pandas(data: Any) -> Any:
    """Materialise a Spark frame as pandas; return anything else unchanged.

    Safe to call on a plain DataFrame, so a function that may receive either can
    normalise its input in one line without importing anything Spark-related.
    """
    return _spark.to_pandas(data)


# --------------------------------------------------------------------------- #
# combining several frames into one
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CombinedData:
    """Several frames stacked into one, with a column recording their origin."""

    frame: Any
    feature_names: list[str]
    tag_column: str | None
    sources: dict[str, int] = field(default_factory=dict)

    @property
    def is_combined(self) -> bool:
        """False when a single frame was passed through untouched."""
        return self.tag_column is not None

    @property
    def features(self) -> Any:
        """The frame restricted to the feature columns, i.e. without the tag."""
        return self.frame[self.feature_names]


def combine_datasets(
    data: Any,
    tag_column: str = DEFAULT_TAG_COLUMN,
) -> CombinedData:
    """Stack a mapping of frames into one frame tagged by key.

    ``{"Train": train, "Test": test}`` becomes a single frame with an extra
    column -- ``Dataset`` by default -- holding ``"Train"`` or ``"Test"``. That
    column is excluded from ``feature_names``, so the analyses never profile the
    tag itself, and it is exactly what :func:`profiling.grouping.make_grouping`
    then groups on to compare the two.

    A frame passed in on its own is returned untouched, with ``tag_column`` set
    to ``None``. That means a caller can always route its input through this
    function without special-casing the single-frame case.

    Parameters
    ----------
    data
        A DataFrame, or a mapping of name to DataFrame. pandas and Spark frames
        are both accepted; mixing the two in one mapping is not.
    tag_column
        Name of the column that records the origin of each row. It must not
        collide with an existing column.
    """
    if not isinstance(data, Mapping):
        return CombinedData(
            frame=data,
            feature_names=[str(column) for column in data.columns],
            tag_column=None,
            sources={},
        )

    if not data:
        raise ValueError("combine_datasets() was given an empty mapping of frames.")

    frames: dict[str, Any] = {}
    for key, value in data.items():
        label = str(key)
        if label in frames:
            raise ValueError(
                "Dataset labels must remain unique when converted to text; "
                f"more than one key becomes {label!r}."
            )
        if not hasattr(value, "columns") or not hasattr(value, "shape"):
            raise TypeError(f"Frame {label!r} is not a DataFrame-like object.")
        frames[label] = value

    spark_flags = {_spark.is_spark(frame) for frame in frames.values()}
    if len(spark_flags) != 1:
        raise TypeError("Cannot combine pandas and Spark frames in the same mapping.")

    for key, frame in frames.items():
        if tag_column in list(frame.columns):
            raise ValueError(
                f"Frame {key!r} already has a column named {tag_column!r}; "
                "pass a different tag_column."
            )

    first = next(iter(frames.values()))
    if _spark.is_spark(first):
        combined = _spark.spark_concat(frames, tag_column)
    else:
        combined = pd.concat(list(frames.values()), ignore_index=True)
        lengths = np.fromiter(
            (int(frame.shape[0]) for frame in frames.values()),
            dtype=np.intp,
            count=len(frames),
        )
        max_code = max(len(frames) - 1, 0)
        code_dtype = np.min_scalar_type(max_code)
        codes = np.repeat(np.arange(len(frames), dtype=code_dtype), lengths)
        combined[tag_column] = pd.Categorical.from_codes(
            codes,
            categories=list(frames),
        )

    feature_names = [str(column) for column in combined.columns if str(column) != tag_column]
    return CombinedData(
        frame=combined,
        feature_names=feature_names,
        tag_column=tag_column,
        sources={key: int(frame.shape[0]) for key, frame in frames.items()},
    )


# --------------------------------------------------------------------------- #
# one-hot encoding
# --------------------------------------------------------------------------- #
def _infer_dummy_groups(columns: Sequence[str], separator: str) -> dict[str, list[str]]:
    """Group ``Colour_red``/``Colour_blue`` under ``Colour`` using the separator."""
    groups: dict[str, list[str]] = {}
    for column in columns:
        text = str(column)
        position = text.rfind(separator)
        if position > 0:
            groups.setdefault(text[:position], []).append(column)
    return groups


def _dummy_groups_by_prefix(columns: Sequence[str], features: Iterable[str]) -> dict[str, list[str]]:
    """Group by explicit original feature names, matched as a substring."""
    return {
        str(feature): [column for column in columns if str(feature) in str(column)]
        for feature in features
    }


def merge_one_hot_encoded_columns(
    data: pd.DataFrame,
    features: Mapping[str, Sequence[str]] | Iterable[str] | str = "auto",
    separator: str = "__",
    *,
    strip_prefix: bool = False,
) -> pd.DataFrame:
    """Collapse one-hot-encoded columns back into single categorical columns.

    Profiling a wide block of 0/1 dummies says very little; profiling the one
    categorical column they encode says a lot. Each group of dummies is replaced,
    in the position of its first member, by one column holding the name of the
    winning dummy for that row.

    Parameters
    ----------
    data
        Input frame. It is not modified; a new frame is returned.
    features
        ``"auto"`` infers the groups from ``separator``. A list of original
        feature names groups by substring match on the column names. A mapping
        of ``{original: [dummy, ...]}`` states the groups explicitly.
    separator
        Separator used in the dummy names. Only consulted for ``"auto"``.
    strip_prefix
        Drop the ``"<feature><separator>"`` prefix from the merged values, so
        ``Colour__red`` becomes ``red``. Off by default, which keeps the full
        dummy name.

    Returns
    -------
    pandas.DataFrame
        A copy with each dummy group replaced by a single column.

    Notes
    -----
    The winner of a row is its largest dummy, so a row in which every dummy is
    zero is attributed to the first column of the group rather than reported as
    unknown. Rows where the whole group is null stay null.
    """
    if not data.columns.is_unique:
        raise ValueError("merge_one_hot_encoded_columns() requires unique column names.")
    if not separator and isinstance(features, str) and features.lower() == "auto":
        raise ValueError("separator cannot be empty when features='auto'.")

    if isinstance(features, Mapping):
        groups = {str(key): list(value) for key, value in features.items()}
    elif isinstance(features, str):
        if features.lower() != "auto":
            raise ValueError(
                f"features must be 'auto', a list of feature names or a mapping; got {features!r}."
            )
        groups = _infer_dummy_groups(list(data.columns), separator)
    else:
        groups = _dummy_groups_by_prefix(list(data.columns), features)

    replacements: dict[Any, tuple[str, pd.Series]] = {}
    consumed: set[Any] = set()

    for feature, dummies in groups.items():
        present = [column for column in dummies if column in data.columns]
        if len(present) < 2:
            continue

        overlap = consumed.intersection(present)
        if overlap:
            raise ValueError(
                f"Dummy columns cannot belong to more than one feature group: {sorted(map(str, overlap))}."
            )
        if feature in data.columns and feature not in present:
            raise ValueError(
                f"Cannot create merged feature {feature!r}: a column with that name already exists."
            )

        block = data[present]
        if any(
            not (
                pd.api.types.is_numeric_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
            )
            for dtype in block.dtypes
        ):
            raise TypeError(
                f"Dummy group {feature!r} must contain only numeric or boolean columns."
            )

        all_missing = block.isna().all(axis=1)
        winner = pd.Series(pd.NA, index=data.index, dtype=object)
        present_rows = ~all_missing
        if bool(present_rows.any()):
            winner.loc[present_rows] = block.loc[present_rows].idxmax(
                axis=1,
                skipna=True,
            )
        if strip_prefix:
            prefix = f"{feature}{separator}"
            winner = winner.map(
                {
                    dummy: (
                        dummy[len(prefix) :]
                        if isinstance(dummy, str) and dummy.startswith(prefix)
                        else dummy
                    )
                    for dummy in present
                },
                na_action="ignore",
            )

        replacements[present[0]] = (feature, winner.rename(feature))
        consumed.update(present)

    if not replacements:
        return data.copy()

    # Assemble the result once. Repeated drop/insert operations copy a wide
    # DataFrame for every dummy group and become quadratic in the column count.
    columns: list[pd.Series] = []
    for name, series in data.items():
        replacement = replacements.get(name)
        if replacement is not None:
            columns.append(replacement[1])
        elif name not in consumed:
            columns.append(series)
    return pd.concat(columns, axis=1)

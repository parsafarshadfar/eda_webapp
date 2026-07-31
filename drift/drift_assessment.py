"""Memory-conscious data-drift assessment.

The public class and helper signatures in this module intentionally match the
original notebook API.  Internally, work is organised feature-by-feature so the
largest temporary allocation is one column (plus the small result matrices),
and every expensive conversion is reused across all requested drift methods.

Two supporting modules keep this file focused on the statistics:
``drift.timeperiods`` understands date columns of any type and cuts them into
snapshots (monthly by default), and ``drift.storage`` decides where results are
saved and writes each file in a single operation so local folders, DBFS and ADLS
all behave the same.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any
import os
import re
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    energy_distance,
    gaussian_kde,
    ks_2samp,
    wasserstein_distance,
)

try:  # Package import: ``from drift import DriftAssessment``.
    from . import storage
    from .drift_util import DEFAULT_DPI, PSI_THRESHOLDS, df_to_img
    from .timeperiods import DEFAULT_INTERVAL, normalise_interval, snapshot_labels
except ImportError:  # Notebook compatibility when ``drift/`` is on sys.path.
    import storage  # type: ignore[no-redef]
    from drift_util import (  # type: ignore[no-redef]
        DEFAULT_DPI,
        PSI_THRESHOLDS,
        df_to_img,
    )
    from timeperiods import (  # type: ignore[no-redef]
        DEFAULT_INTERVAL,
        normalise_interval,
        snapshot_labels,
    )

__all__ = [
    "DriftAssessment",
    "MyException",
    "merge_one_hot_encoded_columns",
    "psi_coloring",
    "yes_no",
]


_METHOD_ALIASES = {
    "ks2": "ks2",
    "2s-ks": "ks2",
    "two_sample_ks": "ks2",
    "wasserstein": "ws",
    "ws": "ws",
    "cm": "cm",
    "psi": "psi",
    "variables_psi": "psi",
    "scores_psi": "psi",
    "chisquare": "chisquare",
    "chi-square": "chisquare",
    "chi_square": "chisquare",
}
_NUMERIC_METHODS = frozenset({"ks2", "ws", "cm"})
_INVALID_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: Snapshot name used when the current data is compared as one population.
#: It covers the current/monitored rows only, never the baseline.
CURRENT_SNAPSHOT_LABEL = "Current data"

#: Decimals kept in the descriptive mean/std tables.
_MEAN_DECIMALS = 4

#: Short, stable file stems for the saved distance tables. Long names ("chisquare",
#: "missing_percentages") push a nested export past the 260-character path limit
#: Windows still enforces, so every name is abbreviated once, here.
DISTANCE_FILE_NAMES = {
    "psi": "psi",
    "ks2": "ks2",
    "ws": "ws",
    "cm": "cm",
    "chisquare": "chisq",
}

#: One folder per kind of result, numbered so they sort in reading order --
#: the same layout the Streamlit app writes, so a bundle from a notebook and a
#: bundle from the app can be read the same way.
DISTANCE_FOLDER = "01_distance_measures"
DESCRIPTIVE_FOLDER = "02_descriptive_trends"
MISSING_FOLDER = "03_missing_values"
SCORE_FOLDER = "04_model_score"

#: Inside each of those: "charts" holds one PNG per feature, and PSI's
#: per-snapshot bin tables (the most numerous files) get their own folder.
CHART_FOLDER = "charts"
PSI_BIN_FOLDER = "psi_bins"

#: Threading only pays off once there are a few features to overlap; below
#: this the coordination costs more than it saves.
_MIN_FEATURES_FOR_THREADS = 4

#: Upper bound on worker threads. Past this the memory traffic dominates and
#: extra threads stop helping.
_MAX_WORKERS = 8

# Backends that render to a file and have no window to show a figure in.
_NON_INTERACTIVE_BACKENDS = frozenset({"agg", "pdf", "ps", "svg", "cairo", "template"})


@dataclass
class _FeatureOutcome:
    """Everything one feature produced, ready to be merged on the main thread."""

    index: int
    values: dict[str, np.ndarray]
    ks_p_values: np.ndarray | None
    notes: list[str]
    skipped: list[tuple[str, Any, str]]
    psi_tables: list[tuple[Any, Any, pd.DataFrame]]


def _worker_count(n_jobs: int | None, n_features: int) -> int:
    """How many features to assess at once.

    ``None`` picks a sensible number: one thread per core, capped so a machine
    with 64 cores does not spawn 64 threads for 6 features, and never more than
    there are features to work on. ``1`` keeps everything on this thread, which
    is what a small dataset wants -- the threads would cost more than they save.
    """
    if n_jobs is not None:
        return max(1, min(int(n_jobs), n_features))
    if n_features < _MIN_FEATURES_FOR_THREADS:
        return 1
    return max(1, min(os.cpu_count() or 1, _MAX_WORKERS, n_features))


class MyException(Exception):
    """Backward-compatible custom exception type."""


def _is_path(value: Any) -> bool:
    return isinstance(value, (str, os.PathLike))


def _read_csv(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a CSV using the modern parallel parser when it is available."""
    try:
        return pd.read_csv(path, engine="pyarrow")
    except (ImportError, ModuleNotFoundError, ValueError):
        # The C parser is the supported fallback for environments without
        # PyArrow (including the documented older dependency set).
        return pd.read_csv(path, low_memory=False)


def _as_pandas_frame(value: Any, *, name: str) -> pd.DataFrame:
    """Normalise a supported frame/path without making an unnecessary deep copy."""
    if _is_path(value):
        value = _read_csv(value)

    if isinstance(value, pd.DataFrame):
        return value.copy(deep=False)

    # SciPy's distance functions ultimately require local NumPy arrays.  Spark
    # inputs are therefore materialised once here, rather than once per metric.
    if hasattr(value, "toPandas"):
        value = value.toPandas()
    elif hasattr(value, "to_pandas"):
        value = value.to_pandas()
    elif hasattr(value, "pandas_api"):
        pandas_api = value.pandas_api()
        value = pandas_api.to_pandas() if hasattr(pandas_api, "to_pandas") else pandas_api

    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas/Spark DataFrame or a CSV path")
    return value.copy(deep=False)


def _is_numeric_series(series: pd.Series) -> bool:
    """True for real-valued drift features, excluding booleans and categories."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    if pd.api.types.is_bool_dtype(dtype):
        return False
    return pd.api.types.is_numeric_dtype(dtype)


def _numeric_values(series: pd.Series | np.ndarray) -> np.ndarray:
    """Return finite float64 values, allocating no more than one feature column."""
    if isinstance(series, pd.Series):
        converted = pd.to_numeric(series, errors="coerce")
        values = converted.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    else:
        try:
            values = np.asarray(series, dtype=np.float64)
        except (TypeError, ValueError):
            converted = pd.to_numeric(pd.Series(series, copy=False), errors="coerce")
            values = converted.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)

    values = np.ravel(values)
    finite = np.isfinite(values)
    return values if finite.all() else values[finite]


def _categorical_values(series: pd.Series | np.ndarray) -> np.ndarray:
    """Return non-missing categorical values as a one-dimensional object array."""
    if not isinstance(series, pd.Series):
        series = pd.Series(np.ravel(series), copy=False)
    return series.loc[series.notna()].to_numpy(dtype=object, copy=False)


def _sample_values(series: pd.Series, count: int = 3) -> str:
    """A few real values from a column, for an error message."""
    present = series.dropna().astype("string").head(count).tolist()
    return ", ".join(repr(value) for value in present) if present else "only empty values"


def _safe_component(value: Any, fallback: str = "result") -> str:
    """Make one user-controlled label safe as a filename component."""
    text = _INVALID_FILE_CHARS.sub("_", str(value)).strip().strip(".")
    return text[:120] or fallback


def _save_dataframe_image(
    frame: pd.DataFrame,
    address: str,
    *,
    title: str | None = None,
    show_index: bool = True,
    color_thresholds: Sequence[float] | None = None,
    higher_is_worse: bool = True,
) -> None:
    """Save one blue table image and release its figure immediately.

    ``color_thresholds`` shades risky values amber and red, so a saved PSI table
    carries its own triage without a separate status column.
    """
    if frame.empty:
        return
    figure = df_to_img(
        frame,
        title=title,
        show_index=show_index,
        color_thresholds=color_thresholds,
        higher_is_worse=higher_is_worse,
    )
    try:
        storage.write_figure(figure, address, dpi=DEFAULT_DPI)
    finally:
        plt.close(figure)


def _normalise_bucket_type(bucket_type: str) -> str:
    text = str(bucket_type).strip().lower()
    if text in {"bin", "bins", "equal", "equal_width"}:
        return "bins"
    if text in {"quantile", "quantiles", "qcut"}:
        return "quantiles"
    raise ValueError("psi_bucket_type must be 'bins' or 'quantile(s)'")


class DriftAssessment:
    """Perform distribution-drift and over-time descriptive analysis.

    Parameters
    ----------
    data
        A DataFrame, CSV path, or mapping of snapshot labels to frames/paths.
        A mapping value may also be a list of frames/paths to concatenate.
    col_names
        Feature names included in the analysis.
    reference
        Baseline frame/path.  For dated data or a snapshot mapping, omitting it
        uses the first snapshot as the baseline.
    ohe_columns
        One-hot features in list or mapping form.  Their public behaviour is
        retained, while merged category labels are aligned to plain baselines.
    interval
        How ``date_col`` is cut into snapshots: ``"Monthly"`` (the default),
        ``"Quarterly"``, ``"Yearly"``, ``"Weekly"``, ``"Daily"`` or ``"All"``
        for one combined population.  Any common spelling works -- ``"M"``,
        ``"month"``, ``"monthly"`` all mean the same thing.
    date_col
        Date column used to create snapshots.  It may be text, a number such as
        ``201903``, a quarter label, a timestamp or a pandas ``Period``; see
        :mod:`drift.timeperiods`.
    save_results
        Write CSV/XLSX/PNG artifacts when true.  False performs no disk writes.
    save_folder_name
        Destination folder for saved artifacts.  Defaults to the standard
        ``results/drift`` folder (:func:`drift.output_dir`) and also accepts a
        DBFS or ADLS address.

    Notes
    -----
    The historical attributes (``data_dict``, ``cat_cols``, ``num_cols``,
    ``snapshots_labels`` and the others) remain available.  Result-bearing
    attributes ending in ``_`` are additive and let applications reuse a single
    computation without parsing files or recomputing metrics.
    """

    def __init__(
        self,
        data,
        col_names,
        reference=None,
        ohe_columns=None,
        interval=DEFAULT_INTERVAL,
        date_col=None,
        save_results=False,
        save_folder_name=None,
        score_col=None,
        sample_percent=100.0,
        random_state=0,
    ):
        if isinstance(col_names, (str, bytes)):
            raise TypeError("col_names must be a non-empty sequence of column names")
        try:
            self.col_names = list(col_names)
        except TypeError as exc:
            raise TypeError(
                "col_names must be a non-empty sequence of column names"
            ) from exc
        if not self.col_names:
            raise ValueError("col_names must contain at least one feature")
        if len(set(self.col_names)) != len(self.col_names):
            raise ValueError("col_names must not contain duplicates")

        # Any common spelling of the interval is accepted, and an omitted one
        # means monthly.  Without a date column there is nothing to cut by time,
        # so the data is treated as a single population.
        resolved_interval = normalise_interval(interval)
        if date_col is None:
            resolved_interval = "All"
        if (
            resolved_interval == "All"
            and reference is None
            and not isinstance(data, Mapping)
        ):
            raise ValueError(
                "reference is required when the current data is one population: "
                "pass reference=..., or a date_col to build snapshots from."
            )

        # A mapping copy and pandas shallow copies protect caller-owned structure
        # without doubling every data block in memory.
        self.data = dict(data) if isinstance(data, Mapping) else data
        self.reference = reference
        self.date_col = date_col
        # Carried alongside the features rather than among them: a model score is
        # an output, not an input, so it must not join the feature distances or
        # the missing-value tables. It is retained through preprocessing purely so
        # score_analysis() has it.
        self.score_col = score_col

        try:
            sample_percent = float(sample_percent)
        except (TypeError, ValueError) as exc:
            raise TypeError("sample_percent must be a number between 0 and 100") from exc
        if not 0.0 < sample_percent <= 100.0:
            raise ValueError("sample_percent must be greater than 0 and at most 100")
        self.sample_percent = sample_percent
        self.random_state = random_state
        #: Rows held by each population before subsampling, so a run can always
        #: report what it actually measured against what was supplied.
        self.rows_before_sampling_: dict[Any, int] = {}
        self.save_results = bool(save_results)
        self.OHE_features = ohe_columns
        self.interval = resolved_interval

        self.cat_cols: list[Any] = []
        self.cats_codes: dict[Any, dict[int, Any]] = {}
        self.num_cols: list[Any] = []
        self.data_dict: dict[Any, pd.DataFrame] = {}
        self.snapshots_labels: list[Any] = []

        # Application-facing results populated by analysis calls.
        self.distance_results_: dict[str, pd.DataFrame] = {}
        self.distance_extended_results_: dict[str, pd.DataFrame] = {}
        self.distance_figures_: dict[str, Any] = {}
        self.distance_notes_: list[str] = []
        self.ks_p_values_ = pd.DataFrame()
        self.ks_details_ = pd.DataFrame()
        self.descriptive_results_: dict[str, pd.DataFrame] = {}
        self.descriptive_figures_: dict[str, Any] = {}
        self.missing_percentages_ = pd.DataFrame()
        self.missing_counts_ = pd.DataFrame()
        self.score_distribution_ = pd.DataFrame()
        self.score_summary_ = pd.DataFrame()
        self.score_density_ = pd.DataFrame()

        # One standard place for results, overridable per assessment.  Kept as a
        # plain string so cloud addresses survive intact.
        self._save_path = storage.resolve_dir(save_folder_name)
        self.save_folder = self._save_path
        if self.save_results:
            storage.ensure_dir(self._save_path)

        self._preprocess()

    # ------------------------------------------------------------------ setup
    def _retained_columns(self) -> list[Any]:
        """Feature columns, plus the model-score column when one was named.

        Every frame is narrowed to these columns during preprocessing. Keeping
        the score here — and nowhere near ``col_names`` — is what stops it from
        being treated as a drift feature in its own right.
        """
        columns = list(self.col_names)
        if self.score_col is not None and self.score_col not in columns:
            columns.append(self.score_col)
        return columns

    @staticmethod
    def _score_density(
        populations: Mapping[Any, pd.Series],
        grid_points: int = 256,
        max_points: int = 20_000,
    ) -> pd.DataFrame:
        """Kernel density curves for every population, on one shared grid.

        The point of the curves is to be laid over each other in a single set of
        axes, so they have to be evaluated at the same x values -- a per-population
        grid would make two identical distributions look different. The grid spans
        the union of the populations' ranges, padded by one bandwidth-ish margin so
        the tails are not clipped flat.

        Returns an empty frame when no population has enough spread to estimate a
        density from (fewer than two distinct values), which is the one case
        ``gaussian_kde`` cannot handle.
        """
        usable: dict[Any, np.ndarray] = {}
        for label, series in populations.items():
            values = _numeric_values(series)
            if values.size >= 2 and np.unique(values).size >= 2:
                usable[label] = values
        if not usable:
            return pd.DataFrame()

        low = min(float(values.min()) for values in usable.values())
        high = max(float(values.max()) for values in usable.values())
        if not np.isfinite([low, high]).all() or low == high:
            return pd.DataFrame()
        margin = 0.05 * (high - low)
        grid = np.linspace(low - margin, high + margin, num=int(grid_points))

        curves: dict[Any, np.ndarray] = {}
        for label, values in usable.items():
            if values.size > max_points:
                # A Gaussian KDE costs O(grid x points); past this many rows the
                # curve stops changing but the wait does not. Evenly spaced so the
                # thinned sample still spans the distribution.
                step = values.size / max_points
                picks = (np.arange(max_points) * step).astype(np.int64)
                values = np.sort(values)[picks]
            try:
                curves[label] = gaussian_kde(values)(grid)
            except (np.linalg.LinAlgError, ValueError):
                # A singular covariance means no usable spread; the band table
                # still describes this population.
                continue

        if not curves:
            return pd.DataFrame()
        density = pd.DataFrame(curves, index=pd.Index(grid, name="Score"))
        return density

    @staticmethod
    def _subsample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
        """A seeded random share of ``frame``, keeping at least one row."""
        keep = max(1, int(round(frame.shape[0] * fraction)))
        if keep >= frame.shape[0]:
            return frame
        positions = np.sort(
            np.random.default_rng(seed).choice(frame.shape[0], size=keep, replace=False)
        )
        # Sorted positions preserve the original row order, so a sampled run
        # reads like a thinned version of the data rather than a shuffled one.
        return frame.iloc[positions]

    def _prepare_frame(
        self,
        value: Any,
        *,
        name: str,
        require_date: bool = True,
    ) -> pd.DataFrame:
        frame = _as_pandas_frame(value, name=name)
        if self.OHE_features is not None:
            frame = merge_one_hot_encoded_columns(frame, self.OHE_features)

        required = self._retained_columns()
        if require_date and self.date_col is not None and self.date_col not in required:
            required.append(self.date_col)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"{name} is missing required column(s): {missing}")
        return frame

    def _prepare_snapshot_value(self, value: Any, *, name: str) -> pd.DataFrame:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"{name} contains an empty list of frames")
            frames = [
                self._prepare_frame(item, name=f"{name}[{index}]")
                for index, item in enumerate(value)
            ]
            frame = pd.concat(frames, ignore_index=True)
        else:
            frame = self._prepare_frame(value, name=name)
        return frame.loc[:, self._retained_columns()].copy(deep=False)

    def _preprocess(self):
        """Create the snapshot dictionary while avoiding full-frame copies."""
        if isinstance(self.data, Mapping):
            if not self.data:
                raise ValueError("data snapshot mapping must not be empty")
            snapshots = {
                label: self._prepare_snapshot_value(value, name=f"data[{label!r}]")
                for label, value in self.data.items()
            }
        else:
            frame = self._prepare_frame(self.data, name="data")
            self.data = frame

            if self.interval == "All":
                # One snapshot holding the current/monitored rows only -- never
                # the baseline, which stays separate as the comparison point.
                snapshots = {
                    CURRENT_SNAPSHOT_LABEL: frame.loc[:, self._retained_columns()].copy(
                        deep=False
                    )
                }
            else:
                # The date column may be text, a number, a quarter label or a
                # timestamp; drift.timeperiods sorts that out and returns one
                # sortable label per row ("2019-03" for the monthly default).
                labels = snapshot_labels(
                    frame[self.date_col],
                    self.interval,
                    name=f"{self.date_col!r}",
                )
                if labels.isna().all():
                    # Say exactly what went wrong: "no snapshots" on its own
                    # sends people looking for a problem in their features.
                    raise ValueError(
                        f"The date column {self.date_col!r} could not be read as dates: "
                        f"none of its {len(frame):,} values produced a usable date, so no "
                        "time snapshots could be built. Supported formats include "
                        "'2019-03-15', '2019-03', 'Mar 2019', '15/03/2019', 201903, "
                        "20190315, '2019Q1', timestamps and pandas periods "
                        f"(the column currently holds values such as "
                        f"{_sample_values(frame[self.date_col])}). Pick a different date "
                        "column, fix the format, or compare the current data as one "
                        "population instead."
                    )
                # Only a handful of distinct labels exist, so a categorical
                # keeps this temporary column small on a long dataset.
                labels = labels.astype("category")
                # GroupBy.indices supplies positional arrays even when the input
                # index contains duplicates, avoiding a temporary label column.
                groups = frame.groupby(labels, sort=True, observed=True).indices
                retained = self._retained_columns()
                snapshots = {
                    str(label): frame.iloc[positions].loc[:, retained]
                    for label, positions in groups.items()
                }

        if not snapshots:
            raise ValueError(
                "No snapshots could be built from the current data: it appears to be empty."
            )
        empty_labels = [label for label, frame in snapshots.items() if frame.empty]
        if empty_labels:
            raise ValueError(f"empty snapshot(s) are not supported: {empty_labels}")

        if self.reference is None:
            # The documented fallback is the first snapshot.  It remains in the
            # trend so users can see the expected zero-distance baseline.
            first_label = next(iter(snapshots))
            reference = snapshots[first_label].copy(deep=False)
        else:
            prepared_reference = self._prepare_frame(
                self.reference,
                name="reference",
                require_date=False,
            )
            reference = prepared_reference.loc[:, self._retained_columns()].copy(
                deep=False
            )

        if reference.empty:
            raise ValueError("reference must contain at least one row")

        # Subsample last, once the populations exist. Each is drawn separately so
        # every snapshot keeps the same share of its own rows -- sampling the
        # combined data would let a large month crowd out a small one and change
        # the comparison itself. The draw is seeded, so a rerun of the same
        # settings measures the same rows.
        self.rows_before_sampling_ = {
            "reference": int(reference.shape[0]),
            **{label: int(frame.shape[0]) for label, frame in snapshots.items()},
        }
        if self.sample_percent < 100.0:
            fraction = self.sample_percent / 100.0
            seeds = np.random.default_rng(self.random_state).integers(
                0, 2**31 - 1, size=len(snapshots) + 1
            )
            reference = self._subsample(reference, fraction, int(seeds[0]))
            snapshots = {
                label: self._subsample(frame, fraction, int(seed))
                for (label, frame), seed in zip(snapshots.items(), seeds[1:])
            }
            empty_after = [label for label, frame in snapshots.items() if frame.empty]
            if reference.empty or empty_after:
                raise ValueError(
                    f"sample_percent={self.sample_percent:g} leaves an empty population "
                    f"({'reference' if reference.empty else empty_after}). Raise the "
                    "percentage, or compare over a longer interval."
                )

        self.reference = reference
        self.data_dict = snapshots
        self.snapshots_labels = list(snapshots)

        # The baseline determines feature semantics.  Optimised pandas category,
        # StringDtype and boolean columns are all correctly treated as discrete.
        self.num_cols = [
            feature
            for feature in self.col_names
            if _is_numeric_series(self.reference[feature])
        ]
        self.cat_cols = [
            feature for feature in self.col_names if feature not in self.num_cols
        ]

        # Preserve the legacy category-code metadata without label-encoding and
        # copying every categorical column.  These codes are descriptive only.
        codes: dict[Any, dict[int, Any]] = {}
        for feature in self.cat_cols:
            series = self.reference[feature]
            if isinstance(series.dtype, pd.CategoricalDtype):
                categories = series.cat.categories.tolist()
            else:
                try:
                    categories = pd.unique(series.loc[series.notna()]).tolist()
                except TypeError:
                    categories = [str(value) for value in series.loc[series.notna()]]
            codes[feature] = dict(enumerate(categories))
        self.cats_codes = codes

    # ------------------------------------------------------------ calculations
    @staticmethod
    def _warn_skipped(method: str, feature: Any, reason: str, warned: set[tuple]) -> None:
        key = (method, feature, reason)
        if key in warned:
            return
        warned.add(key)
        warnings.warn(
            f"{method.upper()} was not calculated for {feature!r}: {reason}.",
            RuntimeWarning,
            stacklevel=3,
        )

    def calc_distances(
        self,
        methods=("CM", "KS2", "PSI", "WS"),
        psi_n_buckets=10,
        psi_bucket_type="bins",
        verbose=False,
        plot_trend=True,
        n_jobs=None,
        **plot_kwargs,
    ):
        """Calculate requested drift distances for every feature and snapshot.

        Returns
        -------
        pandas.DataFrame
            The final requested method, with features as rows and snapshots as
            columns (the historical return contract).

        Other results
        -------------
        ``distance_results_`` contains every requested method under canonical
        lowercase keys: ``ks2``, ``ws``, ``cm``, ``psi`` and ``chisquare``.
        ``ks_p_values_`` stores KS p-values in the same orientation.

        ``n_jobs`` sets how many features are assessed at once. ``None`` (the
        default) picks one thread per core, bounded; ``1`` forces the old
        single-threaded path. The numbers are identical either way -- results are
        merged in feature order, never in completion order.
        """
        if isinstance(methods, (str, bytes)) or not isinstance(methods, (list, tuple)):
            raise TypeError("methods must be a list or tuple of drift method names")
        if not methods:
            raise ValueError("methods must contain at least one drift method")

        requested: list[str] = []
        for method in methods:
            key = str(method).strip().lower()
            canonical = _METHOD_ALIASES.get(key)
            if canonical is None:
                valid = "KS2, 2S-KS, Wasserstein/WS, CM, PSI, scores_PSI, Chisquare"
                raise ValueError(f"{method!r} is not a valid drift method. Available: {valid}")
            requested.append(canonical)

        try:
            psi_n_buckets = int(psi_n_buckets)
        except (TypeError, ValueError) as exc:
            raise TypeError("psi_n_buckets must be an integer") from exc
        if psi_n_buckets < 1:
            raise ValueError("psi_n_buckets must be at least 1")
        bucket_type = _normalise_bucket_type(psi_bucket_type)

        unique_methods = list(dict.fromkeys(requested))
        n_snapshots = len(self.snapshots_labels)
        n_features = len(self.col_names)
        metric_values = {
            method: np.full((n_snapshots, n_features), np.nan, dtype=np.float64)
            for method in unique_methods
        }
        ks_p_values = (
            np.full((n_snapshots, n_features), np.nan, dtype=np.float64)
            if "ks2" in unique_methods
            else None
        )
        warned: set[tuple] = set()
        notes: list[str] = []
        started = time.perf_counter()

        if verbose:
            print("Calculating drift metrics:", ", ".join(unique_methods))

        # One feature at a time, and features in parallel: NumPy and SciPy
        # release the GIL for the sorting and binning that dominate here, so
        # threads use the cores without the memory cost of separate processes.
        # Each worker writes only its own column, and everything that touches
        # Matplotlib or the filesystem happens back on this thread.
        workers = _worker_count(n_jobs, n_features)
        collected: list[_FeatureOutcome] = []

        def compute(index: int) -> _FeatureOutcome:
            return self._assess_feature(
                index,
                self.col_names[index],
                unique_methods=unique_methods,
                psi_buckets=psi_n_buckets,
                bucket_type=bucket_type,
                keep_psi_tables=self.save_results,
            )

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(compute, index): index for index in range(n_features)}
                for future in as_completed(futures):
                    collected.append(future.result())
        else:
            collected = [compute(index) for index in range(n_features)]

        # Merge in a fixed order, so the result never depends on which thread
        # finished first.
        for outcome in sorted(collected, key=lambda item: item.index):
            for method, column in outcome.values.items():
                metric_values[method][:, outcome.index] = column
            if ks_p_values is not None and outcome.ks_p_values is not None:
                ks_p_values[:, outcome.index] = outcome.ks_p_values
            for note in outcome.notes:
                if note not in notes:
                    notes.append(note)
            for method, feature, reason in outcome.skipped:
                self._warn_skipped(method, feature, reason, warned)
            for snapshot_label, feature, psi_table in outcome.psi_tables:
                # Short names on purpose: these live several folders deep, and
                # Windows still rejects a path over 260 characters.
                stem = "_".join(
                    [
                        "psi_bins",
                        _safe_component(snapshot_label, "snapshot"),
                        _safe_component(feature, "feature"),
                    ]
                )
                target = storage.join_address(
                    self._save_path, DISTANCE_FOLDER, PSI_BIN_FOLDER, stem
                )
                storage.write_frame_csv(psi_table, f"{target}.csv")
                _save_dataframe_image(
                    psi_table,
                    f"{target}.png",
                    title=f"PSI: {feature} / {snapshot_label}",
                    show_index=True,
                )

        # Build compact result frames only after all numeric work is complete.
        self.distance_results_ = {}
        self.distance_extended_results_ = {}
        self.distance_figures_ = {}
        self.distance_notes_ = notes
        for method in unique_methods:
            snapshot_by_feature = pd.DataFrame(
                metric_values[method],
                columns=self.col_names,
                index=self.snapshots_labels,
            ).round(4)
            feature_by_snapshot = snapshot_by_feature.T
            self.distance_results_[method] = feature_by_snapshot

            descriptions = snapshot_by_feature.describe().T
            summary_columns = [
                column
                for column in ("mean", "std", "min", "max")
                if column in descriptions.columns
            ]
            extended = feature_by_snapshot.join(descriptions.loc[:, summary_columns])
            self.distance_extended_results_[method] = extended

            if plot_trend and not snapshot_by_feature.empty:
                saving_path = (
                    storage.join_address(
                        self._save_path, DISTANCE_FOLDER, CHART_FOLDER, method
                    )
                    if self.save_results
                    else None
                )
                figure, axes = self._plot_trends(
                    snapshot_by_feature,
                    saving_path=saving_path,
                    y_label=method,
                    **plot_kwargs,
                )
                self.distance_figures_[method] = figure

        if ks_p_values is not None:
            self.ks_p_values_ = pd.DataFrame(
                ks_p_values,
                columns=self.col_names,
                index=self.snapshots_labels,
            ).T.round(4)

            details = pd.DataFrame(index=self.snapshots_labels, columns=self.col_names, dtype=object)
            ks_statistics = metric_values["ks2"]
            for row in range(n_snapshots):
                for column in range(n_features):
                    statistic = ks_statistics[row, column]
                    p_value = ks_p_values[row, column]
                    details.iat[row, column] = (
                        (f"stat: {statistic:.4f}", f"P: {p_value:.4f}")
                        if np.isfinite(statistic) and np.isfinite(p_value)
                        else np.nan
                    )
            baseline_n = int(self.reference.shape[0])
            critical_regions = []
            for label in self.snapshots_labels:
                snapshot_n = int(self.data_dict[label].shape[0])
                if baseline_n and snapshot_n:
                    critical_regions.append(
                        1.731
                        * np.sqrt((baseline_n + snapshot_n) / (baseline_n * snapshot_n))
                    )
                else:
                    critical_regions.append(np.nan)
            details["critical region"] = np.round(critical_regions, 4)
            self.ks_details_ = details
        else:
            self.ks_p_values_ = pd.DataFrame()
            self.ks_details_ = pd.DataFrame()

        if self.save_results:
            self._save_distance_outputs(
                unique_methods,
                psi_n_buckets=psi_n_buckets,
                bucket_type=bucket_type,
            )

        if verbose:
            for note in self.distance_notes_:
                print(f"Note: {note}")
            print(f"Elapsed time: {time.perf_counter() - started:.3f} seconds")

        return self.distance_results_[requested[-1]]


    def _assess_feature(
        self,
        feature_index: int,
        feature: Any,
        *,
        unique_methods: Sequence[str],
        psi_buckets: int,
        bucket_type: str,
        keep_psi_tables: bool,
    ) -> "_FeatureOutcome":
        """Every requested metric for one feature, across every snapshot.

        Pure computation: it reads the baseline and the snapshots, and returns
        its results. Nothing shared is written, which is what makes it safe to
        run several of these at once.
        """
        n_snapshots = len(self.snapshots_labels)
        values = {
            method: np.full(n_snapshots, np.nan, dtype=np.float64)
            for method in unique_methods
        }
        ks_column = (
            np.full(n_snapshots, np.nan, dtype=np.float64)
            if "ks2" in unique_methods
            else None
        )
        notes: list[str] = []
        skipped: list[tuple[str, Any, str]] = []
        psi_tables: list[tuple[Any, Any, pd.DataFrame]] = []

        is_numeric = feature in self.num_cols
        expected = (
            _numeric_values(self.reference[feature])
            if is_numeric
            else _categorical_values(self.reference[feature])
        )
        psi_reference = (
            self._prepare_psi_reference(
                expected,
                numeric=is_numeric,
                psi_buckets=psi_buckets,
                bucket_type=bucket_type,
            )
            if "psi" in unique_methods and expected.size
            else None
        )
        chi_expected_counts = (
            pd.Series(expected, copy=False).value_counts(dropna=False, sort=False)
            if "chisquare" in unique_methods and not is_numeric and expected.size
            else None
        )

        for snapshot_index, snapshot_label in enumerate(self.snapshots_labels):
            snapshot = self.data_dict[snapshot_label]
            actual = (
                _numeric_values(snapshot[feature])
                if is_numeric
                else _categorical_values(snapshot[feature])
            )

            if expected.size == 0 or actual.size == 0:
                for method in unique_methods:
                    skipped.append(
                        (
                            method,
                            feature,
                            "the baseline or snapshot contains no usable values",
                        )
                    )
                continue

            for method in unique_methods:
                if method in _NUMERIC_METHODS and not is_numeric:
                    note = (
                        f"{method.upper()} is numeric-only; categorical feature "
                        f"{feature!r} was left as NaN."
                    )
                    if note not in notes:
                        notes.append(note)
                    continue
                if method == "chisquare" and is_numeric:
                    note = (
                        f"CHISQUARE is categorical-only; numeric feature "
                        f"{feature!r} was left as NaN."
                    )
                    if note not in notes:
                        notes.append(note)
                    continue

                try:
                    if method == "ks2":
                        statistic, p_value = ks_2samp(expected, actual)
                        values[method][snapshot_index] = statistic
                        assert ks_column is not None
                        ks_column[snapshot_index] = p_value
                    elif method == "ws":
                        values[method][snapshot_index] = wasserstein_distance(expected, actual)
                    elif method == "cm":
                        # Retain historical semantics: "CM" uses SciPy's energy
                        # distance, as the original implementation did.
                        values[method][snapshot_index] = energy_distance(expected, actual)
                    elif method == "psi":
                        assert psi_reference is not None
                        psi_value, psi_table = self._calculate_psi_from_reference(
                            psi_reference,
                            actual,
                        )
                        values[method][snapshot_index] = psi_value
                        if keep_psi_tables:
                            psi_tables.append((snapshot_label, feature, psi_table))
                    elif method == "chisquare":
                        assert chi_expected_counts is not None
                        actual_counts = pd.Series(actual, copy=False).value_counts(
                            dropna=False, sort=False
                        )
                        categories = chi_expected_counts.index.union(
                            actual_counts.index, sort=False
                        )
                        contingency = np.vstack(
                            [
                                chi_expected_counts.reindex(
                                    categories, fill_value=0
                                ).to_numpy(dtype=np.float64),
                                actual_counts.reindex(categories, fill_value=0).to_numpy(
                                    dtype=np.float64
                                ),
                            ]
                        )
                        # Columns with zero total carry no information and make
                        # scipy's expected-frequency matrix invalid.
                        contingency = contingency[:, contingency.sum(axis=0) > 0]
                        values[method][snapshot_index] = (
                            0.0
                            if contingency.shape[1] <= 1
                            else float(chi2_contingency(contingency, correction=False)[0])
                        )
                except (TypeError, ValueError, FloatingPointError, ZeroDivisionError) as exc:
                    skipped.append((method, feature, str(exc)))

        return _FeatureOutcome(
            index=feature_index,
            values=values,
            ks_p_values=ks_column,
            notes=notes,
            skipped=skipped,
            psi_tables=psi_tables,
        )

    def _save_distance_outputs(
        self,
        methods: Sequence[str],
        *,
        psi_n_buckets: int,
        bucket_type: str,
    ) -> None:
        """Persist aggregate results, one single-shot write per file."""
        # The workbook is assembled in memory and written once, so a firewalled
        # ADLS destination never sees an incremental upload.
        sheets = {
            (
                f"PSI_{psi_n_buckets}_{bucket_type}" if method == "psi" else method.upper()
            ): self.distance_extended_results_[method]
            for method in methods
        }
        storage.write_excel(
            sheets,
            storage.join_address(
                self._save_path, DISTANCE_FOLDER, "distance_measures.xlsx"
            ),
        )

        for method in methods:
            frame = self.distance_results_[method]
            # "psi_10bins" beats "PSI_10buckets_bins": the settings are still
            # visible, in a name short enough to survive a deep folder.
            suffix = (
                f"_{psi_n_buckets}{'bins' if bucket_type == 'bins' else 'qcut'}"
                if method == "psi"
                else ""
            )
            target = storage.join_address(
                self._save_path,
                DISTANCE_FOLDER,
                f"{DISTANCE_FILE_NAMES.get(method, method)}{suffix}",
            )
            storage.write_frame_csv(frame, f"{target}.csv")
            _save_dataframe_image(
                frame,
                f"{target}.png",
                title=f"{method.upper()} drift",
                show_index=True,
                # Only PSI has established bands; shading a Wasserstein
                # distance against 0.10 would be meaningless.
                color_thresholds=PSI_THRESHOLDS if method == "psi" else None,
            )

        if not self.ks_details_.empty:
            target = storage.join_address(self._save_path, DISTANCE_FOLDER, "ks2_detail")
            storage.write_frame_csv(self.ks_details_, f"{target}.csv")
            _save_dataframe_image(
                self.ks_details_,
                f"{target}.png",
                title="KS statistics and p-values",
                show_index=True,
            )
        if not self.ks_p_values_.empty:
            target = storage.join_address(self._save_path, DISTANCE_FOLDER, "ks2_p")
            storage.write_frame_csv(self.ks_p_values_, f"{target}.csv")
            _save_dataframe_image(
                self.ks_p_values_,
                f"{target}.png",
                title="KS p-values",
                show_index=True,
                # A small p-value is the notable one, so the bands invert.
                color_thresholds=(0.10, 0.05),
                higher_is_worse=False,
            )

    @staticmethod
    def _prepare_psi_reference(
        expected,
        *,
        numeric: bool,
        psi_buckets: int,
        bucket_type: str,
    ) -> dict[str, Any]:
        """Pre-aggregate the baseline PSI distribution once per feature."""
        if numeric:
            expected_values = _numeric_values(expected)
            if expected_values.size == 0:
                return {"numeric": True, "empty": True}
            minimum = float(expected_values.min())
            maximum = float(expected_values.max())
            constant_baseline = minimum == maximum
            if constant_baseline:
                # Three intervals let drift away from a constant baseline be
                # detected on either side of the original value.
                lower = np.nextafter(minimum, -np.inf)
                upper = np.nextafter(maximum, np.inf)
                breakpoints = np.asarray([-np.inf, lower, upper, np.inf])
            elif bucket_type == "bins":
                breakpoints = np.linspace(minimum, maximum, num=psi_buckets + 1)
                breakpoints[0] = -np.inf
                breakpoints[-1] = np.inf
            else:
                quantiles = np.linspace(0.0, 1.0, num=psi_buckets + 1)
                candidates = np.quantile(expected_values, quantiles)
                interior = np.unique(candidates[1:-1])
                breakpoints = np.concatenate(([-np.inf], interior, [np.inf]))

            expected_counts = np.histogram(expected_values, bins=breakpoints)[0]
            expected_percents = expected_counts.astype(np.float64) / expected_values.size
            if constant_baseline:
                intervals = [
                    f"(-inf, {minimum:g})",
                    f"[{minimum:g}]",
                    f"({minimum:g}, inf)",
                ]
            else:
                intervals = [
                    f"({left:.2f}, {right:.2f}]"
                    for left, right in zip(breakpoints[:-1], breakpoints[1:])
                ]
            return {
                "numeric": True,
                "empty": False,
                "breakpoints": breakpoints,
                "intervals": intervals,
                "expected_counts": expected_counts,
                "expected_probabilities": expected_percents,
            }

        expected_values = _categorical_values(expected)
        if expected_values.size == 0:
            return {"numeric": False, "empty": True}
        expected_counts = pd.Series(expected_values, copy=False).value_counts(
            dropna=False, sort=False
        )
        return {
            "numeric": False,
            "empty": False,
            "expected_counts": expected_counts,
            "expected_size": int(expected_values.size),
        }

    @staticmethod
    def _calculate_psi_from_reference(
        reference: Mapping[str, Any],
        actual,
    ) -> tuple[float, pd.DataFrame]:
        """Compare one sample with a pre-aggregated PSI baseline."""
        epsilon = 1e-4
        if reference.get("numeric"):
            columns = [
                "Bin_edges",
                "Baseline count",
                "Actual Count",
                "Baseline%",
                "Actual%",
                "PSI",
            ]
            actual_values = _numeric_values(actual)
            if reference.get("empty") or actual_values.size == 0:
                return np.nan, pd.DataFrame(columns=columns)

            breakpoints = reference["breakpoints"]
            actual_counts = np.histogram(actual_values, bins=breakpoints)[0]
            expected_counts = reference["expected_counts"]
            expected_probabilities = reference["expected_probabilities"]
            actual_probabilities = (
                actual_counts.astype(np.float64) / actual_values.size
            )
            stable_expected = np.clip(expected_probabilities, epsilon, None)
            stable_actual = np.clip(actual_probabilities, epsilon, None)
            with np.errstate(divide="ignore", invalid="ignore"):
                psi_values = (stable_expected - stable_actual) * np.log(
                    stable_expected / stable_actual
                )

            table = pd.DataFrame(
                {
                    "Bin_edges": reference["intervals"],
                    "Baseline count": expected_counts,
                    "Actual Count": actual_counts,
                    "Baseline%": expected_probabilities,
                    "Actual%": actual_probabilities,
                    "PSI": np.round(psi_values, 4),
                }
            )
            return float(np.sum(psi_values)), table

        actual_values = _categorical_values(actual)
        if reference.get("empty") or actual_values.size == 0:
            return np.nan, pd.DataFrame(columns=["Series_1", "Series_2", "PSI"])

        expected_counts = reference["expected_counts"]
        actual_counts = pd.Series(actual_values, copy=False).value_counts(
            dropna=False, sort=False
        )
        categories = expected_counts.index.union(actual_counts.index, sort=False)
        expected_probabilities = (
            expected_counts.reindex(categories, fill_value=0).astype(np.float64)
            / reference["expected_size"]
        )
        actual_probabilities = (
            actual_counts.reindex(categories, fill_value=0).astype(np.float64)
            / actual_values.size
        )
        stable_expected = expected_probabilities.clip(lower=epsilon)
        stable_actual = actual_probabilities.clip(lower=epsilon)
        psi_values = (stable_expected - stable_actual) * np.log(
            stable_expected / stable_actual
        )
        table = pd.DataFrame(
            {
                "Series_1": expected_probabilities,
                "Series_2": actual_probabilities,
                "PSI": psi_values.round(4),
            }
        )
        return float(psi_values.sum()), table

    @staticmethod
    def _calculate_psi_two_series(
        expected,
        actual,
        bucket_type="bins",
        psi_buckets=20,
    ):
        """Calculate PSI for one numeric or categorical feature.

        Category levels are aligned over their union, and zero-probability bins
        are stabilised with a small epsilon.  Numeric boundaries are derived
        once from the baseline and reused for the actual sample.
        """
        try:
            psi_buckets = int(psi_buckets)
        except (TypeError, ValueError) as exc:
            raise TypeError("psi_buckets must be an integer") from exc
        if psi_buckets < 1:
            raise ValueError("psi_buckets must be at least 1")
        bucket_type = _normalise_bucket_type(bucket_type)

        expected_series = (
            expected
            if isinstance(expected, pd.Series)
            else pd.Series(np.ravel(expected), copy=False)
        )
        prepared = DriftAssessment._prepare_psi_reference(
            expected_series,
            numeric=_is_numeric_series(expected_series),
            psi_buckets=psi_buckets,
            bucket_type=bucket_type,
        )
        return DriftAssessment._calculate_psi_from_reference(prepared, actual)

    # ---------------------------------------------------------------- plotting
    @staticmethod
    def _plot_trends(df, saving_path, y_label="distance", show=True, **plot_kwargs):
        """Plot feature trends, creating per-feature figures only when saving.

        ``show`` controls the legacy notebook behaviour of displaying the combined
        figure. A caller that only wants the per-feature PNGs passes ``False``.
        """
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if df.shape[1] == 0:
            return None, np.asarray([], dtype=object)

        destination = storage.ensure_dir(saving_path) if saving_path is not None else None

        def style_axis(ax, feature, *, legend=False):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, color="white")
            ax.xaxis.grid(False)
            ax.set_facecolor("#EFF0F0")
            ax.tick_params(labelsize=7)
            ax.set_ylabel(y_label)
            if not legend and ax.get_legend() is not None:
                ax.get_legend().remove()

            finite = pd.to_numeric(df[feature], errors="coerce")
            finite = finite[np.isfinite(finite)]
            if not finite.empty:
                minimum = float(finite.min())
                maximum = float(finite.max())
                padding = max((maximum - minimum) * 0.15, abs(float(finite.mean())) * 0.05, 1e-9)
                ax.set_ylim(min(0.0, minimum - padding), maximum + padding)

        # The old implementation created and discarded one figure per feature
        # even when nothing was saved.  Avoid that work on dashboard reruns.
        if destination is not None:
            for feature in df.columns:
                figure, axis = plt.subplots(figsize=(6, 3), dpi=72)
                df[[feature]].plot(
                    ax=axis, alpha=0.85, rot=45, marker=".", legend=False
                )
                style_axis(axis, feature)
                axis.set_xlabel(str(feature))
                storage.write_figure(
                    figure,
                    storage.join_address(
                        destination, f"{_safe_component(feature, 'feature')}.png"
                    ),
                    dpi=110,
                )
                plt.close(figure)

        try:
            n_cols = int(plot_kwargs.get("n_cols", 5))
        except (TypeError, ValueError) as exc:
            raise TypeError("n_cols must be an integer") from exc
        if n_cols < 1:
            raise ValueError("n_cols must be at least 1")
        n_cols = min(n_cols, df.shape[1])
        n_rows = int(np.ceil(df.shape[1] / n_cols))
        figure, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(6 * n_cols, n_rows * 3.2),
            dpi=96,
            squeeze=False,
        )
        flat_axes = axes.ravel(order="C")

        for index, feature in enumerate(df.columns):
            axis = flat_axes[index]
            df[[feature]].plot(ax=axis, alpha=0.85, rot=45, marker=".")
            style_axis(axis, feature, legend=True)
            axis.set_xticks(range(len(df.index)))
            axis.set_xticklabels([str(value) for value in df.index])

        for axis in flat_axes[df.shape[1] :]:
            axis.remove()
        figure.tight_layout()

        if destination is not None:
            storage.write_figure(
                figure,
                storage.join_address(destination, "All_features_plot.png"),
                dpi=120,
            )
        # Preserve notebook behaviour: the combined trend appears as cell output.
        # Under a file-only backend (a script, a test, Streamlit) ``show`` does
        # nothing except warn, so it is skipped there.
        if show and plt.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS:
            plt.show()
        return figure, flat_axes[: df.shape[1]]

    # -------------------------------------------------------------- summaries
    def calc_descriptives_overtime(self, measures=("mean", "std"), **plot_kwargs):
        """Calculate mean/mode and standard deviation over snapshots.

        The historical print/save behaviour remains.  The method now also
        returns the computed tables and stores them on ``descriptive_results_``.
        Pass ``plot_trend=False`` to suppress Matplotlib work in applications.
        """
        if isinstance(measures, (str, bytes)):
            measures = (measures,)
        if not isinstance(measures, (list, tuple)):
            raise TypeError("measures must be a list or tuple")
        requested = [str(measure).strip().lower() for measure in measures]
        invalid = [measure for measure in requested if measure not in {"mean", "std"}]
        if invalid:
            raise ValueError(f"Unsupported descriptive measure(s): {invalid}")

        plot_trend = bool(plot_kwargs.pop("plot_trend", True))
        baseline_label = (
            "Baseline" if "Baseline" not in self.data_dict else "Baseline (reference)"
        )
        frames: dict[Any, pd.DataFrame] = {baseline_label: self.reference}
        frames.update(self.data_dict)
        results: dict[str, pd.DataFrame] = {}
        figures: dict[str, Any] = {}

        if "mean" in requested:
            print("Calculating average/mode of features over time")
            mean_columns: dict[Any, list[Any]] = {}
            for label, frame in frames.items():
                values: list[Any] = []
                for feature in self.col_names:
                    series = frame[feature]
                    if feature in self.cat_cols:
                        modes = series.mode(dropna=True)
                        values.append(modes.iloc[0] if not modes.empty else np.nan)
                    else:
                        # Four decimals: enough to see a real move in a mean,
                        # few enough to keep the table readable.
                        average = series.mean()
                        values.append(
                            round(float(average), _MEAN_DECIMALS)
                            if pd.notna(average)
                            else np.nan
                        )
                mean_columns[label] = values

            feature_means = pd.DataFrame(mean_columns, index=self.col_names)
            # The numeric-only view of the same numbers. It exists because the
            # trend charts need a purely numeric frame; it is not saved
            # separately, since every value is already in ``mean``.
            numeric_means = feature_means.loc[self.num_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            results["mean"] = feature_means
            results["numeric_mean"] = numeric_means

            if self.save_results:
                target = storage.join_address(self._save_path, DESCRIPTIVE_FOLDER, "mean")
                storage.write_frame_csv(feature_means, f"{target}.csv")
                _save_dataframe_image(
                    feature_means,
                    f"{target}.png",
                    title="Feature averages and modes over time",
                )

            if plot_trend and not numeric_means.empty and numeric_means.shape[1] > 1:
                save_path = (
                    storage.join_address(
                        self._save_path, DESCRIPTIVE_FOLDER, CHART_FOLDER
                    )
                    if self.save_results
                    else None
                )
                figure, _ = self._plot_trends(
                    numeric_means.T,
                    saving_path=save_path,
                    y_label="Avg. value",
                    **plot_kwargs,
                )
                figures["numeric_mean"] = figure

        if "std" in requested:
            print("Calculating standard deviation of numeric features over time")
            feature_std = pd.DataFrame(
                {
                    label: frame.loc[:, self.num_cols].std().round(_MEAN_DECIMALS)
                    for label, frame in frames.items()
                },
                index=self.num_cols,
            )
            results["std"] = feature_std
            if self.save_results:
                target = storage.join_address(self._save_path, DESCRIPTIVE_FOLDER, "std")
                storage.write_frame_csv(feature_std, f"{target}.csv")
                _save_dataframe_image(
                    feature_std,
                    f"{target}.png",
                    title="Feature standard deviations over time",
                )

        self.descriptive_results_ = results
        self.descriptive_figures_ = figures
        return results

    def missing_analysis(
        self,
        save_file_name="missing",
        plot_trend=None,
        **plot_kwargs,
    ):
        """Return missing percentages and counts for baseline plus snapshots.

        ``plot_trend`` defaults to "only when saving": with ``save_results=True``
        one PNG per feature is written next to the tables, so a rising gap in a
        single column can be looked at on its own.
        """
        reference_label = (
            "Reference" if "Reference" not in self.data_dict else "Reference (baseline)"
        )
        frames: dict[Any, pd.DataFrame] = {reference_label: self.reference}
        frames.update(self.data_dict)

        count_columns: dict[Any, pd.Series] = {}
        percentage_columns: dict[Any, pd.Series] = {}
        for label, frame in frames.items():
            counts = frame.loc[:, self.col_names].isna().sum(axis=0).astype(np.int64)
            count_columns[label] = counts
            denominator = int(frame.shape[0])
            if denominator:
                percentage_columns[label] = (counts / denominator * 100.0).round(3)
            else:
                percentage_columns[label] = pd.Series(
                    np.nan, index=counts.index, dtype=np.float64
                )

        missing_counts = pd.DataFrame(count_columns, index=self.col_names)
        missing_percentages = pd.DataFrame(
            percentage_columns, index=self.col_names
        )
        if reference_label in missing_counts:
            order = missing_counts[reference_label].sort_values(ascending=False).index
            missing_counts = missing_counts.loc[order]
            missing_percentages = missing_percentages.loc[order]

        self.missing_counts_ = missing_counts
        self.missing_percentages_ = missing_percentages

        if self.save_results:
            stem = _safe_component(save_file_name, "missing")
            for table, kind, title in (
                (missing_counts, "count", "Missing-value counts"),
                (missing_percentages, "perc", "Missing-value percentages"),
            ):
                target = storage.join_address(
                    self._save_path, MISSING_FOLDER, f"{stem}_{kind}"
                )
                storage.write_frame_csv(table, f"{target}.csv")
                _save_dataframe_image(table, f"{target}.png", title=title)

        if plot_trend is None:
            plot_trend = self.save_results
        if plot_trend and missing_percentages.shape[1] > 1:
            save_path = (
                storage.join_address(self._save_path, MISSING_FOLDER, CHART_FOLDER)
                if self.save_results
                else None
            )
            # The tables above already carry these numbers, so only the saved
            # per-feature PNGs are wanted here -- no combined figure on screen.
            figure, _ = self._plot_trends(
                missing_percentages.T,
                saving_path=save_path,
                y_label="Missing (%)",
                show=False,
                **plot_kwargs,
            )
            plt.close(figure)

        return missing_percentages, missing_counts

    def score_analysis(
        self,
        n_bins=10,
        bucket_type="bins",
        save_file_name="score",
        plot_trend=None,
        **plot_kwargs,
    ):
        """Compare the model-score distribution in the baseline with each snapshot.

        A feature drifting matters because it may move the model's output; this
        looks at that output directly. The score column is named on the
        assessment (``score_col=...``) and is deliberately not one of
        ``col_names``, so it never appears among the feature distances.

        Returns ``(distribution, summary)``:

        ``distribution``
            One row per score band, one column per population, holding the
            **percentage of that population's rows** falling in the band. Shares
            rather than counts, so a small snapshot is comparable with a large
            baseline. Bands are cut from the baseline alone and reused
            everywhere, which is what makes the columns comparable at all.
        ``summary``
            Row count, missing count, mean, standard deviation and the quartiles
            for each population, plus **PSI against the baseline** — the number
            most model-monitoring processes act on. The baseline's own PSI is 0
            by construction and is shown so the column reads consistently.

        Both are also stored as ``score_distribution_`` and ``score_summary_``.

        For a numeric score, ``score_density_`` additionally holds kernel density
        curves for every population on one shared x grid, ready to be drawn over
        each other in a single chart: two distributions that still match trace the
        same curve, and one that has moved is obvious at a glance.
        """
        if self.score_col is None:
            raise MyException(
                "No score column was given. Pass score_col=... when creating the "
                "assessment to compare model scores."
            )
        try:
            n_bins = int(n_bins)
        except (TypeError, ValueError) as exc:
            raise TypeError("n_bins must be an integer") from exc
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        bucket_type = _normalise_bucket_type(bucket_type)

        baseline_label = (
            "Baseline" if "Baseline" not in self.data_dict else "Baseline (reference)"
        )
        populations: dict[Any, pd.Series] = {baseline_label: self.reference[self.score_col]}
        populations.update(
            {label: frame[self.score_col] for label, frame in self.data_dict.items()}
        )

        baseline_scores = populations[baseline_label]
        numeric = _is_numeric_series(baseline_scores)

        # Bands come from the baseline and are reused for every snapshot. Cutting
        # each population on its own quantiles would compare different bands and
        # report a flat distribution however far the score had moved.
        prepared = self._prepare_psi_reference(
            baseline_scores,
            numeric=numeric,
            psi_buckets=n_bins,
            bucket_type=bucket_type,
        )
        if prepared.get("empty"):
            raise MyException(
                f"The score column {self.score_col!r} has no usable values in the "
                "baseline, so there is no distribution to compare against."
            )

        if numeric:
            # The prepared reference already carries printable interval labels,
            # and its outer edges are -inf/+inf, so every value lands in a band
            # and the shares always total 100.
            band_labels = list(prepared["intervals"])
            edges = np.asarray(prepared["breakpoints"], dtype=np.float64)
        else:
            band_labels = [str(level) for level in prepared["expected_counts"].index]

        shares: dict[Any, pd.Series] = {}
        summaries: dict[Any, pd.Series] = {}
        for label, series in populations.items():
            if numeric:
                values = _numeric_values(series)
                counts = (
                    np.histogram(values, bins=edges)[0].astype(np.float64)
                    if values.size
                    else np.zeros(len(band_labels), dtype=np.float64)
                )
                total = float(counts.sum())
                shares[label] = pd.Series(
                    (counts / total * 100.0) if total else np.full(len(counts), np.nan),
                    index=band_labels,
                    dtype=np.float64,
                )
                described = pd.Series(
                    {
                        "Rows": float(series.shape[0]),
                        "Missing": float(series.isna().sum()),
                        "Mean": float(np.mean(values)) if values.size else np.nan,
                        "Std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                        "Min": float(np.min(values)) if values.size else np.nan,
                        "P25": float(np.percentile(values, 25)) if values.size else np.nan,
                        "Median": float(np.median(values)) if values.size else np.nan,
                        "P75": float(np.percentile(values, 75)) if values.size else np.nan,
                        "Max": float(np.max(values)) if values.size else np.nan,
                    }
                )
            else:
                present = _categorical_values(series)
                observed = pd.Series(present, dtype=object).value_counts()
                observed.index = observed.index.map(str)
                total = float(observed.sum())
                # Categories seen only in a snapshot are appended, so a brand-new
                # score band is visible rather than quietly dropped.
                extra = [str(level) for level in observed.index if level not in band_labels]
                band_labels.extend(extra)
                aligned = observed.reindex(band_labels).fillna(0.0).astype(np.float64)
                shares[label] = (
                    aligned / total * 100.0
                    if total
                    else pd.Series(np.nan, index=band_labels, dtype=np.float64)
                )
                described = pd.Series(
                    {
                        "Rows": float(series.shape[0]),
                        "Missing": float(series.isna().sum()),
                        "Mean": np.nan,
                        "Std": np.nan,
                        "Min": np.nan,
                        "P25": np.nan,
                        "Median": np.nan,
                        "P75": np.nan,
                        "Max": np.nan,
                    }
                )

            if label == baseline_label:
                described["PSI vs baseline"] = 0.0
            else:
                # _calculate_psi_from_reference returns (psi, per-bin table); only
                # the headline number belongs in the summary.
                psi_value, _ = self._calculate_psi_from_reference(prepared, series)
                described["PSI vs baseline"] = float(psi_value)
            summaries[label] = described

        distribution = pd.DataFrame(shares)
        if not numeric:
            # A band a population never produced is a real 0%, not a gap in the
            # data. Only a population with no scores at all stays blank.
            populated = [
                column for column in distribution.columns if distribution[column].notna().any()
            ]
            distribution[populated] = distribution[populated].fillna(0.0)
        distribution = distribution.round(4)
        distribution.index.name = "Score band"
        summary = pd.DataFrame(summaries).round(_MEAN_DECIMALS)
        summary.index.name = "Statistic"

        # Kernel density curves for the overlay chart: one shared x grid so the
        # populations can be drawn on a single set of axes and read against each
        # other. Numeric scores only -- a categorical score has no density, and
        # the band table already shows its shape.
        density = self._score_density(populations) if numeric else pd.DataFrame()

        self.score_distribution_ = distribution
        self.score_summary_ = summary
        self.score_density_ = density

        if self.save_results:
            stem = _safe_component(save_file_name, "score")
            for table, kind, title in (
                (distribution, "distribution", f"{self.score_col}: score distribution (%)"),
                (summary, "summary", f"{self.score_col}: score summary"),
            ):
                target = storage.join_address(self._save_path, SCORE_FOLDER, f"{stem}_{kind}")
                storage.write_frame_csv(table, f"{target}.csv")
                _save_dataframe_image(table, f"{target}.png", title=title)

        if plot_trend is None:
            plot_trend = self.save_results
        if plot_trend and distribution.shape[1] > 1:
            save_path = (
                storage.join_address(self._save_path, SCORE_FOLDER, CHART_FOLDER)
                if self.save_results
                else None
            )
            figure, _ = self._plot_trends(
                distribution,
                saving_path=save_path,
                y_label="Share of rows (%)",
                show=False,
                **plot_kwargs,
            )
            plt.close(figure)

        return distribution, summary

    def complete_drift_analysis(
        self,
        include_distance_measures=True,
        include_descriptives=True,
    ):
        """Run the same distance and descriptive stages as the legacy helper."""
        if include_distance_measures:
            self.calc_distances()
        if include_descriptives:
            self.calc_descriptives_overtime()


def _get_ohe_dict(
    columns: Sequence[Any],
    features: Sequence[Any],
    separator: str,
) -> dict[Any, list[Any]]:
    """Resolve explicit original feature names to their dummy columns."""
    groups: dict[Any, list[Any]] = {}
    for feature in features:
        prefix = f"{feature}{separator}"
        exact = [column for column in columns if str(column).startswith(prefix)]
        # Preserve the original substring matching as a fallback for unusual
        # naming schemes, while preferring the safer prefix match.
        groups[feature] = exact or [
            column for column in columns if str(feature) in str(column)
        ]
    return groups


def merge_one_hot_encoded_columns(df, ohe_features="Auto", dummy_separator="_"):
    """Merge groups of one-hot columns into categorical features.

    The signature and column-position behaviour are retained.  The input frame
    is never modified, missing/all-zero rows remain missing, and values such as
    ``CardType_Elite`` are converted to ``Elite`` so they align with an ordinary
    categorical baseline.
    """
    frame = _as_pandas_frame(df, name="df")
    columns = list(frame.columns)

    if isinstance(ohe_features, Mapping):
        groups = {
            feature: list(dummy_features)
            for feature, dummy_features in ohe_features.items()
        }
    elif isinstance(ohe_features, (list, tuple, set, pd.Index, np.ndarray)):
        groups = _get_ohe_dict(columns, list(ohe_features), dummy_separator)
    elif isinstance(ohe_features, str) and ohe_features.lower() == "auto":
        groups: dict[Any, list[Any]] = {}
        for column in columns:
            text = str(column)
            position = text.rfind(dummy_separator)
            if position > 0:
                groups.setdefault(text[:position], []).append(column)
    else:
        raise TypeError(
            "ohe_features must be 'Auto', a feature list, or a mapping of dummy columns"
        )

    merged = frame
    for merged_feature, requested_dummies in groups.items():
        dummy_features = [
            column for column in requested_dummies if column in merged.columns
        ]
        if len(dummy_features) <= 1:
            continue
        if merged_feature in merged.columns and merged_feature not in dummy_features:
            raise ValueError(
                f"Cannot merge {merged_feature!r}: a column with that name already exists"
            )

        position = int(merged.columns.get_loc(dummy_features[0]))
        numeric_dummies = merged.loc[:, dummy_features].apply(
            pd.to_numeric, errors="coerce"
        )
        has_value = numeric_dummies.notna().any(axis=1)
        has_active_dummy = numeric_dummies.max(axis=1, skipna=True).gt(0)
        # Filling only the temporary OHE block avoids pandas' deprecated
        # all-NA ``idxmax`` behaviour; the validity mask restores those rows.
        winner = numeric_dummies.fillna(-np.inf).idxmax(axis=1).astype("object")
        winner.loc[~(has_value & has_active_dummy)] = pd.NA

        prefix = f"{merged_feature}{dummy_separator}"
        winner = winner.map(
            lambda value, p=prefix: (
                value[len(p) :]
                if isinstance(value, str) and value.startswith(p)
                else value
            ),
            na_action="ignore",
        )

        # ``drop`` creates the independent frame before ``insert`` mutates its
        # metadata, leaving the caller-owned dataframe untouched.
        merged = merged.drop(columns=dummy_features)
        merged.insert(min(position, merged.shape[1]), merged_feature, winner)

    return merged if merged is not frame else frame.copy(deep=False)


def psi_coloring(
    value,
    stable_color="green",
    moderate_color="gold",
    significant_color="orange",
):
    """Return a CSS colour declaration for standard PSI severity bands."""
    if pd.isna(value):
        return "color: gray"
    if value <= 0.1:
        color = stable_color
    elif value < 0.25:
        color = moderate_color
    else:
        color = significant_color
    return f"color: {color}"


def yes_no(answer):
    """Backward-compatible interactive yes/no helper."""
    yes = {"yes", "y", "ye", "yeah"}
    no = {"no", "n"}
    choice = input(answer).strip().lower()
    if choice in yes:
        return None
    if choice in no:
        sys.exit()
    raise ValueError("Please answer yes or no")

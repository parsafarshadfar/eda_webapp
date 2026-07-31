"""Core data-profiling engine.

One package, two front ends: the Streamlit dashboard imports it, and so does a
notebook or a batch script. Nothing here knows about Streamlit, so every
capability is available from either.

The package is split so that every expensive computation happens exactly once
and is then reused by every consumer:

``dataio``      loading, delimiter detection, lossless memory optimisation,
                stacking several frames, collapsing one-hot encodings
``stats``       descriptive statistics, missing-value and correlation analysis
``summaries``   compact, pre-aggregated summaries of distributions/outliers
``grouping``    splitting a dataset into segments by value, interval, equal-width
                bin or quantile
``plots``       interactive (Plotly) figures rendered from the summaries
``static``      static (Matplotlib) figures rendered from the same summaries
``pipeline``    orchestration: one run, or one run per segment
``report``      PDF, ZIP and folder export built from the summaries
``storage``     where results are saved, and single-shot writing to local
                folders, DBFS and ADLS

Because the interactive charts and the exported charts are both derived from
the same pre-aggregated summaries, the web app and the downloadable report can
never disagree with one another.

Getting started
---------------
::

    import profiling

    loaded = profiling.load_table(
        "profiling_examples/Data/HousingData_TrainData.csv"
    )
    settings = profiling.ProfilingSettings(correlation_methods=("spearman",))

    result = profiling.run_profiling(loaded.frame, settings)
    profiling.write_report_dir(result)          # -> results/profiling/...

    # ...or one report per segment of the data:
    segmented = profiling.run_segmented_profiling(
        loaded.frame, settings, by="HouseAge", n_quantiles=4
    )
    segmented.describe_comparison()

Where results are saved
-----------------------
Anything saved with ``save=True`` -- and any export given no address -- lands in
``results/profiling/`` beside this repository. One call redirects all of it,
which is what the example notebook does so its results appear next to it::

    profiling.set_output_dir("profiling_examples/results")
    profiling.output_dir()                      # where things go right now

Any single call can still be pointed elsewhere with ``save=<address>``. Cloud
addresses (``dbfs:/...``, ``abfss://...``) are supported: files are rendered in
memory and written in one operation, never appended to, so a locked-down
Databricks/ADLS destination accepts them.

Importing ``profiling`` pulls in pandas and NumPy only. Matplotlib and Plotly
are loaded the first time a plotting or export name is actually used, so a
script that only wants numbers does not pay for them.
"""

from __future__ import annotations

from typing import Any

__version__ = "2.1.0"

# Submodules that cost nothing beyond pandas/NumPy are imported eagerly; the
# names they export are re-exported here so a notebook needs one import.
from . import dataio, grouping, stats, storage, summaries  # noqa: E402
from .dataio import (  # noqa: E402
    CombinedData,
    LoadedData,
    combine_datasets,
    dtype_summary,
    is_categorical_series,
    is_numeric_series,
    load_table,
    memory_usage_mb,
    merge_one_hot_encoded_columns,
    numeric_columns,
    optimize_memory,
    to_pandas,
)
from .grouping import (  # noqa: E402
    BY_EQUAL_WIDTH,
    BY_INTERVALS,
    BY_QUANTILE,
    BY_VALUE,
    NO_GROUPING,
    Grouping,
    Segment,
    make_grouping,
    segment_frames,
)
from .pipeline import (  # noqa: E402
    ProfilingResult,
    ProfilingSettings,
    SegmentedResult,
    environment_report,
    run_profiling,
    run_segmented_profiling,
)
from .storage import (  # noqa: E402
    default_output_dir,
    join_address,
    output_dir,
    reset_output_dir,
    set_output_dir,
)
from .stats import (  # noqa: E402
    correlation_matrices,
    correlation_matrix,
    describe_data,
    missing_count,
    top_absolute_correlations,
)
from .summaries import (  # noqa: E402
    EQUAL_WIDTH,
    KIND_CATEGORICAL,
    KIND_NUMERIC,
    QUANTILE,
    BoxSummary,
    CategoricalDistribution,
    NumericDistribution,
    box_summaries_to_frame,
    box_summary,
    build_box_summaries,
    build_distributions,
    categorical_distribution,
    distribution_of,
    distributions_to_frame,
    numeric_distribution,
)

#: Names served on first access from the submodules that pull in Matplotlib or
#: Plotly, keeping ``import profiling`` cheap for a numbers-only script.
_LAZY_NAMES: dict[str, str] = {
    "plots": "plots",
    "static": "static",
    "report": "report",
    # profiling.static
    "PALETTE": "static",
    "box_figure": "static",
    "box_grid_figures": "static",
    "dataframe_figure": "static",
    "df_to_img": "static",
    "figure_to_png": "static",
    "heatmap_figure": "static",
    "histogram_figure": "static",
    "histogram_grid_figures": "static",
    "missing_bar_figure": "static",
    "save_figure": "static",
    "save_table_png": "static",
    "table_figures": "static",
    # profiling.report
    "build_pdf": "report",
    "build_zip": "report",
    "bundle_files": "report",
    "safe_dirname": "report",
    "safe_filename": "report",
    "suggested_basename": "report",
    "write_report_dir": "report",
    "write_segmented_report_dir": "report",
}

__all__ = [
    "__version__",
    # submodules
    "dataio",
    "grouping",
    "plots",
    "report",
    "static",
    "stats",
    "storage",
    "summaries",
    # where results are saved
    "default_output_dir",
    "join_address",
    "output_dir",
    "reset_output_dir",
    "set_output_dir",
    # loading and preparation
    "CombinedData",
    "LoadedData",
    "combine_datasets",
    "dtype_summary",
    "is_categorical_series",
    "is_numeric_series",
    "load_table",
    "memory_usage_mb",
    "merge_one_hot_encoded_columns",
    "numeric_columns",
    "optimize_memory",
    "to_pandas",
    # segmentation
    "BY_EQUAL_WIDTH",
    "BY_INTERVALS",
    "BY_QUANTILE",
    "BY_VALUE",
    "NO_GROUPING",
    "Grouping",
    "Segment",
    "make_grouping",
    "segment_frames",
    # statistics
    "correlation_matrices",
    "correlation_matrix",
    "describe_data",
    "missing_count",
    "top_absolute_correlations",
    # summaries
    "EQUAL_WIDTH",
    "KIND_CATEGORICAL",
    "KIND_NUMERIC",
    "QUANTILE",
    "BoxSummary",
    "CategoricalDistribution",
    "NumericDistribution",
    "box_summaries_to_frame",
    "box_summary",
    "build_box_summaries",
    "build_distributions",
    "categorical_distribution",
    "distribution_of",
    "distributions_to_frame",
    "numeric_distribution",
    # orchestration
    "ProfilingResult",
    "ProfilingSettings",
    "SegmentedResult",
    "environment_report",
    "run_profiling",
    "run_segmented_profiling",
    # figures and export (loaded on first use)
    "PALETTE",
    "box_figure",
    "box_grid_figures",
    "build_pdf",
    "build_zip",
    "bundle_files",
    "dataframe_figure",
    "df_to_img",
    "figure_to_png",
    "heatmap_figure",
    "histogram_figure",
    "histogram_grid_figures",
    "missing_bar_figure",
    "safe_dirname",
    "safe_filename",
    "save_figure",
    "save_table_png",
    "suggested_basename",
    "table_figures",
    "write_report_dir",
    "write_segmented_report_dir",
]


def __getattr__(name: str) -> Any:
    """Resolve the Matplotlib/Plotly-backed names on first access (PEP 562)."""
    module_name = _LAZY_NAMES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = module if name == module_name else getattr(module, name)
    globals()[name] = value  # cache, so this runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))

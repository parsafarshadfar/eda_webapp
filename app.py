"""Unified profiling and drift-assessment dashboard.

The Streamlit layer delegates calculations to the independent :mod:`profiling`
and :mod:`drift` packages. Expensive work runs once per content fingerprint and
configuration; the page and its exports consume the same compact cached result
tables, so interaction never silently recomputes an analysis.
"""

from __future__ import annotations

import html
import hashlib
import inspect
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from drift import (
    INTERVALS as DRIFT_INTERVALS,
    DriftAssessment,
    __version__ as drift_engine_version,
    df_to_img as drift_df_to_img,
    merge_one_hot_encoded_columns as merge_drift_ohe,
    storage as drift_storage,
)
from drift.drift_assessment import DISTANCE_FILE_NAMES
from drift.drift_util import ALERT_COLOR as drift_alert_color
from drift.drift_util import PSI_THRESHOLDS
from drift.drift_util import WARN_COLOR as drift_warn_color
from profiling import __version__ as engine_version
from profiling import plots
from profiling import storage as profiling_storage
from profiling.dataio import LoadedData, load_table
from profiling.pipeline import ProfilingResult, ProfilingSettings, run_profiling
from profiling.report import build_pdf, build_zip, suggested_basename, write_report_dir
from profiling.summaries import EQUAL_WIDTH, KIND_CATEGORICAL, QUANTILE

PROJECT_ROOT = Path(__file__).resolve().parent

#: Saved results land in ``results/`` beside this file: ``results/profiling/``
#: for the profiling workspace and ``results/drift/`` for the drift workspace.
#: The two packages resolve those folders themselves, so a notebook that
#: redirects its own output folder cannot affect the app. Set
#: ``EDA_RESULTS_ROOT`` to keep a deployment's results somewhere else.
RESULTS_ROOT = Path(
    os.environ.get("EDA_RESULTS_ROOT")
    or PROJECT_ROOT / profiling_storage.RESULTS_DIRNAME
)
EXAMPLE_DATASETS = {
    "California Housing (train)": str(
        PROJECT_ROOT / "profiling_examples" / "Data" / "HousingData_TrainData.csv"
    ),
    "California Housing (test)": str(
        PROJECT_ROOT / "profiling_examples" / "Data" / "HousingData_TestData.csv"
    ),
    "Breast Cancer (train)": str(
        PROJECT_ROOT / "profiling_examples" / "Data" / "BreastCancer_TrainData.csv"
    ),
}
DRIFT_EXAMPLE_DIR = PROJECT_ROOT / "drift_examples" / "data"
DRIFT_EXAMPLES = {
    "Standard categorical data": (
        str(DRIFT_EXAMPLE_DIR / "data_sample_for_drift_baseline.csv"),
        str(DRIFT_EXAMPLE_DIR / "data_sample_for_drift.csv"),
        (),
    ),
    "One-hot encoded data": (
        str(DRIFT_EXAMPLE_DIR / "data_sample_for_drift_baseline_OHE.csv"),
        str(DRIFT_EXAMPLE_DIR / "data_sample_for_drift_OHE.csv"),
        ("CardType",),
    ),
}

#: First entry of every feature picker: "everything", so the default selection
#: is complete and narrowing it is a deliberate act.
ALL_FEATURES = "All features"
#: First option of the model-score picker: no score column at all, which is the
#: right answer for most datasets.
SCORE_NONE = "None"

#: Beyond this many features the result sections start collapsed — a hundred
#: expanded sections is a page nobody can navigate. Their content is still
#: rendered, so opening one is instant.
COLLAPSE_SECTIONS_ABOVE = 12

#: Ceiling on the per-feature PNGs written for ONE result table, so a very wide
#: dataset cannot turn an export into thousands of images. Counted per table:
#: a shared budget let the distance charts starve the descriptive and
#: missing-value ones.
MAX_EXPORTED_CHARTS = 200

#: Short file stems for the exported drift tables, shared with the engine
#: (``drift.drift_assessment.DISTANCE_FILE_NAMES``). A bundle nests
#: folder/charts/name_feature.png, and Windows still rejects a path over 260
#: characters, so every name in it is kept short.
DRIFT_FILE_NAMES = dict(DISTANCE_FILE_NAMES)
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

DRIFT_METHOD_LABELS = {
    "psi": "Population Stability Index (PSI)",
    "ks2": "Kolmogorov–Smirnov statistic",
    "ws": "Wasserstein distance",
    "cm": "Energy distance (legacy CM)",
    "chisquare": "Chi-square statistic",
}

DESCRIBE_DISPLAY_LABELS = {
    "DataType": "Data type",
    "n_uniques (excl. Nulls)": "Unique values\n(excl. nulls)",
    "n_Missing": "Missing\ncount",
    "Perc. of Missing": "Missing\n(%)",
    "n_Zeros": "Zero\ncount",
    "Perc. of Zeros": "Zeros\n(%)",
    "1st Most Freq": "Most frequent\nvalue",
    "Perc. of 1st Most Freq": "Most frequent\n(%)",
}


st.set_page_config(
    page_title="EDA Studio",
    layout="wide",
    page_icon="🔎",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Streamlit API compatibility
# --------------------------------------------------------------------------- #
def _stretch(func) -> dict:
    """Width keyword for full-width widgets, across Streamlit versions."""
    if "width" in inspect.signature(func).parameters:
        try:
            major, minor = (int(part) for part in st.__version__.split(".")[:2])
            if (major, minor) >= (1, 49):
                return {"width": "stretch"}
        except (ValueError, AttributeError):
            pass
    return {"use_container_width": True}


_CHART_WIDTH = _stretch(st.plotly_chart)
_BUTTON_WIDTH = _stretch(st.download_button)


def chart(figure, key: str | None = None) -> None:
    """Render a Plotly figure at full container width."""
    st.plotly_chart(figure, config=PLOTLY_CONFIG, key=key, **_CHART_WIDTH)


#: Cell fills for risky values, matching ``drift.df_to_img`` exactly so a table
#: on screen and its exported PNG flag the same cells in the same colours.
WARN_FILL = drift_warn_color
ALERT_FILL = drift_alert_color


def _threshold_style(
    value: object,
    thresholds: tuple[float, float],
    higher_is_worse: bool,
) -> str:
    """Cell CSS for one value, or an empty string to leave it alone."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    warn, alert = float(thresholds[0]), float(thresholds[1])
    if higher_is_worse:
        fill = ALERT_FILL if number >= alert else (WARN_FILL if number >= warn else None)
    else:  # small values are the notable ones, as with a p-value
        fill = ALERT_FILL if number <= alert else (WARN_FILL if number <= warn else None)
    return f"background-color: {fill}; font-weight: 600" if fill else ""


#: Past this many columns, squeezing everything into the viewport makes the
#: cells too narrow to read, so the table scrolls sideways instead.
SCROLL_AFTER_COLUMNS = 20

#: Minimum width per column once a table scrolls, in rem.
SCROLL_COLUMN_WIDTH = 7.0


def professional_table(
    frame: pd.DataFrame,
    *,
    formats: dict[str, object] | None = None,
    gradient: dict[str, object] | None = None,
    height: int = 600,
    label: str = "Data table",
    thresholds: tuple[float, float] | None = None,
    higher_is_worse: bool = True,
    scroll: bool | None = None,
    column_width: float = SCROLL_COLUMN_WIDTH,
) -> None:
    """Render one centred, responsive table with safe HTML content.

    The whole table is always rendered as a single block: splitting a wide result
    into "part 1 of 2" made it impossible to read a feature across its snapshots.
    Up to :data:`SCROLL_AFTER_COLUMNS` columns the table fits the page and the
    font tightens as columns are added. Beyond that — a 100-column upload, say —
    every column keeps a readable width and the table scrolls sideways inside its
    own frame, which beats a hundred unreadable slivers.

    ``scroll`` forces that behaviour on or off; ``None`` decides by column count.
    ``thresholds`` shades risky numeric cells amber and red, in the same two
    colours ``drift.df_to_img`` uses for the exported PNG.
    """
    safe_label = html.escape(str(label), quote=True)
    n_columns = len(frame.columns)
    wide = n_columns > SCROLL_AFTER_COLUMNS if scroll is None else bool(scroll)

    styled = frame.style.format(
        {key: value for key, value in (formats or {}).items() if key in frame.columns},
        na_rep="—",
        escape="html",
        precision=3,
    )
    if gradient:
        numeric_columns = [
            column
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column].dtype)
        ]
        if numeric_columns:
            styled = styled.background_gradient(subset=numeric_columns, **gradient)
    if thresholds is not None and n_columns > 1:
        # The first column carries the feature name, so it is never shaded.
        styled = styled.map(
            lambda value: _threshold_style(value, thresholds, higher_is_worse),
            subset=list(frame.columns[1:]),
        )
    styled = (
        styled.format_index(escape="html", axis="columns")
        .hide(axis="index")
        .set_properties(**{"text-align": "center"})
        .set_table_attributes(f'class="profile-table" aria-label="{safe_label}"')
    )

    # Three densities keep a 4-column and a 16-column table equally legible.
    if n_columns >= 12:
        density = " profile-table-wrap--tight"
    elif n_columns >= 7:
        density = " profile-table-wrap--dense"
    else:
        density = ""

    if wide:
        # A minimum width per column makes the table wider than its frame, which
        # is what turns on the horizontal scrollbar. The font stays as it is.
        density += " profile-table-wrap--scroll"
        table_style = f"min-width:{n_columns * column_width:.1f}rem"
        st.caption(
            f"{n_columns:,} columns — scroll sideways inside the table to see them all."
        )
    else:
        table_style = ""

    st.markdown(
        (
            f'<div class="profile-table-wrap{density}" role="region" tabindex="0" '
            f'aria-label="{safe_label}" style="max-height:{int(height)}px">'
            f'<div style="{table_style}">{styled.to_html()}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def sidebar_run_button(label: str, key: str, *, help: str) -> bool:
    """The one primary action, always at the very end of the sidebar.

    Every control that changes the result lives in the sidebar, so the button
    that acts on them belongs there too -- directly under the last setting rather
    than back up on the page.
    """
    st.sidebar.markdown('<div class="run-anchor"></div>', unsafe_allow_html=True)
    return st.sidebar.button(
        label,
        type="primary",
        key=key,
        help=help,
        **_BUTTON_WIDTH,
    )


def stale_results_banner(what: str, run_label: str) -> None:
    """Say loudly that the settings moved on since the results were computed.

    A plain ``st.warning`` scrolls away and reads like a caption. This banner
    sticks to the top of the results area while the user scrolls, pulses gently,
    and is backed by a toast so the message also arrives if the banner is off
    screen when the setting changes.
    """
    st.markdown(
        f"""
        <div class="stale-banner" role="alert">
          <span class="stale-icon">⚠️</span>
          <span><strong>These results are out of date.</strong> The {what} changed after
          this run. Press <strong>{run_label}</strong> at the end of the sidebar to refresh —
          until then, everything below reflects the previous settings.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.toast(f"Settings changed — press **{run_label}** to refresh.", icon="⚠️")


def resolve_feature_choice(chosen: list[str], available: list[str]) -> list[str]:
    """Turn a picker's selection into the real list of features.

    ``ALL_FEATURES`` anywhere in the selection means all of them; otherwise the
    picked names are used, in the order they appear in the data. An empty
    selection stays empty, so the caller can tell the user to choose something.
    """
    if ALL_FEATURES in chosen:
        return list(available)
    picked = {name for name in chosen if name != ALL_FEATURES}
    return [name for name in available if name in picked]


def _compact_value(value: object) -> str:
    """Format a numeric mode compactly while keeping text modes safe."""
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        if pd.isna(value):
            return "—"
        return f"{value:,.4g}"
    return str(value)


def _available_column_label(columns, preferred: str) -> str:
    """Return a display label that cannot collide with an existing column."""
    existing = set(columns)
    if preferred not in existing:
        return preferred
    suffix = 2
    while f"{preferred} {suffix}" in existing:
        suffix += 1
    return f"{preferred} {suffix}"


def render_footer(workspace: str = "profiling") -> None:
    """Page footer.

    Called on every exit path, including the early ``st.stop()`` branches, so
    the credit is present whether or not a dataset has been analysed yet.
    """
    st.divider()
    left, right = st.columns([3, 2])
    left.caption("**EDA Studio** · developed by Parsa Farshadfar")
    engine_label = (
        f"profiling engine v{engine_version}"
        if workspace == "profiling"
        else f"drift engine v{drift_engine_version}"
    )
    right.caption(
        f"{engine_label} · "
        f"pandas {pd.__version__} · numpy {np.__version__} · streamlit {st.__version__}"
    )


st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
      [data-testid="stSidebar"] .stSlider {padding-bottom: 0.4rem;}
      div[data-testid="stExpander"] details summary p {font-weight: 600;}
      .profile-table-wrap {
        border: 1px solid #D6DEE5;
        border-radius: 0.5rem;
        max-width: 100%;
        /* A very wide table scrolls sideways inside its own frame instead of
           being split into two halves that cannot be read across. */
        overflow-x: auto;
        overflow-y: auto;
        margin: 0.4rem 0 0.8rem;
        width: 100%;
      }
      .profile-table-wrap:focus-visible {
        outline: 3px solid #72B7B2;
        outline-offset: 2px;
      }
      table.profile-table {
        border-collapse: separate;
        border-spacing: 0;
        color: #22303C;
        font-size: clamp(0.70rem, 0.62rem + 0.20vw, 0.88rem);
        max-width: 100%;
        min-width: 0;
        table-layout: fixed;
        width: 100% !important;
      }
      .profile-table-wrap--dense table.profile-table {
        font-size: clamp(0.62rem, 0.55rem + 0.18vw, 0.78rem);
      }
      .profile-table-wrap--tight table.profile-table {
        font-size: clamp(0.55rem, 0.48rem + 0.16vw, 0.70rem);
      }
      .profile-table-wrap--tight table.profile-table th,
      .profile-table-wrap--tight table.profile-table td {
        padding: clamp(0.18rem, 0.16rem + 0.14vw, 0.36rem);
      }
      /* Wide tables: keep every column readable and scroll instead of shrinking. */
      .profile-table-wrap--scroll table.profile-table {
        table-layout: auto;
        width: 100% !important;
      }
      .profile-table-wrap--scroll table.profile-table th,
      .profile-table-wrap--scroll table.profile-table td {
        overflow-wrap: normal;
        white-space: nowrap;
      }
      .profile-table-wrap--scroll table.profile-table th {
        white-space: pre-line;
      }
      /* A visible scrollbar, so it is obvious the table continues sideways. */
      .profile-table-wrap--scroll {scrollbar-color: #9FB3C4 #EEF3F7; scrollbar-width: thin;}
      .profile-table-wrap--scroll::-webkit-scrollbar {height: 10px;}
      .profile-table-wrap--scroll::-webkit-scrollbar-thumb {
        background: #9FB3C4;
        border-radius: 5px;
      }
      .profile-table-wrap--scroll::-webkit-scrollbar-track {background: #EEF3F7;}
      /* Stale-result banner: sticks to the top of the results area while the
         user scrolls, and breathes so it cannot be mistaken for a caption. */
      .stale-banner {
        align-items: center;
        background: linear-gradient(90deg, #FFF0BF 0%, #FFF7E0 100%);
        border: 1px solid #E0A800;
        border-left: 6px solid #E0A800;
        border-radius: 0.55rem;
        box-shadow: 0 6px 18px rgba(31, 45, 61, 0.16);
        color: #4A3600;
        display: flex;
        font-size: 0.94rem;
        gap: 0.7rem;
        margin: 0.2rem 0 1rem;
        padding: 0.75rem 1rem;
        position: sticky;
        top: 0.3rem;
        z-index: 60;
        animation: stale-pulse 2.4s ease-in-out infinite;
      }
      .stale-banner strong {color: #7A4E00;}
      .stale-banner .stale-icon {font-size: 1.25rem; line-height: 1;}
      @keyframes stale-pulse {
        0%, 100% {box-shadow: 0 6px 18px rgba(224, 168, 0, 0.20);}
        50% {box-shadow: 0 6px 26px rgba(224, 168, 0, 0.55);}
      }
      @media (prefers-reduced-motion: reduce) {
        .stale-banner {animation: none;}
      }
      /* The primary run button lives at the end of the sidebar. */
      [data-testid="stSidebar"] .run-anchor {
        border-top: 1px solid #D6DEE5;
        margin: 0.6rem 0 0.4rem;
        padding-top: 0.7rem;
      }
      table.profile-table th {
        background: #2F4B63;
        border-bottom: 1px solid #D6DEE5;
        border-right: 1px solid #D6DEE5;
        color: white;
        font-weight: 650;
        line-height: 1.15;
        overflow-wrap: anywhere;
        padding: clamp(0.32rem, 0.28rem + 0.25vw, 0.65rem);
        position: sticky;
        text-align: center !important;
        top: 0;
        vertical-align: middle;
        white-space: pre-line;
        word-break: normal;
        z-index: 2;
      }
      table.profile-table td {
        border-bottom: 1px solid #D6DEE5;
        border-right: 1px solid #D6DEE5;
        font-variant-numeric: tabular-nums;
        overflow-wrap: anywhere;
        padding: clamp(0.28rem, 0.24rem + 0.20vw, 0.55rem);
        text-align: center !important;
        vertical-align: middle;
        white-space: normal;
        word-break: normal;
      }
      table.profile-table tbody tr:nth-child(odd) {background: #FFFFFF;}
      table.profile-table tbody tr:nth-child(even) {background: #EEF3F7;}
      table.profile-table tbody tr:hover {background: #E1EBF3;}
      table.profile-table th:last-child,
      table.profile-table td:last-child {border-right: 0;}
      table.profile-table tbody tr:last-child td {border-bottom: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# cached work
# --------------------------------------------------------------------------- #
# ``cache_resource`` rather than ``cache_data`` on purpose: ``cache_data``
# serialises its return value and deserialises a fresh copy on every access,
# which would mean re-materialising the whole DataFrame on every widget
# interaction. Nothing downstream mutates the loaded frame or the result, so
# sharing one instance is both correct and dramatically cheaper.
@st.cache_resource(show_spinner=False, max_entries=8)
def load_example(path: str, optimize: bool) -> LoadedData:
    return load_table(path, optimize=optimize)


@st.cache_resource(show_spinner=False, max_entries=8)
def load_upload(_upload, cache_key: str, optimize: bool) -> LoadedData:
    return load_table(_upload, optimize=optimize)


@st.cache_resource(show_spinner=False, max_entries=4)
def profile(
    _frame: pd.DataFrame,
    _settings: ProfilingSettings,
    cache_key: str,
    dataset_name: str,
    _dtype_labels: dict[str, str],
    _source_info: dict,
) -> ProfilingResult:
    return run_profiling(
        _frame,
        _settings,
        dataset_name=dataset_name,
        dtype_labels=_dtype_labels,
        source_info=_source_info,
    )


@dataclass
class DriftRun:
    """Small, UI-ready tables produced by one cached drift assessment."""

    distance_results: dict[str, pd.DataFrame]
    ks_p_values: pd.DataFrame | None
    descriptives: dict[str, pd.DataFrame]
    missing_percentages: pd.DataFrame | None
    missing_counts: pd.DataFrame | None
    elapsed_seconds: float
    snapshots: tuple[str, ...]
    snapshot_sizes: dict[str, int]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    notes: tuple[str, ...] = ()
    score_distribution: pd.DataFrame | None = None
    score_summary: pd.DataFrame | None = None
    score_density: pd.DataFrame | None = None
    score_column: str | None = None
    rows_before_sampling: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftSettings:
    """Hashable drift controls captured at the moment a run starts."""

    features: tuple[str, ...]
    methods: tuple[str, ...]
    date_column: str | None
    interval: str
    ohe_columns: tuple[str, ...]
    psi_buckets: int = 10
    psi_bucket_type: str = "bins"
    #: Which descriptive trends to compute: any of ("mean", "std"). Empty skips
    #: the section entirely -- some users only ever want the mean.
    descriptive_measures: tuple[str, ...] = ("mean",)
    include_missing: bool = True
    #: Column holding the model's output. ``None`` skips the score section
    #: entirely -- most datasets have no score to compare.
    score_column: str | None = None
    #: Share of each population actually assessed, as a percentage.
    sample_percent: float = 100.0

    @property
    def include_descriptives(self) -> bool:
        return bool(self.descriptive_measures)

    @property
    def include_score(self) -> bool:
        return self.score_column is not None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@st.cache_resource(show_spinner=False, max_entries=3)
def assess_drift(
    _current: pd.DataFrame,
    _reference: pd.DataFrame,
    cache_key: str,
    features: tuple[str, ...],
    methods: tuple[str, ...],
    *,
    date_column: str | None,
    interval: str,
    ohe_columns: tuple[str, ...],
    psi_buckets: int,
    psi_bucket_type: str,
    descriptive_measures: tuple[str, ...],
    include_missing: bool,
    score_column: str | None,
    sample_percent: float,
) -> DriftRun:
    """Run the engine once and retain only compact aggregate result tables."""
    started = datetime.now(timezone.utc)
    assessment = DriftAssessment(
        _current,
        col_names=list(features),
        reference=_reference,
        ohe_columns=list(ohe_columns) or None,
        date_col=date_column,
        interval=interval,
        save_results=False,
        score_col=score_column,
        sample_percent=sample_percent,
    )
    assessment.calc_distances(
        methods=list(methods),
        psi_n_buckets=int(psi_buckets),
        psi_bucket_type=psi_bucket_type,
        plot_trend=False,
        verbose=False,
    )

    descriptives: dict[str, pd.DataFrame] = {}
    if descriptive_measures:
        calculated = assessment.calc_descriptives_overtime(
            measures=tuple(descriptive_measures),
            plot_trend=False,
        )
        descriptives = {
            str(name): table.copy()
            for name, table in (calculated or {}).items()
            if isinstance(table, pd.DataFrame)
        }

    score_distribution = None
    score_summary = None
    score_density = None
    if score_column is not None:
        score_distribution, score_summary = assessment.score_analysis(
            n_bins=int(psi_buckets),
            bucket_type=psi_bucket_type,
            plot_trend=False,
        )
        score_distribution = score_distribution.copy()
        score_summary = score_summary.copy()
        score_density = assessment.score_density_.copy()

    missing_percentages = None
    missing_counts = None
    if include_missing:
        missing_percentages, missing_counts = assessment.missing_analysis()
        missing_percentages = missing_percentages.copy()
        missing_counts = missing_counts.copy()

    distances = {
        str(method).lower(): table.copy()
        for method, table in assessment.distance_results_.items()
    }

    # Chi-square is defined for categorical features only. The engine keeps a row
    # for every feature and marks the numeric ones unavailable, which reached the
    # page as all-NaN rows and a grid of blank charts. Drop them here, at the one
    # place the results are assembled, so the table, the charts and the exported
    # bundle cannot disagree about which features were tested.
    chisquare = distances.get("chisquare")
    if chisquare is not None:
        categorical = {str(value) for value in (assessment.cat_cols or ())}
        distances["chisquare"] = chisquare.loc[
            [index for index in chisquare.index if str(index) in categorical]
        ]
    ks_details = getattr(assessment, "ks_p_values_", None)
    if isinstance(ks_details, pd.DataFrame):
        ks_details = ks_details.copy()

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return DriftRun(
        distance_results=distances,
        ks_p_values=ks_details,
        descriptives=descriptives,
        missing_percentages=missing_percentages,
        missing_counts=missing_counts,
        elapsed_seconds=elapsed,
        snapshots=tuple(str(value) for value in assessment.snapshots_labels),
        snapshot_sizes={
            str(label): int(snapshot.shape[0])
            for label, snapshot in assessment.data_dict.items()
        },
        numeric_features=tuple(str(value) for value in (assessment.num_cols or ())),
        categorical_features=tuple(str(value) for value in (assessment.cat_cols or ())),
        notes=tuple(str(note) for note in getattr(assessment, "distance_notes_", ())),
        score_distribution=score_distribution,
        score_summary=score_summary,
        score_density=score_density,
        score_column=score_column,
        rows_before_sampling={
            str(label): int(count)
            for label, count in getattr(assessment, "rows_before_sampling_", {}).items()
        },
    )


def settings_key(settings: ProfilingSettings, token: str) -> str:
    """Stable identity for a (dataset, settings) pair, used as a cache key."""
    return json.dumps({"token": token, **asdict(settings)}, sort_keys=True, default=str)


def upload_cache_key(upload: object) -> str:
    """Content identity for an uploaded file without materialising another copy."""
    try:
        payload = upload.getbuffer()
    except (AttributeError, TypeError):
        payload = upload.getvalue()
    digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
    return f"{getattr(upload, 'name', 'upload')}:{len(payload)}:{digest}"


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
def dataset_controls() -> LoadedData | None:
    st.sidebar.header("1 · Dataset")
    source = st.sidebar.radio(
        "Source",
        ("Example dataset", "Upload a file"),
        index=0,
        horizontal=True,
        help=(
            "**Example dataset** — a bundled file, ready to explore without uploading anything.\n\n"
            "**Upload a file** — your own CSV, TSV or delimited text. The separator is detected "
            "from the header row, so `,`, `;`, tab and `|` all work."
        ),
    )

    optimize = st.sidebar.toggle(
        "Optimise memory on load",
        value=True,
        help=(
            "Shrink the table in memory using conversions that cannot change a value: integers "
            "are narrowed to the smallest exact type and repetitive text becomes a category. "
            "Floating-point precision is never reduced, so every statistic still comes from the "
            "numbers in your file.\n\n"
            "Example: a 16,512-row file drops from 1.28 MB to 0.93 MB. Turn it off only if you "
            "want the dtypes exactly as pandas first read them."
        ),
    )

    if source == "Upload a file":
        upload = st.sidebar.file_uploader(
            "CSV, TSV or delimited text", type=["csv", "tsv", "txt"], label_visibility="collapsed"
        )
        if upload is None:
            return None
        key = upload_cache_key(upload)
        try:
            return load_upload(upload, key, optimize)
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            st.sidebar.error(f"Could not read the file: {error}")
            return None

    name = st.sidebar.selectbox("Example dataset", list(EXAMPLE_DATASETS))
    try:
        return load_example(EXAMPLE_DATASETS[name], optimize)
    except Exception as error:  # noqa: BLE001 - surfaced to the user
        st.sidebar.error(f"Could not read the example dataset: {error}")
        return None


def analysis_controls(loaded: LoadedData) -> tuple[ProfilingSettings, bool]:
    """Collect every profiling setting, and say whether Run was pressed.

    The whole panel is one :func:`streamlit.form`. Nothing on the page reacts
    while settings are being changed -- Streamlit does not even rerun the script
    for a widget inside a form -- so a 1 GB dataset is never re-rendered because
    a checkbox moved. One click on the button at the bottom applies everything at
    once.

    Controls that used to appear and disappear are always drawn now, for the same
    reason: revealing a widget would need a rerun. Each says when it applies.
    """
    columns = [str(column) for column in loaded.frame.columns]
    form = st.sidebar.form("profiling_settings", border=False)

    with form:
        st.header("2 · Features")
        chosen_columns = st.multiselect(
            "Columns to analyse",
            [ALL_FEATURES, *columns],
            default=[ALL_FEATURES],
            key="profiling_columns",
            help=(
                f"**{ALL_FEATURES}** is selected by default: all {len(columns):,} columns are "
                "profiled.\n\n"
                "Remove it and pick individual columns to narrow the run — useful on a wide "
                "table, where correlation over hundreds of columns is the slow part.\n\n"
                "Text and categorical columns are described and counted, but they are skipped "
                "by correlation and box plots, which need numbers."
            ),
        )
        selected = resolve_feature_choice(chosen_columns, columns)
        st.caption(
            f"All {len(columns):,} columns selected."
            if ALL_FEATURES in chosen_columns
            else f"{len(selected):,} of {len(columns):,} columns selected."
        )

        st.header("3 · Analyses")
        include_describe = st.checkbox(
            "Descriptive statistics",
            value=True,
            key="profiling_describe",
            help=(
                "Per column: dtype, distinct values, missing and zero counts, the most frequent "
                "value, and min / 1% / median / 99% / max.\n\n"
                "Example: `AveRooms · float64 · 14,850 distinct · 875 missing (5.299%) · "
                "median 5.224`."
            ),
        )
        include_missing = st.checkbox(
            "Missing values",
            value=True,
            key="profiling_missing",
            help=(
                "Counts and percentages of nulls per column, ranked. Percentages are of the "
                "total row count, not of the non-null count, so 875 missing out of 16,512 rows "
                "reads as 5.299%."
            ),
        )
        include_correlation = st.checkbox(
            "Correlation",
            value=True,
            key="profiling_correlation",
            help=(
                "Pairwise relationships between numeric columns, as a heatmap plus a ranked "
                "list of the strongest pairs. Text columns are excluded.\n\n"
                "This is the step whose cost grows fastest with row count -- see "
                "**Performance** below."
            ),
        )
        include_histograms = st.checkbox(
            "Distributions",
            value=True,
            key="profiling_histograms",
            help=(
                "A histogram per numeric column and a value-count chart per categorical one. "
                "Charts are drawn from pre-computed bin counts, so a 10-million-row file sends "
                "no more data to the browser than a 1,000-row one."
            ),
        )
        include_box = st.checkbox(
            "Outliers (box plots)",
            value=True,
            key="profiling_box_plots",
            help=(
                "Quartiles, whiskers and outlier counts per numeric column. A value is an "
                "outlier when it sits further than 1.5 × IQR beyond the nearer quartile -- the "
                "standard Tukey rule. Counts are exact even when only a sample of points is "
                "drawn."
            ),
        )

        st.divider()
        st.caption("**Correlation settings** — used when *Correlation* is ticked.")
        chosen_methods = st.multiselect(
            "Correlation methods",
            ["Pearson", "Spearman", "Kendall"],
            default=["Spearman"],
            key="profiling_correlation_methods",
            help=(
                "**Pearson** — straight-line relationship between the raw values; sensitive to "
                "outliers.\n\n"
                "**Spearman** — Pearson on ranks, so it catches any consistently increasing or "
                "decreasing relationship and shrugs off outliers.\n\n"
                "**Kendall** — rank agreement (tau-b), the most robust and by far the slowest: "
                "every pair costs its own O(n log n) pass. On 100 columns that is 4,950 pairs."
            ),
        )
        methods = tuple(method.lower() for method in chosen_methods)
        threshold = st.slider(
            "Report pairs above |r|",
            0.0,
            1.0,
            0.65,
            0.01,
            key="profiling_correlation_threshold",
            help=(
                "Only feature pairs whose absolute coefficient exceeds this appear in the "
                "ranked list and get outlined on the heatmap. 0.65 keeps the list short; lower "
                "it to see weaker relationships.\n\n"
                "Example: at 0.65, a pair at r = −0.88 is listed and a pair at r = 0.41 is not."
            ),
        )

        st.divider()
        st.header("4 · Binning")
        st.caption("How the distribution charts divide each numeric column.")
        binning_label = st.radio(
            "Binning method",
            ["Equal-width", "Quantile"],
            horizontal=True,
            key="profiling_binning",
            help=(
                "**Equal-width** — bins of the same width, so the shape of the distribution is "
                "visible but a skewed column crowds into one bar.\n\n"
                "**Quantile** — bins holding roughly the same number of rows, so the tails are "
                "readable and the widths differ. Example: an income column with a long tail is "
                "far more informative with quantile bins."
            ),
        )
        binning = EQUAL_WIDTH if binning_label == "Equal-width" else QUANTILE
        n_bins = st.slider(
            "Number of bins",
            2,
            100,
            10,
            key="profiling_n_bins",
            help=(
                "How many bars each numeric histogram gets. Fewer bins show the broad shape; "
                "more bins reveal detail and gaps. A column with fewer distinct values than "
                "this gets one bin per value instead of empty bars."
            ),
        )
        show_percentage = st.toggle(
            "Show distributions as percentages",
            value=False,
            key="profiling_show_percentage",
            help=(
                "Label the bars with each bin's share of the non-missing values instead of its "
                "row count -- the comparable view when two datasets differ in size."
            ),
        )

        with st.expander("Performance", expanded=False):
            sample_percent = float(
                st.slider(
                    "Analyse this % of rows",
                    min_value=1,
                    max_value=100,
                    value=100,
                    step=1,
                    key="profiling_sample_percent",
                    help=(
                        "Profile a random share of the rows instead of all of them — the "
                        "quickest way to make a very large file workable.\n\n"
                        "**100% analyses everything and is the default.** Below that, the "
                        "draw is seeded, so the same percentage and seed always profile the "
                        "same rows, and *every* table and chart in the run describes that "
                        "same sample — nothing is computed on the full data and nothing on "
                        "a different subset.\n\n"
                        "Example: a 20-million-row extract at 10% profiles 2 million rows in "
                        "roughly a tenth of the time. Distribution shapes and correlations "
                        "survive sampling well; exact counts do not, so read the row and "
                        "missing counts as belonging to the sample. The percentage, the rows "
                        "used and the seed are all recorded in the exported report."
                    ),
                )
            )
            st.caption(
                "Correlation is the step whose cost grows fastest with row count. Pearson and "
                "Spearman reduce to a single matrix product, but Kendall's tau needs an "
                "O(n log n) pass for every feature pair, so it dominates on large files. "
                "Subsampling applies to **every** selected method, and all of them are computed "
                "from the same sampled rows, so the methods stay comparable. The draw is "
                "seeded, and the sample size and seed are recorded in the report."
            )
            limit_rows = st.checkbox(
                "Subsample rows for correlation",
                value=False,
                key="profiling_limit_rows",
                help="Tick this, then set the sample size below.",
            )
            max_rows = int(
                st.number_input(
                    "Maximum rows",
                    min_value=1_000,
                    max_value=max(1_000, int(loaded.n_rows)),
                    value=int(min(100_000, max(1_000, loaded.n_rows))),
                    step=1_000,
                    key="profiling_max_rows",
                    help="Used only when subsampling is ticked.",
                )
            )
            random_state = int(
                st.number_input(
                    "Random seed",
                    min_value=0,
                    value=0,
                    step=1,
                    key="profiling_seed",
                    help="Same seed, same sample, same numbers.",
                )
            )

        run = st.form_submit_button(
            "▶  Run profiling",
            type="primary",
            **_BUTTON_WIDTH,
        )

    settings = ProfilingSettings(
        selected_features=tuple(selected),
        include_describe=include_describe,
        include_missing=include_missing,
        include_correlation=include_correlation,
        include_histograms=include_histograms,
        include_box_plots=include_box,
        correlation_methods=methods if include_correlation else (),
        correlation_threshold=float(threshold),
        correlation_max_rows=max_rows if limit_rows else None,
        sample_percent=sample_percent,
        binning=binning,
        n_bins=int(n_bins),
        show_percentage=bool(show_percentage),
        random_state=random_state if limit_rows else 0,
    )
    return settings, run


def drift_dataset_controls() -> tuple[LoadedData, LoadedData, tuple[str, ...], bool] | None:
    """Load a reference/current pair, defaulting to the bundled example."""
    st.sidebar.header("1 · Datasets")
    st.sidebar.caption(
        "Drift needs two populations: a **baseline** you trust and the **current** data you are "
        "monitoring."
    )
    source = st.sidebar.radio(
        "Source",
        ("Example datasets", "Upload two files"),
        index=0,
        horizontal=True,
        key="drift_source",
        help=(
            "**Example datasets** — a bundled baseline/current pair with a `Month` column, ready "
            "to run.\n\n"
            "**Upload two files** — your own pair. Both need the same column names; columns "
            "present on only one side cannot be compared and are reported as conflicts."
        ),
    )
    optimize = st.sidebar.toggle(
        "Optimise memory on load",
        value=True,
        key="drift_optimize",
        help=(
            "The same lossless shrinking the profiling workspace uses: integers narrowed to the "
            "smallest exact type, repetitive text stored as categories, floating-point precision "
            "untouched. The drift engine reads the optimised dtypes natively, and every distance "
            "is computed from the values in your files."
        ),
    )

    if source == "Example datasets":
        example_name = st.sidebar.selectbox(
            "Example format",
            list(DRIFT_EXAMPLES),
            index=0,
            key="drift_example_name",
            help=(
                "**Standard categorical data** — `CardType` is already one plain text column.\n\n"
                "**One-hot encoded data** — the same data with `CardType_Elite`, "
                "`CardType_Platinum`, … instead, to show the collapsing step below."
            ),
        )
        baseline_path, current_path, ohe_columns = DRIFT_EXAMPLES[example_name]
        try:
            baseline = load_example(baseline_path, optimize)
            current = load_example(current_path, optimize)
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            st.sidebar.error(f"Could not read the drift examples: {error}")
            return None
        return baseline, current, tuple(ohe_columns), True

    baseline_upload = st.sidebar.file_uploader(
        "Baseline / reference file",
        type=["csv", "tsv", "txt"],
        key="drift_baseline_upload",
        help=(
            "The population you trust and want to compare against — typically the data a model "
            "was trained or validated on, or a period known to be healthy.\n\n"
            "It is never split by time and never mixed into the current data: it stays the single "
            "reference point for every comparison. No date column needed."
        ),
    )
    current_upload = st.sidebar.file_uploader(
        "Current / monitored file",
        type=["csv", "tsv", "txt"],
        key="drift_current_upload",
        help=(
            "The newer data you are checking for drift. Include a date column and it can be cut "
            "into monthly (or daily, weekly, quarterly, yearly) snapshots, each compared with the "
            "baseline; without one it is compared as a single population.\n\n"
            "Column names must match the baseline's — mismatches are reported so you can fix them."
        ),
    )
    if baseline_upload is None or current_upload is None:
        return None

    try:
        baseline = load_upload(
            baseline_upload,
            upload_cache_key(baseline_upload),
            optimize,
        )
        current = load_upload(
            current_upload,
            upload_cache_key(current_upload),
            optimize,
        )
    except Exception as error:  # noqa: BLE001 - surfaced to the user
        st.sidebar.error(f"Could not read the uploaded datasets: {error}")
        return None
    return baseline, current, (), False


def _parse_ohe_columns(raw: str) -> tuple[str, ...]:
    """Parse a compact comma-separated OHE prefix list without duplicates."""
    return tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


def _logical_drift_columns(frame: pd.DataFrame, ohe_columns: tuple[str, ...]) -> list[str]:
    """Return post-OHE column names using only a one-row schema sample."""
    if not ohe_columns:
        return [str(column) for column in frame.columns]
    sample = frame.head(1).copy()
    merged = merge_drift_ohe(sample, list(ohe_columns))
    return [str(column) for column in merged.columns]


def drift_analysis_controls(
    baseline: LoadedData,
    current: LoadedData,
    default_ohe: tuple[str, ...],
    *,
    is_example: bool,
) -> tuple[DriftSettings, tuple[str, ...], tuple[str, ...], bool] | None:
    """Collect the drift settings, and say whether Run was pressed.

    Like the profiling panel, everything lives in one :func:`streamlit.form`:
    changing a setting does not rerun the app, so the results on the right stay
    exactly as they were until the button at the bottom is pressed. Controls that
    only apply in some situations are always drawn -- hiding them would need the
    rerun this design exists to avoid -- and each one says when it is used.
    """
    ohe_key_suffix = "_".join(default_ohe) or ("example" if is_example else "upload")
    form = st.sidebar.form("drift_settings", border=False)

    with form:
        st.header("2 · Structure")
        collapse_ohe = st.checkbox(
            "Collapse one-hot encoded groups",
            value=bool(default_ohe),
            key=f"drift_collapse_ohe_{ohe_key_suffix}",
            help=(
                "Turn a block of 0/1 indicator columns back into the single categorical "
                "feature they encode, before any drift is measured.\n\n"
                "Comparing five `CardType_*` flags one at a time hides the thing that actually "
                "moved: which card type customers hold. Collapsed, PSI is calculated once over "
                "the whole category and reads like the business question.\n\n"
                "Your files are not modified — the rebuild happens on internal copies."
            ),
        )
        raw_ohe = st.text_input(
            "Original feature names",
            value=", ".join(default_ohe),
            key=f"drift_ohe_columns_{ohe_key_suffix}",
            placeholder="CardType",
            help=(
                "Used when **Collapse one-hot encoded groups** is ticked. Enter the name "
                "**before** the underscore in the dummy columns — the feature as it existed "
                "before encoding.\n\n"
                "**Example.** Your file has these four columns:\n\n"
                "`CardType_Elite`, `CardType_Platinum`, `CardType_Smart Cash`, "
                "`CardType_True Line`\n\n"
                "Enter `CardType`. They become one `CardType` feature whose values are "
                "`Elite`, `Platinum`, `Smart Cash`, `True Line`.\n\n"
                "Separate several encoded features with commas. A row where every flag is 0 "
                "(or all are null) stays missing rather than being assigned an arbitrary "
                "category."
            ),
        )
        ohe_columns = _parse_ohe_columns(raw_ohe) if collapse_ohe else ()
        if collapse_ohe and not ohe_columns:
            st.warning("Enter at least one original feature name, or turn collapsing off.")

        st.divider()
        # Both options measure the *current* data against the baseline. The choice
        # is only whether the current data is cut into periods first.
        snapshot_options = (
            "Split the current data by time",
            "Current data as one population",
        )
        dated_intervals = [
            "Monthly",
            *(name for name in DRIFT_INTERVALS if name not in {"Monthly", "All"}),
        ]
        default_snapshot = 0 if is_example and "Month" in current.frame.columns else 1
        snapshot_layout = st.radio(
            "Comparison layout",
            snapshot_options,
            index=default_snapshot,
            key=f"drift_snapshot_layout_{'example' if is_example else 'upload'}",
            help=(
                "How the **current / monitored** file is grouped before it is compared. The "
                "baseline is never split and never mixed in — it stays the single reference "
                "point for every comparison.\n\n"
                "**Split the current data by time** — cut it into periods using a date column "
                "and measure each period against the baseline. This is what shows *when* a "
                "feature started moving. Example: six monthly snapshots, six PSI values per "
                "feature, and a trend chart per feature.\n\n"
                "**Current data as one population** — no date column needed; the whole current "
                "file is compared with the baseline as a single group, giving one number per "
                "feature. Use it for a quick check, or when the file has no usable date."
            ),
        )
        split_by_time = snapshot_layout != snapshot_options[1]
        date_options = [str(column) for column in current.frame.columns]
        preferred = date_options.index("Month") if "Month" in date_options else 0
        date_column = st.selectbox(
            "Snapshot date column",
            date_options,
            index=preferred,
            key=f"drift_date_column_{'example' if is_example else 'upload'}",
            help=(
                "⚠️ **Only used when *Comparison layout* is set to "
                "\"Split the current data by time\".** With \"Current data as one "
                "population\" this control is ignored.\n\n"
                "It is the column that says *when* each row happened, and it is parsed rather "
                "than assumed, so almost any shape works:\n\n"
                "`2019-03-15` · `2019-03` · `Mar 2019` · `15/03/2019` · `201903` · `20190315` "
                "· `2019Q1` · a real timestamp · a pandas period\n\n"
                "Rows whose date cannot be read are reported and left out, rather than being "
                "guessed at. The baseline needs no date column."
            ),
        )
        interval = st.selectbox(
            "Snapshot length",
            dated_intervals,
            index=0,
            key=f"drift_interval_{'example' if is_example else 'upload'}",
            help=(
                "⚠️ **Only used when *Comparison layout* is set to "
                "\"Split the current data by time\".** With \"Current data as one "
                "population\" this control is ignored.\n\n"
                "It sets how long one snapshot covers before it is compared with the "
                "baseline.\n\n"
                "**Monthly** (default) — labels like `2019-03`, the usual choice for "
                "monitoring.\n\n"
                "**Daily** `2019-03-15` · **Weekly** `2019-W11` (ISO weeks) · "
                "**Quarterly** `2019-Q1` · **Yearly** `2019`\n\n"
                "Shorter periods react faster but hold fewer rows, which makes every measure "
                "noisier; longer periods are steadier but slower to reveal a change. A year of "
                "data gives 12 monthly snapshots or 4 quarterly ones."
            ),
        )
        if not split_by_time:
            date_column = None
            interval = "All"

        try:
            baseline_columns = _logical_drift_columns(baseline.frame, ohe_columns)
            current_columns = _logical_drift_columns(current.frame, ohe_columns)
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"Could not reconstruct the one-hot encoded columns: {error}")
            st.form_submit_button("▶  Run drift assessment", disabled=True, **_BUTTON_WIDTH)
            return None

        current_set = set(current_columns)
        baseline_set = set(baseline_columns)
        shared = [
            column
            for column in baseline_columns
            if column in current_set and column != date_column
        ]
        baseline_only = tuple(
            column for column in baseline_columns if column not in current_set
        )
        current_only = tuple(
            column
            for column in current_columns
            if column not in baseline_set and column != date_column
        )

        st.header("3 · Features")
        if not shared:
            st.error(
                "The baseline and current datasets share no analysable columns, so there is "
                "nothing to compare. Check that both files use the same column names — and if "
                "one side is one-hot encoded, tick **Collapse one-hot encoded groups** above."
            )
            st.form_submit_button("▶  Run drift assessment", disabled=True, **_BUTTON_WIDTH)
            return None
        if baseline_only or current_only:
            # Named in full on the page; flagged here because this is where it
            # can still be fixed.
            st.warning(
                f"{len(baseline_only) + len(current_only)} column(s) exist in only one file "
                "and cannot be compared — the full list is at the top of the page.",
                icon="⚠️",
            )
        chosen_features = st.multiselect(
            "Features to assess",
            [ALL_FEATURES, *shared],
            default=[ALL_FEATURES],
            key="drift_features",
            help=(
                f"**{ALL_FEATURES}** is selected by default: every one of the "
                f"{len(shared)} columns both files share is assessed.\n\n"
                f"Remove it and pick individual features to narrow the run — a shorter list "
                "means a faster assessment and a smaller export. A column present in only one "
                "of the two files can never be assessed and is listed as a conflict instead."
            ),
        )
        selected = resolve_feature_choice(chosen_features, shared)
        st.caption(
            f"All {len(shared)} shared features selected."
            if ALL_FEATURES in chosen_features
            else f"{len(selected)} of {len(shared)} shared features selected."
        )

        st.header("4 · Analyses")
        selected_methods = st.multiselect(
            "Distance measures",
            list(DRIFT_METHOD_LABELS),
            # PSI only by default: it is the one measure that works for both
            # numeric and categorical features and has recognised triage bands.
            default=["psi"],
            format_func=lambda value: DRIFT_METHOD_LABELS[value],
            key="drift_methods",
            help=(
                "How the distance between the baseline and each snapshot is measured. **PSI "
                "alone is selected by default** — it covers numeric and categorical features "
                "and has recognised triage bands. Add others when you want a second "
                "opinion:\n\n"
                "**PSI** — compares binned shares of the two distributions. Numeric *and* "
                "categorical. Rule of thumb: < 0.10 stable, 0.10–0.25 moderate, ≥ 0.25 "
                "substantial.\n\n"
                "**Kolmogorov–Smirnov** — numeric only. The largest gap between the two "
                "cumulative distributions, plus a p-value. Sensitive on large samples: with a "
                "million rows, a tiny, harmless difference still produces p < 0.001.\n\n"
                "**Wasserstein** — numeric only. \"How far the mass moved\", in the feature's "
                "own units, so a shift of 0.4 on an income column means 0.4 of income. Compare "
                "a feature with itself over time, never one feature with another.\n\n"
                "**Energy distance (legacy CM)** — numeric only. Same idea as Wasserstein with "
                "a different weighting; kept for continuity with older reports.\n\n"
                "**Chi-square** — categorical only. Tests whether the category frequencies "
                "differ.\n\n"
                "A measure that does not apply to a feature's type is reported as unavailable, "
                "with the reason, rather than silently producing a meaningless number."
            ),
        )

        st.divider()
        st.caption("**PSI settings** — used when PSI is selected.")
        psi_buckets = st.slider(
            "PSI buckets",
            min_value=2,
            max_value=50,
            value=10,
            key="drift_psi_buckets",
            help=(
                "How many buckets a numeric feature is split into before the two distributions "
                "are compared. 10 is the industry default.\n\n"
                "More buckets notice smaller, more local shifts but inflate PSI on modest "
                "samples; fewer buckets are steadier but can hide a shift inside one wide "
                "bucket. Keep it fixed when comparing runs — a PSI at 20 buckets is not "
                "comparable with a PSI at 10.\n\n"
                "Categorical features ignore this: their categories are the buckets."
            ),
        )
        psi_label = st.radio(
            "PSI bucket edges",
            ("Equal-width", "Quantile"),
            horizontal=True,
            key="drift_psi_type",
            help=(
                "Where the bucket edges are placed, using the **baseline** to define them so "
                "both populations are binned the same way.\n\n"
                "**Equal-width** — buckets of equal size across the baseline's range. Simple "
                "and the usual choice.\n\n"
                "**Quantile** — buckets holding equally many baseline rows, so a skewed "
                "feature spreads over all buckets instead of piling into one. Example: an "
                "income column where 90% of customers sit in the first equal-width bucket."
            ),
        )
        psi_bucket_type = "bins" if psi_label == "Equal-width" else "quantiles"

        st.divider()
        st.caption("**Descriptive trends** — plain summaries next to the distances.")
        include_mean = st.checkbox(
            "Mean / most frequent value",
            value=True,
            key="drift_descriptive_mean",
            help=(
                "Track the centre of each feature over the snapshots, next to the baseline: "
                "the **mean** for a numeric feature and the **most frequent value** for a "
                "categorical one, to four decimals.\n\n"
                "A distance measure tells you *that* a feature moved; this tells you *what* "
                "moved. Example: PSI on `Feature2` jumps in March, and the mean shows why — it "
                "fell from 0.7618 to 0.5231.\n\n"
                "Each feature also gets its own chart, and a PNG per feature in the export."
            ),
        )
        include_std = st.checkbox(
            "Standard deviation",
            value=False,
            key="drift_descriptive_std",
            help=(
                "Track the spread of each numeric feature over the snapshots.\n\n"
                "Read it together with the mean: a mean that falls while the standard "
                "deviation holds steady means the whole distribution shifted down; a standard "
                "deviation that grows on a steady mean means the population became more "
                "varied. Categorical features have no standard deviation and are left out."
            ),
        )
        include_missing = st.checkbox(
            "Missing-value comparison",
            value=True,
            key="drift_missing",
            help=(
                "Compare how much of each feature is null in the baseline and in every "
                "snapshot, as both a percentage and a row count.\n\n"
                "Worth keeping on: a broken upstream feed usually shows up here first, as a "
                "column that quietly goes from 0.1% to 30% missing. That also depresses every "
                "distance measure for the feature, since only the non-null rows are compared."
            ),
        )
        # Off by default: plenty of datasets carry no model output at all, and a
        # score column is not something the app can reliably guess.
        include_score = st.checkbox(
            "Model score comparison",
            value=False,
            key="drift_include_score",
            help=(
                "Compare the distribution of a **model's output** between the baseline and "
                "each snapshot, instead of only its inputs.\n\n"
                "Leave this off if the dataset holds no model score — most do not, which is "
                "why it starts unticked."
            ),
        )
        # Never disabled by the checkbox above it. Every control here lives inside
        # one st.form, and a form does not rerun until it is submitted, so a
        # `disabled=` bound to a sibling widget stays stuck at its previous value
        # until the next run -- which is exactly the trap this used to fall into.
        # "None" carries the same meaning without depending on a rerun.
        score_options = [SCORE_NONE] + [
            column for column in shared if column != date_column
        ]
        score_column = st.selectbox(
            "Model score column",
            options=score_options,
            index=0,  # None: no score column, the right default for most datasets
            key="drift_score_column",
            help=(
                "The column holding the model's output: a predicted probability, a credit "
                "score, a risk band, a predicted class.\n\n"
                "**What it does.** The baseline's scores are cut into bands, and every "
                "snapshot is measured against *those* bands, so the columns are comparable. "
                "You get the share of rows in each band, the mean and quartiles per "
                "snapshot, and **PSI of the score against the baseline** — the number most "
                "monitoring processes act on (< 0.10 stable, 0.10–0.25 moderate, ≥ 0.25 "
                "substantial).\n\n"
                "**Why separately from the features.** Score drift is the thing that "
                "actually costs money, and it does not follow from feature drift: inputs can "
                "each move a little and leave the score alone, or sit still while a "
                "re-trained model shifts it. The column is excluded from the feature "
                "distances so it is never double-counted as an input.\n\n"
                "Example: predicted default probability averaging 0.041 in the baseline and "
                "0.058 last month, PSI 0.31 — the population being scored has changed enough "
                "to justify a look, whatever the individual features say.\n\n"
                "Numeric scores are banded by value; a categorical score (`low`/`medium`/"
                "`high`) uses its own categories as the bands."
            ),
        )
        if include_score and score_column == SCORE_NONE:
            st.caption("Pick a score column above to run the model score comparison.")

        with st.expander("Performance", expanded=False):
            sample_percent = float(
                st.slider(
                    "Assess this % of rows",
                    min_value=1,
                    max_value=100,
                    value=100,
                    step=1,
                    key="drift_sample_percent",
                    help=(
                        "Assess a random share of the rows instead of all of them — the "
                        "quickest way to make a very large pair of files workable.\n\n"
                        "**100% uses everything and is the default.** Below that, the "
                        "baseline and *each snapshot* are sampled separately, so every "
                        "population keeps the same share of its own rows and the comparison "
                        "is not tilted by one busy month. The draw is seeded, so the same "
                        "percentage always measures the same rows.\n\n"
                        "Sampling adds noise to every distance: PSI on a thinned sample "
                        "wobbles, and small snapshots wobble most. Use it to explore a large "
                        "file quickly, then confirm anything you plan to act on at 100%."
                    ),
                )
            )

        measures = tuple(
            name
            for name, wanted in (("mean", include_mean), ("std", include_std))
            if wanted
        )

        run = st.form_submit_button(
            "▶  Run drift assessment",
            type="primary",
            **_BUTTON_WIDTH,
        )

    settings = DriftSettings(
        features=tuple(selected),
        methods=tuple(selected_methods),
        date_column=date_column,
        interval=interval,
        ohe_columns=tuple(ohe_columns),
        psi_buckets=int(psi_buckets),
        psi_bucket_type=psi_bucket_type,
        descriptive_measures=measures,
        include_missing=bool(include_missing),
        score_column=(
            str(score_column)
            if include_score and score_column != SCORE_NONE
            else None
        ),
        sample_percent=sample_percent,
    )
    return settings, baseline_only, current_only, run


# --------------------------------------------------------------------------- #
def overview_section(result: ProfilingResult, loaded: LoadedData) -> None:
    columns = st.columns(5)
    columns[0].metric("Rows", f"{result.n_rows:,}")
    columns[1].metric("Columns analysed", f"{len(result.settings.selected_features):,}")
    columns[2].metric("Numeric features", f"{len(result.numeric_features):,}")
    columns[3].metric("In memory", f"{result.memory_mb:,.1f} MB")
    columns[4].metric("Compute time", f"{result.total_seconds:.2f} s")

    if result.notes:
        for note in result.notes:
            st.info(note, icon="ℹ️")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Column types")
        types = pd.DataFrame(
            {"Column": list(result.dtypes), "Type in source file": list(result.dtypes.values())}
        )
        types["Type in memory"] = [
            str(loaded.frame[column].dtype) if column in loaded.frame.columns else "-"
            for column in types["Column"]
        ]
        professional_table(types, height=320, label="Column types")

    with right:
        st.subheader("Load details")
        st.markdown(
            f"""
            - **File** `{loaded.source_name}`
            - **Parser** `{loaded.parser}` · **delimiter** `{loaded.delimiter}`
            - **Fingerprint** `blake2b-128 {loaded.token}`
            - **Memory** {loaded.memory_before_mb:,.2f} MB → {loaded.memory_after_mb:,.2f} MB
              ({loaded.memory_saved_mb:,.2f} MB reclaimed)
            """
        )
        if loaded.optimizations and st.checkbox(
            f"Show the {len(loaded.optimizations)} lossless dtype changes", value=False
        ):
            professional_table(
                pd.DataFrame(
                    {
                        "Column": list(loaded.optimizations),
                        "Change": list(loaded.optimizations.values()),
                    }
                ),
                label="Lossless dtype changes",
            )

        st.subheader("Step timings")
        timings = pd.DataFrame(
            {
                "Step": [key.replace("_", " ").title() for key in result.timings],
                "Seconds": [round(value, 3) for value in result.timings.values()],
            }
        )
        professional_table(
            timings,
            formats={"Seconds": "{:.3f}"},
            label="Step timings",
        )


def describe_section(result: ProfilingResult) -> None:
    if result.describe is None or result.describe.empty:
        st.info("No numeric features were selected.")
        return

    st.caption(
        "Percentages are relative to the total row count. The most frequent value counts nulls "
        "as a candidate, so a mostly-empty column reports a null mode."
    )
    display = (
        result.describe.rename(columns=DESCRIBE_DISPLAY_LABELS)
        .rename_axis("Feature")
        .reset_index()
    )
    formats = {
        "Missing\n(%)": "{:.3f}%",
        "Zeros\n(%)": "{:.3f}%",
        "Most frequent\n(%)": "{:.3f}%",
        "Most frequent\nvalue": _compact_value,
        "Min": "{:,.4g}",
        "1%": "{:,.4g}",
        "50%": "{:,.4g}",
        "99%": "{:,.4g}",
        "Max": "{:,.4g}",
        "Unique values\n(excl. nulls)": "{:,.0f}",
        "Missing\ncount": "{:,.0f}",
        "Zero\ncount": "{:,.0f}",
    }
    professional_table(
        display,
        formats={key: value for key, value in formats.items() if key in display},
        height=min(680, 60 + 35 * len(display)),
        label="Descriptive statistics",
    )


def missing_section(result: ProfilingResult) -> None:
    if result.missing_all is None:
        st.info("Missing-value analysis was not enabled.")
        return

    total_missing = int(result.missing_all["number_of_missing"].sum())
    affected = int((result.missing_all["number_of_missing"] > 0).sum())

    columns = st.columns(3)
    columns[0].metric("Columns with gaps", f"{affected:,}")
    columns[1].metric("Missing cells", f"{total_missing:,}")
    denominator = result.n_rows * max(len(result.missing_all), 1)
    columns[2].metric(
        "Share of all cells", f"{(100.0 * total_missing / denominator) if denominator else 0:.3f}%"
    )

    if result.missing.empty:
        st.success("No missing values were found in the analysed columns.")
    else:
        chart(plots.missing_values_figure(result.missing))

    st.subheader("Per column")
    show_all = st.toggle("Include columns with no missing values", value=result.missing.empty)
    frame = result.missing_all if show_all else result.missing
    display = (
        frame.rename(
            columns={"number_of_missing": "Missing rows", "percentage_of_missing": "Missing (%)"}
        )
        .rename_axis("Feature")
        .reset_index()
    )
    professional_table(
        display,
        formats={"Missing rows": "{:,.0f}", "Missing (%)": "{:.3f}%"},
        height=min(600, 60 + 35 * len(frame)),
        label="Missing values by feature",
    )


def correlation_section(result: ProfilingResult) -> None:
    if not result.correlations:
        st.info(
            "Correlation analysis was not run. It needs at least two numeric features and at "
            "least one selected method."
        )
        return

    threshold = result.settings.correlation_threshold
    st.caption(
        f"Cells outlined in black exceed |r| > {threshold:g}. Pairs are formed from rows where "
        "both features are present (pairwise-complete), matching pandas."
    )

    for position, (method, matrix) in enumerate(result.correlations.items()):
        if position:
            st.divider()
        st.markdown(f"### {method.capitalize()}")
        chart(plots.correlation_heatmap(matrix, method, threshold=threshold), key=f"heat_{method}")

        top = result.top_correlations.get(method)
        value_column = f"absolute {method} correlation values"
        if top is None or top.empty:
            st.success(f"No feature pair exceeds |r| > {threshold:g}.")
        else:
            st.markdown(f"**{len(top):,} pair{'s' if len(top) != 1 else ''} above the threshold**")
            display_label = f"Absolute {method.capitalize()}\ncorrelation"
            professional_table(
                top.rename(columns={value_column: display_label}),
                formats={display_label: "{:.4f}"},
                height=min(480, 60 + 35 * len(top)),
                label=f"{method.capitalize()} correlations above threshold",
            )
        # The heatmap above already shows the whole matrix, and the export
        # carries it as a CSV, so it is not repeated here as a table.


def chart_grid(items: list, render, *, label: str, columns_per_row: int = 3) -> None:
    """Draw every chart, three to a row.

    No pagination: a feature you have to click "next page" to reach is a feature
    nobody looks at. Each chart carries a pre-aggregated summary of a few hundred
    bytes, so the whole set stays small however many rows the dataset has.
    """
    if not items:
        st.info(f"No {label} to display for the current selection.")
        return

    st.caption(f"One chart per feature, {len(items):,} in total.")
    for start in range(0, len(items), columns_per_row):
        row = st.columns(columns_per_row)
        chunk = items[start : start + columns_per_row]
        for offset, (slot, item) in enumerate(zip(row, chunk)):
            with slot:
                render(item, start + offset)


def distributions_section(result: ProfilingResult) -> None:
    summaries = list(result.distributions.values())
    settings = result.settings
    st.caption(
        f"{settings.binning.replace('-', ' ').capitalize()} binning with {settings.n_bins} bins. "
        "Charts are drawn from pre-computed bin counts, so the payload sent to the browser does "
        "not grow with the size of the dataset."
    )

    n_categorical = sum(1 for s in summaries if s.kind == KIND_CATEGORICAL)
    if n_categorical:
        plural = "s are" if n_categorical != 1 else " is"
        st.caption(f"{n_categorical} categorical feature{plural} shown as value counts.")

    def render(summary, index: int) -> None:
        chart(
            plots.histogram_figure(
                summary,
                color=plots.color_for(index),
                show_percentage=settings.show_percentage,
            ),
            key=f"hist_{summary.feature}_{index}",
        )

    chart_grid(summaries, render, label="distributions")


def outliers_section(result: ProfilingResult) -> None:
    summaries = list(result.box_summaries.values())
    st.caption(
        "Each box plot spans Q1 to Q3 and reaches out to the most extreme observation within "
        "1.5 x IQR of the quartiles. Outlier counts are exact; when a feature has thousands of "
        "them an evenly spaced subset is drawn."
    )

    if summaries:
        flagged = [s for s in sorted(summaries, key=lambda s: -s.n_outliers)[:5] if s.n_outliers]
        if flagged:
            details = ", ".join(
                f"`{s.feature}` ({s.n_outliers:,}, {100.0 * s.n_outliers / max(s.n_valid, 1):.3f}%)"
                for s in flagged
            )
            st.markdown(f"**Most outlier-heavy features:** {details}")
        else:
            st.markdown("**No feature has values beyond 1.5 x IQR.**")

    def render(summary, index: int) -> None:
        chart(
            plots.box_figure(summary, color=plots.color_for(index)),
            key=f"box_{summary.feature}_{index}",
        )

    chart_grid(summaries, render, label="box plots")


def _display_path(path: Path) -> str:
    """Project-relative form of a path, or the full path when it lies outside."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def save_to_project_folder(
    workspace: str,
    writer,
    *,
    target_name: str,
    note: str,
    storage_module,
) -> None:
    """Offer to write the results into a folder, editable by the user.

    The download buttons hand a bundle to the browser; this keeps a copy where
    the app is running. The default is ``results/<workspace>/`` beside
    ``app.py`` -- the same folder the example notebooks use -- and the address is
    editable, so a Databricks user can send the bundle straight to DBFS or ADLS.
    ``writer`` receives the target folder and returns the addresses it wrote.
    """
    default_parent = str(RESULTS_ROOT / workspace)
    st.markdown("#### Save the results to a folder")
    st.caption(note)

    parent = st.text_input(
        "Destination folder",
        value=st.session_state.get(f"{workspace}_save_parent", default_parent),
        key=f"{workspace}_save_parent",
        help=(
            "Where to write this run. The default is the project's own results folder, "
            "next to app.py.\n\n"
            "**Local:** `C:/reports/january` or `/mnt/reports/january`\n\n"
            "**Databricks DBFS:** `dbfs:/FileStore/eda/drift` (or the `/dbfs/...` mount)\n\n"
            "**ADLS / cloud:** `abfss://container@account.dfs.core.windows.net/eda/drift`\n\n"
            "Cloud addresses are written through the cluster's own credentials, in a single "
            "operation per file — nothing is streamed or appended, which is what a locked-down "
            "storage account requires. A sub-folder named after this run is created inside."
        ),
    )
    parent = (parent or default_parent).strip()
    remote = storage_module.is_remote(parent)
    folder = storage_module.join_address(parent, target_name)
    st.code(f"{folder if remote else _display_path(Path(folder))}/", language="text")

    state_key = f"{workspace}_saved_folder"
    if st.button(
        "Save to this folder",
        key=f"{workspace}_save_folder",
        **_BUTTON_WIDTH,
    ):
        try:
            with st.spinner(f"Writing results to {target_name}…"):
                written = writer(folder)
            st.session_state[state_key] = (folder, len(written))
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            st.session_state.pop(state_key, None)
            st.error(f"Could not write the results: {error}")
            if storage_module.is_remote(parent):
                st.caption(
                    "Cloud addresses need Databricks' `dbutils`, which only exists on a "
                    "cluster. Running locally? Use a local folder instead."
                )

    saved = st.session_state.get(state_key)
    if saved:
        path, count = saved
        shown = path if storage_module.is_remote(path) else f"{_display_path(Path(path))}/"
        st.success(f"{count} file(s) written to `{shown}`", icon="💾")


def download_section(result: ProfilingResult, cache_key: str) -> None:
    st.subheader("Export the full analysis")
    st.markdown(
        """
        Both bundles contain the same numbers you see on screen, rendered as static charts.
        Each one carries a cover page and a `metadata.json` recording the dataset fingerprint,
        the exact settings, the step timings and the library versions, so the report can be
        traced and reproduced.
        """
    )

    n_charts = len(result.distributions) + len(result.box_summaries)
    if n_charts > 60:
        st.warning(
            f"{n_charts} charts will be rendered. Building the export may take a minute or two.",
            icon="⏳",
        )

    base = suggested_basename(result)
    left, right = st.columns(2)

    with left:
        st.markdown("#### PDF report")
        st.caption(
            "Paginated document: cover, dataset overview, one section per analysis, a methodology "
            "page defining every statistic, and a reproducibility page. Text stays selectable."
        )
        if st.button("Build PDF", **_BUTTON_WIDTH):
            with st.spinner("Rendering the PDF report…"):
                st.session_state["profiling_pdf"] = (cache_key, build_pdf(result))
        stored = st.session_state.get("profiling_pdf")
        if stored and stored[0] == cache_key:
            st.download_button(
                f"Download {base}.pdf  ({len(stored[1]) / 1024:,.0f} KB)",
                data=stored[1],
                file_name=f"{base}.pdf",
                mime="application/pdf",
                type="primary",
                **_BUTTON_WIDTH,
            )

    with right:
        st.markdown("#### ZIP archive")
        st.caption(
            "Numbered folders with the underlying CSVs, an Excel workbook of the correlations, "
            "PNG charts and the PDF. Every chart has a CSV beside it with the values behind it."
        )
        per_feature = st.checkbox(
            "Include one PNG per feature",
            value=True,
            help="Uncheck for a smaller archive containing only the grid overviews.",
        )
        if st.button("Build ZIP", **_BUTTON_WIDTH):
            with st.spinner("Rendering charts and assembling the archive…"):
                cached = st.session_state.get("profiling_pdf")
                # Reuse the PDF when it has already been built: rendering it is
                # the most expensive single part of the export.
                payload = cached[1] if cached and cached[0] == cache_key else build_pdf(result)
                st.session_state["profiling_pdf"] = (cache_key, payload)
                st.session_state["profiling_zip"] = (
                    (cache_key, per_feature),
                    build_zip(result, pdf_bytes=payload, include_individual_charts=per_feature),
                )
        stored = st.session_state.get("profiling_zip")
        if stored and stored[0] == (cache_key, per_feature):
            st.download_button(
                f"Download {base}.zip  ({len(stored[1]) / 1024:,.0f} KB)",
                data=stored[1],
                file_name=f"{base}.zip",
                mime="application/zip",
                type="primary",
                **_BUTTON_WIDTH,
            )

    if st.checkbox("Show what is inside the ZIP", value=False):
        st.code(
            f"""{base}/
├── README.txt                      what each folder holds
├── metadata.json                   fingerprint, settings, timings, versions
├── report.pdf                      the full paginated report
├── 01_descriptive_statistics/      per-feature profile (CSV + PNG)
├── 02_missing_values/              counts and percentages (CSV + PNG)
├── 03_correlation/                 matrices, ranked pairs, heatmaps, XLSX
├── 04_distributions/               bin edges and counts (CSV) + one PNG per feature
└── 05_outliers/                    quartiles, fences, counts (CSV) + one PNG per feature""",
            language="text",
        )

    # One folder name per run, remembered so a rerun of the page does not invent
    # a new timestamp. Toronto time, with a graceful fallback where no timezone
    # database is installed.
    stamps = st.session_state.setdefault("profiling_folder_stamps", {})
    stamp = stamps.setdefault(cache_key, profiling_storage.run_stamp("profiling"))

    st.divider()
    save_to_project_folder(
        "profiling",
        lambda folder: write_report_dir(
            result,
            folder,
            include_pdf=False,
            include_individual_charts=True,
        ),
        target_name=stamp,
        note=(
            "Writes the same CSVs and one PNG per feature as the ZIP, unpacked into one "
            "folder. Only the PDF is left out — build the ZIP above if you want that too."
        ),
        storage_module=profiling_storage,
    )


def _table_with_feature_column(table: pd.DataFrame, label: str = "Feature") -> pd.DataFrame:
    """Move a result index into a collision-safe, readable first column."""
    index_label = _available_column_label(table.columns, label)
    return table.rename_axis(index_label).reset_index()


def _numeric_table(table: pd.DataFrame) -> pd.DataFrame:
    """Coerce a compact result table for plotting without changing the source."""
    return table.apply(lambda column: pd.to_numeric(column, errors="coerce"))


def feature_trend_figure(
    values: pd.Series,
    *,
    title: str,
    y_label: str,
    color_index: int = 0,
    thresholds: tuple[float, float] | None = None,
) -> go.Figure:
    """One feature, one chart: its value across the snapshots.

    A chart per feature rather than every feature on shared axes -- drift
    measures live on different scales, so one large feature used to flatten
    everything else into a line along the bottom.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    labels = [str(index) for index in numeric.index]
    series = numeric.to_numpy(dtype=float, copy=False)
    color = plots.color_for(color_index)

    figure = go.Figure()
    if len(labels) == 1:  # a single snapshot reads better as one bar
        figure.add_bar(
            x=labels,
            y=series,
            marker_color=color,
            width=0.4,
            hovertemplate="%{x}<br>%{y:.5g}<extra></extra>",
        )
    else:
        figure.add_scatter(
            x=labels,
            y=series,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 7},
            hovertemplate="Snapshot %{x}<br>%{y:.5g}<extra></extra>",
        )
    if thresholds is not None:
        # The same amber/red bands the tables are shaded with.
        for level, fill in zip(thresholds, (WARN_FILL, ALERT_FILL)):
            figure.add_hline(
                y=float(level),
                line={"color": fill, "width": 2, "dash": "dash"},
                annotation_text=f"{float(level):g}",
                annotation_position="top left",
                annotation_font={"size": 10, "color": "#7A4E00"},
            )
    figure.update_layout(
        title={"text": title, "font": {"size": 14}},
        xaxis_title="Snapshot" if len(labels) > 1 else "",
        yaxis_title=y_label,
        height=300,
        margin={"l": 60, "r": 20, "t": 50, "b": 60},
        showlegend=False,
        template="plotly_white",
    )
    figure.update_xaxes(type="category", tickangle=-30)
    figure.update_yaxes(rangemode="tozero")
    return figure


def feature_trend_png(
    values: pd.Series,
    *,
    title: str,
    y_label: str,
    thresholds: tuple[float, float] | None = None,
) -> bytes:
    """The same single-feature chart as a PNG for the saved bundle."""
    from matplotlib.figure import Figure

    numeric = pd.to_numeric(values, errors="coerce")
    labels = [str(index) for index in numeric.index]
    series = numeric.to_numpy(dtype=float, copy=False)

    figure = Figure(figsize=(7.4, 3.2), dpi=150, facecolor="white")
    axes = figure.add_subplot(111)
    if len(labels) == 1:
        axes.bar(labels, series, color="#4C78A8", width=0.35)
    else:
        axes.plot(labels, series, marker="o", markersize=4, color="#4C78A8", linewidth=1.8)
    if thresholds is not None:
        for level, color in zip(thresholds, ("#E0A800", "#C0392B")):
            axes.axhline(float(level), color=color, linewidth=1.1, linestyle="--")
            axes.annotate(
                f"{float(level):g}",
                xy=(0.002, float(level)),
                xycoords=("axes fraction", "data"),
                fontsize=7,
                color=color,
                va="bottom",
            )
    axes.set_title(title, fontsize=10, color="#22303C")
    axes.set_ylabel(y_label, fontsize=8)
    axes.tick_params(labelsize=7)
    axes.tick_params(axis="x", rotation=30)
    axes.grid(axis="y", color="#D6DEE5", linewidth=0.6)
    axes.set_axisbelow(True)
    axes.set_ylim(bottom=min(0.0, float(np.nanmin(series)) if series.size else 0.0))
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    figure.tight_layout()

    payload = BytesIO()
    figure.savefig(payload, format="png", facecolor="white")
    return payload.getvalue()


def score_density_figure(density: pd.DataFrame, score_name: str) -> go.Figure:
    """Every population's score density on one set of axes.

    A single chart on purpose: the question is whether the distributions still sit
    on top of each other, and that is only answerable when they share the axes.
    The baseline is drawn heavier so it reads as the thing being compared against.
    """
    figure = go.Figure()
    grid = density.index.to_numpy(dtype=float)
    for position, column in enumerate(density.columns):
        baseline = position == 0
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=density[column].to_numpy(dtype=float),
                mode="lines",
                name=str(column),
                line=dict(
                    color=plots.color_for(position),
                    width=3.0 if baseline else 1.9,
                    dash=None if baseline else "solid",
                ),
                # Filled baseline, outlined snapshots: the shaded shape is the one
                # everything else is being read against.
                fill="tozeroy" if baseline else None,
                fillcolor="rgba(47, 75, 99, 0.16)" if baseline else None,
                hovertemplate=f"{column}<br>%{{x:,.4g}}<br>density %{{y:,.4g}}<extra></extra>",
            )
        )
    figure.update_xaxes(title_text=score_name)
    figure.update_yaxes(title_text="Density", rangemode="tozero")
    figure.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=25, t=55, b=50),
        hoverlabel=dict(font_size=12),
        separators=".,",
        height=420,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=dict(text=f"{score_name}: distribution by population", font=dict(size=15)),
    )
    return figure


def feature_chart_grid(
    table: pd.DataFrame,
    features: list[str],
    *,
    y_label: str,
    key_prefix: str,
    thresholds: tuple[float, float] | None = None,
    columns_per_row: int = 3,
) -> None:
    """Lay the per-feature charts out two to a row.

    Owns its own count caption, so the number on screen is the number of charts
    actually drawn even when a measure does not apply to some of the features in
    the table.
    """
    numeric = _numeric_table(table)
    plotted = [feature for feature in features if feature in numeric.index]
    if not plotted:
        st.caption("No feature in this table has a value to chart.")
        return

    st.caption(f"One chart per feature, {len(plotted):,} in total, largest first.")
    for start in range(0, len(plotted), columns_per_row):
        row = st.columns(columns_per_row)
        for slot, feature in zip(row, plotted[start : start + columns_per_row]):
            with slot:
                chart(
                    feature_trend_figure(
                        numeric.loc[feature],
                        title=str(feature),
                        y_label=y_label,
                        color_index=plotted.index(feature),
                        thresholds=thresholds,
                    ),
                    key=f"{key_prefix}_{_safe_export_name(str(feature))}",
                )


def column_conflict_warning(
    baseline_only: tuple[str, ...],
    current_only: tuple[str, ...],
    baseline: LoadedData,
    current: LoadedData,
) -> None:
    """List the columns that exist in only one of the two datasets.

    Drift can only be measured on a column both populations have, so a mismatch
    silently shrinks the analysis. The collapsed list below names them exactly;
    it carries the whole message on its own, so there is no banner above it
    saying the same thing in prose.
    """
    if not baseline_only and not current_only:
        return

    conflicts = pd.DataFrame(
        [
            {
                "Column": name,
                "Present in": f"baseline only ({baseline.source_name})",
                "Missing from": current.source_name,
            }
            for name in baseline_only
        ]
        + [
            {
                "Column": name,
                "Present in": f"current only ({current.source_name})",
                "Missing from": baseline.source_name,
            }
            for name in current_only
        ]
    )
    # The label carries the finding now that no banner precedes it.
    with st.expander(
        f"{len(conflicts)} column(s) appear in only one dataset and were left out",
        expanded=False,
    ):
        professional_table(
            conflicts,
            height=min(420, 60 + 35 * len(conflicts)),
            label="Conflicting columns",
        )


def drift_overview_section(
    result: DriftRun,
    baseline: LoadedData,
    current: LoadedData,
    settings: DriftSettings,
) -> None:
    columns = st.columns(5)
    columns[0].metric("Baseline rows", f"{baseline.n_rows:,}")
    columns[1].metric("Current rows", f"{current.n_rows:,}")
    columns[2].metric("Snapshots", f"{len(result.snapshots):,}")
    columns[3].metric("Features", f"{len(settings.features):,}")
    columns[4].metric("Compute time", f"{result.elapsed_seconds:.2f} s")

    # ``distance_notes_`` only ever says "this measure does not apply to this
    # feature type", which the empty cell already shows. It is left out rather
    # than repeated as a wall of info boxes.
    if baseline.token == current.token:
        st.warning(
            "The baseline and current files have identical fingerprints. Near-zero drift is expected.",
            icon="⚠️",
        )
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Rows compared")
        st.caption(
            "The baseline is the fixed comparison point; every current snapshot is measured "
            "against it. Shares are of that dataset's own row count."
        )
        # Both populations in one table, each row saying which side it belongs
        # to, so "300 rows" can never be mistaken for part of the current data.
        rows = [
            {
                "Population": "Baseline (reference)",
                "Snapshot": baseline.source_name,
                "Rows": baseline.n_rows,
                "Share of its dataset (%)": 100.0,
            }
        ]
        rows.extend(
            {
                "Population": "Current (monitored)",
                "Snapshot": str(label),
                "Rows": int(size),
                "Share of its dataset (%)": 100.0 * size / max(current.n_rows, 1),
            }
            for label, size in result.snapshot_sizes.items()
        )
        snapshot_table = pd.DataFrame(rows)
        professional_table(
            snapshot_table,
            formats={"Rows": "{:,.0f}", "Share of its dataset (%)": "{:.2f}%"},
            height=min(460, 60 + 35 * len(snapshot_table)),
            label="Rows compared",
        )
    with right:
        st.subheader("Run details")
        st.markdown(
            f"""
            - **Baseline:** `{baseline.source_name}` · {baseline.memory_after_mb:,.2f} MB
            - **Current:** `{current.source_name}` · {current.memory_after_mb:,.2f} MB
            - **Numeric features:** {len(result.numeric_features):,}
            - **Categorical features:** {len(result.categorical_features):,}
            - **Methods:** {", ".join(DRIFT_METHOD_LABELS.get(value, value) for value in settings.methods)}
            """
        )


def drift_summary_section(result: DriftRun) -> None:
    psi = result.distance_results.get("psi")
    if psi is None or psi.empty:
        st.info("Enable PSI to add threshold-based drift triage to this summary.")
        return

    long = (
        psi.rename_axis("Feature")
        .reset_index()
        .melt(id_vars="Feature", var_name="Snapshot", value_name="PSI")
    )
    valid = pd.to_numeric(long["PSI"], errors="coerce")
    finite = valid[np.isfinite(valid)]
    highest = float(finite.max()) if not finite.empty else np.nan
    warn_level, alert_level = PSI_THRESHOLDS
    shifted = int(long.loc[valid >= warn_level, "Feature"].nunique())
    substantial = int(long.loc[valid >= alert_level, "Feature"].nunique())

    columns = st.columns(4)
    columns[0].metric("Highest PSI", "—" if pd.isna(highest) else f"{highest:.3f}")
    columns[1].metric(f"Features ≥ {warn_level:g}", f"{shifted:,}")
    columns[2].metric(f"Features ≥ {alert_level:g}", f"{substantial:,}")
    columns[3].metric("Feature-snapshot tests", f"{len(long):,}")

    psi_threshold_guide()
    ordered = long.assign(_psi=valid).sort_values(
        "_psi",
        ascending=False,
        na_position="last",
    )
    professional_table(
        ordered.drop(columns="_psi"),
        formats={"PSI": "{:.4f}"},
        height=min(620, 60 + 35 * len(ordered)),
        label="PSI drift summary",
        thresholds=PSI_THRESHOLDS,
    )


def psi_threshold_guide() -> None:
    """Explain the amber/red shading wherever PSI is shown.

    The tables shade risky cells instead of adding a status column, so the
    meaning of the two colours has to be stated next to them.
    """
    warn_level, alert_level = PSI_THRESHOLDS
    st.markdown(
        f"""
        <div style="border:1px solid #D6DEE5;border-radius:0.5rem;padding:0.6rem 0.9rem;
                    margin:0.2rem 0 0.8rem;font-size:0.88rem;line-height:1.6;">
          <b>How to read the shading</b><br>
          <span style="background:#FFFFFF;border:1px solid #D6DEE5;padding:0 0.4rem;">
            PSI &lt; {warn_level:g}</span>
          &nbsp;little or no shift — no action expected.<br>
          <span style="background:{WARN_FILL};padding:0 0.4rem;font-weight:600;">
            {warn_level:g} ≤ PSI &lt; {alert_level:g}</span>
          &nbsp;moderate shift — worth investigating, especially if it repeats across snapshots.<br>
          <span style="background:{ALERT_FILL};padding:0 0.4rem;font-weight:600;">
            PSI ≥ {alert_level:g}</span>
          &nbsp;substantial shift — investigate before relying on a model trained on the baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "These bands are the common industry triage heuristic, not p-values or universal "
        "business thresholds. PSI also grows with the number of buckets and shrinks on small "
        "samples, so calibrate the limits against your own history before acting on them."
    )


_DRIFT_METHOD_NOTES = {
    "psi": (
        "PSI compares binned probability distributions. It works for numeric and categorical "
        "features and is best interpreted with domain-specific thresholds."
    ),
    "ks2": (
        "The two-sample KS statistic is the largest separation between empirical cumulative "
        "distributions. The companion p-values are evidence measures, not effect sizes."
    ),
    "ws": (
        "Wasserstein distance is expressed in the feature's own units. Compare a feature across "
        "snapshots, not unrelated features with different scales."
    ),
    "cm": (
        "This legacy CM option uses SciPy's energy distance. Like Wasserstein distance, its "
        "magnitude depends on the feature scale."
    ),
    "chisquare": (
        "Chi-square applies to categorical features, so only those are listed here — numeric "
        "features are left out rather than shown as unavailable rows with nothing in them."
    ),
}


def ranked_features(table: pd.DataFrame) -> list[str]:
    """Every measurable feature in the table, largest value first.

    All of them are charted -- a feature left out of the charts is a feature
    nobody looks at -- but the ones that moved most come first, so the important
    charts are on screen before any scrolling.

    The one exception is a feature with no finite value in any snapshot, which a
    measure reports when it does not apply to that feature's type. There is
    nothing to draw, so charting it yields a blank pair of axes; it stays in the
    table, marked unavailable, and is left out of the chart grid.
    """
    numeric = _numeric_table(table)
    ranked = numeric.max(axis=1).sort_values(ascending=False, na_position="last").dropna()
    return [str(feature) for feature in ranked.index]


def drift_distances_section(result: DriftRun) -> None:
    if not result.distance_results:
        st.info("No distance measures were produced.")
        return

    for position, (method, table) in enumerate(result.distance_results.items()):
        if position:
            st.divider()
        label = DRIFT_METHOD_LABELS.get(method, method.upper())
        thresholds = PSI_THRESHOLDS if method == "psi" else None
        st.markdown(f"### {label}")
        st.caption(_DRIFT_METHOD_NOTES.get(method, "Distance values by feature and snapshot."))

        if table.empty:
            st.info(
                "No categorical features were selected, so chi-square has nothing to test."
                if method == "chisquare"
                else f"{label} produced no results for the current selection."
            )
            continue

        # With a single snapshot every chart would be one bar repeating the
        # number already in the table, so the table alone is shown.
        if table.shape[1] > 1:
            feature_chart_grid(
                table,
                ranked_features(table),
                y_label=label,
                key_prefix=f"drift_chart_{method}",
                thresholds=thresholds,
            )
        display = _table_with_feature_column(table)
        professional_table(
            display,
            formats={
                column: "{:.5g}"
                for column in display.columns
                if column != display.columns[0]
            },
            height=min(620, 60 + 35 * len(display)),
            label=f"{label} values",
            thresholds=thresholds,
        )
        if method == "psi":
            psi_threshold_guide()

        if method == "ks2" and isinstance(result.ks_p_values, pd.DataFrame):
            st.markdown("#### KS p-values")
            st.caption(
                "Small p-values indicate evidence against equal distributions — shaded here "
                "at 0.10 and 0.05. When many feature-snapshot pairs are tested, account for "
                "multiple comparisons before treating one small value as conclusive."
            )
            p_values = _table_with_feature_column(result.ks_p_values)
            professional_table(
                p_values,
                formats={
                    column: "{:.5g}"
                    for column in p_values.columns
                    if column != p_values.columns[0]
                },
                height=min(620, 60 + 35 * len(p_values)),
                label="KS p-values",
                thresholds=(0.10, 0.05),
                higher_is_worse=False,  # a small p-value is the notable one
            )


#: Descriptive tables the app shows. ``numeric_mean`` is deliberately absent: it
#: repeats the numeric rows of ``mean`` and is only kept by the engine because
#: the charts need a purely numeric frame.
DRIFT_DESCRIPTIVE_LABELS = {
    "mean": "Mean (numeric) or most frequent value (categorical) by snapshot",
    "std": "Standard deviation by snapshot",
}


def drift_descriptives_section(result: DriftRun) -> None:
    shown = {
        name: table
        for name, table in result.descriptives.items()
        if name in DRIFT_DESCRIPTIVE_LABELS
    }
    if not shown:
        st.info("Descriptive trends were not requested.")
        return

    st.caption(
        "Distances say *that* a feature moved; these tables usually say *what* moved. "
        "Numbers are rounded to four decimals, and the baseline column is the same "
        "reference every snapshot is compared with."
    )
    for position, (name, table) in enumerate(shown.items()):
        if position:
            st.divider()
        label = DRIFT_DESCRIPTIVE_LABELS[name]
        st.markdown(f"### {label}")
        normalized = table.rename(columns={"Baseline": "Reference"})
        display = _table_with_feature_column(normalized)
        professional_table(
            display,
            formats={
                column: "{:,.4f}"
                for column in display.columns[1:]
                if pd.api.types.is_numeric_dtype(display[column].dtype)
            },
            height=min(600, 60 + 35 * len(normalized)),
            label=label,
        )
        numeric = _numeric_table(normalized).dropna(how="all")
        if not numeric.empty and len(numeric.columns) > 1:
            feature_chart_grid(
                numeric,
                ranked_features(numeric),
                y_label="Average value" if name == "mean" else "Std. deviation",
                key_prefix=f"drift_descriptive_{name}",
            )


def drift_score_section(result: DriftRun) -> None:
    """Baseline-versus-current comparison of the model's own output."""
    distribution = result.score_distribution
    summary = result.score_summary
    if distribution is None or summary is None:
        st.info("No model score column was selected.")
        return

    st.caption(
        f"Distribution of **{result.score_column}** — the model's output — in the baseline "
        "and in every snapshot. Bands are cut from the baseline alone and reused for each "
        "snapshot, which is what makes the columns comparable."
    )

    psi_row = summary.loc["PSI vs baseline"] if "PSI vs baseline" in summary.index else None
    if psi_row is not None:
        worst = psi_row.drop(labels=[psi_row.index[0]], errors="ignore")
        worst = worst.dropna()
        if not worst.empty:
            peak = float(worst.max())
            band = (
                "substantial" if peak >= 0.25 else "moderate" if peak >= 0.10 else "stable"
            )
            st.markdown(
                f"**Highest score PSI against the baseline: {peak:.4f}** "
                f"({band}) — in `{worst.idxmax()}`."
            )

    density = result.score_density
    if density is not None and not density.empty:
        st.markdown("### Score distributions, overlaid")
        st.caption(
            "Every population's score density on one set of axes. Curves that sit on top of "
            "each other mean the score has held its shape; a curve shifted sideways or changed "
            "in height is drift you can see before reading a single number."
        )
        chart(score_density_figure(density, result.score_column or "score"))

    st.markdown("### Score summary by snapshot")
    summary_display = _table_with_feature_column(summary, label="Statistic")
    professional_table(
        summary_display,
        formats={
            column: "{:,.4f}"
            for column in summary_display.columns
            if column != summary_display.columns[0]
        },
        height=min(460, 60 + 35 * len(summary_display)),
        label="Model score summary",
    )
    psi_threshold_guide()

    st.markdown("### Share of rows per score band (%)")
    band_display = _table_with_feature_column(distribution, label="Score band")
    professional_table(
        band_display,
        formats={
            column: "{:.3f}%"
            for column in band_display.columns
            if column != band_display.columns[0]
        },
        height=min(600, 60 + 35 * len(band_display)),
        label="Model score distribution",
    )


def drift_missing_section(result: DriftRun) -> None:
    percentages = result.missing_percentages
    counts = result.missing_counts
    if percentages is None or counts is None:
        st.info("Missing-value comparison was not requested.")
        return

    percentages = percentages.rename(columns={"Baseline": "Reference"})
    counts = counts.rename(columns={"Baseline": "Reference"})
    st.markdown("### Missing rows (%)")
    st.caption(
        "A value is missing when pandas reports it as null. Percentages are of each "
        "population's own row count, so a small snapshot is comparable with a large baseline."
    )
    percentage_display = _table_with_feature_column(percentages)
    professional_table(
        percentage_display,
        formats={
            column: "{:.3f}%"
            for column in percentage_display.columns
            if column != percentage_display.columns[0]
        },
        height=min(600, 60 + 35 * len(percentage_display)),
        label="Missing percentages",
    )
    feature_chart_grid(
        percentages,
        ranked_features(percentages),
        y_label="Missing rows (%)",
        key_prefix="drift_missing_chart",
    )

    if st.checkbox("Show missing row counts", value=False, key="drift_show_missing_counts"):
        count_display = _table_with_feature_column(counts)
        professional_table(
            count_display,
            formats={
                column: "{:,.0f}"
                for column in count_display.columns
                if column != count_display.columns[0]
            },
            height=min(600, 60 + 35 * len(count_display)),
            label="Missing row counts",
        )


def _safe_export_name(value: str) -> str:
    """Portable filename component for generated drift artifacts."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "table"


@dataclass(frozen=True)
class DriftExportTable:
    """One result table, with everything needed to export it consistently."""

    folder: str
    name: str
    table: pd.DataFrame
    title: str
    y_label: str
    thresholds: tuple[float, float] | None = None
    higher_is_worse: bool = True
    chart_features: bool = True


def _drift_export_tables(result: DriftRun) -> list[DriftExportTable]:
    """Enumerate every drift result table once for CSV, PNG and chart export."""
    tables: list[DriftExportTable] = []
    for method, table in result.distance_results.items():
        label = DRIFT_METHOD_LABELS.get(method, method.upper())
        tables.append(
            DriftExportTable(
                "01_distance_measures",
                DRIFT_FILE_NAMES.get(method, method),
                table,
                title=f"{label} by snapshot",
                y_label=label,
                thresholds=PSI_THRESHOLDS if method == "psi" else None,
            )
        )
        if method == "ks2" and isinstance(result.ks_p_values, pd.DataFrame):
            tables.append(
                DriftExportTable(
                    "01_distance_measures",
                    "ks2_p",
                    result.ks_p_values,
                    title="KS p-values by snapshot",
                    y_label="KS p-value",
                    thresholds=(0.10, 0.05),
                    higher_is_worse=False,
                    chart_features=False,
                )
            )
    for name, table in result.descriptives.items():
        if name not in DRIFT_DESCRIPTIVE_LABELS:
            continue  # numeric_mean repeats the numeric rows of mean
        tables.append(
            DriftExportTable(
                "02_descriptive_trends",
                name,
                table,
                title=DRIFT_DESCRIPTIVE_LABELS[name],
                y_label="Average value" if name == "mean" else "Std. deviation",
            )
        )
    if result.missing_percentages is not None:
        tables.append(
            DriftExportTable(
                "03_missing_values",
                "missing_perc",
                result.missing_percentages,
                title="Missing rows (%) by snapshot",
                y_label="Missing rows (%)",
            )
        )
    if result.missing_counts is not None:
        tables.append(
            DriftExportTable(
                "03_missing_values",
                "missing_count",
                result.missing_counts,
                title="Missing row counts by snapshot",
                y_label="Missing rows",
                chart_features=False,
            )
        )
    if result.score_summary is not None:
        tables.append(
            DriftExportTable(
                "04_model_score",
                "score_summary",
                result.score_summary,
                title=f"{result.score_column}: score summary by snapshot",
                y_label="Value",
                # Rows here are statistics on different scales, not features, so
                # a chart per row would put a row count and a PSI on one axis.
                chart_features=False,
            )
        )
    if result.score_distribution is not None:
        tables.append(
            DriftExportTable(
                "04_model_score",
                "score_distribution",
                result.score_distribution,
                title=f"{result.score_column}: share of rows per score band (%)",
                y_label="Share of rows (%)",
            )
        )
    return tables


def _drift_table_png(
    table: pd.DataFrame,
    title: str,
    *,
    thresholds: tuple[float, float] | None = None,
    higher_is_worse: bool = True,
) -> bytes:
    """Render one exported table through the required ``df_to_img`` helper."""
    from matplotlib import pyplot as plt

    # df_to_img is blue by default, matching the profiling tables, and shades
    # risky cells in the same two colours the app uses on screen.
    figure = drift_df_to_img(
        table,
        title=title,
        show_index=True,
        color_thresholds=thresholds,
        higher_is_worse=higher_is_worse,
    )
    try:
        payload = BytesIO()
        figure.savefig(payload, format="png", dpi=240, bbox_inches="tight", facecolor="white")
        return payload.getvalue()
    finally:
        plt.close(figure)


def _drift_bundle_files(
    result: DriftRun,
    settings: DriftSettings,
    baseline: LoadedData,
    current: LoadedData,
):
    """Yield ``(relative path, bytes)`` for every file in a drift bundle.

    One definition of the contents, used by both the ZIP download and the folder
    copy, so an unpacked archive and a written folder are the same tree.
    """
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "name": baseline.source_name,
            "fingerprint": baseline.token,
            "rows": baseline.n_rows,
            "columns": baseline.n_cols,
        },
        "current": {
            "name": current.source_name,
            "fingerprint": current.token,
            "rows": current.n_rows,
            "columns": current.n_cols,
        },
        "settings": settings.as_dict(),
        "snapshots": result.snapshot_sizes,
        "elapsed_seconds": result.elapsed_seconds,
        "library_versions": {
            "profiling_engine": engine_version,
            "drift_engine": drift_engine_version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
            "plotly": package_version("plotly"),
            "pyarrow": package_version("pyarrow"),
            "openpyxl": package_version("openpyxl"),
            "streamlit": st.__version__,
        },
    }

    warn_level, alert_level = PSI_THRESHOLDS
    yield "metadata.json", json.dumps(metadata, indent=2, default=str).encode("utf-8")
    yield "README.txt", (
        "EDA Studio drift assessment\n\n"
        "Every result table is included twice: CSV for analysis and PNG for reporting.\n"
        "Each table also has a charts/ folder holding one PNG per feature, so a single\n"
        "feature's trend can be dropped into a document on its own.\n\n"
        f"PSI tables are shaded amber from {warn_level:g} and red from {alert_level:g}; KS\n"
        "p-value tables are shaded at 0.10 and 0.05. Those bands are heuristic triage\n"
        "thresholds, not universal statistical significance thresholds.\n\n"
        "All PNG tables were rendered with drift.df_to_img.\n"
    ).encode("utf-8")

    for export in _drift_export_tables(result):
        charts_written = 0  # counted per table, so every section gets its charts
        safe_name = _safe_export_name(export.name)
        yield (
            f"{export.folder}/{safe_name}.csv",
            export.table.to_csv(index=True).encode("utf-8"),
        )
        yield (
            f"{export.folder}/{safe_name}.png",
            _drift_table_png(
                export.table,
                export.title,
                thresholds=export.thresholds,
                higher_is_worse=export.higher_is_worse,
            ),
        )
        if not export.chart_features:
            continue
        # One chart per feature, so a single feature's trend can be used on its
        # own. Bounded so a very wide dataset cannot produce thousands of PNGs.
        numeric = _numeric_table(export.table)
        for feature in numeric.index:
            if charts_written >= MAX_EXPORTED_CHARTS:
                break
            values = numeric.loc[feature]
            if values.isna().all():
                continue
            charts_written += 1
            yield (
                f"{export.folder}/charts/{safe_name}_{_safe_export_name(str(feature))}.png",
                feature_trend_png(
                    values,
                    title=f"{feature} — {export.title}",
                    y_label=export.y_label,
                    thresholds=export.thresholds if export.higher_is_worse else None,
                ),
            )


def build_drift_zip(
    result: DriftRun,
    settings: DriftSettings,
    baseline: LoadedData,
    current: LoadedData,
) -> bytes:
    """Build an in-memory, reproducible drift bundle with CSV and PNG twins."""
    payload = BytesIO()
    with ZipFile(payload, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path, content in _drift_bundle_files(result, settings, baseline, current):
            archive.writestr(path, content)
    return payload.getvalue()


def write_drift_dir(
    result: DriftRun,
    settings: DriftSettings,
    baseline: LoadedData,
    current: LoadedData,
    folder: str,
) -> list[str]:
    """Write the drift bundle into ``folder`` as ordinary files.

    Each file is finished in memory and then written once through
    ``drift.storage``, which is what lets the same call target a local folder,
    DBFS or ADLS.
    """
    return [
        drift_storage.write_bytes(
            drift_storage.join_address(folder, *path.split("/")), content
        )
        for path, content in _drift_bundle_files(result, settings, baseline, current)
    ]


def drift_download_section(
    result: DriftRun,
    settings: DriftSettings,
    baseline: LoadedData,
    current: LoadedData,
    run_key: str,
) -> None:
    st.subheader("Export the drift evidence")
    st.markdown(
        "Build a ZIP containing metadata, one CSV per result table, and a matching PNG rendered "
        "through `df_to_img`. The bundle is assembled in memory and does not leave server-side files."
    )
    if st.button("Build drift ZIP", key="drift_build_zip", **_BUTTON_WIDTH):
        try:
            with st.spinner("Rendering drift tables and assembling the ZIP…"):
                st.session_state["drift_export"] = (
                    run_key,
                    build_drift_zip(result, settings, baseline, current),
                )
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            st.error(f"Could not build the drift export: {error}")

    # The ZIP the browser downloads can afford a descriptive name; the folder on
    # disk cannot -- it is the parent of every nested chart, and Windows still
    # rejects a path over 260 characters.
    base = (
        f"drift_{_safe_export_name(Path(baseline.source_name).stem)}_vs_"
        f"{_safe_export_name(Path(current.source_name).stem)}"
    )
    stored = st.session_state.get("drift_export")
    if stored and stored[0] == run_key:
        st.download_button(
            f"Download {base}.zip  ({len(stored[1]) / 1024:,.0f} KB)",
            data=stored[1],
            file_name=f"{base}.zip",
            mime="application/zip",
            type="primary",
            key="drift_download_zip",
            **_BUTTON_WIDTH,
        )

    # One folder name per assessment: remembered against the run key so a rerun
    # of the page does not invent a new timestamp under the user's feet. The
    # clock is Toronto's, and falls back gracefully where no tz database exists.
    stamps = st.session_state.setdefault("drift_folder_stamps", {})
    stamp = stamps.setdefault(run_key, drift_storage.run_stamp())

    st.divider()
    save_to_project_folder(
        "drift",
        lambda folder: write_drift_dir(result, settings, baseline, current, folder),
        target_name=f"drift_{stamp}",
        note=(
            "Writes the same `metadata.json`, CSVs, `df_to_img` PNGs and per-feature charts "
            "as the ZIP, unpacked into one folder."
        ),
        storage_module=drift_storage,
    )


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
def render_profiling_app() -> None:
    """Render the original profiling experience inside the unified app."""
    st.title("📊 Data Profiler")
    st.caption(
        "Exploratory data analysis and data-quality profiling with reproducible, "
        "exportable evidence."
    )

    with st.expander("How to use profiling"):
        st.markdown(
            """
            **1 · Load a dataset.** Upload a delimited file or pick one of the bundled examples.
            The delimiter is detected from the header, and the file is loaded with only lossless
            dtype conversions so every value matches the source exactly.

            **2 · Choose features and analyses.** Descriptive statistics, missing values,
            correlation, distributions and outliers can each be switched on or off.

            **3 · Run the profiling.** Results are computed once and cached. Switching sections,
            paging through charts and building exports never recompute the analysis.

            **4 · Export.** The PDF and ZIP contain the same numbers as the screen, plus a
            methodology page, `metadata.json`, and the raw values behind every chart. The same
            results can also be written straight into `results/profiling/` beside `app.py`.

            Correlations use pairwise-complete observations. Quantiles use NumPy/pandas linear
            interpolation, and outliers use the standard 1.5 × IQR fences. Constant features have
            undefined correlation and are left blank rather than reported as zero.
            """
        )

    loaded = dataset_controls()
    if loaded is None:
        st.info("Upload a file or choose an example dataset in the sidebar to begin.", icon="👈")
        render_footer("profiling")
        st.stop()

    settings, run = analysis_controls(loaded)
    st.success(
        f"**{loaded.source_name}** · {loaded.n_rows:,} rows × {loaded.n_cols:,} columns · "
        f"{loaded.memory_after_mb:,.1f} MB in memory"
    )

    with st.expander("Preview the raw data"):
        n_preview = st.slider("Rows to preview", 5, 100, 10, key="profiling_preview_rows")
        index_label = _available_column_label(loaded.frame.columns, "Row")
        preview = loaded.frame.head(n_preview).rename_axis(index_label).reset_index()
        professional_table(
            preview,
            height=min(600, 60 + 35 * len(preview)),
            label="Raw data preview",
            # A raw file can be very wide; scroll rather than squeeze.
            scroll=True,
        )

    if not settings.selected_features:
        st.warning("Select at least one column in the sidebar.", icon="⚠️")
        render_footer("profiling")
        st.stop()

    key = settings_key(settings, loaded.token)
    if st.session_state.get("profiling_active_token") != loaded.token:
        st.session_state.pop("profiling_active_run", None)
        st.session_state.pop("profiling_pdf", None)
        st.session_state.pop("profiling_zip", None)
        st.session_state["profiling_active_token"] = loaded.token

    if run:
        st.session_state["profiling_active_run"] = (key, settings)

    active = st.session_state.get("profiling_active_run")
    if active is None:
        st.info(
            "Press **▶ Run profiling** at the end of the sidebar to analyse the current selection.",
            icon="👈",
        )
        render_footer("profiling")
        st.stop()

    active_key, active_settings = active
    if active_key != key:
        stale_results_banner("profiling settings", "▶ Run profiling")

    with st.spinner("Profiling the dataset…"):
        result = profile(
            loaded.frame,
            active_settings,
            active_key,
            loaded.source_name,
            loaded.original_dtypes,
            {
                "parser": loaded.parser,
                "delimiter": loaded.delimiter,
                "token": loaded.token,
                "memory_saved_mb": round(loaded.memory_saved_mb, 4),
            },
        )
    settings = active_settings

    sections: list[tuple[str, object]] = [("Overview", lambda: overview_section(result, loaded))]
    if settings.include_describe:
        sections.append(("Descriptive statistics", lambda: describe_section(result)))
    if settings.include_missing:
        sections.append(("Missing values", lambda: missing_section(result)))
    if settings.include_correlation:
        sections.append(("Correlation", lambda: correlation_section(result)))
    if settings.include_histograms:
        sections.append(("Distributions", lambda: distributions_section(result)))
    if settings.include_box_plots:
        sections.append(("Outliers", lambda: outliers_section(result)))
    sections.append(("Download the report", lambda: download_section(result, active_key)))

    # With a wide selection, every section open at once is a page nobody can
    # navigate. The content is still built, so opening one is instant.
    expanded = len(settings.selected_features) <= COLLAPSE_SECTIONS_ABOVE
    st.divider()
    if not expanded:
        st.caption(
            f"{len(settings.selected_features):,} features selected — the sections below the "
            "overview start collapsed. Open any one of them; the charts inside are already "
            "rendered."
        )
    for index, (label, render) in enumerate(sections, start=1):
        # The overview is the cheap orientation panel, so it stays open even when
        # a wide selection collapses everything below it.
        with st.expander(f"{index}. {label}", expanded=expanded or label == "Overview"):
            render()
    render_footer("profiling")


def render_drift_app() -> None:
    """Render a guided baseline-versus-current drift assessment."""
    st.title("〽️ Drift Assessment")
    st.caption(
        "Compare a trusted baseline with current data, isolate changing features, and export "
        "the evidence behind every conclusion."
    )
    with st.expander("How to use drift assessment"):
        st.markdown(
            """
            **1 · Choose two datasets.** The bundled example is ready by default. For your own
            data, provide a baseline/reference file and a current/monitored file.

            **2 · Define snapshots.** Compare the current file as one population, or split it
            over time using a date column — monthly by default, and daily, weekly, quarterly
            or yearly if you prefer. The date column may be text, a number such as `201903`,
            a quarter label like `2019Q1`, or a timestamp; it is parsed either way.

            **3 · Choose features and measures.** PSI is a practical distribution-shift screen;
            KS adds a statistical test for numeric features; Wasserstein and energy distance
            quantify effect size in the feature's own units. Chi-square is intended for
            categorical features.

            **4 · Review and export.** The summary prioritizes PSI signals, while the detailed
            sections retain every raw metric, p-value, descriptive trend, and missingness result.
            The ZIP contains CSV and `df_to_img` PNG versions of every table, and the same
            bundle can be written straight into `results/drift/` beside `app.py`.
            """
        )

    pair = drift_dataset_controls()
    if pair is None:
        st.info(
            "Choose the bundled example or upload both a baseline and current file to begin.",
            icon="👈",
        )
        render_footer("drift")
        st.stop()
    baseline, current, default_ohe, is_example = pair

    if baseline.n_rows == 0 or current.n_rows == 0:
        st.error("Both datasets must contain at least one data row.")
        render_footer("drift")
        st.stop()

    controlled = drift_analysis_controls(
        baseline,
        current,
        default_ohe,
        is_example=is_example,
    )
    if controlled is None:
        render_footer("drift")
        st.stop()
    settings, baseline_only, current_only, run = controlled

    st.success(
        f"**{baseline.source_name}** → **{current.source_name}** · "
        f"{baseline.n_rows:,} baseline rows · {current.n_rows:,} current rows · "
        f"{len(settings.features):,} selected features"
    )
    # Right at the top of the page, where it cannot be missed: a column that only
    # one file has is usually a typo or a renamed field, and it silently shrinks
    # the analysis.
    column_conflict_warning(baseline_only, current_only, baseline, current)

    with st.expander("Preview the two datasets"):
        n_preview = st.slider("Rows to preview", 3, 50, 5, key="drift_preview_rows")
        baseline_tab, current_tab = st.tabs(["Baseline / reference", "Current / monitored"])
        with baseline_tab:
            preview = baseline.frame.head(n_preview)
            professional_table(
                preview.rename_axis(_available_column_label(preview.columns, "Row")).reset_index(),
                height=min(420, 60 + 35 * len(preview)),
                label="Baseline preview",
                scroll=True,
            )
        with current_tab:
            preview = current.frame.head(n_preview)
            professional_table(
                preview.rename_axis(_available_column_label(preview.columns, "Row")).reset_index(),
                height=min(420, 60 + 35 * len(preview)),
                label="Current preview",
                scroll=True,
            )

    if not settings.features:
        st.warning("Select at least one shared feature.", icon="⚠️")
        render_footer("drift")
        st.stop()
    if not settings.methods:
        st.warning("Select at least one distance measure.", icon="⚠️")
        render_footer("drift")
        st.stop()

    key = json.dumps(
        {
            "baseline": baseline.token,
            "current": current.token,
            **settings.as_dict(),
        },
        sort_keys=True,
        default=str,
    )
    pair_token = (baseline.token, current.token)
    if st.session_state.get("drift_active_pair") != pair_token:
        st.session_state.pop("drift_active_run", None)
        st.session_state.pop("drift_export", None)
        st.session_state["drift_active_pair"] = pair_token

    if run:
        st.session_state["drift_active_run"] = (key, settings)

    active = st.session_state.get("drift_active_run")
    if active is None:
        st.info(
            "Press **▶ Run drift assessment** at the end of the sidebar to compare the selected "
            "populations.",
            icon="👈",
        )
        render_footer("drift")
        st.stop()

    active_key, active_settings = active
    if active_key != key:
        stale_results_banner("drift settings", "▶ Run drift assessment")

    try:
        with st.spinner("Assessing distribution drift…"):
            result = assess_drift(
                current.frame,
                baseline.frame,
                active_key,
                active_settings.features,
                active_settings.methods,
                date_column=active_settings.date_column,
                interval=active_settings.interval,
                ohe_columns=active_settings.ohe_columns,
                psi_buckets=active_settings.psi_buckets,
                psi_bucket_type=active_settings.psi_bucket_type,
                descriptive_measures=active_settings.descriptive_measures,
                include_missing=active_settings.include_missing,
                score_column=active_settings.score_column,
                sample_percent=active_settings.sample_percent,
            )
    except Exception as error:  # noqa: BLE001 - surfaced to the user
        st.error(f"The drift assessment could not be completed: {error}")
        st.caption(
            "Check the selected date column, shared feature types, one-hot prefixes, and whether "
            "both datasets contain usable non-null observations."
        )
        render_footer("drift")
        st.stop()

    settings = active_settings
    sections: list[tuple[str, object]] = [
        (
            "Overview",
            lambda: drift_overview_section(result, baseline, current, settings),
        ),
        ("Drift summary", lambda: drift_summary_section(result)),
        ("Distance measures", lambda: drift_distances_section(result)),
    ]
    if settings.include_descriptives:
        sections.append(("Descriptive trends", lambda: drift_descriptives_section(result)))
    if settings.include_missing:
        sections.append(("Missing values", lambda: drift_missing_section(result)))
    # Last of the assessments, matching its position at the end of the sidebar,
    # and still ahead of the download section.
    if settings.include_score:
        sections.append(("Model score", lambda: drift_score_section(result)))
    sections.append(
        (
            "Download the report",
            lambda: drift_download_section(
                result,
                settings,
                baseline,
                current,
                active_key,
            ),
        )
    )

    expanded = len(settings.features) <= COLLAPSE_SECTIONS_ABOVE
    st.divider()
    if not expanded:
        st.caption(
            f"{len(settings.features):,} features selected — the sections below the overview "
            "start collapsed. Open any one of them; the charts inside are already rendered."
        )
    for index, (label, render) in enumerate(sections, start=1):
        # The overview is the cheap orientation panel, so it stays open even when
        # a wide selection collapses everything below it.
        with st.expander(f"{index}. {label}", expanded=expanded or label == "Overview"):
            render()
    render_footer("drift")


workspace = st.sidebar.radio(
    "Analysis workspace",
    ("Profiling", "Drift assessment"),
    index=0,
    key="workspace",
    help="Switch between single-dataset profiling and baseline-versus-current drift analysis.",
)
st.sidebar.divider()

if workspace == "Profiling":
    render_profiling_app()
else:
    render_drift_app()

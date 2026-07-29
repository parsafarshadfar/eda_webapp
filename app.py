"""Data Profiler Dashboard.

A Streamlit front end over the :mod:`profiling` package. The expensive work
happens once per configuration and is cached; everything the page renders --
tables, interactive charts, the PDF and the ZIP -- is derived from that single
cached result, so switching tabs or paging through charts never recomputes
anything.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from profiling import plots
from profiling.dataio import LoadedData, load_table
from profiling.pipeline import ProfilingResult, ProfilingSettings, run_profiling
from profiling.report import build_pdf, build_zip, suggested_basename
from profiling.summaries import EQUAL_WIDTH, KIND_CATEGORICAL, QUANTILE

EXAMPLE_DATASETS = {
    "California Housing (train)": "Examples/Data/HousingData_TrainData.csv",
    "California Housing (test)": "Examples/Data/HousingData_TestData.csv",
    "Breast Cancer (train)": "Examples/Data/BreastCancer_TrainData.csv",
}

CHARTS_PER_PAGE = 9
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


st.set_page_config(
    page_title="Data Profiler Dashboard",
    layout="wide",
    page_icon="📊",
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
_TABLE_WIDTH = _stretch(st.dataframe)
_BUTTON_WIDTH = _stretch(st.download_button)


def chart(figure, key: str | None = None) -> None:
    """Render a Plotly figure at full container width."""
    st.plotly_chart(figure, config=PLOTLY_CONFIG, key=key, **_CHART_WIDTH)


def table(frame, **kwargs) -> None:
    """Render a dataframe at full container width."""
    st.dataframe(frame, **_TABLE_WIDTH, **kwargs)


st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
      [data-testid="stSidebar"] .stSlider {padding-bottom: 0.4rem;}
      div[data-testid="stExpander"] details summary p {font-weight: 600;}
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
@st.cache_resource(show_spinner=False, max_entries=3)
def load_example(path: str, optimize: bool) -> LoadedData:
    return load_table(path, optimize=optimize)


@st.cache_resource(show_spinner=False, max_entries=3)
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


def settings_key(settings: ProfilingSettings, token: str) -> str:
    """Stable identity for a (dataset, settings) pair, used as a cache key."""
    return json.dumps({"token": token, **asdict(settings)}, sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
def dataset_controls() -> LoadedData | None:
    st.sidebar.header("1 · Dataset")
    source = st.sidebar.radio(
        "Source",
        ("Upload a file", "Example dataset"),
        horizontal=True,
        label_visibility="collapsed",
    )

    optimize = st.sidebar.toggle(
        "Optimise memory on load",
        value=True,
        help=(
            "Applies only lossless conversions: integers are narrowed to the smallest type that "
            "still represents every value exactly, and repetitive text columns become categories. "
            "Floating point precision is never reduced."
        ),
    )

    if source == "Upload a file":
        upload = st.sidebar.file_uploader(
            "CSV, TSV or delimited text", type=["csv", "tsv", "txt"], label_visibility="collapsed"
        )
        if upload is None:
            return None
        key = f"{upload.name}:{upload.size}:{getattr(upload, 'file_id', '')}"
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


def analysis_controls(loaded: LoadedData) -> ProfilingSettings:
    columns = [str(column) for column in loaded.frame.columns]

    st.sidebar.header("2 · Features")
    if st.sidebar.checkbox("Select all", value=True):
        selected = columns
        st.sidebar.caption(f"All {len(columns)} columns selected.")
    else:
        selected = st.sidebar.multiselect(
            "Columns to analyse", columns, default=columns[: min(len(columns), 12)]
        )

    st.sidebar.header("3 · Analyses")
    include_describe = st.sidebar.checkbox("Descriptive statistics", value=True)
    include_missing = st.sidebar.checkbox("Missing values", value=True)
    include_correlation = st.sidebar.checkbox("Correlation", value=True)
    include_histograms = st.sidebar.checkbox("Distributions", value=True)
    include_box = st.sidebar.checkbox("Outliers (box plots)", value=True)

    methods: tuple[str, ...] = ()
    threshold = 0.65
    if include_correlation:
        chosen = st.sidebar.multiselect(
            "Correlation methods", ["Pearson", "Spearman", "Kendall"], default=["Kendall"]
        )
        methods = tuple(method.lower() for method in chosen)
        threshold = st.sidebar.slider("Report pairs above |r|", 0.0, 1.0, 0.65, 0.01)

    st.sidebar.header("4 · Binning")
    binning_label = st.sidebar.radio(
        "Method", ["Equal-width", "Quantile"], horizontal=True, label_visibility="collapsed"
    )
    binning = EQUAL_WIDTH if binning_label == "Equal-width" else QUANTILE
    n_bins = st.sidebar.slider("Number of bins", 2, 100, 8)
    show_percentage = st.sidebar.toggle("Show distributions as percentages", value=False)

    with st.sidebar.expander("Performance", expanded=False):
        st.caption(
            "Kendall's tau is exact but costs O(n log n) per feature pair. On very large files a "
            "seeded subsample keeps it usable; the sample size and seed are recorded in the report."
        )
        limit_rows = st.checkbox("Subsample rows for correlation", value=False)
        max_rows = None
        random_state = 0
        if limit_rows:
            max_rows = int(
                st.number_input(
                    "Maximum rows",
                    min_value=1_000,
                    max_value=max(1_000, int(loaded.n_rows)),
                    value=int(min(100_000, max(1_000, loaded.n_rows))),
                    step=1_000,
                )
            )
            random_state = int(st.number_input("Random seed", min_value=0, value=0, step=1))

    return ProfilingSettings(
        selected_features=tuple(selected),
        include_describe=include_describe,
        include_missing=include_missing,
        include_correlation=include_correlation,
        include_histograms=include_histograms,
        include_box_plots=include_box,
        correlation_methods=methods,
        correlation_threshold=float(threshold),
        correlation_max_rows=max_rows,
        binning=binning,
        n_bins=int(n_bins),
        show_percentage=bool(show_percentage),
        random_state=random_state,
    )


# --------------------------------------------------------------------------- #
# result rendering
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
        table(types, hide_index=True, height=320)

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
            table(
                pd.DataFrame(
                    {
                        "Column": list(loaded.optimizations),
                        "Change": list(loaded.optimizations.values()),
                    }
                ),
                hide_index=True,
            )

        st.subheader("Step timings")
        timings = pd.DataFrame(
            {
                "Step": [key.replace("_", " ").title() for key in result.timings],
                "Seconds": [round(value, 3) for value in result.timings.values()],
            }
        )
        table(timings, hide_index=True)


def describe_section(result: ProfilingResult) -> None:
    if result.describe is None or result.describe.empty:
        st.info("No numeric features were selected.")
        return

    st.caption(
        "Percentages are relative to the total row count. The most frequent value counts nulls "
        "as a candidate, so a mostly-empty column reports a null mode."
    )
    formats = {
        "Perc. of Missing": "{:.3f}%",
        "Perc. of Zeros": "{:.3f}%",
        "Perc. of 1st Most Freq": "{:.3f}%",
        "1st Most Freq": "{:,.4g}",
        "Min": "{:,.4g}",
        "1%": "{:,.4g}",
        "50%": "{:,.4g}",
        "99%": "{:,.4g}",
        "Max": "{:,.4g}",
        "n_uniques (excl. Nulls)": "{:,.0f}",
        "n_Missing": "{:,.0f}",
        "n_Zeros": "{:,.0f}",
    }
    styled = result.describe.style.format(
        {k: v for k, v in formats.items() if k in result.describe}, na_rep="-"
    )
    table(styled, height=min(680, 60 + 35 * len(result.describe)))


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
    table(
        frame.rename(
            columns={"number_of_missing": "Missing rows", "percentage_of_missing": "Missing (%)"}
        ).style.format({"Missing rows": "{:,.0f}", "Missing (%)": "{:.4f}%"}),
        height=min(600, 60 + 35 * len(frame)),
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
            table(
                top.style.format({value_column: "{:.4f}"}),
                hide_index=True,
                height=min(480, 60 + 35 * len(top)),
            )

        if st.checkbox("Show the full correlation matrix", value=False, key=f"matrix_{method}"):
            if matrix.shape[0] <= 40:
                table(
                    matrix.style.format("{:.3f}", na_rep="-").background_gradient(
                        cmap="RdBu_r", vmin=-1, vmax=1
                    )
                )
            else:
                table(matrix.round(4))


@st.fragment
def _chart_pager(items: list, render, *, label: str, state_key: str) -> None:
    """Paged chart list.

    Marked as a fragment so paging redraws only this block instead of rerunning
    the whole script, and so only the charts on the current page are ever sent
    to the browser.
    """
    if not items:
        st.info(f"No {label} to display for the current selection.")
        return

    names = [item.feature for item in items]
    chosen = st.multiselect(
        f"Features ({len(names)} available)",
        names,
        default=names,
        key=f"{state_key}_features",
    )
    visible = [item for item in items if item.feature in set(chosen)]
    if not visible:
        st.info("Select at least one feature.")
        return

    n_pages = (len(visible) + CHARTS_PER_PAGE - 1) // CHARTS_PER_PAGE
    page = 1
    if n_pages > 1:
        page = st.slider(
            f"Page (showing {CHARTS_PER_PAGE} of {len(visible)} at a time)",
            1,
            n_pages,
            1,
            key=f"{state_key}_page",
        )
    start = (page - 1) * CHARTS_PER_PAGE
    render(visible[start : start + CHARTS_PER_PAGE], start)


def distributions_section(result: ProfilingResult) -> None:
    summaries = list(result.distributions.values())
    settings = result.settings
    st.caption(
        f"{settings.binning.replace('-', ' ').capitalize()} binning with {settings.n_bins} bins. "
        "Charts are drawn from pre-computed bin counts, so the payload sent to the browser does "
        "not grow with the size of the dataset."
    )

    def render(items, offset: int) -> None:
        columns = st.columns(3)
        for index, summary in enumerate(items):
            with columns[index % 3]:
                chart(
                    plots.histogram_figure(
                        summary,
                        color=plots.color_for(offset + index),
                        show_percentage=settings.show_percentage,
                    ),
                    key=f"hist_{summary.feature}_{offset + index}",
                )

    n_categorical = sum(1 for s in summaries if s.kind == KIND_CATEGORICAL)
    if n_categorical:
        plural = "s are" if n_categorical != 1 else " is"
        st.caption(f"{n_categorical} categorical feature{plural} shown as value counts.")
    _chart_pager(summaries, render, label="distributions", state_key="dist")


def outliers_section(result: ProfilingResult) -> None:
    summaries = list(result.box_summaries.values())
    st.caption(
        "Whiskers reach the most extreme observation within 1.5 x IQR of the quartiles. Outlier "
        "counts are exact; when a feature has thousands of them an evenly spaced subset is drawn."
    )

    if summaries:
        flagged = [s for s in sorted(summaries, key=lambda s: -s.n_outliers)[:5] if s.n_outliers]
        if flagged:
            details = ", ".join(
                f"`{s.feature}` ({s.n_outliers:,}, {100.0 * s.n_outliers / max(s.n_valid, 1):.2f}%)"
                for s in flagged
            )
            st.markdown(f"**Most outlier-heavy features:** {details}")
        else:
            st.markdown("**No feature has values beyond 1.5 x IQR.**")

    def render(items, offset: int) -> None:
        columns = st.columns(2)
        for index, summary in enumerate(items):
            with columns[index % 2]:
                chart(
                    plots.box_figure(summary, color=plots.color_for(offset + index)),
                    key=f"box_{summary.feature}_{offset + index}",
                )

    _chart_pager(summaries, render, label="box plots", state_key="box")


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
                st.session_state["pdf"] = (cache_key, build_pdf(result))
        stored = st.session_state.get("pdf")
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
                cached = st.session_state.get("pdf")
                # Reuse the PDF when it has already been built: rendering it is
                # the most expensive single part of the export.
                payload = cached[1] if cached and cached[0] == cache_key else build_pdf(result)
                st.session_state["pdf"] = (cache_key, payload)
                st.session_state["zip"] = (
                    (cache_key, per_feature),
                    build_zip(result, pdf_bytes=payload, include_individual_charts=per_feature),
                )
        stored = st.session_state.get("zip")
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
├── 04_distributions/               bin edges and counts (CSV) + charts
└── 05_outliers/                    quartiles, fences, counts (CSV) + charts""",
            language="text",
        )


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
st.title("📊 Data Profiler Dashboard")
st.caption(
    "Exploratory data analysis and data-quality profiling with reproducible, exportable evidence."
)

with st.expander("How to use this dashboard"):
    st.markdown(
        """
        **1 · Load a dataset.** Upload a delimited file or pick one of the bundled examples. The
        delimiter is detected from the header, and the file is loaded with only lossless dtype
        conversions so every value matches the source exactly.

        **2 · Choose features and analyses.** Descriptive statistics, missing values, correlation,
        distributions and outliers can each be switched on or off.

        **3 · Run the profiling.** Results are computed once and cached. Switching tabs, paging
        through charts and building exports never recompute the analysis.

        **4 · Export.** The PDF and ZIP contain the same numbers as the screen, plus a methodology
        page, a machine-readable `metadata.json` and the raw values behind every chart.

        ---

        **A note on the statistics.** Correlations use pairwise-complete observations, matching
        `pandas.DataFrame.corr`. Quantiles use linear interpolation. Outlier fences are the
        standard 1.5 x IQR rule. A constant feature has an undefined correlation and is reported
        blank rather than zero. Every definition is repeated on the methodology page of the export.
        """
    )

loaded = dataset_controls()

if loaded is None:
    st.info("Upload a file or choose an example dataset in the sidebar to begin.", icon="👈")
    st.stop()

settings = analysis_controls(loaded)

top_left, top_right = st.columns([3, 1])
with top_left:
    st.success(
        f"**{loaded.source_name}** · {loaded.n_rows:,} rows × {loaded.n_cols:,} columns · "
        f"{loaded.memory_after_mb:,.1f} MB in memory"
    )
with top_right:
    run = st.button("Run profiling", type="primary", **_BUTTON_WIDTH)

with st.expander("Preview the raw data"):
    n_preview = st.slider("Rows to preview", 5, 100, 10, key="preview_rows")
    table(loaded.frame.head(n_preview))

if not settings.selected_features:
    st.warning("Select at least one column in the sidebar.", icon="⚠️")
    st.stop()

key = settings_key(settings, loaded.token)

# A completed run is remembered so that touching a sidebar widget does not wipe
# the page. Results from a different dataset are dropped rather than shown,
# because a stale table next to a new file would be actively misleading.
if st.session_state.get("active_token") != loaded.token:
    st.session_state.pop("active_run", None)
    st.session_state["active_token"] = loaded.token

if run:
    st.session_state["active_run"] = (key, settings)

active = st.session_state.get("active_run")
if active is None:
    st.info("Press **Run profiling** to analyse the current selection.", icon="▶️")
    st.stop()

active_key, active_settings = active
if active_key != key:
    st.warning(
        "Showing the previous results — the settings have changed since this run. "
        "Press **Run profiling** to refresh.",
        icon="⚠️",
    )

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

# Sections are stacked down the page as expanders, open by default, so the
# whole profile can be read in one pass and any section collapsed out of the
# way. Streamlit does not allow an expander inside an expander, which is why
# the secondary panels inside each section are checkboxes rather than nested
# expanders.
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

st.divider()
for index, (label, render) in enumerate(sections, start=1):
    with st.expander(f"{index}. {label}", expanded=True):
        render()

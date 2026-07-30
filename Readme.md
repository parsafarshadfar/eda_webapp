# 📊 Data Profiler Dashboard

A Streamlit application for exploratory data analysis and data-quality profiling, built to
produce **reproducible, exportable evidence** rather than throwaway charts. Every number on
screen can be downloaded as a paginated PDF or as a ZIP of the underlying tables, together
with a record of the exact settings and library versions that produced it.

![An overview of the EDA Webapp](./screenshot.png)

---

## Features

**Analysis**

- **Descriptive statistics** — dtype, distinct values, missing and zero counts, most frequent
  value, and the minimum, 1st, 50th, 99th percentiles and maximum for every column.
- **Missing-value analysis** — counts and percentages per column, with a ranked chart.
- **Correlation** — Pearson, Spearman and Kendall matrices, plus the ranked list of feature
  pairs whose absolute coefficient exceeds a threshold.
- **Distributions** — histograms with equal-width or quantile binning; categorical columns
  get value-count plots.
- **Outliers** — box plots with quartiles, Tukey fences and exact outlier counts.

**Export**

- **PDF report** — cover page, dataset overview, one section per analysis, a methodology page
  defining every statistic, and a reproducibility page listing settings and library versions.
  Text stays selectable and searchable.
- **ZIP archive** — numbered folders containing the CSV behind every chart, an Excel workbook
  of the correlations, PNG charts, the PDF, and a machine-readable `metadata.json` carrying the
  dataset fingerprint, settings, step timings and versions.
- **Folder** — the same bundle written straight to disk instead of into an archive, for a
  notebook or a batch job. The ZIP and the folder share one definition of their contents, so an
  unpacked archive and a written folder are the same tree.

The interactive charts in the browser are Plotly; the exported charts are Matplotlib. Both are
rendered from the *same* pre-aggregated summaries, so an exported page can never disagree with
what was shown on screen.

**Beyond the dashboard** — the engine is a plain Python package with no Streamlit import, so a
notebook or script gets everything above plus **segmentation** (profile each segment of a dataset
separately, by value, interval, equal-width bin or quantile), **dataset comparison** and
**one-hot-encoding collapse**. See [Using the package directly](#using-the-package-directly).

---

## How to use

1. **Load a dataset** — upload a delimited file or pick one of the bundled examples. The
   delimiter is detected from the header line.
2. **Choose features and analyses** — each of the five analyses can be switched on or off.
3. **Run profiling** — results are computed once and cached. Switching sections, paging through
   charts and building exports never recompute the analysis.
4. **Export** — build the PDF or the ZIP from the last section of the page.

---

## Installation

```bash
git clone https://github.com/parsafarshadfar/eda_webapp.git
cd eda_webapp
pip install -r requirements.txt
```

Requires **Python 3.12 or 3.13**.

## Running

```bash
streamlit run app.py
```

If `streamlit` is not on your `PATH`:

```bash
python -m streamlit run app.py
```

---

## Project layout

```
app.py                     Streamlit user interface
profiling/                 analysis engine — the whole toolkit
├── dataio.py              loading, delimiter detection, lossless memory tuning,
│                          stacking frames, collapsing one-hot encodings
├── stats.py               descriptive statistics, missing values, correlation
├── summaries.py           pre-aggregated distribution and outlier summaries
├── grouping.py            splitting a dataset into segments
├── plots.py               interactive Plotly figures
├── static.py              static Matplotlib figures
├── pipeline.py            orchestration — one run, or one run per segment
├── report.py              PDF, ZIP and folder export
└── _spark.py              optional Spark bridge (no Spark dependency)
Examples/Data/             sample datasets
Examples/Profiling_walkthrough.ipynb
                           annotated notebook covering every feature
```

`profiling` is the single source of truth. It has no Streamlit import anywhere, so everything the
dashboard can do is also available from a notebook, a script or a Databricks job — and so is a
good deal it does not expose, chiefly segmentation.

---

## Methodology

The definitions below are repeated on the methodology page of every exported report.

| Statistic | Definition |
|---|---|
| **Quantiles** | Linear interpolation between the two bracketing order statistics — the convention used by `numpy.quantile` and `pandas.Series.quantile`. |
| **Missing** | A value pandas reports as null (`NaN`, `NaT`, `None`). Percentages are relative to the total row count, not the non-null count. |
| **Most frequent value** | Nulls count as a candidate. Ties resolve to the smallest value, so repeated runs agree. |
| **Correlation** | Pearson on raw values; Spearman as Pearson on average ranks; Kendall as tau-b with tie correction. Pairs use rows where both features are present (pairwise-complete), matching `pandas.DataFrame.corr`. A constant feature gives an undefined coefficient and is left blank rather than reported as zero. |
| **Equal-width bins** | The observed range, padded by 0.1% at each end so the maximum falls inside the last bin, split into equal intervals. |
| **Quantile bins** | Empirical quantiles with duplicate edges dropped; right-closed intervals with the lowest edge included. |
| **Outliers** | A value further than 1.5 × the interquartile range (IQR, the distance from the first to the third quartile) beyond the nearer quartile. Each box plot spans Q1 to Q3 and reaches the most extreme observation still inside the fences. Outlier counts are exact. |

**Subsampling.** Correlation can optionally be computed on a subsample of rows. The draw is
seeded and applies to *every* selected method, all of which are computed from the same sampled
rows, so the methods stay comparable. The sample size and seed are recorded in the report.

**Memory optimisation.** Loading applies only lossless dtype conversions: integers are narrowed
to the smallest type that represents every value exactly, and repetitive text columns become
categories. Floating-point precision is never reduced.

---

## Performance notes

- Descriptive statistics read cardinality, mode, zeros, min/max and all quantiles from a
  **single sorted copy** of each column instead of making a separate pass per statistic.
- Correlation methods share one extracted matrix. With no missing values, Pearson and Spearman
  reduce to a single BLAS matrix product; Kendall is spread over a thread pool.
- Charts are built from pre-aggregated summaries, so the payload sent to the browser does not
  grow with the number of rows.
- CSV loading prefers PyArrow's multi-threaded parser, falling back to the pandas C parser.

---

## Using the package directly

The dashboard is one front end onto `profiling`; a notebook or a script is another. Import the
package and the whole engine is available:

```python
import profiling

loaded = profiling.load_table("data.csv")
settings = profiling.ProfilingSettings(correlation_methods=("pearson", "spearman"))

result = profiling.run_profiling(loaded.frame, settings, dataset_name="data.csv")
profiling.write_report_dir(result, "Results_of_DataProfiling")   # or build_pdf / build_zip
```

Importing `profiling` costs pandas and NumPy only — Matplotlib and Plotly load on first use, so a
numbers-only script does not pay for them.

**[`Examples/Profiling_walkthrough.ipynb`](Examples/Profiling_walkthrough.ipynb)** is an annotated,
runnable tour of every feature: loading and provenance, each analysis on its own, both figure
back ends, the one-call pipeline, all three export shapes, and everything below.

### Segmentation

The main capability the dashboard does not expose. A dataset is split into segments and every
analysis is run per segment, with identical settings so the numbers stay comparable:

```python
segmented = profiling.run_segmented_profiling(
    df,
    settings,
    by="Age",              # column name, Series, or array
    n_quantiles=4,         # or bins=[(1, 20), (20, 50)] / n_bins=4 / nothing, for distinct values
)

segmented.segment_overview()               # rows, features and time per segment
segmented.describe_comparison()            # long format, one 'segment' column in front
segmented.correlation_comparison("pearson")  # one row per pair, one column per segment
segmented["(20.0, 35.0]"]                  # a plain ProfilingResult, exportable on its own

profiling.write_segmented_report_dir(segmented, "Results_of_DataProfiling")
```

`profiling.make_grouping(...)` builds the split on its own if you only want the row indices — it
holds no data, so it is cheap to build and reuse.

### Comparing datasets

`combine_datasets` stacks a mapping of frames into one, tagged by origin. Segmenting on the tag
turns every comparison table above into a train/test drift report:

```python
both = profiling.combine_datasets({"Train": train_df, "Test": test_df})   # adds a 'Dataset' column

drift = profiling.run_segmented_profiling(both.frame, settings, by="Dataset")
drift.describe_comparison().pivot(index="feature", columns="segment", values="50%")
```

### Other preparation helpers

- `merge_one_hot_encoded_columns(df)` collapses dummy blocks back into single categorical
  columns, so the profile describes the feature rather than thirty flags.
- `optimize_memory(df)` applies the lossless dtype conversions on a frame you loaded yourself.
- `dataframe_figure(table)` renders any DataFrame as one table image.

Every static renderer — tables, histograms, box plots, missing-value charts, heatmaps and their
paginated grid builders — supports the same optional image-saving contract:

```python
profiling.dataframe_figure(table, save=True)
# -> Results_of_DataProfiling/dataframe.png

profiling.histogram_figure(summary, save="images/income_distribution.png")
# Parent folders are created automatically.
```

`save=False` (the default) only returns the Matplotlib figure, `save=True` chooses a descriptive
PNG name under `Results_of_DataProfiling/`, and `save=path` writes to the supplied address.
Multipage builders append `_01`, `_02`, and so on. Table images use the dashboard's navy-blue
palette, wrap long headers onto at most two lines, align numbers consistently and display
percentage columns to three decimal places.

### Spark and Databricks

The package has **no** Spark dependency and imports nothing Spark-related until a Spark object is
actually passed in. When one is, `make_grouping` bucketises in Spark — `Bucketizer` for
`bins`/`n_bins`, `QuantileDiscretizer` for `n_quantiles` — rather than pulling the grouping column
onto the driver, and `run_segmented_profiling` materialises one segment at a time. **Spark SQL**
and **PySpark pandas** frames are both accepted.

---

## Usage in industry

Data profiling underpins data quality work in finance, healthcare, retail and beyond: detecting
quality issues, understanding distributions, identifying outliers and anomalies, and documenting
the state of a dataset before it is used. The export bundle is designed so that a profiling run
can be attached to a review or validation file and reproduced later from the recorded settings.

---

Developed by **Parsa Farshadfar**.

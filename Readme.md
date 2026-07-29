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

The interactive charts in the browser are Plotly; the exported charts are Matplotlib. Both are
rendered from the *same* pre-aggregated summaries, so an exported page can never disagree with
what was shown on screen.

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

Requires **Python 3.12 or 3.13**. To also run `Examples/Example.ipynb`, install
`requirements-dev.txt` instead.

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
profiling/                 analysis engine
├── dataio.py              loading, delimiter detection, lossless memory tuning
├── stats.py               descriptive statistics, missing values, correlation
├── summaries.py           pre-aggregated distribution and outlier summaries
├── plots.py               interactive Plotly figures
├── static.py              static Matplotlib figures
├── pipeline.py            orchestration and the result object
└── report.py              PDF and ZIP export
DataProfiler.py            grouped/segmented profiling driver (notebook API)
DataProfiling_utils.py     helper functions (notebook API)
Examples/                  sample datasets and a usage notebook
```

`DataProfiler.py` and `DataProfiling_utils.py` keep their original public signatures and are
what the example notebook uses; internally they delegate to the `profiling` package.

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
| **Outliers** | A value further than 1.5 × the interquartile range (IQR, the distance from the first to the third quartile) beyond the nearer quartile. Whiskers reach the most extreme observation still inside the fences. Outlier counts are exact. |

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

## Notes on the notebook

Some features are available only in `Examples/Example.ipynb` and not in the web app, chiefly
**groupby** segmentation, which profiles each segment of a dataset separately. The notebook also
supports **Spark SQL** and **PySpark pandas** DataFrames and has been tested on Databricks.

---

## Usage in industry

Data profiling underpins data quality work in finance, healthcare, retail and beyond: detecting
quality issues, understanding distributions, identifying outliers and anomalies, and documenting
the state of a dataset before it is used. The export bundle is designed so that a profiling run
can be attached to a review or validation file and reproduced later from the recorded settings.

---

Developed by **Parsa Farshadfar**.

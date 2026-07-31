# Data profiling engine

`profiling` computes and renders a full exploratory profile of a dataset: descriptive statistics,
missing values, correlation, distributions and outliers — for the whole table, or once per segment
of it. It is the calculation layer behind the **Profiling** workspace in [`app.py`](../app.py) and
is equally usable from a notebook, a script, a batch job or a Databricks cluster.

Nothing here imports Streamlit, and importing the package costs pandas and NumPy only: Matplotlib
and Plotly load the first time a figure or export name is touched.

## Modules

| Module | What lives there |
|---|---|
| `dataio` | loading, delimiter detection, lossless memory tuning, stacking frames, collapsing one-hot encodings |
| `stats` | descriptive statistics, missing values, correlation |
| `summaries` | pre-aggregated distribution and outlier summaries |
| `grouping` | splitting a dataset into segments |
| `plots` | interactive Plotly figures |
| `static` | static Matplotlib figures, including `df_to_img` |
| `pipeline` | orchestration — one run, or one run per segment |
| `report` | PDF, ZIP and folder export |
| `storage` | where results are saved; single-shot writing to local folders, DBFS and ADLS |
| `_spark` | optional Spark bridge (no Spark dependency) |

The rule that holds it together: **every chart, interactive or exported, is drawn from the same
pre-aggregated summary**, never from the raw column. A Plotly chart on screen and a Matplotlib
chart in the PDF therefore cannot disagree, and the payload per chart is a few hundred bytes
regardless of how many rows the file has.

## Quick start

```python
import profiling

loaded = profiling.load_table("profiling_examples/Data/HousingData_TrainData.csv")
settings = profiling.ProfilingSettings(correlation_methods=("pearson", "spearman"))

result = profiling.run_profiling(loaded.frame, settings, dataset_name="HousingData.csv")

result.describe                      # per-feature statistics
result.missing                       # counts and percentages
result.correlations["spearman"]      # full matrix
result.top_correlations["spearman"]  # ranked pairs above the threshold
result.distributions["MedInc"]       # bin edges and counts
result.box_summaries["MedInc"]       # quartiles, fences, exact outlier counts

profiling.write_report_dir(result)   # -> results/profiling/<run>/
```

Or one analysis at a time:

```python
profiling.describe_data(frame)
profiling.missing_count(frame)
profiling.correlation_matrices(frame, ("pearson", "kendall"))
profiling.build_distributions(frame, binning=profiling.QUANTILE, n_bins=10)
profiling.build_box_summaries(frame)
```

## Where results are saved

One rule for the whole repository: **results go to a `results` folder beside whatever you ran.**

| You ran | Results land in |
|---|---|
| `app.py` (Streamlit) | `results/profiling/` next to `app.py` |
| `profiling_examples/profiling_walkthrough.ipynb` | `profiling_examples/results/` |
| your own script | `results/profiling/`, until you change it |

```python
profiling.output_dir()                                # where results go right now
profiling.set_output_dir("profiling_examples/results")  # change it for the session
```

Any single call can still be sent elsewhere, and the folder is created for you:

```python
profiling.df_to_img(table, save=True)                        # the standard folder
profiling.df_to_img(table, save="reports/january/psi.png")   # this one only
profiling.histogram_figure(summary, save="reports/hist.png")
profiling.write_report_dir(result, "reports/january")
```

`save=False` (the default) returns the figure without writing anything; multipage grid builders
append `_01`, `_02`, and so on.

## Table images

`df_to_img` renders any DataFrame as a navy-blue, publication-ready image — the same blue the
drift package uses, so a report can mix the two. Long headers wrap onto at most two lines at word
boundaries, numbers are aligned and percentage columns show three decimals. Images are rendered at
240 dpi, well above screen resolution, because they get scaled up in slides and documents.
`dataframe_figure` is the same renderer with its full interface exposed, and `table_figures`
paginates a long table.

## Segmentation

The main capability the dashboard does not expose. A dataset is split into segments and every
analysis runs per segment with identical settings, so the numbers stay comparable:

```python
segmented = profiling.run_segmented_profiling(
    frame,
    settings,
    by="HouseAge",      # column name, Series or array
    n_quantiles=4,      # or bins=[(1, 20), (20, 50)] / n_bins=4 / nothing, for distinct values
)

segmented.segment_overview()                  # rows, features and time per segment
segmented.describe_comparison()               # long format, 'segment' column in front
segmented.missing_comparison(affected_only=True)
segmented.box_comparison()
segmented.correlation_comparison("spearman")  # one row per pair, one column per segment
segmented["(20.0, 35.0]"]                     # a plain ProfilingResult, exportable on its own

profiling.write_segmented_report_dir(segmented)   # one folder per segment + 00_comparison/
```

`profiling.make_grouping(...)` builds the split on its own when you only want the row positions;
it holds no data, so it is cheap to build and reuse.

## Comparing two datasets

`combine_datasets` stacks a mapping of frames into one, tagged by origin. Segmenting on that tag
turns every comparison table above into a train/test report:

```python
drift = profiling.run_segmented_profiling(
    {"Train": train_df, "Test": test_df}, settings, by="Dataset"
)
drift.describe_comparison().pivot(index="feature", columns="segment", values="50%")
```

## Exporting

Three shapes of the same bundle, sharing one definition of the contents (`bundle_files`), so an
unpacked archive and a written folder are the same tree:

```python
profiling.build_pdf(result)                      # -> bytes
profiling.build_zip(result, pdf_bytes=pdf)       # -> bytes (pass the PDF in to render it once)
profiling.write_report_dir(result)               # -> [addresses written]
profiling.write_segmented_report_dir(segmented)  # -> {segment: [addresses]}
```

The PDF carries a cover page, one section per analysis, a methodology page defining every
statistic, and a reproducibility page. Both bundles include `metadata.json` with the dataset
fingerprint, the exact settings, step timings and library versions.

Charts are exported as **one PNG per feature** under each section's `charts/` folder — a nine-up
grid is right for a report page but useless as a file, since a single feature cannot be lifted out
of it. (Beyond 200 features the grids come back, to keep the bundle a sane size.)

## Methodology

| Statistic | Definition |
|---|---|
| **Quantiles** | Linear interpolation between the two bracketing order statistics — the `numpy.quantile` / `pandas.Series.quantile` convention. |
| **Missing** | A value pandas reports as null (`NaN`, `NaT`, `None`). Percentages are relative to the total row count, not the non-null count. |
| **Most frequent value** | Nulls count as a candidate. Ties resolve to the smallest value, so repeated runs agree. |
| **Correlation** | Pearson on raw values; Spearman as Pearson on average ranks; Kendall as tau-b with tie correction. Pairs use rows where both features are present (pairwise-complete), matching `pandas.DataFrame.corr`. A constant feature gives an undefined coefficient and is left blank rather than reported as zero. |
| **Equal-width bins** | The observed range, padded by 0.1% at each end so the maximum falls inside the last bin, split into equal intervals. |
| **Quantile bins** | Empirical quantiles with duplicate edges dropped; right-closed intervals with the lowest edge included. |
| **Outliers** | A value further than 1.5 × IQR beyond the nearer quartile. Each box plot spans Q1 to Q3 and reaches the most extreme observation still inside the fences. Counts are exact even when only a subset of points is drawn. |

**Subsampling.** Correlation can be computed on a seeded subsample of rows. The draw applies to
*every* selected method, all computed from the same sampled rows, so the coefficients stay
comparable; the size and seed are recorded in the report.

**Memory optimisation.** Loading applies only lossless dtype conversions: integers are narrowed to
the smallest type that represents every value exactly, and repetitive text becomes `category`.
Floating-point precision is never reduced.

## Performance notes

- Descriptive statistics read cardinality, mode, zeros, min/max and every quantile from a **single
  sorted copy** of each column instead of one pass per statistic.
- Correlation methods share one extracted numeric matrix. With no gaps, Pearson and Spearman
  reduce to a single BLAS product; Kendall is spread over a bounded thread pool.
- Charts are built from pre-aggregated summaries, so the browser payload does not grow with the
  row count.
- CSV loading prefers PyArrow's multi-threaded parser and falls back to the pandas C parser.
- Figures use Matplotlib's object API, never `pyplot`, so nothing accumulates in a global registry
  in a long-running process.
- `sample_percent=` profiles a random share of the rows when a file is too large to profile whole.
  **100 is the default and uses everything.**

  ```python
  profiling.ProfilingSettings(sample_percent=10, random_state=0)
  ```

  The draw happens once, before anything is measured, so every table and chart in the result
  describes the same rows — nothing is computed on the full data and nothing on a different
  subset. It is seeded, and the percentage, the rows used and the seed land in `result.notes` and
  the exported metadata, so a sampled report says so on its face. It composes with
  `correlation_max_rows`, which thins only the correlation step. Shapes and correlations survive
  sampling; exact row and missing counts belong to the sample, not the file.

## Spark, Databricks, DBFS and ADLS

No Spark dependency, and nothing Spark-related is imported until a Spark object is passed in. When
one is, `make_grouping` bucketises **in Spark** (`Bucketizer` for `bins`/`n_bins`,
`QuantileDiscretizer` for `n_quantiles`) instead of pulling the grouping column onto the driver,
`combine_datasets` concatenates with `pyspark.pandas`, and `run_segmented_profiling` materialises
one segment at a time. Spark SQL and PySpark pandas frames are both accepted.

Save addresses may be cloud URIs:

```python
profiling.set_output_dir("abfss://container@account.dfs.core.windows.net/eda/profiling")
profiling.write_report_dir(result)
```

Three properties make that work behind a corporate firewall:

- **every file is finished before it is written** — tables are serialised to bytes and figures
  rendered to PNG in memory, then written with a single call. Nothing is opened once and appended
  to in pieces, which is the pattern a locked-down ADLS destination refuses;
- **cloud writes go through `dbutils.fs`** — the driver performs one `cp` of a staged file, using
  the cluster's own credentials, so Python never opens a connection to the storage account;
- **remote addresses never touch `pathlib`** — `Path("abfss://a/b")` collapses the double slash,
  so `profiling.storage` handles URIs as text.

## Learn by example

[`profiling_examples/profiling_walkthrough.ipynb`](../profiling_examples/profiling_walkthrough.ipynb)
walks through every capability in the order an analyst works, showing each result three ways: the
DataFrame, the same table as a blue image, and the address it was saved to — and each chart as
Plotly, Matplotlib and a saved PNG. It ships already executed, with an `.html` twin beside it as a
fixed record of that run, and covers the things the dashboard does not expose at all:
segmentation, dataset comparison and one-hot collapsing.

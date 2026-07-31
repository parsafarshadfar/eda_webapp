# 🔎 EDA Studio

A single Streamlit app for exploratory **data profiling** and baseline-versus-current **drift
assessment**. It is built to produce *reproducible, exportable evidence* rather than throwaway
charts: pick a workspace in the sidebar, run the analysis, and export the tables, images, settings,
fingerprints and library versions behind every number.

![The profiling workspace](./screenshot.png)

---

## Run it

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Requires **Python 3.12 or 3.13**. If `streamlit` is not on your `PATH`, use
`python -m streamlit run app.py`.

---

## The two workspaces

Choose one at the top of the sidebar.

### 📊 Profiling — one dataset

Upload a delimited file or pick a bundled example, choose features and analyses, press **Run
profiling**:

- **descriptive statistics** — dtype, distinct values, missing and zero counts, most frequent
  value, min / 1% / 50% / 99% / max;
- **missing values** — counts and percentages per column, with a ranked chart;
- **correlation** — Pearson, Spearman and Kendall matrices plus the ranked list of pairs above a
  threshold;
- **distributions** — histograms with equal-width or quantile binning; value counts for
  categorical columns;
- **outliers** — box plots with quartiles, Tukey fences and exact outlier counts.

### 〽️ Drift assessment — baseline vs current

Use the bundled example or upload a baseline/current pair, define snapshots, press **▶ Run drift
assessment**:

- **snapshots over time** — the *current* data is cut into periods and each one is compared with
  the baseline, which is never split or mixed in. Monthly by default; daily, weekly, quarterly and
  yearly are a keyword away. The date column can be text, a number such as `201903`, a quarter
  label like `2019Q1`, or a timestamp; it is parsed, not assumed;
- **PSI by default** — the one measure that covers numeric *and* categorical features. Add
  Kolmogorov–Smirnov (with p-values), Wasserstein, energy distance (the historical `CM`) or
  chi-square when you want a second opinion; a measure that does not apply to a feature's type is
  reported as unavailable instead of aborting the run;
- **risk shading, not a status column** — values are shaded amber from `0.10` and red from `0.25`
  (KS p-values at `0.10` and `0.05`), on screen *and* in the exported PNGs, with the bands
  explained next to every table;
- **one chart per feature** — drift measures live on different scales, so each feature gets its own
  chart rather than one crowded set of axes, and a PNG per feature lands in the export;
- **supporting evidence** — mean/most-frequent-value and standard-deviation trends to four
  decimals, row counts labelled baseline vs current, and missing-value counts and percentages;
- **model score drift** — optional and **off by default**, since most datasets carry no model
  output. Name a score column and its distribution is compared directly: the share of rows in
  each score band, the mean and quartiles per snapshot, and PSI of the score against the
  baseline. Bands are cut from the baseline and reused for every snapshot, so the columns are
  comparable, and every population's density is drawn on **one set of axes** so a score that has
  kept its shape is visible at a glance. The score is carried *beside* the features, never among
  them, so it is never double-counted against the model it came from;
- **one-hot groups** — `CardType_*` columns can be rebuilt into a single `CardType` feature first.

Every control carries a detailed help icon with an example, and columns that exist in only one of
the two files are called out as conflicts rather than quietly dropped.

**Nothing recomputes until you ask it to.** Every setting lives in one sidebar form, so changing
a checkbox, a slider or a feature list does not rerun anything — the results on the right stay
exactly as they were. One click on the orange **▶ Run** button at the end of the sidebar applies
the lot. Results are then cached by content fingerprint and settings, so switching sections and
building exports never recompute an analysis.

**Sampling, when a file is too big to wait for.** Both workspaces carry an *Analyse this % of
rows* slider under **Performance**, at 100% unless you move it. Below that, the run works from a
seeded random share: one draw, taken before anything is measured, so every table and chart in the
result describes the same rows. Drift samples the baseline and each snapshot separately, so no
population is crowded out by a busier one. The percentage, the rows used and the seed are recorded
in the export, and a sampled run says so on its face. Shapes and rankings survive sampling; exact
counts do not, so confirm anything you plan to act on at 100%.

**Built for wide, heavy data.** Every feature is charted, none hidden behind pagination; with more
than 12 features the sections start collapsed (their contents are already rendered, so opening one
is instant). Wide tables keep a readable column width and scroll sideways inside their own frame
rather than being squeezed or split. Drift assesses features across threads — NumPy and SciPy
release the GIL for the sorting and binning that dominate, so the cores are used without the memory
cost of extra processes, and the numbers are identical to a single-threaded run.

---

## Exporting, and where results go

Each workspace ends with an export section offering the same results three ways:

| | Profiling | Drift |
|---|---|---|
| **PDF** | cover, dataset overview, one section per analysis, a methodology page defining every statistic, a reproducibility page | — |
| **ZIP** | numbered folders with the CSV behind every chart, a correlations workbook, PNG charts, the PDF, `metadata.json` | one CSV and one shaded PNG per result table, one PNG per feature under `charts/`, plus `metadata.json` |
| **Folder** | the same bundle written to a folder you choose | the same bundle written to a folder you choose |

One rule for the whole project: **results go to a `results` folder beside whatever you ran**, in
the same numbered sections whether they came from the app, a notebook or the engine directly:

```
01_distance_measures/   tables, psi_bins/, charts/   (drift)
02_descriptive_trends/  tables + charts/
03_missing_values/      tables + charts/
04_model_score/         tables + charts/   (only when a score column is named)
```

Run folders are stamped in Toronto time (`drift_20260730_161500`), falling back gracefully where a
container or locked-down cluster has no timezone database.

```
results/profiling/     the app's profiling exports        (beside app.py)
results/drift/         the app's drift exports            (beside app.py)
profiling_examples/results/   the profiling notebook's results
drift_examples/results/       the drift notebook's results
```

The destination is editable in the app: type any local path, or a `dbfs:/…` or
`abfss://…` address on Databricks, and the bundle is written there instead. Setting
`EDA_RESULTS_ROOT` changes the default a deployment starts from.

The interactive charts are Plotly and the exported charts are Matplotlib, both rendered from the
*same* pre-aggregated summaries — so an exported page can never disagree with the screen.

---

## Beyond the dashboard

The app is one front end. Both engines are plain Python packages with no Streamlit dependency, and
they expose more than the dashboard does — profiling **segmentation**, folder-based report writing,
direct control over saved drift artifacts, and Spark/Databricks support.

| | Read this | Learn by example |
|---|---|---|
| Profiling | [`profiling/Readme.md`](profiling/Readme.md) | [`profiling_examples/profiling_walkthrough.ipynb`](profiling_examples/profiling_walkthrough.ipynb) |
| Drift | [`drift/Readme.md`](drift/Readme.md) | [`drift_examples/drift_walkthrough.ipynb`](drift_examples/drift_walkthrough.ipynb) |

```python
import profiling

loaded = profiling.load_table("data.csv")
settings = profiling.ProfilingSettings(correlation_methods=("pearson", "spearman"))
result = profiling.run_profiling(loaded.frame, settings, dataset_name="data.csv")
profiling.write_report_dir(result)                 # -> results/profiling/<run>/
```

```python
from drift import DriftAssessment

assessment = DriftAssessment(current, col_names=features, reference=baseline, date_col="month")
assessment.calc_distances(methods=["PSI", "KS2", "WS"], plot_trend=False)
assessment.distance_results_["psi"]                # features x snapshots
```

```python
# The model's own output, compared directly. score_col is carried beside the
# features, never among them, and sample_percent thins a file too big to wait for.
scored = DriftAssessment(
    current, col_names=features, reference=baseline, date_col="month",
    score_col="ModelScore", sample_percent=25,
)
distribution, summary = scored.score_analysis(n_bins=10)
summary.loc["PSI vs baseline"]                     # score PSI per snapshot
```

Both notebooks show every result three ways — the DataFrame, the same table as a blue image, and
the address it was saved to — and every chart as Plotly, Matplotlib and a saved PNG. They ship
**already executed**, so every cell has its output, and each has an `.html` twin beside it
(`drift_walkthrough.html`, `profiling_walkthrough.html`) as a fixed record of that run.

**Databricks, DBFS and ADLS.** Point the output folder at a cloud address and nothing else changes:

```python
profiling.set_output_dir("abfss://container@account.dfs.core.windows.net/eda/profiling")
drift.set_output_dir("dbfs:/FileStore/eda/drift")
```

Every file is serialised in memory and written in a **single operation** — never opened once and
appended to in pieces — and cloud writes go through `dbutils.fs`, so a locked-down storage account
accepts them. Details in each package README.

---

## Project layout

```
app.py                     the Streamlit app (both workspaces)
profiling/                 profiling engine  — see profiling/Readme.md
drift/                     drift engine      — see drift/Readme.md
profiling_examples/        sample data + the profiling walkthrough notebook
drift_examples/            sample data + the drift walkthrough notebook
results/                   results saved by the app (created on demand)
tests/                     engine, rendering, storage and app-integration tests
requirements.txt           pinned runtime dependencies
```

`profiling` and `drift` are the sources of truth. Neither imports Streamlit, so the app, the
notebooks, scripts and batch jobs all run the same calculations.

Run the tests with:

```bash
python -m pytest tests -q
```

---

## Usage in industry

Data profiling and drift monitoring underpin data-quality work in finance, healthcare, retail and
beyond: catching quality issues, understanding distributions, spotting outliers, and documenting the
state of a dataset before and after it is used. The export bundles are designed so a run can be
attached to a review or validation file and reproduced later from the recorded settings.

---

Developed by **Parsa Farshadfar**.

# Data drift engine

`drift` compares a trusted reference population with one or more current-data snapshots. It is the
calculation layer behind the **Drift assessment** workspace in [`app.py`](../app.py) and is equally
usable from a notebook, a script, a batch job or a Databricks cluster.

The package is self-contained: it does not import Streamlit, and it does not depend on the
`profiling` package, so this folder can be copied into a workspace on its own.

## Modules

| Module | What lives there |
|---|---|
| `drift_assessment` | validation, snapshots, distances, descriptive trends, missingness |
| `timeperiods` | understanding a date column of any type and cutting it into snapshots |
| `drift_util` | `df_to_img` — the blue table renderer |
| `storage` | where results are saved; single-shot writing to local folders, DBFS and ADLS |

## Capabilities

- Population Stability Index (PSI) for numeric **and** categorical features
- Two-sample Kolmogorov–Smirnov statistics and p-values for numeric features
- Wasserstein and energy distance (the historical `CM` option)
- Chi-square statistics for categorical features
- Snapshots by **month (default)**, quarter, year, week or day, from a date column of any type —
  or one whole-data population, or a mapping you supply yourself
- Mean/mode, standard-deviation and missing-value trends, rounded to four decimals
- Model-score drift: the distribution of a model's *output* compared against the baseline,
  band by band, with score PSI per snapshot
- Seeded subsampling of every population, for a pair of files too large to assess whole
- Non-mutating one-hot-encoded column reconstruction
- CSV/XLSX output, blue table images rendered through `df_to_img` with risky values shaded
  amber and red, and one trend PNG per feature

Work is organised one feature at a time: the baseline PSI distribution is prepared once per feature
and reused for every snapshot, and each snapshot column is released before the next one is read.
Features are assessed **in parallel** — NumPy and SciPy release the GIL for the sorting and binning
that dominate, so the cores are used without the memory cost of separate processes.
`calc_distances(..., n_jobs=1)` forces the single-threaded path; the numbers are identical either
way, because results are merged in feature order rather than completion order.
Invalid cells are reported as `NaN` with an explanatory note instead of aborting unrelated results,
and caller-owned DataFrames are never mutated.

## Quick start

```python
import pandas as pd
from drift import DriftAssessment

baseline = pd.read_csv("drift_examples/data/data_sample_for_drift_baseline.csv")
current = pd.read_csv("drift_examples/data/data_sample_for_drift.csv")

assessment = DriftAssessment(
    current,
    col_names=["Feature1", "Feature2", "Feature3", "Feature4", "CardType"],
    reference=baseline,
    date_col="Month",        # interval="Monthly" is the default
    save_results=False,
)

# The historical return value is the final requested method.
assessment.calc_distances(
    methods=["PSI", "KS2", "WS"],
    psi_n_buckets=10,
    psi_bucket_type="bins",  # or "quantiles"
    plot_trend=False,
)

# Every requested method is available without recomputation.
psi = assessment.distance_results_["psi"]      # features x snapshots
ks = assessment.distance_results_["ks2"]
ks_p_values = assessment.ks_p_values_
notes = assessment.distance_notes_             # why a cell is NaN
descriptives = assessment.calc_descriptives_overtime(plot_trend=False)
missing_percentages, missing_counts = assessment.missing_analysis()
```

Canonical keys in `distance_results_` are `psi`, `ks2`, `ws`, `cm` and `chisquare`; every table has
features as rows and snapshots as columns. `distance_notes_` explains cells deliberately left
unavailable, such as KS for a categorical feature or chi-square for a numeric one.

## Time is understood, not assumed

A real date column is rarely a clean timestamp, so `date_col` accepts whatever you have:

| The column holds | Example |
|---|---|
| ISO or free text | `"2019-03-15"`, `"2019-03"`, `"Mar 2019"`, `"15/03/2019"` |
| compact numbers | `201903`, `20190315`, `2019` |
| quarter labels | `"2019Q1"`, `"2019-Q1"`, `"Q1 2019"` |
| timestamps | `datetime64`, with or without a timezone |
| pandas periods | `PeriodDtype` |

Each strategy is tried in order of certainty and the one that reads the most values wins, so a
column does not have to be internally consistent. Values that cannot be read become `NaT`, are
excluded from the snapshots, and are reported once as a warning — never guessed.

`interval` decides how long a snapshot is. **Monthly is the default**, and any common spelling
works (`"M"`, `"month"`, `"Monthly"`):

| Interval | Label | Meaning |
|---|---|---|
| `"Daily"` | `2019-03-15` | one snapshot per calendar day |
| `"Weekly"` | `2019-W11` | ISO week, so a year never splits a week |
| `"Monthly"` | `2019-03` | **default** |
| `"Quarterly"` | `2019-Q1` | calendar quarter |
| `"Yearly"` | `2019` | calendar year |
| `"All"` | `Current data` | no time axis: the current rows as one population |

Labels sort chronologically as plain text, so snapshots stay in order everywhere they appear. The
same helpers are available on their own:

```python
import drift

drift.snapshot_labels(column, "Monthly")   # -> "2019-03" per row
drift.parse_time_column(column)            # -> timestamps
drift.normalise_interval("q")              # -> "Quarterly"
```

Omitting `date_col` compares the current data as one population and requires a `reference`. With a
date column, omitting `reference` makes the first snapshot the baseline — it stays in the trend, so
you can see the expected zero-distance row.

## Model score drift

A feature moving matters because it may move the model's output — but that does not follow
automatically, so the output is worth measuring directly. Name it with `score_col=`:

```python
assessment = DriftAssessment(
    current, col_names=features, reference=baseline, date_col="month",
    score_col="ModelScore",
)
distribution, summary = assessment.score_analysis(n_bins=10)

summary.loc["PSI vs baseline"]     # the number monitoring usually acts on
distribution                       # share of rows per band, one column per snapshot
```

The score is carried **beside** the features, never among them: it never enters
`num_cols`/`cat_cols`, the feature distances or the missing-value tables, so it is not
double-counted against the model it came from.

`distribution` holds each population's *share of rows* per band — shares, not counts, so a small
snapshot is comparable with a large baseline. Bands are cut from the baseline alone and reused for
every snapshot; cutting each population on its own quantiles would compare different bands and
report a flat distribution however far the score had moved. A numeric score is banded by value, a
categorical one uses its own categories, and a category seen only in a snapshot is appended rather
than dropped.

`summary` adds rows, missing, mean, standard deviation and quartiles per population. Both are also
kept as `score_distribution_` and `score_summary_`, and saved under `04_model_score/` when
`save_results=True`.

For a numeric score, `score_density_` holds a kernel density curve per population, all evaluated on
**one shared x grid** so they can be laid over each other in a single chart — two distributions that
still match trace the same curve, and one that has moved is obvious without reading a number. A
per-population grid would make identical distributions look different, which is why the grid spans
the union of their ranges. A categorical score has no density and returns an empty frame; so does a
constant one, which has no spread to estimate from.

## Subsampling a large pair of files

`sample_percent=` assesses a random share of the rows. **100 is the default and uses everything.**

```python
DriftAssessment(current, col_names=features, reference=baseline,
                date_col="month", sample_percent=25, random_state=0)
```

The baseline and *each snapshot* are drawn separately, so every population keeps the same share of
its own rows — sampling the combined data would let a busy month crowd out a quiet one and change
the comparison itself. The draw is seeded, so the same percentage always measures the same rows,
and `rows_before_sampling_` records what each population held beforehand. Row order is preserved.

Sampling adds noise to every distance, and small snapshots wobble most: use it to explore, then
confirm at 100% anything you intend to act on.

## One-hot encoded data

Pass the original feature names:

```python
assessment = DriftAssessment(
    current_ohe,
    col_names=["Feature1", "Feature2", "Feature3", "Feature4", "CardType"],
    reference=baseline_ohe,
    ohe_columns=["CardType"],
)
```

Columns such as `CardType_Elite` and `CardType_Platinum` are rebuilt as the logical `CardType`
feature on internal copies, and values are stripped to `Elite`, `Platinum`, … so they align with an
ordinary categorical baseline. All-zero and all-missing rows stay missing rather than being
assigned an arbitrary category.

## Snapshots supplied by hand

```python
assessment = DriftAssessment(
    {"2021_Q2": [frame_a, frame_b], "2021_Q3": frame_c},   # a list is concatenated
    col_names=features,
    reference=baseline,
)
```

Values may also be CSV paths, and Spark frames are materialised once, up front, because SciPy needs
local arrays.

## Where results are saved

One rule for the whole repository: **results go to a `results` folder beside whatever you ran.**

| You ran | Results land in |
|---|---|
| `app.py` (Streamlit) | `results/drift/` next to `app.py` |
| `drift_examples/drift_walkthrough.ipynb` | `drift_examples/results/` |
| your own script | `results/drift/`, until you change it |

```python
drift.output_dir()                                # where results go right now
drift.set_output_dir("drift_examples/results")    # change it for the session

assessment = DriftAssessment(..., save_results=True)                              # standard folder
assessment = DriftAssessment(..., save_results=True, save_folder_name="reports/january")  # this run
```

With `save_results=True` the engine writes, per assessment, in three numbered folders —
`01_distance_measures/`, `02_descriptive_trends/`, `03_missing_values/`, the same layout the
Streamlit app exports:

- one CSV and one blue PNG per result table — `psi_10bins`, `ks2`, `ks2_p`, `ws`, `cm`, `chisq`,
  `mean`, `std`, `missing_perc`, `missing_count`. PSI tables are shaded amber from `0.10` and red
  from `0.25`, KS p-value tables at `0.10` and `0.05`;
- an XLSX workbook with a sheet per measure, including the mean/std/min/max summary columns;
- a CSV + PNG bin table per feature and snapshot under `psi_bins/`;
- **one trend PNG per feature** under each section's `charts/` folder, plus a combined
  `All_features_plot.png` — a single feature's trend can be dropped into a document on its own.

File names are deliberately terse. An export nests folder → charts → file, and Windows still
rejects a path over 260 characters, so a long name is the difference between a report that opens
and one that does not.

Table images always go through `df_to_img`, and every figure is closed immediately after it is
written.

## Table images

`df_to_img` renders a DataFrame as a navy-blue table image — the same blue
`profiling.df_to_img` produces, so a report can mix the two:

```python
figure = drift.df_to_img(psi, title="Monthly PSI", show_index=True, save=True)
figure = drift.df_to_img(psi, save="reports/january/psi.png")   # or a specific address

# Shade risky values instead of adding a status column. A legend is drawn under the
# table, so the image explains its own colours.
figure = drift.df_to_img(psi, color_thresholds=drift.PSI_THRESHOLDS)   # (0.10, 0.25)
figure = drift.df_to_img(p_values, color_thresholds=(0.10, 0.05), higher_is_worse=False)
```

Headers and cells wrap at word boundaries, empty and duplicate-column frames are handled, and the
returned figure stays the caller's to close.

## Databricks, DBFS and ADLS

Save addresses may be cloud URIs:

```python
drift.set_output_dir("abfss://container@account.dfs.core.windows.net/eda/drift")
```

Three properties make that work behind a corporate firewall:

- **every file is finished before it is written** — tables are serialised to bytes, the workbook is
  built in a buffer and figures are rendered to PNG in memory, then each is written with a single
  call. Nothing is opened once and appended to in pieces, which is the pattern a locked-down ADLS
  destination refuses;
- **cloud writes go through `dbutils.fs`** — the driver performs one `cp` of a staged file, using
  the cluster's own credentials, so Python never opens a connection to the storage account;
- **remote addresses never touch `pathlib`** — `Path("abfss://a/b")` collapses the double slash, so
  `drift.storage` handles URIs as text.

Keep `col_names` to the features you actually need: only those columns are copied into the
snapshots, so a wide source table costs no more than a narrow one.

## Interpretation

- PSI below `0.10`, from `0.10` to `0.25`, and at or above `0.25` are common triage bands, not
  universal statistical thresholds.
- KS p-values are affected by sample size and multiple testing.
- Wasserstein and energy distances use the feature's own scale, so compare one feature over time
  rather than comparing unrelated features with each other.
- Chi-square is intended for categorical frequency distributions.

## Learn by example

[`drift_examples/drift_walkthrough.ipynb`](../drift_examples/drift_walkthrough.ipynb) covers all of
the above — baseline comparison, monthly and quarterly snapshots, date columns of every type,
one-hot encoded categories, supplied snapshots and saved artifacts — showing each result three
ways: the DataFrame, the same table as a blue image, and the address it was saved to.

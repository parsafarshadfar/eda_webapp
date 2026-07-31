from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from drift import (
    HEADER_COLOR,
    DriftAssessment,
    df_to_img,
    merge_one_hot_encoded_columns,
    normalise_interval,
    snapshot_labels,
)


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, np.nan],
            "segment": pd.Series(["A", "A", "B", None], dtype="category"),
        }
    )


def _actual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [2.0, 3.0, 4.0, 5.0],
            "segment": pd.Series(["A", "B", "C", "C"], dtype="string"),
        }
    )


def test_public_result_contract_and_optimised_categories() -> None:
    baseline = _baseline()
    actual = _actual()
    original = actual.copy(deep=True)

    assessment = DriftAssessment(
        actual,
        col_names=["amount", "segment"],
        reference=baseline,
    )
    returned = assessment.calc_distances(
        methods=["PSI", "KS2", "WS", "CM", "Chisquare"],
        plot_trend=False,
    )

    assert list(assessment.distance_results_) == [
        "psi",
        "ks2",
        "ws",
        "cm",
        "chisquare",
    ]
    assert returned is assessment.distance_results_["chisquare"]
    assert returned.index.tolist() == ["amount", "segment"]
    assert returned.columns.tolist() == ["Current data"]
    assert np.isnan(assessment.distance_results_["ks2"].loc["segment", "Current data"])
    assert np.isnan(
        assessment.distance_results_["chisquare"].loc["amount", "Current data"]
    )
    assert assessment.distance_results_["psi"].loc["segment", "Current data"] > 0
    assert assessment.ks_p_values_.shape == (2, 1)
    pd.testing.assert_frame_equal(actual, original)


def test_psi_handles_constant_numeric_and_disjoint_categories() -> None:
    numeric_value, numeric_table = DriftAssessment._calculate_psi_two_series(
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 2.0, 2.0]),
        psi_buckets=10,
        bucket_type="bins",
    )
    category_value, category_table = DriftAssessment._calculate_psi_two_series(
        pd.Series(["A", "A"], dtype="category"),
        pd.Series(["B", "B"], dtype="category"),
    )

    assert np.isfinite(numeric_value) and numeric_value > 0
    assert numeric_table["Baseline count"].sum() == 3
    assert numeric_table["Actual Count"].sum() == 3
    assert np.isfinite(category_value) and category_value > 0
    assert category_table.index.tolist() == ["A", "B"]
    assert category_table["PSI"].notna().all()


def test_dated_data_can_infer_first_snapshot_as_reference() -> None:
    data = pd.DataFrame(
        {
            "value": [1.0, 2.0, 10.0, 11.0],
            "when": ["2025-01-01", "2025-01-20", "2025-02-01", "bad date"],
        }
    )
    # The unreadable row is reported rather than guessed at, and excluded.
    with pytest.warns(RuntimeWarning, match="could not be read as a date"):
        assessment = DriftAssessment(
            data,
            col_names=["value"],
            reference=None,
            date_col="when",
            interval="Monthly",
        )

    assert assessment.snapshots_labels == ["2025-01", "2025-02"]
    assert assessment.reference["value"].tolist() == [1.0, 2.0]
    result = assessment.calc_distances(methods=["PSI"], plot_trend=False)
    assert result.loc["value", "2025-01"] == 0.0


def test_unreadable_date_column_says_so_in_plain_words() -> None:
    data = pd.DataFrame({"value": [1.0, 2.0, 3.0], "when": ["abc", "def", "ghi"]})

    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError) as failure:
            DriftAssessment(
                data,
                col_names=["value"],
                reference=pd.DataFrame({"value": [1.0, 2.0]}),
                date_col="when",
            )

    message = str(failure.value)
    assert "date column 'when' could not be read as dates" in message
    assert "'abc'" in message  # shows what the column actually holds
    assert "2019Q1" in message  # and what it could hold instead


def test_one_hot_merge_is_non_mutating_and_aligns_labels() -> None:
    encoded = pd.DataFrame(
        {
            "CardType_Elite": [1, 0, 0, np.nan],
            "CardType_Smart Cash": [0, 1, 0, np.nan],
            "other": [1, 2, 3, 4],
        }
    )
    before = encoded.copy(deep=True)

    merged = merge_one_hot_encoded_columns(encoded, ["CardType"])

    pd.testing.assert_frame_equal(encoded, before)
    assert merged.columns.tolist() == ["CardType", "other"]
    assert merged["CardType"].iloc[:2].tolist() == ["Elite", "Smart Cash"]
    assert pd.isna(merged["CardType"].iloc[2])
    assert pd.isna(merged["CardType"].iloc[3])


def test_descriptives_missing_and_saved_tables(tmp_path) -> None:
    assessment = DriftAssessment(
        _actual(),
        col_names=["amount", "segment"],
        reference=_baseline(),
        save_results=True,
        save_folder_name=tmp_path,
    )
    assessment.calc_distances(methods=["PSI"], plot_trend=False)
    descriptives = assessment.calc_descriptives_overtime(plot_trend=False)
    percentages, counts = assessment.missing_analysis()

    assert set(descriptives) == {"mean", "numeric_mean", "std"}
    assert percentages.shape == counts.shape == (2, 2)
    # One folder per kind of result, and short file names inside them: an export
    # nests several levels deep and Windows still caps a path at 260 characters.
    distances = tmp_path / "01_distance_measures"
    assert (distances / "distance_measures.xlsx").is_file()
    assert (distances / "psi_10bins.png").is_file()
    assert (distances / "psi_bins" / "psi_bins_Current data_amount.csv").is_file()
    assert (tmp_path / "02_descriptive_trends" / "mean.png").is_file()
    assert (tmp_path / "02_descriptive_trends" / "std.png").is_file()
    assert (tmp_path / "03_missing_values" / "missing_count.png").is_file()
    assert (tmp_path / "03_missing_values" / "missing_perc.png").is_file()
    assert plt.get_fignums() == []


def test_df_to_img_handles_empty_and_duplicate_columns() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["duplicate", "duplicate"])
    figure = df_to_img(frame, show_index=True)
    empty_figure = df_to_img(pd.DataFrame())
    try:
        assert figure.axes
        assert empty_figure.axes
    finally:
        plt.close(figure)
        plt.close(empty_figure)


def test_df_to_img_tables_are_blue_like_profiling(tmp_path) -> None:
    figure = df_to_img(
        pd.DataFrame({"value": [1.0, 2.0]}),
        title="Blue table",
        save=tmp_path / "table.png",
    )
    try:
        header = figure.axes[0].tables[0][(0, 0)]
        body = figure.axes[0].tables[0][(1, 0)]

        assert to_hex(header.get_facecolor()) == HEADER_COLOR.lower()
        assert to_hex(body.get_facecolor()) == "#eef3f7"
        assert (tmp_path / "table.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "values",
    [
        ["2019-01-31", "2019-02-15", "2019-03-01"],  # ISO text
        ["31/01/2019", "15/02/2019", "01/03/2019"],  # day-first text
        ["Jan 2019", "Feb 2019", "March 2019"],  # month names
        [201901, 201902, 201903],  # compact numbers
        [20190131, 20190215, 20190301],  # numeric timestamps
        pd.to_datetime(["2019-01-31", "2019-02-15", "2019-03-01"]),  # timestamps
        pd.to_datetime(["2019-01-31", "2019-02-15", "2019-03-01"]).tz_localize("UTC"),
        pd.PeriodIndex(["2019-01", "2019-02", "2019-03"], freq="M"),  # periods
        pd.Series(["2019-01", "2019-02", "2019-03"], dtype="category"),
    ],
)
def test_any_date_column_type_produces_monthly_snapshots(values) -> None:
    labels = snapshot_labels(pd.Series(list(values)))  # monthly is the default

    assert labels.tolist() == ["2019-01", "2019-02", "2019-03"]


def test_interval_spellings_and_other_grains() -> None:
    dates = pd.Series(["2019-01-31", "2019-05-15", "2020-02-01"])

    assert normalise_interval(None) == "Monthly"
    assert normalise_interval("m") == normalise_interval("MONTHLY") == "Monthly"
    assert normalise_interval("q") == "Quarterly"
    assert normalise_interval("annual") == "Yearly"
    assert snapshot_labels(dates, "quarterly").tolist() == [
        "2019-Q1",
        "2019-Q2",
        "2020-Q1",
    ]
    assert snapshot_labels(dates, "Y").tolist() == ["2019", "2019", "2020"]
    assert snapshot_labels(dates, "weekly").tolist() == [
        "2019-W05",
        "2019-W20",
        "2020-W05",
    ]
    with pytest.raises(ValueError, match="not a recognised interval"):
        normalise_interval("fortnightly")


def test_quarterly_snapshots_from_a_quarter_labelled_column() -> None:
    data = pd.DataFrame(
        {
            "value": [1.0, 2.0, 9.0, 10.0],
            "quarter": ["2019Q1", "2019-Q1", "Q2 2019", "2019Q2"],
        }
    )

    assessment = DriftAssessment(
        data,
        col_names=["value"],
        date_col="quarter",
        interval="quarterly",
    )

    assert assessment.snapshots_labels == ["2019-Q1", "2019-Q2"]
    assert assessment.reference["value"].tolist() == [1.0, 2.0]


def test_saving_accepts_a_custom_folder_and_writes_readable_files(tmp_path) -> None:
    assessment = DriftAssessment(
        _actual(),
        col_names=["amount"],
        reference=_baseline(),
        save_results=True,
        save_folder_name=tmp_path / "custom_place",
    )
    assessment.calc_distances(methods=["PSI"], plot_trend=False)

    workbook = tmp_path / "custom_place" / "01_distance_measures" / "distance_measures.xlsx"
    values = tmp_path / "custom_place" / "01_distance_measures" / "psi_10bins.csv"

    assert workbook.read_bytes().startswith(b"PK")  # a complete, valid xlsx
    assert "amount" in values.read_text(encoding="utf-8")


def _scored_pair(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A baseline and a current frame whose model score has clearly moved."""
    rng = np.random.default_rng(seed)
    n = 3_000
    baseline = pd.DataFrame(
        {
            "amount": rng.normal(50.0, 10.0, n),
            "segment": rng.choice(["A", "B"], n),
            "score": rng.beta(2.0, 5.0, n),
        }
    )
    current = pd.DataFrame(
        {
            "amount": rng.normal(50.0, 10.0, n),
            "segment": rng.choice(["A", "B"], n),
            # Beta(4, 5) sits well to the right of Beta(2, 5): the inputs are
            # unchanged and only the model's output has moved.
            "score": rng.beta(4.0, 5.0, n),
            "month": rng.choice(["2024-01", "2024-02"], n),
        }
    )
    return baseline, current


def test_score_column_is_compared_without_becoming_a_drift_feature() -> None:
    baseline, current = _scored_pair()

    assessment = DriftAssessment(
        current,
        col_names=["amount", "segment"],
        reference=baseline,
        date_col="month",
        score_col="score",
    )
    assessment.calc_distances(methods=["PSI"], plot_trend=False, verbose=False)
    distribution, summary = assessment.score_analysis(n_bins=10)

    # A score is an output, not an input. It must never be measured as a feature
    # in its own right, or it would be double-counted against the model it came
    # from -- but it must still be there for the score comparison itself.
    assert "score" not in assessment.num_cols
    assert "score" not in assessment.cat_cols
    assert "score" not in assessment.distance_results_["psi"].index
    assert "score" in assessment.reference.columns

    # Each column is one population's share of rows, so each sums to 100%.
    assert np.allclose(distribution.sum(axis=0), 100.0)
    # The baseline is its own reference, and the shifted snapshots are flagged.
    assert summary.loc["PSI vs baseline"].iloc[0] == 0.0
    assert (summary.loc["PSI vs baseline"].iloc[1:] > 0.25).all()
    assert (summary.loc["Mean"].iloc[1:] > summary.loc["Mean"].iloc[0]).all()


def test_score_analysis_needs_a_score_column() -> None:
    baseline, current = _scored_pair()
    assessment = DriftAssessment(
        current, col_names=["amount"], reference=baseline, date_col="month"
    )

    with pytest.raises(Exception, match="No score column"):
        assessment.score_analysis()


def test_subsampling_thins_every_population_by_the_same_share() -> None:
    baseline, current = _scored_pair()

    full = DriftAssessment(
        current, col_names=["amount", "segment"], reference=baseline, date_col="month"
    )
    quarter = DriftAssessment(
        current,
        col_names=["amount", "segment"],
        reference=baseline,
        date_col="month",
        sample_percent=25.0,
    )

    assert quarter.reference.shape[0] == round(full.reference.shape[0] * 0.25)
    for label, frame in quarter.data_dict.items():
        # Each snapshot keeps a quarter of *its own* rows, so a busy month cannot
        # crowd out a quiet one the way sampling the combined data would.
        assert frame.shape[0] == round(full.data_dict[label].shape[0] * 0.25)
    assert quarter.rows_before_sampling_["reference"] == full.reference.shape[0]

    # Seeded: the same percentage measures the same rows every time.
    repeat = DriftAssessment(
        current,
        col_names=["amount", "segment"],
        reference=baseline,
        date_col="month",
        sample_percent=25.0,
    )
    assert quarter.reference.index.equals(repeat.reference.index)


@pytest.mark.parametrize("bad", [0, -1, 101, "half"])
def test_subsampling_rejects_impossible_percentages(bad) -> None:
    baseline, current = _scored_pair()

    with pytest.raises((ValueError, TypeError)):
        DriftAssessment(
            current,
            col_names=["amount"],
            reference=baseline,
            date_col="month",
            sample_percent=bad,
        )

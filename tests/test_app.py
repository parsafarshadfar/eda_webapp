from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def _assert_clean(app: AppTest) -> None:
    assert not app.exception, [exception.message for exception in app.exception]


def _button(app: AppTest, label: str):
    """Find a button by a distinctive part of its label, page or sidebar."""
    return next(
        button
        for button in list(app.button) + list(app.sidebar.button)
        if label in button.label
    )


def _selectbox(app: AppTest, label: str):
    return next(box for box in app.sidebar.selectbox if box.label == label)


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    """Point the app's default save folder at tmp_path."""
    monkeypatch.setenv("EDA_RESULTS_ROOT", str(tmp_path))
    return tmp_path


def test_default_profiling_workspace_loads_the_renamed_example() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()

    _assert_clean(app)
    assert app.sidebar.radio[0].label == "Analysis workspace"
    assert app.sidebar.radio[0].value == "Profiling"
    assert app.title[0].value == "📊 Data Profiler"
    assert any("HousingData_TrainData.csv" in item.value for item in app.success)

    _button(app, "Run profiling").click().run(timeout=60)

    _assert_clean(app)
    assert "1. Overview" in [expander.label for expander in app.expander]


def test_default_drift_example_runs_end_to_end() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Drift assessment").run()

    _assert_clean(app)
    assert app.title[0].value == "〽️ Drift Assessment"
    assert any("baseline" in item.value.lower() for item in app.success)

    _button(app, "Run drift assessment").click().run(timeout=60)

    _assert_clean(app)
    labels = [expander.label for expander in app.expander]
    assert "2. Drift summary" in labels
    assert "3. Distance measures" in labels
    assert any(metric.label == "Snapshots" for metric in app.metric)

    _button(app, "Build drift ZIP").click().run(timeout=60)

    _assert_clean(app)
    export_key, payload = app.session_state["drift_export"]
    assert export_key
    assert payload.startswith(b"PK")


def test_one_hot_encoded_drift_example_runs_with_logical_feature_names() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Drift assessment").run()
    _selectbox(app, "Example format").set_value("One-hot encoded data").run()

    _assert_clean(app)
    _button(app, "Run drift assessment").click().run(timeout=60)

    _assert_clean(app)
    assert any(metric.label == "Features" and metric.value == "6" for metric in app.metric)


def test_snapshot_length_control_offers_every_interval_and_regroups() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Drift assessment").run()

    lengths = _selectbox(app, "Snapshot length")
    assert lengths.value == "Monthly"  # the default the engine also uses
    assert set(lengths.options) == {"Monthly", "Daily", "Weekly", "Quarterly", "Yearly"}

    _button(app, "Run drift assessment").click().run(timeout=60)
    _assert_clean(app)
    monthly = next(metric for metric in app.metric if metric.label == "Snapshots")
    assert monthly.value == "6"

    _selectbox(app, "Snapshot length").set_value("Quarterly").run()
    _button(app, "Run drift assessment").click().run(timeout=60)

    _assert_clean(app)
    quarterly = next(metric for metric in app.metric if metric.label == "Snapshots")
    assert quarterly.value == "2"


def test_changing_a_setting_after_a_run_raises_the_sticky_stale_banner() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    _button(app, "Run profiling").click().run(timeout=60)

    # The class name also appears in the stylesheet, so the banner is identified
    # by its text.
    def banners(rendered: AppTest) -> list[str]:
        return [item.value for item in rendered.markdown if "out of date" in item.value]

    _assert_clean(app)
    assert not banners(app)

    # Any change to a setting invalidates the visible results.
    next(box for box in app.sidebar.checkbox if box.label == "Missing values").set_value(
        False
    ).run(timeout=60)

    _assert_clean(app)
    banner = banners(app)[0]
    assert 'class="stale-banner"' in banner  # sticky, animated, top of the results
    assert "Run profiling" in banner


def test_all_features_is_the_default_and_wide_runs_collapse_their_sections() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()

    columns = next(box for box in app.sidebar.multiselect if box.label == "Columns to analyse")
    assert columns.value == ["All features"]  # everything, without ticking anything

    _button(app, "Run profiling").click().run(timeout=60)
    _assert_clean(app)
    # 10 columns: comfortably readable, so the sections stay open.
    assert not any("start collapsed" in item.value for item in app.caption)

    _selectbox(app, "Example dataset").set_value("Breast Cancer (train)").run(timeout=60)
    _button(app, "Run profiling").click().run(timeout=120)

    _assert_clean(app)
    # 31 columns: every section open at once is unreadable, so they start closed.
    assert any("start collapsed" in item.value for item in app.caption)


def test_descriptive_trends_can_be_narrowed_to_the_mean_alone() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Drift assessment").run()

    means = next(
        box for box in app.sidebar.checkbox if box.label == "Mean / most frequent value"
    )
    deviations = next(
        box for box in app.sidebar.checkbox if box.label == "Standard deviation"
    )
    assert means.value is True and deviations.value is False  # mean only by default

    _button(app, "Run drift assessment").click().run(timeout=60)
    _assert_clean(app)
    headings = "".join(item.value for item in app.markdown)
    assert "Standard deviation by snapshot" not in headings

    deviations.set_value(True).run()
    _button(app, "Run drift assessment").click().run(timeout=60)

    _assert_clean(app)
    headings = "".join(item.value for item in app.markdown)
    assert "Standard deviation by snapshot" in headings


def test_every_feature_is_charted_and_a_single_snapshot_skips_the_bars() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Drift assessment").run()
    _button(app, "Run drift assessment").click().run(timeout=60)

    _assert_clean(app)
    # Monthly: a chart for each of the 5 features per section, none left out.
    charted = [item.value for item in app.caption if "One chart per feature" in item.value]
    assert charted and all("in total" in caption for caption in charted)
    monthly_charts = len(list(app.get("plotly_chart")))
    assert monthly_charts == 17  # 6 distance + 5 descriptive + 6 missing

    layout = next(
        radio for radio in app.sidebar.radio if radio.label == "Comparison layout"
    )
    layout.set_value("Current data as one population").run()
    _button(app, "Run drift assessment").click().run(timeout=60)

    _assert_clean(app)
    assert any(metric.label == "Snapshots" and metric.value == "1" for metric in app.metric)
    # One snapshot means one bar repeating the table, so the distance section
    # drops its charts and keeps the table.
    assert len(list(app.get("plotly_chart"))) == monthly_charts - 6


def test_profiling_results_can_be_saved_into_the_project_folder(results_root) -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    _button(app, "Run profiling").click().run(timeout=60)
    _button(app, "Save to this folder").click().run(timeout=120)

    _assert_clean(app)
    written = sorted(path for path in (results_root / "profiling").rglob("*") if path.is_file())
    names = {path.name for path in written}

    assert "metadata.json" in names
    assert "descriptive_statistics.csv" in names
    assert any(path.suffix == ".png" for path in written)
    assert any("file(s) written to" in item.value for item in app.success)


def test_drift_results_can_be_saved_into_the_project_folder(results_root) -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Drift assessment").run()
    _button(app, "Run drift assessment").click().run(timeout=60)
    _button(app, "Save to this folder").click().run(timeout=120)

    _assert_clean(app)
    written = sorted(path for path in (results_root / "drift").rglob("*") if path.is_file())
    names = {path.name for path in written}

    assert {"metadata.json", "README.txt", "psi.csv", "psi.png"} <= names
    # Every table is exported as a CSV and a PNG twin...
    tables = [path for path in written if path.parent.name != "charts"]
    assert sum(path.suffix == ".csv" for path in tables) == sum(
        path.suffix == ".png" for path in tables
    )
    # ...and every charted feature also gets its own PNG.
    charts = [path for path in written if path.parent.name == "charts"]
    assert charts and all(path.suffix == ".png" for path in charts)
    assert any("psi_Feature1" in path.name for path in charts)


def _checkbox(app: AppTest, label: str):
    return next(box for box in app.sidebar.checkbox if box.label == label)


def test_model_score_comparison_is_off_until_asked_for() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Drift assessment").run()

    score = _checkbox(app, "Model score comparison")
    # Off by default: most datasets carry no model output, and the app cannot
    # guess which column would be one.
    assert score.value is False

    picker = _selectbox(app, "Model score column")
    # "None" first and selected: no score column is the honest default, and the
    # picker is never disabled. Everything here lives in one st.form, which does
    # not rerun until submitted, so a `disabled=` bound to the checkbox above
    # would stay stuck until the next run.
    assert picker.options[0] == "None"
    assert picker.value == "None"
    assert picker.disabled is False

    _button(app, "Run drift assessment").click().run(timeout=60)
    _assert_clean(app)
    assert not any("Model score" in expander.label for expander in app.expander)

    # Ticking the box must not need a run before a column can be chosen.
    score.set_value(True).run()
    picker = _selectbox(app, "Model score column")
    assert picker.disabled is False
    # Feature1 stands in for a score column here; the point is the wiring, not
    # the meaning of the number.
    picker.set_value("Feature1").run()
    _button(app, "Run drift assessment").click().run(timeout=120)

    _assert_clean(app)
    assert any("Model score" in expander.label for expander in app.expander)
    assert any("overlaid" in item.value.lower() for item in app.markdown)


def test_model_score_is_the_last_assessment_before_the_download() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Drift assessment").run()
    _checkbox(app, "Model score comparison").set_value(True).run()
    _selectbox(app, "Model score column").set_value("Feature1").run()
    _button(app, "Run drift assessment").click().run(timeout=120)

    _assert_clean(app)
    labels = [expander.label for expander in app.expander]
    score_at = next(i for i, label in enumerate(labels) if "Model score" in label)
    download_at = next(i for i, label in enumerate(labels) if "Download" in label)
    missing_at = next(i for i, label in enumerate(labels) if "Missing values" in label)

    # Last of the assessments, mirroring its place at the end of the sidebar,
    # and still ahead of the download section.
    assert missing_at < score_at < download_at


def test_score_bands_are_tabulated_not_charted() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Drift assessment").run()
    _checkbox(app, "Model score comparison").set_value(True).run()
    _selectbox(app, "Model score column").set_value("Feature1").run()
    _button(app, "Run drift assessment").click().run(timeout=120)

    _assert_clean(app)
    headings = "".join(item.value for item in app.markdown)
    # The band table stays; the per-band chart grid it used to carry does not.
    assert "Share of rows per score band" in headings
    assert not any(
        "One chart per feature" in item.value and "band" in item.value
        for item in app.caption
    )
    # One overlay chart instead: every population on a single set of axes.
    assert "Score distributions, overlaid" in headings


def test_subsampling_analyses_fewer_rows_but_still_reports() -> None:
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    _button(app, "Run profiling").click().run(timeout=60)
    _assert_clean(app)
    full_rows = next(item for item in app.metric if item.label == "Rows").value

    sampler = next(
        slider for slider in app.sidebar.slider if slider.label == "Analyse this % of rows"
    )
    assert sampler.value == 100  # the whole dataset unless asked otherwise
    sampler.set_value(25).run()
    _button(app, "Run profiling").click().run(timeout=60)

    _assert_clean(app)
    sampled_rows = next(item for item in app.metric if item.label == "Rows").value
    assert int(str(sampled_rows).replace(",", "")) < int(str(full_rows).replace(",", ""))
    assert any("subsample" in item.value for item in app.info)

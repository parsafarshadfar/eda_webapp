from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import profiling


class DataPreparationTests(unittest.TestCase):
    def test_path_and_file_like_loads_have_the_same_fingerprint(self) -> None:
        payload = b"group;value\nA;1\nA;2\nB;3\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_bytes(payload)

            from_path = profiling.load_table(
                path,
                prefer_pyarrow=False,
            )
            upload = io.BytesIO(payload)
            upload.seek(4)
            from_upload = profiling.load_table(
                upload,
                prefer_pyarrow=False,
            )

        self.assertEqual(from_path.token, from_upload.token)
        self.assertEqual(upload.tell(), 4)
        self.assertEqual(from_path.delimiter, ";")
        assert_frame_equal(from_path.frame, from_upload.frame)

    def test_memory_optimisation_is_lossless_and_does_not_mutate_input(self) -> None:
        frame = pd.DataFrame(
            {
                "integer": pd.Series([0, 1, 2, 3], dtype="int64"),
                "label": ["same", "same", "other", "same"],
                "precise": [0.1, 0.2, 0.3, 0.4],
            }
        )
        original = frame.copy(deep=True)

        optimised, applied = profiling.optimize_memory(frame)

        assert_frame_equal(frame, original)
        self.assertEqual(str(optimised["integer"].dtype), "int8")
        self.assertIsInstance(optimised["label"].dtype, pd.CategoricalDtype)
        self.assertEqual(optimised["precise"].dtype, np.dtype("float64"))
        self.assertEqual(set(applied), {"integer", "label"})

    def test_combined_origin_column_is_compact_and_source_order_is_preserved(
        self,
    ) -> None:
        train = pd.DataFrame({"value": [1, 2]})
        test = pd.DataFrame({"value": [3]})

        combined = profiling.combine_datasets({"Train": train, "Test": test})

        self.assertIsInstance(combined.frame["Dataset"].dtype, pd.CategoricalDtype)
        self.assertEqual(
            combined.frame["Dataset"].astype(str).tolist(), ["Train", "Train", "Test"]
        )
        self.assertEqual(combined.sources, {"Train": 2, "Test": 1})
        self.assertEqual(combined.feature_names, ["value"])

    def test_one_hot_merge_handles_all_missing_rows_and_keeps_input_unchanged(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "colour__red": [1.0, 0.0, np.nan],
                "colour__blue": [0.0, 1.0, np.nan],
                "score": [0.2, 0.4, 0.6],
            }
        )
        original = frame.copy(deep=True)

        merged = profiling.merge_one_hot_encoded_columns(
            frame,
            separator="__",
            strip_prefix=True,
        )

        assert_frame_equal(frame, original)
        self.assertEqual(list(merged.columns), ["id", "colour", "score"])
        self.assertEqual(merged["colour"].iloc[:2].tolist(), ["red", "blue"])
        self.assertTrue(pd.isna(merged["colour"].iloc[2]))


class ProfilingRobustnessTests(unittest.TestCase):
    def test_duplicate_index_segmentation_does_not_duplicate_rows(self) -> None:
        frame = pd.DataFrame(
            {"group": ["A", "B", "A", "B"], "value": [1, 2, 3, 4]},
            index=[0, 0, 1, 1],
        )

        grouping = profiling.make_grouping(frame, "group")
        subsets = {segment.key: subset for segment, subset in grouping.frames(frame)}

        self.assertEqual(subsets["A"]["value"].tolist(), [1, 3])
        self.assertEqual(subsets["B"]["value"].tolist(), [2, 4])
        self.assertEqual(sum(len(subset) for subset in subsets.values()), len(frame))

    def test_constant_quantile_grouping_drops_duplicate_edges(self) -> None:
        frame = pd.DataFrame({"group": [1, 1, 1], "value": [2, 3, 4]})

        grouping = profiling.make_grouping(frame, "group", n_quantiles=4)

        self.assertEqual(len(grouping), 0)
        self.assertEqual(grouping.n_rows_unassigned, 3)

    def test_outlier_plot_sample_is_bounded_but_counts_remain_exact(self) -> None:
        values = pd.Series([0.0] * 1_000 + list(range(100, 200)), name="measure")

        summary = profiling.box_summary(values, max_outlier_points=7)

        self.assertIsNotNone(summary)
        self.assertEqual(summary.n_outliers, 100)
        self.assertEqual(summary.outliers.size, 7)
        self.assertTrue(summary.outliers_truncated)

    def test_interactive_plots_handle_categories_and_non_finite_boxes(self) -> None:
        categorical = profiling.categorical_distribution(
            pd.Series(["A", "B", "A"], name="segment")
        )
        category_figure = profiling.plots.histogram_figure(categorical)

        non_finite = profiling.box_summary(
            pd.Series([np.inf, -np.inf], name="measure")
        )
        box_figure = profiling.plots.box_figure(non_finite)

        self.assertEqual(list(category_figure.data[0].x), ["A", "B"])
        self.assertIn("no finite values", box_figure.layout.annotations[0].text)

    def test_no_interactive_figure_uses_a_webgl_trace(self) -> None:
        """WebGL traces must never come back.

        A ``*gl`` trace builds its canvas from a WebGL context. Created inside a
        container that is hidden or zero-width at mount — a collapsed section, a
        column mid-layout — that canvas comes up blank and only paints once
        something forces a resize, so the chart looked empty until the viewer hit
        Plotly's full-screen button. Every figure the app draws must stay on the
        SVG renderer.
        """
        frame = pd.DataFrame(
            {
                # 0.0 x1000 then 100..199: the tail lands outside 1.5 x IQR, so
                # the box plot gets a populated outlier overlay to inspect.
                "measure": [0.0] * 1_000 + list(range(100, 200)),
                "segment": ["A", "B"] * 550,
            }
        )
        boxes = profiling.build_box_summaries(frame)
        distributions = profiling.build_distributions(frame)
        with_gaps = frame.copy()
        with_gaps.loc[:10, "measure"] = np.nan
        missing = profiling.missing_count(with_gaps)

        figures = [profiling.plots.box_figure(summary) for summary in boxes.values()]
        figures += [
            profiling.plots.histogram_figure(summary) for summary in distributions.values()
        ]
        figures.append(profiling.plots.missing_values_figure(missing))
        figures.append(
            profiling.plots.correlation_heatmap(frame[["measure"]].corr(), "pearson")
        )
        figures.append(profiling.plots.empty_figure("nothing to show"))

        self.assertTrue(
            any(trace.type == "scatter" for trace in figures[0].data),
            "expected an SVG outlier overlay on the box plot",
        )
        offenders = {
            trace.type
            for figure in figures
            for trace in figure.data
            if str(trace.type).endswith("gl")
        }
        self.assertEqual(offenders, set())

    def test_top_correlations_skip_undefined_pairs(self) -> None:
        matrix = pd.DataFrame(
            [
                [1.0, 0.8, np.nan],
                [0.8, 1.0, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            columns=["a", "b", "constant"],
            index=["a", "b", "constant"],
        )

        ranked = profiling.top_absolute_correlations(matrix)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked.loc[0, "Feature 1"], "a")
        self.assertEqual(ranked.loc[0, "Feature 2"], "b")

    def test_invalid_settings_fail_before_profiling(self) -> None:
        frame = pd.DataFrame({"value": [1, 2, 3]})
        settings = profiling.ProfilingSettings(n_bins=0)

        with self.assertRaisesRegex(ValueError, "n_bins"):
            profiling.run_profiling(frame, settings)

    def test_saving_statistics_creates_the_folder_and_uses_table_images(self) -> None:
        frame = pd.DataFrame({"value": [1.0, np.nan, 3.0]})
        with tempfile.TemporaryDirectory() as directory:
            describe_folder = Path(directory) / "describe"
            missing_folder = Path(directory) / "missing"

            profiling.describe_data(frame, saving_path=str(describe_folder))
            profiling.missing_count(frame, saving_path=str(missing_folder))

            self.assertTrue((describe_folder / "Data_Description.csv").is_file())
            self.assertTrue((describe_folder / "Data_Description.png").is_file())
            self.assertTrue((missing_folder / "Missing_analysis.csv").is_file())
            self.assertTrue((missing_folder / "Missing_analysis.png").is_file())


if __name__ == "__main__":
    unittest.main()

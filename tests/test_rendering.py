from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex

import profiling


@contextmanager
def output_directory(path: Path):
    """Point profiling's standard output folder at ``path`` for one block."""
    profiling.set_output_dir(path)
    try:
        yield
    finally:
        profiling.reset_output_dir()


class StaticRenderingTests(unittest.TestCase):
    def test_professional_table_formatting(self) -> None:
        frame = pd.DataFrame(
            {
                "DataType": ["float64"],
                "n_uniques (excl. Nulls)": [12_345],
                "Perc. of Missing": [5.29876],
                "Perc. of 1st Most Freq": [1.23456],
                "Max": [1_243.33333],
            },
            index=["A_feature_name_that_needs_wrapping"],
        )

        figure = profiling.dataframe_figure(
            frame,
            title="Descriptive statistics",
            show_index=True,
            index_wrap=16,
        )
        rendered = figure.axes[0].tables[0]
        headers = [
            rendered[(0, column)].get_text().get_text()
            for column in range(len(frame.columns) + 1)
        ]
        first_row = [
            rendered[(1, column)].get_text().get_text()
            for column in range(len(frame.columns) + 1)
        ]

        self.assertEqual(headers[0], "Feature")
        self.assertEqual(headers[2], "Unique values\n(excl. nulls)")
        self.assertEqual(headers[3], "Missing\n(%)")
        self.assertEqual(first_row[3], "5.299%")
        self.assertEqual(first_row[4], "1.235%")
        self.assertEqual(first_row[5], "1,243.333")
        self.assertIn("\n", first_row[0])
        self.assertEqual(to_hex(rendered[(0, 1)].get_facecolor()), "#2f4b63")

    def test_text_stays_within_cells(self) -> None:
        loaded = profiling.load_table(
            Path(__file__).resolve().parents[1]
            / "profiling_examples"
            / "Data"
            / "HousingData_TrainData.csv"
        )
        frame = profiling.describe_data(loaded.frame).head(7)
        figure = profiling.table_figures(
            frame,
            title="Descriptive statistics",
            rows_per_page=7,
            show_index=True,
            index_label="Feature",
        )[0]

        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        for cell in figure.axes[0].tables[0].get_celld().values():
            text = cell.get_text()
            if not text.get_text():
                continue
            cell_bounds = cell.get_window_extent(renderer)
            text_bounds = text.get_window_extent(renderer)
            self.assertGreaterEqual(text_bounds.x0, cell_bounds.x0 - 1.0)
            self.assertLessEqual(text_bounds.x1, cell_bounds.x1 + 1.0)
            self.assertGreaterEqual(text_bounds.y0, cell_bounds.y0 - 1.0)
            self.assertLessEqual(text_bounds.y1, cell_bounds.y1 + 1.0)

    def test_custom_save_address_and_parent_creation(self) -> None:
        frame = pd.DataFrame({"Missing (%)": [1.23456]})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "table"
            figure = profiling.dataframe_figure(frame, save=target)
            written = target.with_suffix(".png")

            self.assertEqual(figure.__class__.__name__, "Figure")
            self.assertTrue(written.is_file())
            self.assertTrue(written.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_save_true_uses_standard_folder(self) -> None:
        summary = profiling.numeric_distribution(
            pd.Series([1.0, 2.0, 3.0], name="Income")
        )
        with tempfile.TemporaryDirectory() as directory:
            with output_directory(Path(directory)):
                profiling.histogram_figure(summary, save=True)
                self.assertTrue((Path(directory) / "Income_histogram.png").is_file())

    def test_standard_output_folder_defaults_beside_the_repository(self) -> None:
        default = Path(profiling.output_dir())

        self.assertEqual(default.name, "profiling")
        self.assertEqual(default.parent.name, "results")
        self.assertEqual(default.parent.parent, Path(__file__).resolve().parents[1])

    def test_paginated_save_uses_numbered_names(self) -> None:
        frame = pd.DataFrame({"value": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "pages.png"
            figures = profiling.table_figures(
                frame,
                rows_per_page=1,
                show_index=False,
                save=target,
            )

            self.assertEqual(len(figures), 3)
            self.assertFalse(target.exists())
            self.assertTrue((target.parent / "pages_01.png").is_file())
            self.assertTrue((target.parent / "pages_02.png").is_file())
            self.assertTrue((target.parent / "pages_03.png").is_file())

    def test_every_chart_renderer_supports_custom_save_addresses(self) -> None:
        numeric = [
            profiling.numeric_distribution(
                pd.Series([1.0, 2.0, 3.0, 4.0], name=f"Measure {index}")
            )
            for index in range(1, 3)
        ]
        boxes = [
            profiling.box_summary(
                pd.Series([1.0, 2.0, 3.0, 20.0], name=f"Measure {index}")
            )
            for index in range(1, 3)
        ]
        correlation = pd.DataFrame(
            [[1.0, 0.25], [0.25, 1.0]],
            index=["Measure 1", "Measure 2"],
            columns=["Measure 1", "Measure 2"],
        )
        missing = pd.DataFrame(
            {
                "number_of_missing": [2],
                "percentage_of_missing": [12.34567],
            },
            index=["Measure 1"],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = {
                "box": root / "charts" / "box.png",
                "heatmap": root / "charts" / "heatmap.png",
                "missing": root / "charts" / "missing.png",
            }
            profiling.box_figure(boxes[0], save=targets["box"], dpi=60)
            profiling.heatmap_figure(
                correlation,
                "pearson",
                fit_page=False,
                save=targets["heatmap"],
                dpi=60,
            )
            profiling.missing_bar_figure(
                missing,
                fit_page=False,
                save=targets["missing"],
                dpi=60,
            )

            histogram_target = root / "grids" / "histograms.png"
            box_target = root / "grids" / "boxes.png"
            profiling.histogram_grid_figures(
                numeric,
                n_cols=1,
                n_rows=1,
                save=histogram_target,
                dpi=60,
            )
            profiling.box_grid_figures(
                boxes,
                n_rows=1,
                save=box_target,
                dpi=60,
            )

            for target in targets.values():
                self.assertTrue(target.is_file())
                self.assertTrue(target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertFalse(histogram_target.exists())
            self.assertFalse(box_target.exists())
            for target in (
                root / "grids" / "histograms_01.png",
                root / "grids" / "histograms_02.png",
                root / "grids" / "boxes_01.png",
                root / "grids" / "boxes_02.png",
            ):
                self.assertTrue(target.is_file())
                self.assertTrue(target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_dataframe_renderer_and_convenience_api_use_blue_save_contract(self) -> None:
        frame = pd.DataFrame({"percentage": [2.34567]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "table.png"
            figure = profiling.dataframe_figure(
                frame,
                font_size=10,
                index_font_size=9,
                header_color="#2F4B63",
                row_colors=("#EEF3F7", "#FFFFFF"),
                wrap=10,
                index_wrap=10,
                save=target,
            )
            rendered = figure.axes[0].tables[0]

            self.assertTrue(target.is_file())
            self.assertFalse((target.parent / "table.png.png").exists())
            self.assertEqual(to_hex(rendered[(0, 0)].get_facecolor()), "#2f4b63")
            named_target = root / "named" / "convenience.png"
            profiling.df_to_img(frame, save=True, save_name=named_target)
            self.assertTrue(named_target.is_file())
            with output_directory(root / "standard"):
                profiling.df_to_img(frame, save=True)
                self.assertTrue((root / "standard" / "dataframe.png").is_file())

    def test_pdf_and_bundle_descriptive_tables_use_df_to_img(self) -> None:
        frame = pd.DataFrame(
            {
                "Income": [1.0, 2.0, 3.0, 4.0],
                "Long numeric feature name": [10.0, 12.0, 14.0, 16.0],
            }
        )
        settings = profiling.ProfilingSettings(
            selected_features=tuple(frame.columns),
            include_missing=False,
            include_correlation=False,
            include_histograms=False,
            include_box_plots=False,
        )
        result = profiling.run_profiling(frame, settings, dataset_name="example.csv")
        report_module = profiling.report

        with patch.object(
            report_module,
            "df_to_img",
            wraps=profiling.df_to_img,
        ) as renderer:
            pdf = report_module.build_pdf(result)

            self.assertTrue(pdf.startswith(b"%PDF-"))
            self.assertEqual(renderer.call_count, 1)
            self.assertTrue(renderer.call_args.kwargs["show_index"])
            self.assertTrue(renderer.call_args.kwargs["fit_page"])
            self.assertEqual(renderer.call_args.kwargs["index_label"], "Feature")

            renderer.reset_mock()
            files = dict(
                report_module.bundle_files(
                    result,
                    include_pdf=False,
                    include_individual_charts=False,
                )
            )
            image = files["01_descriptive_statistics/descriptive_statistics.png"]
            self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(renderer.call_count, 1)


if __name__ == "__main__":
    unittest.main()

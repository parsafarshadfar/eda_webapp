"""The save-address contract shared by the profiling and drift packages.

Both packages expose the same small module (``profiling.storage`` and
``drift.storage``) so either folder can be copied into a Databricks workspace on
its own. These tests run every case against both, which is what keeps the two
implementations in step.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.figure import Figure

from drift import storage as drift_storage
from profiling import storage as profiling_storage

MODULES = (profiling_storage, drift_storage)
MODULE_IDS = ("profiling", "drift")

ADLS = "abfss://container@account.dfs.core.windows.net/eda"


@pytest.fixture(params=MODULES, ids=MODULE_IDS)
def storage(request):
    """Run a test against both packages' storage modules."""
    module = request.param
    yield module
    module.reset_output_dir()


def test_default_output_dir_is_results_beside_the_repository(storage) -> None:
    default = Path(storage.default_output_dir())

    assert default.name == storage.PROJECT_NAME
    assert default.parent.name == "results"
    assert default.parent.parent == Path(__file__).resolve().parents[1]


def test_set_output_dir_redirects_and_resets(storage, tmp_path) -> None:
    assert storage.set_output_dir(tmp_path) == str(tmp_path)
    assert storage.resolve_dir() == str(tmp_path)
    assert storage.resolve_dir(tmp_path / "custom") == str(tmp_path / "custom")

    storage.reset_output_dir()
    assert Path(storage.output_dir()).parent.name == "results"


def test_cloud_addresses_survive_address_arithmetic(storage) -> None:
    assert storage.is_remote(ADLS)
    assert storage.is_remote("dbfs:/FileStore/eda")
    assert not storage.is_remote("/dbfs/FileStore/eda")  # FUSE mount: a real path

    joined = storage.join_address(ADLS, "monthly", "psi.csv")
    assert joined == f"{ADLS}/monthly/psi.csv"
    assert storage.parent_address(joined) == f"{ADLS}/monthly"
    assert storage.parent_address("dbfs:/FileStore/eda/psi.csv") == "dbfs:/FileStore/eda"
    assert storage.with_suffix(f"{ADLS}/psi", ".csv") == f"{ADLS}/psi.csv"
    assert storage.with_suffix(f"{ADLS}/psi.csv", ".csv") == f"{ADLS}/psi.csv"


def test_local_writes_are_single_shot_and_create_parents(storage, tmp_path) -> None:
    frame = pd.DataFrame({"value": [1.5, 2.5]}, index=["a", "b"])
    nested = tmp_path / "deep" / "folder"

    csv_address = storage.write_frame_csv(frame, storage.join_address(nested, "table"))
    text_address = storage.write_text(storage.join_address(nested, "note.txt"), "hello")
    excel_address = storage.write_excel(
        {"values": frame}, storage.join_address(nested, "book")
    )
    figure_address = storage.write_figure(
        Figure(figsize=(1, 1)), storage.join_address(nested, "chart")
    )

    assert Path(csv_address).name == "table.csv"
    assert pd.read_csv(csv_address, index_col=0).equals(frame)
    assert Path(text_address).read_text(encoding="utf-8") == "hello"
    assert Path(excel_address).name == "book.xlsx"
    assert Path(excel_address).read_bytes().startswith(b"PK")
    assert Path(figure_address).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cloud_write_without_databricks_explains_itself(storage) -> None:
    with pytest.raises(RuntimeError, match="dbutils"):
        storage.write_text(f"{ADLS}/note.txt", "hello")


def test_cloud_write_stages_one_file_and_copies_it_once(storage, monkeypatch) -> None:
    """A remote write must be one ``cp`` of a finished file, never an append."""
    calls: list[tuple[str, ...]] = []

    def staged_file(uri: str) -> Path:
        """The local file behind a ``file:`` URI, on any platform."""
        return Path(url2pathname(urlparse(uri).path))

    class FakeFs:
        def mkdirs(self, address):
            calls.append(("mkdirs", address))

        def cp(self, source, target, recurse=False):
            calls.append(("cp", source, target))
            # The staged file must exist and hold the complete payload at the
            # moment it is handed over.
            local = staged_file(source)
            assert local.is_file()
            assert local.read_bytes() == b"complete payload"

    class FakeDbutils:
        fs = FakeFs()

    monkeypatch.setattr(storage, "_dbutils", lambda: FakeDbutils())
    written = storage.write_bytes(f"{ADLS}/monthly/psi.csv", b"complete payload")

    assert written == f"{ADLS}/monthly/psi.csv"
    assert calls[0] == ("mkdirs", f"{ADLS}/monthly")
    assert len([call for call in calls if call[0] == "cp"]) == 1
    # Nothing is left behind on the driver.
    assert not staged_file(calls[1][1]).exists()

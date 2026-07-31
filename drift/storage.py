"""Where drift results are saved, and how they are written.

The same rule the profiling package follows:

    **results are written to a ``results`` folder beside whatever you ran.**

* ``app.py`` (the Streamlit app) saves under ``<repo root>/results/drift/``;
* the example notebook saves under ``drift_examples/results/`` -- it sets that
  once with :func:`set_output_dir`.

``DriftAssessment(..., save_folder_name=<address>)`` overrides it for one
assessment, and any single call that takes an address can be pointed elsewhere.

Databricks, DBFS and ADLS
-------------------------
``dbfs:/...``, ``abfss://container@account.dfs.core.windows.net/...`` and the
other cloud schemes are accepted as save addresses. Two properties make that
work on a locked-down cluster:

* **nothing is streamed.** Every CSV, workbook and image is serialised to bytes
  in memory and then written with a single call -- never opened once and
  appended to in pieces, which is what a firewalled ADLS destination refuses.
  Files written here are aggregated result tables and figures, not raw data, so
  the memory cost of holding one is small.
* **no pathlib on remote addresses.** ``Path("abfss://a/b")`` collapses the
  double slash, so remote addresses are handled as plain strings.

This module is deliberately self-contained: the drift package has no dependency
on the profiling package, so either folder can be copied into a Databricks
workspace on its own.
"""

from __future__ import annotations

import datetime
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "RESULTS_DIRNAME",
    "PROJECT_NAME",
    "default_output_dir",
    "ensure_dir",
    "is_remote",
    "join_address",
    "output_dir",
    "parent_address",
    "project_root",
    "report_time",
    "run_stamp",
    "relative_label",
    "reset_output_dir",
    "resolve_dir",
    "set_output_dir",
    "with_suffix",
    "write_bytes",
    "write_excel",
    "write_figure",
    "write_frame_csv",
    "write_text",
]

#: Name of the folder that collects saved results.
RESULTS_DIRNAME = "results"

#: Sub-folder of ``results/`` used by this package.
PROJECT_NAME = "drift"

# Schemes that must go through ``dbutils.fs`` rather than the local filesystem.
# ``/dbfs/...`` is deliberately absent: that is the FUSE mount and behaves like
# an ordinary path as long as the whole file is written at once, which is exactly
# what :func:`write_bytes` does.
_REMOTE_SCHEMES = (
    "dbfs:",
    "abfss://",
    "abfs://",
    "wasbs://",
    "wasb://",
    "adl://",
    "s3://",
    "s3a://",
    "gs://",
)

# Session-level override installed by set_output_dir().
_override: str | None = None


# --------------------------------------------------------------------------- #
# the default address
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Folder that contains this package -- the repository root."""
    return Path(__file__).resolve().parent.parent


def default_output_dir() -> str:
    """``<repo root>/results/drift``, or whatever was last set."""
    if _override is not None:
        return _override
    return join_address(str(project_root() / RESULTS_DIRNAME), PROJECT_NAME)


#: Readable alias: ``drift.output_dir()`` answers "where do results go?".
output_dir = default_output_dir


def set_output_dir(address: str | os.PathLike[str] | None) -> str:
    """Point every default save address in this package at ``address``.

    A notebook calls this once, in its setup cell::

        drift.set_output_dir(NOTEBOOK_DIR / "results")
        drift.set_output_dir("abfss://data@account.dfs.core.windows.net/drift")

    ``None`` restores the default. The resolved address is returned so it can be
    printed straight away.
    """
    global _override
    _override = None if address is None else _as_address(address)
    return default_output_dir()


def reset_output_dir() -> str:
    """Forget a previous :func:`set_output_dir` call."""
    return set_output_dir(None)


def resolve_dir(address: str | os.PathLike[str] | None = None) -> str:
    """``address`` when one is given, otherwise the default output folder."""
    if address is None:
        return default_output_dir()
    return _as_address(address)


# --------------------------------------------------------------------------- #
# address arithmetic that is safe for URIs
# --------------------------------------------------------------------------- #
def _as_address(value: str | os.PathLike[str]) -> str:
    """Normalise an address without disturbing a URI scheme."""
    if isinstance(value, Path):
        return str(value)
    text = os.fspath(value)
    if isinstance(text, bytes):
        text = text.decode()
    return text if is_remote(text) else str(Path(text))


def is_remote(address: str | os.PathLike[str]) -> bool:
    """True for a cloud/DBFS URI that must be written through ``dbutils``."""
    text = str(address).replace("\\", "/").lower()
    return text.startswith(_REMOTE_SCHEMES)


def join_address(base: str | os.PathLike[str], *parts: str) -> str:
    """Append path components to a local path or to a remote URI."""
    address = _as_address(base)
    for part in parts:
        piece = str(part).replace("\\", "/").strip("/")
        if not piece:
            continue
        if is_remote(address):
            address = f"{address.rstrip('/')}/{piece}"
        else:
            address = str(Path(address) / piece)
    return address


def parent_address(address: str | os.PathLike[str]) -> str:
    """Folder holding ``address``."""
    text = _as_address(address)
    if not is_remote(text):
        return str(Path(text).parent)

    # Split the scheme off first so the "//" in "abfss://" is never mistaken for
    # a path separator, then drop the last component of what remains.
    trimmed = text.rstrip("/")
    scheme, separator, rest = trimmed.partition("://")
    if not separator:  # the "dbfs:/folder/file" spelling
        scheme, separator, rest = trimmed.partition(":")
    if "/" not in rest.strip("/"):  # already the root of the store
        return trimmed
    return f"{scheme}{separator}{rest.rsplit('/', 1)[0]}"


def with_suffix(address: str | os.PathLike[str], suffix: str) -> str:
    """Add ``suffix`` (for example ``".csv"``) when the address has none."""
    text = _as_address(address)
    name = text.replace("\\", "/").rsplit("/", 1)[-1]
    return text if "." in name else f"{text}{suffix}"


def relative_label(address: str | os.PathLike[str]) -> str:
    """Short, printable form of an address: repo-relative when it is local."""
    text = _as_address(address)
    if is_remote(text):
        return text
    try:
        return Path(text).resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return Path(text).as_posix()


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def _dbutils() -> Any | None:
    """Return Databricks' ``dbutils``, or ``None`` when not on a cluster."""
    try:  # In a notebook it is already injected into the user namespace.
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            found = shell.user_ns.get("dbutils")
            if found is not None:
                return found
    except Exception:  # noqa: BLE001 - absence of IPython is not an error
        pass
    try:  # In a job or an imported module it is built from the Spark session.
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        session = SparkSession.getActiveSession()
        if session is not None:
            return DBUtils(session)
    except Exception:  # noqa: BLE001 - no Spark here, so no dbutils
        pass
    return None


def ensure_dir(address: str | os.PathLike[str]) -> str:
    """Create a folder (local or remote) and return its address."""
    text = _as_address(address)
    if is_remote(text):
        tools = _dbutils()
        if tools is not None:
            tools.fs.mkdirs(text)
        return text
    Path(text).mkdir(parents=True, exist_ok=True)
    return text


def write_bytes(address: str | os.PathLike[str], payload: bytes) -> str:
    """Write ``payload`` to ``address`` in exactly one operation.

    Local and ``/dbfs`` addresses are written with a single ``open``/``write``.
    A remote URI is staged as one local temporary file and handed to
    ``dbutils.fs.cp``, so the driver performs a single upload with the cluster's
    own credentials and no incremental connection to the storage account is
    opened from Python.
    """
    text = _as_address(address)
    if not is_remote(text):
        target = Path(text)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)

    tools = _dbutils()
    if tools is None:
        raise RuntimeError(
            f"{text!r} is a cloud address, which needs Databricks' dbutils. "
            "Run this on a Databricks cluster, or pass a local folder such as "
            f"{default_output_dir()!r}."
        )

    tools.fs.mkdirs(parent_address(text))
    suffix = os.path.splitext(text)[1] or ".tmp"
    handle, staged = tempfile.mkstemp(prefix="drift_upload_", suffix=suffix)
    os.close(handle)
    try:
        Path(staged).write_bytes(payload)
        local_uri = "file:" + os.path.abspath(staged).replace(os.sep, "/")
        if not local_uri.startswith("file:/"):
            local_uri = "file:/" + local_uri[len("file:") :]
        tools.fs.cp(local_uri, text, recurse=False)
    finally:
        os.unlink(staged)
    return text


def write_text(
    address: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
) -> str:
    """Single-shot write of a text file."""
    return write_bytes(address, text.encode(encoding))


def write_frame_csv(
    frame: Any,
    address: str | os.PathLike[str],
    *,
    index: bool = True,
) -> str:
    """Serialise a DataFrame to CSV in memory, then write it once.

    Deliberately not ``frame.to_csv(path)``: that holds a handle open on the
    destination and appends the file in chunks, which a firewalled ADLS mount
    rejects.
    """
    payload = frame.to_csv(index=index, lineterminator="\n").encode("utf-8")
    return write_bytes(with_suffix(address, ".csv"), payload)


def write_excel(
    sheets: Mapping[str, Any],
    address: str | os.PathLike[str],
) -> str:
    """Build a multi-sheet workbook in memory, then write it once."""
    import pandas as pd

    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for name, table in sheets.items():
                table.to_excel(writer, sheet_name=str(name)[:31], index=True)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("Saving drift workbooks requires the 'openpyxl' package") from exc
    return write_bytes(with_suffix(address, ".xlsx"), buffer.getvalue())


def write_figure(
    figure: Any,
    address: str | os.PathLike[str],
    *,
    dpi: int = 160,
    crop: bool = True,
    facecolor: str = "white",
) -> str:
    """Render a Matplotlib figure to PNG bytes, then write it once."""
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=dpi,
        facecolor=facecolor,
        **({"bbox_inches": "tight"} if crop else {}),
    )
    return write_bytes(with_suffix(address, ".png"), buffer.getvalue())

# --------------------------------------------------------------------------- #
# naming a run
# --------------------------------------------------------------------------- #
#: Where the people reading these reports are. Used for the timestamp in a
#: result folder name.
REPORT_TIMEZONE = "America/Toronto"

# Toronto is UTC-5, and UTC-4 during daylight saving. Used only if the zone
# database is unavailable -- a stripped-down container, or a locked-down cluster
# with no tzdata package.
_FALLBACK_STANDARD_OFFSET = -5
_FALLBACK_DAYLIGHT_OFFSET = -4


def _fallback_daylight(moment: "datetime.datetime") -> bool:
    """Is ``moment`` (UTC) inside North American daylight saving time?

    Second Sunday in March to first Sunday in November. Approximate at the hour
    level, which is all a folder name needs.
    """
    year = moment.year
    march = datetime.datetime(year, 3, 8)
    start = march + datetime.timedelta(days=(6 - march.weekday()) % 7)
    november = datetime.datetime(year, 11, 1)
    end = november + datetime.timedelta(days=(6 - november.weekday()) % 7)
    return start <= moment.replace(tzinfo=None) < end


def report_time(zone: str = REPORT_TIMEZONE) -> "datetime.datetime":
    """Current wall-clock time in ``zone``, never raising.

    Nothing here reaches the network: the conversion uses the zone database that
    ships with Python. When that database is missing -- some slim containers and
    locked-down clusters have no ``tzdata`` -- a fixed offset with the standard
    daylight-saving rule is used instead, and if even the clock misbehaves the
    machine's own time is returned. A result folder always gets a name.
    """
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        return utc_now.astimezone(ZoneInfo(zone))
    except Exception:  # noqa: BLE001 - no tz database, or an unknown zone
        pass
    try:
        offset = (
            _FALLBACK_DAYLIGHT_OFFSET
            if _fallback_daylight(utc_now)
            else _FALLBACK_STANDARD_OFFSET
        )
        return utc_now.astimezone(datetime.timezone(datetime.timedelta(hours=offset)))
    except Exception:  # noqa: BLE001 - fall back to whatever the machine says
        return datetime.datetime.now()


def run_stamp(prefix: str = "", zone: str = REPORT_TIMEZONE) -> str:
    """``"drift_20260730_161500"`` -- a short, sortable name for one run."""
    stamp = f"{report_time(zone):%Y%m%d_%H%M%S}"
    return f"{prefix}_{stamp}" if prefix else stamp

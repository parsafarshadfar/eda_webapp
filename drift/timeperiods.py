"""Understanding a date column, whatever shape it arrives in.

Drift is measured over time, so the date column decides how the data is cut into
snapshots. In practice that column is rarely a clean timestamp: it may be text
(``"2019-01"``, ``"Jan 2019"``, ``"15/03/2019"``), a number (``201901``,
``20190115``, ``2019``), a quarter (``"2019Q1"``), a pandas ``Period``, or a
timestamp with a timezone. :func:`parse_time_column` accepts all of those and
returns ordinary timestamps.

:func:`period_labels` then turns those timestamps into snapshot labels.
**Monthly is the default** -- the interval most drift monitoring uses -- and
``Quarterly``, ``Yearly``, ``Weekly`` and ``Daily`` are available with the same
flexibility. Every label sorts chronologically as plain text, so snapshots stay
in order wherever they are displayed or saved.

Only the date column is ever converted, one column at a time, and the parsed
result replaces nothing in the caller's dataframe.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd

__all__ = [
    "DEFAULT_INTERVAL",
    "INTERVALS",
    "describe_span",
    "normalise_interval",
    "parse_time_column",
    "period_labels",
    "snapshot_labels",
]

#: Interval used when none is given: monthly.
DEFAULT_INTERVAL = "Monthly"

#: Supported intervals, in ascending order of length. ``"All"`` means "do not
#: split by time at all: the data is one population".
INTERVALS = ("Daily", "Weekly", "Monthly", "Quarterly", "Yearly", "All")

# Spellings users actually type, mapped to the canonical interval names above.
_INTERVAL_ALIASES = {
    "d": "Daily", "day": "Daily", "days": "Daily", "daily": "Daily",
    "w": "Weekly", "week": "Weekly", "weeks": "Weekly", "weekly": "Weekly",
    "m": "Monthly", "mon": "Monthly", "month": "Monthly", "months": "Monthly",
    "monthly": "Monthly",
    "q": "Quarterly", "quarter": "Quarterly", "quarters": "Quarterly",
    "quarterly": "Quarterly",
    "y": "Yearly", "year": "Yearly", "years": "Yearly", "yearly": "Yearly",
    "a": "Yearly", "annual": "Yearly", "annually": "Yearly",
    "all": "All", "none": "All", "whole": "All", "wholedata": "All",
    "total": "All", "single": "All",
}

# "2019Q1", "2019-Q1", "Q1 2019" and "Q1-2019" all describe the same quarter.
_YEAR_FIRST_QUARTER = re.compile(r"^\s*(\d{4})\s*[-_/ ]?\s*[Qq]\s*([1-4])\s*$")
_QUARTER_FIRST = re.compile(r"^\s*[Qq]\s*([1-4])\s*[-_/ ]?\s*(\d{4})\s*$")

# Digit-only values, by length: 2019 / 201903 / 20190315.
_DIGITS_ONLY = re.compile(r"^\s*\d{4}(\d{2})?(\d{2})?\s*$")


def normalise_interval(interval: Any, *, default: str = DEFAULT_INTERVAL) -> str:
    """Return one of :data:`INTERVALS` from whatever the caller wrote.

    ``None`` and ``""`` fall back to ``default`` (monthly), so an omitted
    interval always has the sensible meaning rather than raising.
    """
    if interval is None:
        return default
    text = str(interval).strip()
    if not text:
        return default
    canonical = _INTERVAL_ALIASES.get(text.replace(" ", "").replace("_", "").lower())
    if canonical is None:
        raise ValueError(
            f"{interval!r} is not a recognised interval. Use one of: {', '.join(INTERVALS)}."
        )
    return canonical


def _from_digits(series: pd.Series) -> pd.Series:
    """Parse compact numeric dates: ``2019``, ``201903`` or ``20190315``.

    The value's own length decides the meaning, so a mixture of lengths in one
    column is still read correctly.
    """
    text = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    lengths = text.str.len()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for length, fmt in ((4, "%Y"), (6, "%Y%m"), (8, "%Y%m%d")):
        selected = lengths == length
        if bool(selected.any()):
            parsed.loc[selected] = pd.to_datetime(
                text.loc[selected], format=fmt, errors="coerce"
            )
    return parsed


def _from_quarters(series: pd.Series) -> pd.Series:
    """Parse quarter labels such as ``2019Q1`` into the quarter's first day."""
    text = series.astype("string").fillna("")
    year_first = text.str.extract(_YEAR_FIRST_QUARTER)
    quarter_first = text.str.extract(_QUARTER_FIRST).rename(columns={0: 1, 1: 0})
    years = year_first[0].fillna(quarter_first[0])
    quarters = year_first[1].fillna(quarter_first[1])

    usable = years.notna() & quarters.notna()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if bool(usable.any()):
        months = (quarters.loc[usable].astype(int) - 1) * 3 + 1
        parsed.loc[usable] = pd.to_datetime(
            {
                "year": years.loc[usable].astype(int),
                "month": months,
                "day": 1,
            }
        )
    return parsed


def _to_datetime(series: pd.Series, **kwargs: Any) -> pd.Series:
    """``pd.to_datetime`` that never raises and never warns about formats."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, errors="coerce", **kwargs)
        except (TypeError, ValueError, OverflowError):
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")


def parse_time_column(values: Any, *, name: str = "date column") -> pd.Series:
    """Convert any reasonable date column into timestamps.

    Strategies are tried in order of certainty and the one that recognises the
    most values wins, so a column does not have to be internally consistent:

    1. already a datetime, ``Period`` or date-like dtype -- used directly;
    2. compact numbers or digit strings (``2019``, ``201903``, ``20190315``);
    3. quarter labels (``2019Q1``, ``Q1-2019``);
    4. general text parsing, retried with ``dayfirst=True`` for
       ``day/month/year`` data.

    Values that cannot be read become ``NaT``; the count is reported once as a
    warning rather than aborting the analysis.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)

    # 1. Nothing to guess.
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.to_timestamp()
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        parsed = series
        # A timezone-aware column is compared against itself only, so dropping
        # the offset keeps labels stable without shifting any observation.
        if getattr(series.dtype, "tz", None) is not None:
            parsed = series.dt.tz_localize(None)
        return _report_unparsed(parsed, series, name)
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype("object")

    # 2. Numbers, and text that is only digits, are compact date codes.
    if pd.api.types.is_numeric_dtype(series.dtype):
        return _report_unparsed(_from_digits(series), series, name)

    text = series.astype("string").str.strip()
    filled = text.notna() & (text.str.len() > 0)
    n_filled = int(filled.sum())
    if n_filled == 0:
        return _report_unparsed(
            pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]"), series, name
        )

    digits = text.str.match(_DIGITS_ONLY).fillna(False)
    if int(digits.sum()) == n_filled:
        return _report_unparsed(_from_digits(text), series, name)

    # 3./4. Try the remaining strategies and keep whichever reads the most.
    candidates = [
        _to_datetime(text),
        _to_datetime(text, dayfirst=True),
        _from_quarters(text),
    ]
    best = max(candidates, key=lambda parsed: int(parsed.notna().sum()))

    # A mixed column (some quarters, some dates) is filled from the runners-up.
    for candidate in candidates:
        if best.isna().any():
            best = best.fillna(candidate)
    return _report_unparsed(best, series, name)


def _report_unparsed(parsed: pd.Series, original: pd.Series, name: str) -> pd.Series:
    """Warn once about values that could not be read as a date."""
    n_missing_before = int(original.isna().sum())
    n_missing_after = int(parsed.isna().sum())
    unreadable = n_missing_after - n_missing_before
    if unreadable > 0:
        warnings.warn(
            f"{unreadable:,} value(s) in {name} could not be read as a date and are "
            "excluded from the time-based snapshots.",
            RuntimeWarning,
            stacklevel=3,
        )
    return parsed


def period_labels(timestamps: pd.Series, interval: Any = DEFAULT_INTERVAL) -> pd.Series:
    """Turn timestamps into sortable snapshot labels.

    ==============  ==================  =========================================
    Interval        Example label       Meaning
    ==============  ==================  =========================================
    ``Daily``       ``2019-03-15``      one snapshot per calendar day
    ``Weekly``      ``2019-W11``        ISO week, so the year never splits a week
    ``Monthly``     ``2019-03``        **default**: one snapshot per month
    ``Quarterly``   ``2019-Q1``         calendar quarter
    ``Yearly``      ``2019``            calendar year
    ==============  ==================  =========================================

    Rows whose timestamp is missing get ``NA`` and are dropped by the caller.
    """
    canonical = normalise_interval(interval)
    if canonical == "All":
        raise ValueError("period_labels() needs a real interval, not 'All'")

    stamps = pd.to_datetime(timestamps, errors="coerce")
    if canonical == "Weekly":
        # ISO week numbering keeps late-December and early-January rows in the
        # same week instead of splitting them across two labels.
        iso = stamps.dt.isocalendar()
        labels = (
            iso["year"].astype("Int64").astype("string")
            + "-W"
            + iso["week"].astype("Int64").astype("string").str.zfill(2)
        )
    else:
        frequency = {
            "Daily": "D",
            "Monthly": "M",
            "Quarterly": "Q",
            "Yearly": "Y",
        }[canonical]
        # ``to_period`` is a vectorised C conversion; ``astype(str)`` on the
        # result already produces "2019-03", "2019Q1" or "2019".
        labels = stamps.dt.to_period(frequency).astype("string")
        if canonical == "Quarterly":
            labels = labels.str.replace("Q", "-Q", regex=False)

    return labels.mask(stamps.isna(), other=pd.NA)


def snapshot_labels(
    values: Any,
    interval: Any = DEFAULT_INTERVAL,
    *,
    name: str = "date column",
) -> pd.Series:
    """Parse a date column and label it in one step.

    This is the entry point :class:`drift.DriftAssessment` uses: it accepts a
    column of any of the supported types and returns the snapshot label for each
    row, ``NA`` where the date could not be read.
    """
    canonical = normalise_interval(interval)
    if canonical == "All":
        raise ValueError("snapshot_labels() needs a real interval, not 'All'")
    return period_labels(parse_time_column(values, name=name), canonical)


def describe_span(timestamps: pd.Series) -> str:
    """``"2019-01-01 -> 2019-03-31"`` for messages and notebook output."""
    stamps = pd.to_datetime(timestamps, errors="coerce").dropna()
    if stamps.empty:
        return "no readable dates"
    return f"{stamps.min():%Y-%m-%d} -> {stamps.max():%Y-%m-%d}"

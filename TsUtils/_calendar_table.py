"""Calendar-oriented reshaping for one dated time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BusinessDay,
    Day,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    Week,
    YearBegin,
    YearEnd,
)

from ._calendar import resolve_frequency, resolve_time_index


def _select_series(data, *, col):
    """Select one public time series without mutating caller-owned data."""
    if isinstance(data, pd.Series):
        if col is not None:
            raise ValueError("col is only valid for DataFrame input")
        return data.copy(deep=False)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas Series or DataFrame")
    if not data.columns.is_unique:
        raise ValueError("DataFrame columns must be unique")
    if col is None:
        if data.shape[1] != 1:
            raise ValueError("col is required for a multi-column DataFrame")
        col = data.columns[0]
    if col not in data.columns:
        raise ValueError(f"col {col!r} is not a DataFrame column")
    return data[col].copy(deep=False)


def _validate_series_values(series):
    """Require finite real numeric observations while allowing missing values."""
    if is_bool_dtype(series.dtype):
        raise TypeError("data must not contain Boolean values")
    if is_complex_dtype(series.dtype):
        raise TypeError("data must not contain complex values")
    if not is_numeric_dtype(series.dtype):
        raise TypeError("data must contain numeric values")

    observed = series.dropna()
    if not np.isfinite(observed.to_numpy()).all():
        raise ValueError("non-missing data values must be finite")


def _complete_series(series, offset):
    """Return a series aligned to its complete resolved calendar."""
    index = series.index
    if isinstance(index, pd.PeriodIndex):
        complete_index = pd.period_range(index[0], index[-1], freq=offset)
    else:
        complete_index = pd.date_range(index[0], index[-1], freq=offset)
    if not index.isin(complete_index).all():
        raise ValueError(f"time index is incompatible with frequency {offset.freqstr!r}")
    return series.reindex(complete_index)


def _classify_frequency(offset):
    """Return the supported unit calendar family for a pandas offset."""
    if offset.n != 1:
        raise ValueError(
            f"calendar_table does not support frequency {offset.freqstr!r}"
        )
    if isinstance(offset, BusinessDay):
        return "business_daily"
    if isinstance(offset, Day):
        return "daily"
    if isinstance(offset, Week):
        return "weekly"
    if isinstance(offset, (BMonthBegin, BMonthEnd, MonthBegin, MonthEnd)):
        return "monthly"
    if isinstance(
        offset,
        (BQuarterBegin, BQuarterEnd, QuarterBegin, QuarterEnd),
    ):
        if isinstance(offset, (BQuarterBegin, QuarterBegin)):
            natural = offset.startingMonth == 1
        else:
            natural = offset.startingMonth == 12
        if natural:
            return "quarterly"
    if isinstance(offset, YearBegin) and offset.month == 1:
        return "annual"
    if isinstance(offset, YearEnd) and offset.month == 12:
        return "annual"
    raise ValueError(f"calendar_table does not support frequency {offset.freqstr!r}")


def _calendar_timestamps(index):
    """Return timestamps carrying natural calendar coordinates."""
    return index.start_time if isinstance(index, pd.PeriodIndex) else index


def _weekly_wednesdays(index):
    """Return the unique Wednesday within each anchored weekly period."""
    if isinstance(index, pd.PeriodIndex):
        starts = index.start_time.normalize()
    else:
        starts = index.normalize() - pd.to_timedelta(6, unit="D")
    days_to_wednesday = (2 - starts.dayofweek) % 7
    return starts + pd.to_timedelta(days_to_wednesday, unit="D")


def _reshape(series, family):
    """Arrange values using the canonical schema for one frequency family."""
    if family == "weekly":
        timestamps = _weekly_wednesdays(series.index)
    else:
        timestamps = _calendar_timestamps(series.index)
    years = timestamps.year
    if family == "annual":
        row_name = "period"
        rows = ["value"] * len(series)
        row_schema = ["value"]
        column_names = ["year"]
        coordinates = {"year": years}
    elif family == "quarterly":
        row_name = "quarter"
        rows = timestamps.quarter
        row_schema = range(1, 5)
        column_names = ["year"]
        coordinates = {"year": years}
    elif family == "monthly":
        row_name = "month"
        rows = timestamps.month
        row_schema = range(1, 13)
        column_names = ["year"]
        coordinates = {"year": years}
    elif family == "weekly":
        row_name = "week"
        rows = ((timestamps.day - 1) // 7) + 1
        row_schema = range(1, 6)
        column_names = ["year", "month"]
        coordinates = {"year": years, "month": timestamps.month}
    elif family in {"daily", "business_daily"}:
        row_name = "day"
        rows = timestamps.day
        row_schema = range(1, 32)
        column_names = ["year", "month"]
        coordinates = {"year": years, "month": timestamps.month}
    else:
        raise ValueError(f"calendar_table does not yet support family {family!r}")

    frame = pd.DataFrame(
        {
            row_name: rows,
            **coordinates,
            "value": series.to_numpy(),
        }
    )
    keys = [row_name, *column_names]
    if frame.duplicated(keys).any():
        raise ValueError("multiple observations map to the same calendar table cell")
    result = frame.set_index(keys)["value"].unstack(column_names)
    return result.reindex(pd.Index(row_schema, name=row_name))


def _apply_label_style(table, *, family, label_style):
    """Apply optional presentation labels without changing table values."""
    if label_style == "numeric":
        return table
    if label_style != "text":
        raise ValueError("label_style must be 'numeric' or 'text'")

    result = table.copy(deep=False)
    if family == "quarterly":
        row_labels = [f"Q{quarter}" for quarter in result.index]
    elif family == "monthly":
        row_labels = [f"{month}月" for month in result.index]
    elif family == "weekly":
        row_labels = [f"第{week}周" for week in result.index]
    elif family in {"daily", "business_daily"}:
        row_labels = [f"{day}日" for day in result.index]
    else:
        row_labels = result.index
    result.index = pd.Index(row_labels, name=result.index.name)

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = pd.MultiIndex.from_tuples(
            [(f"{year}年", f"{month}月") for year, month in result.columns],
            names=result.columns.names,
        )
    else:
        result.columns = pd.Index(
            [f"{year}年" for year in result.columns],
            name=result.columns.name,
        )
    return result


def calendar_table(data, *, col=None, freq=None, label_style="numeric") -> pd.DataFrame:
    """Reshape one dated series into a calendar-oriented table."""
    if label_style not in {"numeric", "text"}:
        raise ValueError("label_style must be 'numeric' or 'text'")
    series = _select_series(data, col=col)
    _validate_series_values(series)
    index = resolve_time_index(series)
    offset = resolve_frequency(index, freq=freq)
    family = _classify_frequency(offset)
    complete = _complete_series(series, offset)
    table = _reshape(complete, family)
    return _apply_label_style(table, family=family, label_style=label_style)

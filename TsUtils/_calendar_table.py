"""Calendar-oriented reshaping for one dated time series."""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import BMonthBegin, BMonthEnd, MonthBegin, MonthEnd

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


def _reshape_monthly(series):
    """Arrange monthly values by natural month and year."""
    index = series.index
    frame = pd.DataFrame(
        {
            "month": index.month,
            "year": index.year,
            "value": series.to_numpy(),
        }
    )
    result = frame.set_index(["month", "year"])["value"].unstack("year")
    return result.reindex(pd.Index(range(1, 13), name="month"))


def calendar_table(data, *, col=None, freq=None, label_style="numeric") -> pd.DataFrame:
    """Reshape one dated monthly series into a calendar-oriented table."""
    series = _select_series(data, col=col)
    index = resolve_time_index(series)
    offset = resolve_frequency(index, freq=freq)
    if offset.n != 1 or not isinstance(
        offset,
        (BMonthBegin, BMonthEnd, MonthBegin, MonthEnd),
    ):
        raise ValueError(f"calendar_table does not support frequency {offset.freqstr!r}")
    complete = _complete_series(series, offset)
    return _reshape_monthly(complete)

"""Recurring calendar-position dummy variables for dated time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
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
)


def _resolve_time_index(data) -> pd.DatetimeIndex | pd.PeriodIndex:
    """Return and validate the time index carried by a public input."""
    if isinstance(data, (pd.DatetimeIndex, pd.PeriodIndex)):
        index = data
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        index = data.index
    else:
        raise TypeError(
            "data must be a Series, DataFrame, DatetimeIndex, or PeriodIndex"
        )

    if not isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("data must have a DatetimeIndex or PeriodIndex")
    if index.empty:
        raise ValueError("time index must not be empty")
    if index.hasnans:
        raise ValueError("time index must not contain missing dates")
    if not index.is_unique:
        raise ValueError("time index must contain unique dates")
    if not index.is_monotonic_increasing:
        raise ValueError("time index must be sorted in increasing order")
    return index


def _resolve_frequency(index):
    """Return a supported seasonal family and its unit pandas offset."""
    frequency = index.freq
    if frequency is None:
        try:
            frequency = pd.infer_freq(index)
        except ValueError as error:
            raise ValueError(
                "time index must have a regular frequency that can be inferred"
            ) from error
    if frequency is None:
        raise ValueError("time index must have a regular frequency that can be inferred")

    offset = to_offset(frequency)
    if offset.n != 1:
        raise ValueError(f"seasonal_dummies does not support frequency {frequency!r}")

    if isinstance(offset, BusinessDay):
        family = "business_daily"
    elif isinstance(offset, Day):
        family = "daily"
    elif isinstance(offset, Week):
        family = "weekly"
    elif isinstance(offset, (BMonthBegin, BMonthEnd, MonthBegin, MonthEnd)):
        family = "monthly"
    elif isinstance(
        offset,
        (BQuarterBegin, BQuarterEnd, QuarterBegin, QuarterEnd),
    ):
        family = "quarterly"
    else:
        raise ValueError(f"seasonal_dummies does not support frequency {frequency!r}")
    return family, offset


def _seasonal_categories(index, family, offset):
    """Return observed labels, the canonical schema, and a column prefix."""
    if family in {"daily", "business_daily"}:
        category_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if family == "business_daily":
            category_names = category_names[:5]
        positions = np.asarray(index.dayofweek, dtype=int)
        labels = [category_names[position] for position in positions]
        return labels, category_names, "weekday"

    if family == "weekly":
        timestamps = index.start_time if isinstance(index, pd.PeriodIndex) else index
        positions = np.asarray(timestamps.isocalendar().week, dtype=int)
        categories = [f"{week:02d}" for week in range(1, 54)]
        labels = [f"{position:02d}" for position in positions]
        return labels, categories, "week"

    if family == "monthly":
        positions = np.asarray(index.month, dtype=int)
        categories = [f"{month:02d}" for month in range(1, 13)]
        labels = [f"{position:02d}" for position in positions]
        return labels, categories, "month"

    months = np.asarray(index.month, dtype=int)
    if isinstance(offset, (BQuarterBegin, QuarterBegin)):
        positions = ((months - offset.startingMonth) % 12) // 3 + 1
    else:
        positions = ((months - offset.startingMonth - 1) % 12) // 3 + 1
    categories = [f"Q{quarter}" for quarter in range(1, 5)]
    labels = [f"Q{position}" for position in positions]
    return labels, categories, "quarter"


def seasonal_dummies(data, *, drop_first=True) -> pd.DataFrame:
    """Return recurring calendar-position dummies aligned to a time series.

    Parameters
    ----------
    data : pandas.Series, pandas.DataFrame, DatetimeIndex, or PeriodIndex
        Dated input whose regular frequency determines the seasonal positions.
        Series and DataFrame values are not read or modified.
    drop_first : bool, default True
        Remove the fixed first category (Monday, ISO week 01, January, or Q1)
        so the matrix can be combined with a model intercept.

    Returns
    -------
    pandas.DataFrame
        ``int8`` dummy columns aligned to the original index. The complete
        canonical schema is retained even when the sample omits categories.

    Raises
    ------
    TypeError
        If the input is unsupported, lacks a pandas time index, or
        ``drop_first`` is not Boolean.
    ValueError
        If the index is empty, invalid, irregular, or has an unsupported
        annual, intraday, multiplied, or other frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> from Ts.TsUtils import seasonal_dummies
    >>> monthly = pd.date_range("2024-01-01", periods=3, freq="MS")
    >>> dummies = seasonal_dummies(monthly)
    >>> list(dummies.columns[:2])
    ['month_02', 'month_03']
    >>> dummies.loc[monthly[0]].sum()
    np.int64(0)

    >>> quarters = pd.period_range("2024Q1", periods=4, freq="Q-DEC")
    >>> seasonal_dummies(quarters, drop_first=False).sum(axis=1).tolist()
    [1, 1, 1, 1]
    """
    if not isinstance(drop_first, (bool, np.bool_)):
        raise TypeError("drop_first must be a boolean")
    index = _resolve_time_index(data)
    family, offset = _resolve_frequency(index)
    labels, categories, prefix = _seasonal_categories(index, family, offset)
    categorical = pd.Categorical(labels, categories=categories, ordered=True)
    result = pd.get_dummies(
        categorical,
        prefix=prefix,
        prefix_sep="_",
        dtype=np.int8,
    )
    result.index = index
    if drop_first:
        result = result.iloc[:, 1:]
    return result

"""Shared private validation for calendar-aware utilities."""

from __future__ import annotations

import pandas as pd
from pandas.tseries.frequencies import to_offset


def resolve_time_index(data) -> pd.DatetimeIndex | pd.PeriodIndex:
    """Return and validate the time index carried by an internal input."""
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


def resolve_frequency(
    index: pd.DatetimeIndex | pd.PeriodIndex,
    *,
    freq: str | None = None,
):
    """Return a pandas offset from explicit, stored, or inferred frequency."""
    if freq is not None:
        if not isinstance(freq, str):
            raise TypeError("freq must be a pandas frequency string or None")
        frequency = freq
    else:
        frequency = index.freq
        if frequency is None:
            try:
                frequency = pd.infer_freq(index)
            except ValueError as error:
                raise ValueError(
                    "time index must have a regular frequency that can be inferred"
                ) from error
        if frequency is None:
            raise ValueError(
                "time index must have a regular frequency that can be inferred"
            )

    try:
        return to_offset(frequency)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid frequency {frequency!r}") from error

"""Resolution of explicit estimation and validation periods."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validated_model_dates(model, data):
    """Return a strict model calendar or None for position-based data."""
    values = getattr(model, "dates", None)
    if values is None:
        return None
    try:
        dates = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as error:
        raise TypeError("model.dates must be datetime-like") from error
    if len(dates) != len(data):
        raise ValueError("model.dates must contain one date per observation")
    if dates.hasnans:
        raise ValueError("model.dates must not contain missing dates")
    if not dates.is_unique:
        raise ValueError("model.dates must be unique")
    if not dates.is_monotonic_increasing:
        raise ValueError("model.dates must be strictly increasing")
    return dates.copy()


def _period_pair(name, period):
    if not isinstance(period, (tuple, list)) or len(period) != 2:
        raise TypeError(f"{name} must be a (start, end) pair")
    return period[0], period[1]


def _position_bound(name, bound, nobs):
    if isinstance(bound, (bool, np.bool_)) or not isinstance(
        bound,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer position because the model has no dates"
        )
    position = int(bound)
    if position < 0 or position >= nobs:
        raise ValueError(
            f"{name}={position} is outside the observed positions 0..{nobs - 1}"
        )
    return position


def _date_bound(name, bound, dates):
    try:
        timestamp = pd.Timestamp(bound)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be datetime-like") from error
    if str(timestamp.tz) != str(dates.tz):
        raise ValueError(f"{name} timezone must match model.dates")
    position = int(dates.get_indexer([timestamp])[0])
    if position < 0:
        raise ValueError(
            f"{name}={timestamp.isoformat()} does not exist in model.dates"
        )
    return position


def _resolve_period(name, period, data, dates):
    start_bound, end_bound = _period_pair(name, period)
    if dates is None:
        start = _position_bound(f"{name} start", start_bound, len(data))
        end = _position_bound(f"{name} end", end_bound, len(data))
    else:
        start = _date_bound(f"{name} start", start_bound, dates)
        end = _date_bound(f"{name} end", end_bound, dates)
    if start > end:
        raise ValueError(f"{name} start must not be later than its end")
    return start, end + 1

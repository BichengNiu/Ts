"""Dated intervention specifications and policy-effect utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Tick,
    Week,
    YearBegin,
    YearEnd,
)

DateRule = Literal["exact", "period", "next", "previous"]
EventKind = Literal["pulse", "step"]


@dataclass(frozen=True)
class EventSpec:
    """Immutable definition of one named intervention."""

    name: str
    dates: Sequence[object]
    kind: EventKind
    window: tuple[int, int] | None = None
    reference: int | None = None
    date_rule: DateRule = "period"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("event name must be a non-empty string")
        if self.kind not in {"pulse", "step"}:
            raise ValueError("event kind must be 'pulse' or 'step'")
        if self.date_rule not in {
            "exact",
            "period",
            "next",
            "previous",
        }:
            raise ValueError(
                "date_rule must be 'exact', 'period', 'next', or 'previous'"
            )

        try:
            parsed_dates = tuple(pd.Timestamp(value) for value in self.dates)
        except (TypeError, ValueError) as error:
            raise ValueError("event dates must contain valid dates") from error
        if not parsed_dates:
            raise ValueError("event dates must not be empty")
        if len(set(parsed_dates)) != len(parsed_dates):
            raise ValueError(
                f"event {self.name.strip()!r} contains duplicate dates"
            )

        if self.kind == "step" and self.window is not None:
            raise ValueError("window is only valid for pulse events")
        if (self.window is None) != (self.reference is None):
            raise ValueError("window and reference must be specified together")
        if self.window is not None:
            if not isinstance(self.window, tuple) or len(self.window) != 2:
                raise ValueError(
                    "window must be an ordered pair of integers"
                )
            start, end = self.window
            invalid_bound = (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
            )
            if invalid_bound or start > end:
                raise ValueError(
                    "window must be an ordered pair of integers"
                )
            if (
                isinstance(self.reference, bool)
                or not isinstance(self.reference, int)
                or self.reference < start
                or self.reference > end
            ):
                raise ValueError("reference must lie inside window")

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "dates", parsed_dates)


@dataclass(frozen=True)
class EventColumns:
    """Generated design-column metadata for one event."""

    name: str
    columns: tuple[str, ...]
    relative_periods: tuple[int, ...] | None
    mapped_positions: tuple[int, ...]


_START_OFFSETS = (
    BMonthBegin,
    BQuarterBegin,
    BYearBegin,
    MonthBegin,
    QuarterBegin,
    SemiMonthBegin,
    YearBegin,
)
_END_OFFSETS = (
    BMonthEnd,
    BQuarterEnd,
    BYearEnd,
    MonthEnd,
    QuarterEnd,
    SemiMonthEnd,
    Week,
    YearEnd,
)


def _validate_datetime_index(values, name: str) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be valid datetime values") from error
    if index.empty:
        raise ValueError(f"{name} must not be empty")
    if index.hasnans:
        raise ValueError(f"{name} must not contain missing dates")
    if not index.is_unique:
        raise ValueError(f"{name} must contain unique dates")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted in increasing order")
    return index


def _same_timezone(left, right) -> bool:
    return str(left) == str(right)


def _period_label(
    event_date: pd.Timestamp,
    calendar: pd.DatetimeIndex,
) -> pd.Timestamp:
    frequency = calendar.freq or pd.infer_freq(calendar)
    if frequency is None:
        raise ValueError(
            "date_rule='period' requires a regular calendar frequency"
        )
    offset = to_offset(frequency)
    normalized = event_date.normalize()
    if isinstance(offset, Tick):
        return event_date.floor(offset)
    if isinstance(offset, _START_OFFSETS):
        return offset.rollback(normalized)
    if isinstance(offset, _END_OFFSETS):
        return offset.rollforward(normalized)
    raise ValueError(
        f"date_rule='period' does not support frequency {frequency!r}"
    )


def _mapped_position(
    event_date: pd.Timestamp,
    rule: DateRule,
    calendar: pd.DatetimeIndex,
) -> int | None:
    if not _same_timezone(event_date.tz, calendar.tz):
        raise ValueError(
            f"event date {event_date} timezone does not match calendar timezone"
        )

    if rule == "period":
        candidate = _period_label(event_date, calendar)
        position = int(calendar.get_indexer([candidate])[0])
        if position >= 0:
            return position
        if candidate < calendar[0] or candidate > calendar[-1]:
            return None
        raise ValueError(
            f"period mapping for {event_date} is absent from the calendar"
        )

    if event_date < calendar[0] or event_date > calendar[-1]:
        return None
    if rule == "exact":
        position = int(calendar.get_indexer([event_date])[0])
        if position < 0:
            raise ValueError(
                f"exact event date {event_date.date()} is absent "
                "from the calendar"
            )
        return position
    if rule == "next":
        return int(calendar.searchsorted(event_date, side="left"))
    return int(calendar.searchsorted(event_date, side="right") - 1)


def _relative_suffix(relative: int) -> str:
    if relative < 0:
        return f"m{abs(relative)}"
    return f"p{relative}"


def _event_schema(
    event: EventSpec,
) -> tuple[tuple[str, ...], tuple[int, ...] | None]:
    base = f"event__{event.name}"
    if event.window is None:
        return (base,), None
    start, end = event.window
    relative_periods = tuple(
        relative
        for relative in range(start, end + 1)
        if relative != event.reference
    )
    columns = tuple(
        f"{base}__{_relative_suffix(relative)}"
        for relative in relative_periods
    )
    return columns, relative_periods


def build_event_matrix(
    target_dates: pd.DatetimeIndex,
    events: Sequence[EventSpec],
    *,
    calendar: pd.DatetimeIndex | None = None,
    reserved_names: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[str, EventColumns]]:
    """Build event regressors on target dates using a complete calendar."""
    target_index = _validate_datetime_index(target_dates, "target dates")
    calendar_index = _validate_datetime_index(
        target_index if calendar is None else calendar,
        "calendar",
    )
    target_positions = calendar_index.get_indexer(target_index)
    if np.any(target_positions < 0):
        raise ValueError(
            "target dates must be an exact subset of the complete calendar"
        )

    event_specs = tuple(events)
    event_names = [event.name for event in event_specs]
    if len(set(event_names)) != len(event_names):
        raise ValueError("duplicate event name in events")

    reserved = set(reserved_names)
    all_columns: list[str] = []
    schemas: dict[
        str,
        tuple[tuple[str, ...], tuple[int, ...] | None],
    ] = {}
    for event in event_specs:
        columns, relative_periods = _event_schema(event)
        collisions = reserved.intersection(columns)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"event column collision: {names}")
        reserved.update(columns)
        all_columns.extend(columns)
        schemas[event.name] = columns, relative_periods

    full = pd.DataFrame(
        0.0,
        index=calendar_index,
        columns=all_columns,
        dtype=float,
    )
    metadata: dict[str, EventColumns] = {}
    for event in event_specs:
        columns, relative_periods = schemas[event.name]
        positions = tuple(
            position
            for event_date in event.dates
            if (
                position := _mapped_position(
                    event_date,
                    event.date_rule,
                    calendar_index,
                )
            )
            is not None
        )
        if event.kind == "step":
            for position in positions:
                full.iloc[position:, full.columns.get_loc(columns[0])] += 1.0
        elif relative_periods is None:
            for position in positions:
                full.iloc[position, full.columns.get_loc(columns[0])] += 1.0
        else:
            for position in positions:
                for column, relative in zip(
                    columns,
                    relative_periods,
                    strict=True,
                ):
                    shifted = position + relative
                    if 0 <= shifted < len(full):
                        full.iloc[
                            shifted,
                            full.columns.get_loc(column),
                        ] += 1.0
        metadata[event.name] = EventColumns(
            name=event.name,
            columns=columns,
            relative_periods=relative_periods,
            mapped_positions=positions,
        )

    matrix = full.iloc[target_positions].copy()
    matrix.index = target_index
    return matrix, metadata

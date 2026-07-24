"""Dated intervention specifications and policy-effect utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

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

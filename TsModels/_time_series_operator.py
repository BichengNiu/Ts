"""Validated per-variable time-series operators for model exogenous inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Ts.TsUtils import difference
from Ts.TsUtils._validation import validate_int


@dataclass(frozen=True)
class TimeSeriesOperator:
    """Immutable transformation specification for one exogenous variable.

    Parameters
    ----------
    lag : int, default 0
        Non-negative positional lag applied after differencing.
    difference : int, default 0
        Number of ordinary first differences.
    seasonal_difference : int, default 0
        Number of seasonal first differences.
    seasonal_period : int or None, default None
        Seasonal lag required when ``seasonal_difference`` is positive.
    log : bool, default False
        Whether to take the natural logarithm before differencing and lagging.

    Examples
    --------
    >>> from Ts.TsModels import TimeSeriesOperator
    >>> operator = TimeSeriesOperator(lag=1, difference=1)
    >>> operator.required_history
    2
    """

    lag: int = 0
    difference: int = 0
    seasonal_difference: int = 0
    seasonal_period: int | None = None
    log: bool = False

    def __post_init__(self):
        for name in ("lag", "difference", "seasonal_difference"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, validate_int(name, value, minimum=0))
        if self.seasonal_difference:
            period = self.seasonal_period
            if isinstance(period, (bool, np.bool_)):
                raise TypeError("seasonal_period must be an integer or None")
            object.__setattr__(
                self,
                "seasonal_period",
                validate_int("seasonal_period", period, minimum=2),
            )
        elif self.seasonal_period is not None:
            if isinstance(self.seasonal_period, (bool, np.bool_)):
                raise TypeError("seasonal_period must be an integer or None")
            object.__setattr__(
                self,
                "seasonal_period",
                validate_int("seasonal_period", self.seasonal_period, minimum=2),
            )
        if not isinstance(self.log, (bool, np.bool_)):
            raise TypeError("log must be a boolean")
        object.__setattr__(self, "log", bool(self.log))

    @property
    def required_history(self) -> int:
        """Return leading raw observations unavailable after this transformation."""
        seasonal = self.seasonal_difference * (self.seasonal_period or 0)
        return self.lag + self.difference + seasonal

    @property
    def is_identity(self) -> bool:
        """Whether this operator leaves a variable unchanged."""
        return self.required_history == 0 and not self.log

    def transformed_name(self, name: str) -> str:
        """Return the parameter name after applying this operator.

        Parameters
        ----------
        name : str
            Raw exogenous-variable name.

        Returns
        -------
        str
            Fitted parameter name with transformation prefixes.
        """
        prefixes = []
        if self.log:
            prefixes.append("log.")
        if self.lag:
            prefixes.append("L.")
        if self.difference:
            prefixes.append("D.")
        if self.seasonal_difference:
            prefixes.append("S.")
        return "".join(prefixes) + name


def normalise_exog_operators(operators, exog_names) -> dict[str, TimeSeriesOperator]:
    """Validate an operator mapping against named exogenous columns."""
    if operators is None:
        return {}
    if not isinstance(operators, Mapping):
        raise TypeError(
            "exog_operators must be a mapping from exogenous name to "
            "TimeSeriesOperator"
        )
    names = tuple(exog_names)
    resolved = {}
    for name, operator in operators.items():
        if name not in names:
            raise ValueError(f"exog_operators contains unknown exogenous variable {name!r}")
        if not isinstance(operator, TimeSeriesOperator):
            raise TypeError(
                f"exog_operators[{name!r}] must be a TimeSeriesOperator"
            )
        if not operator.is_identity:
            resolved[name] = operator
    return resolved


def operator_burn(operators) -> int:
    """Return the maximum deterministic leading loss across operators."""
    return max((operator.required_history for operator in operators.values()), default=0)


def _transformed_exog_names(exog_names, operators) -> tuple[str, ...]:
    """Return unique fitted-column names for raw exogenous names and operators."""
    names = tuple(
        operators.get(name, TimeSeriesOperator()).transformed_name(name)
        for name in exog_names
    )
    if len(set(names)) != len(names):
        raise ValueError(
            "exogenous transformations produce duplicate parameter names: "
            f"{names}"
        )
    return names


def apply_exog_operators(frame: pd.DataFrame, operators) -> pd.DataFrame:
    """Apply log, seasonal differences, ordinary differences, then lags by column."""
    transformed = frame.astype(float).copy()
    for name, operator in operators.items():
        values = transformed[name]
        if operator.log:
            if (values <= 0.0).any():
                raise ValueError(
                    f"log transformation requires strictly positive exogenous "
                    f"data for {name!r}"
                )
            values = np.log(values)
        if operator.seasonal_difference:
            for _ in range(operator.seasonal_difference):
                values = difference(values, order=1, lag=operator.seasonal_period)
        if operator.difference:
            for _ in range(operator.difference):
                values = difference(values, order=1)
        if operator.lag:
            values = values.shift(operator.lag)
        transformed[name] = values
    return transformed


__all__ = ["TimeSeriesOperator"]

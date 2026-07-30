"""Extended autocorrelation function for nonseasonal ARMA order identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._summary import _as_numeric_series
from ._validation import validate_positive_int


@dataclass(frozen=True)
class EACFResult:
    """Immutable numeric and coded extended-autocorrelation tables."""

    values: np.ndarray
    significant: np.ndarray
    symbols: np.ndarray
    ar_orders: np.ndarray
    ma_orders: np.ndarray
    nobs: int

    def __post_init__(self):
        arrays = (
            ("values", self.values, float),
            ("significant", self.significant, bool),
            ("symbols", self.symbols, str),
            ("ar_orders", self.ar_orders, int),
            ("ma_orders", self.ma_orders, int),
        )
        for name, value, dtype in arrays:
            array = np.array(value, dtype=dtype, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def summary(self) -> str:
        """Return the conventional ``x``/``o`` EACF table."""
        width = max(5, *(len(str(order)) + 1 for order in self.ma_orders))
        header = "AR\\MA".rjust(width) + "".join(
            str(order).rjust(width) for order in self.ma_orders
        )
        lines = [
            "Extended Autocorrelation Function",
            f"Observations: {self.nobs}",
            header,
        ]
        for order, row in zip(self.ar_orders, self.symbols, strict=True):
            lines.append(
                str(order).rjust(width) + "".join(symbol.rjust(width) for symbol in row)
            )
        lines.extend(
            [
                "x = significant",
                "o = not significant",
            ]
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _selected_data(data, variable):
    """Return an unambiguous selection for pre-conversion dtype inspection."""
    if isinstance(data, pd.DataFrame):
        if variable is None:
            return data.iloc[:, 0] if data.shape[1] == 1 else None
        if variable not in data.columns:
            return None
        selected = data.loc[:, variable]
        return selected if isinstance(selected, pd.Series) else None
    if isinstance(data, pd.Series):
        return data if variable is None else None
    return np.asarray(data)


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    """Return the biased sample autocorrelation used by classical EACF."""
    centered = values - np.mean(values)
    denominator = float(centered @ centered)
    if denominator <= np.finfo(float).eps:
        raise ValueError("EACF is not defined for a constant series")
    return float(centered[lag:] @ centered[:-lag] / denominator)


def _initial_ar_coefficients(values: np.ndarray, maximum_order: int) -> np.ndarray:
    """Fit the nested no-intercept AR regressions needed by the recursion."""
    matrix = np.zeros((maximum_order + 1, maximum_order), dtype=float)
    nobs = len(values)
    for order in range(1, maximum_order + 1):
        regressors = np.column_stack(
            [values[order - lag : nobs - lag] for lag in range(1, order + 1)]
        )
        coefficients, _, rank, _ = np.linalg.lstsq(
            regressors,
            values[order:],
            rcond=None,
        )
        if rank < order:
            raise ValueError(
                "EACF is undefined because the lagged AR design matrix "
                f"is rank deficient at order {order}"
            )
        matrix[:order, order - 1] = coefficients
    return matrix


def _reduce_coefficients(matrix: np.ndarray) -> np.ndarray:
    """Advance the iterative least-squares coefficient recursion by one step."""
    nrows, ncols = matrix.shape
    reduced = np.empty((nrows, ncols - 1), dtype=float)
    for column in range(ncols - 1):
        pivot = matrix[column, column]
        scale = max(1.0, float(np.max(np.abs(matrix[:, column]))))
        if abs(pivot) <= np.finfo(float).eps * scale:
            raise ValueError(
                "EACF recursion is undefined because an intermediate "
                f"AR coefficient is zero at order {column + 1}"
            )

        shifted = np.empty(nrows, dtype=float)
        shifted[0] = np.nan
        shifted[1:] = matrix[:-1, column]
        shifted[0] = -1.0
        ratio = matrix[column + 1, column + 1] / pivot
        updated = matrix[:, column + 1] - shifted * ratio
        updated[column + 1] = 0.0
        reduced[:, column] = updated
    return reduced


def eacf(data, ar_max=7, ma_max=13, *, variable=None) -> EACFResult:
    """Compute the classical EACF table for nonseasonal ARMA identification.

    Parameters
    ----------
    data : array-like, pandas.Series, or pandas.DataFrame
        One numeric, complete, nonconstant time series. A multi-column
        DataFrame requires ``variable``.
    ar_max, ma_max : int, default 7 and 13
        Largest AR and MA orders displayed in the table.
    variable : hashable, optional
        DataFrame column to analyse.

    Returns
    -------
    EACFResult
        Numeric EACF values, significance decisions, and ``x``/``o`` codes.

    Notes
    -----
    ``o`` denotes an entry within ``2 / sqrt(n - p - q - 1)`` of zero.
    A triangular wedge of ``o`` entries is an order-identification heuristic,
    not an automatic or unique model-selection rule.
    """
    ar_max = validate_positive_int("ar_max", ar_max, minimum=0)
    ma_max = validate_positive_int("ma_max", ma_max, minimum=0)
    selected = _selected_data(data, variable)
    if selected is not None and pd.api.types.is_bool_dtype(selected.dtype):
        raise TypeError("data must contain numeric, non-boolean values")
    if selected is not None and pd.api.types.is_complex_dtype(selected.dtype):
        raise TypeError("data must contain real numeric values")
    series = _as_numeric_series(data, variable=variable)
    if series.isna().any():
        raise ValueError("data must not contain missing values")

    values = series.to_numpy(dtype=float, copy=True)
    if np.ptp(values) <= np.finfo(float).eps * max(1.0, np.max(np.abs(values))):
        raise ValueError("EACF is not defined for a constant series")

    maximum_order = ar_max + ma_max + 1
    minimum_nobs = ar_max + ma_max + 4
    if ar_max:
        minimum_nobs = max(minimum_nobs, 2 * maximum_order + 1)
    if len(values) < minimum_nobs:
        raise ValueError(
            f"data must contain at least {minimum_nobs} observations for "
            f"ar_max={ar_max} and ma_max={ma_max}"
        )

    centered = values - np.mean(values)
    table = np.empty((ar_max + 1, ma_max + 1), dtype=float)
    if ar_max == 0:
        for ma_order in range(ma_max + 1):
            table[0, ma_order] = _autocorrelation(centered, ma_order + 1)
    else:
        coefficients = _initial_ar_coefficients(centered, maximum_order)

        for ma_order in range(ma_max + 1):
            coefficients = _reduce_coefficients(coefficients)
            lag = ma_order + 1
            table[0, ma_order] = _autocorrelation(centered, lag)
            for ar_order in range(1, ar_max + 1):
                regressors = np.column_stack(
                    [
                        centered[ar_order - ar_lag : len(centered) - ar_lag]
                        for ar_lag in range(1, ar_order + 1)
                    ]
                )
                residuals = centered[ar_order:] - (
                    regressors @ coefficients[:ar_order, ar_order - 1]
                )
                table[ar_order, ma_order] = _autocorrelation(residuals, lag)

    ar_orders = np.arange(ar_max + 1)
    ma_orders = np.arange(ma_max + 1)
    effective_nobs = (
        len(values) - ar_orders[:, np.newaxis] - ma_orders[np.newaxis, :] - 1
    )
    thresholds = 2.0 / np.sqrt(effective_nobs)
    significant = np.abs(table) > thresholds
    symbols = np.where(significant, "x", "o")
    return EACFResult(
        values=table,
        significant=significant,
        symbols=symbols,
        ar_orders=ar_orders,
        ma_orders=ma_orders,
        nobs=len(values),
    )

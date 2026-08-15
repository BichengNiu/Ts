"""Shared input and design-matrix helpers for regression stability tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from Ts.TsUtils._validation import validate_choice

from ._break_utils import _validate_time_axis
from ._utils import _parse_input


@dataclass(frozen=True)
class RegressionBreakDesign:
    """Validated regression data whose rows preserve breakpoint alignment."""

    endog: np.ndarray
    exog: np.ndarray
    time_index: np.ndarray
    column_names: tuple[str, ...]


def _column_values(
    frame: pd.DataFrame,
    columns: list[str | int],
) -> tuple[np.ndarray, list[str]]:
    """Read named/positional DataFrame columns without flattening rows."""
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for column in columns:
        if isinstance(column, bool) or not isinstance(column, (str, int)):
            raise TypeError("exog_cols entries must be column names or indices")
        if isinstance(column, int):
            try:
                series = frame.iloc[:, column]
            except IndexError as exc:
                raise ValueError(f"exog column index {column} is out of range") from exc
            name = str(frame.columns[column])
        else:
            if column not in frame.columns:
                raise ValueError(f"exog column {column!r} was not found")
            series = frame[column]
            name = str(column)
        arrays.append(series.to_numpy(dtype=float))
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("exog_cols must not contain duplicate columns")
    return np.column_stack(arrays), names


def _as_exog_matrix(exog: Any, nobs: int) -> tuple[np.ndarray, list[str]]:
    """Convert explicit exogenous data to a two-dimensional float matrix."""
    if isinstance(exog, pd.DataFrame):
        if not exog.columns.is_unique:
            raise ValueError("exog DataFrame columns must be unique")
        values = exog.to_numpy(dtype=float)
        names = [str(name) for name in exog.columns]
    elif isinstance(exog, pd.Series):
        values = exog.to_numpy(dtype=float)[:, None]
        names = [str(exog.name) if exog.name is not None else "x1"]
    else:
        values = np.asarray(exog, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError(f"exog must be 1-D or 2-D, got shape {values.shape}")
        names = [f"x{i + 1}" for i in range(values.shape[1])]
    if values.shape[0] != nobs:
        raise ValueError(
            f"exog length ({values.shape[0]}) must match data length ({nobs})"
        )
    return values, names


def _coefficient_dict(params: np.ndarray, column_names) -> dict[str, float]:
    """Map one flat coefficient vector onto the named regression columns."""
    return {
        name: float(params[position])
        for position, name in enumerate(column_names)
    }


def _prepare_regression_break_design(
    data,
    *,
    exog=None,
    time_index=None,
    trend: str = "c",
    y_col: str | int | None = None,
    time_col: str | int | None = None,
    exog_cols: list[str | int] | tuple[str | int, ...] | None = None,
) -> RegressionBreakDesign:
    """Build a validated OLS design for Chow, CUSUM, and Bai-Perron tests.

    Observation rows are never dropped or reordered because doing so changes
    breakpoint locations. The deterministic trend is positional; the supplied
    time axis is retained only for labels and breakpoint mapping.
    """
    validate_choice("trend", trend, ("n", "c", "ct"))
    if isinstance(data, pd.DataFrame) and not data.columns.is_unique:
        raise ValueError("data DataFrame columns must be unique")
    if exog is not None and exog_cols is not None:
        raise ValueError("exog and exog_cols cannot be used together")
    if exog_cols is not None and not isinstance(data, pd.DataFrame):
        raise ValueError("exog_cols is valid only when data is a DataFrame")

    endog, labels = _parse_input(
        data,
        time_index=time_index,
        y_col=y_col,
        time_col=time_col,
    )
    _validate_time_axis(labels)
    nobs = len(endog)

    matrices: list[np.ndarray] = []
    names: list[str] = []
    if trend in ("c", "ct"):
        matrices.append(np.ones((nobs, 1), dtype=float))
        names.append("const")
    if trend == "ct":
        matrices.append(np.arange(nobs, dtype=float)[:, None])
        names.append("trend")

    if exog_cols is not None:
        if len(exog_cols) == 0:
            raise ValueError("exog_cols must contain at least one column")
        exog_values, exog_names = _column_values(data, list(exog_cols))
        matrices.append(exog_values)
        names.extend(exog_names)
    elif exog is not None:
        exog_values, exog_names = _as_exog_matrix(exog, nobs)
        if exog_values.shape[1] == 0:
            raise ValueError("exog must contain at least one column")
        matrices.append(exog_values)
        names.extend(exog_names)

    if not matrices:
        raise ValueError("trend='n' requires at least one exogenous regressor")
    design = np.column_stack(matrices).astype(float, copy=False)
    if not np.all(np.isfinite(design)):
        raise ValueError("exog and deterministic regressors must be finite")
    if nobs <= design.shape[1]:
        raise ValueError(
            "regression requires more observations than parameters "
            f"({nobs} observations, {design.shape[1]} parameters)"
        )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("regression design matrix must have full column rank")

    return RegressionBreakDesign(
        endog=np.asarray(endog, dtype=float),
        exog=design,
        time_index=np.asarray(labels, dtype=float),
        column_names=tuple(names),
    )

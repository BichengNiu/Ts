"""Auditable Box-Cox transformations for pandas time-series containers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from scipy import stats

from ._difference import _as_float_data


@dataclass(frozen=True)
class BoxCoxResult:
    """Box-Cox transformed data and the lambda values used.

    Attributes
    ----------
    data : pandas.Series or pandas.DataFrame
        Transformed values with the original pandas metadata and shape.
    lmbda : float or pandas.Series
        The fitted or supplied lambda. DataFrame results use one value per
        column in a Series named ``"lmbda"``.
    """

    data: pd.Series | pd.DataFrame
    lmbda: float | pd.Series


def _validate_scalar_lmbda(value, *, context="lmbda") -> float:
    """Return a finite, real, non-boolean Box-Cox parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Real, np.integer, np.floating),
    ):
        raise TypeError(f"{context} must be a finite real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{context} must be finite, got {value}")
    return value


def _resolve_dataframe_lmbdas(data, lmbda):
    """Return per-column lambdas, using ``None`` for fitted parameters."""
    if lmbda is None:
        return pd.Series(
            [None] * data.shape[1],
            index=data.columns,
            dtype=object,
            name="lmbda",
        )

    if isinstance(lmbda, (Real, np.integer, np.floating)) and not isinstance(
        lmbda,
        (bool, np.bool_),
    ):
        scalar = _validate_scalar_lmbda(lmbda)
        return pd.Series(
            [scalar] * data.shape[1],
            index=data.columns,
            dtype=float,
            name="lmbda",
        )

    if isinstance(lmbda, pd.Series):
        if not lmbda.index.is_unique:
            raise ValueError("lmbda labels must be unique")
        supplied = lmbda.copy()
    elif isinstance(lmbda, Mapping):
        supplied = pd.Series(dict(lmbda), dtype=object)
    else:
        raise TypeError(
            "DataFrame lmbda must be None, a finite real number, "
            "or a mapping indexed by columns"
        )

    if set(supplied.index) != set(data.columns):
        raise ValueError("lmbda labels must exactly match DataFrame columns")

    supplied = supplied.reindex(data.columns)
    values = [
        _validate_scalar_lmbda(value, context=f"lmbda[{column!r}]")
        for column, value in supplied.items()
    ]
    return pd.Series(values, index=data.columns, dtype=float, name="lmbda")


def _transform_values(values, lmbda, *, label):
    """Transform one one-dimensional array and return data plus lambda."""
    observed = values[~np.isnan(values)]
    if lmbda is None:
        if observed.size < 2:
            raise ValueError(
                f"{label} must contain at least two observed values to estimate lmbda"
            )
        if np.unique(observed).size == 1:
            raise ValueError(f"{label} must not be constant to estimate lmbda")
        transformed_observed, fitted_lmbda = stats.boxcox(observed)
        transformed = np.full(values.shape, np.nan, dtype=float)
        transformed[~np.isnan(values)] = transformed_observed
        return transformed, float(fitted_lmbda)

    transformed = stats.boxcox(values, lmbda=lmbda)
    return np.asarray(transformed, dtype=float), float(lmbda)


def boxcox(data, *, lmbda=None):
    """Apply a Box-Cox power transformation to positive pandas data.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Real numeric observations. Missing values are allowed and retain their
        positions. Every observed value must be strictly positive.
    lmbda : float, mapping, pandas.Series, or None, default None
        A finite transformation parameter. ``None`` estimates the
        maximum-likelihood lambda independently for each series. DataFrame
        callers may provide one scalar for every column or a mapping whose
        labels exactly match the columns.

    Returns
    -------
    BoxCoxResult
        The transformed pandas object and the fitted or supplied lambda values.

    Notes
    -----
    The function never shifts non-positive observations automatically. Estimate
    lambda on training data and pass the recorded value to later data when a
    transformation is used in a forecasting workflow.
    """
    converted = _as_float_data(data)
    values = converted.to_numpy(dtype=float, na_value=np.nan)
    observed = values[~np.isnan(values)]
    if np.any(observed <= 0.0):
        raise ValueError("Box-Cox transformation requires strictly positive values")

    if isinstance(data, pd.Series):
        resolved_lmbda = None if lmbda is None else _validate_scalar_lmbda(lmbda)
        transformed, fitted_lmbda = _transform_values(
            values,
            resolved_lmbda,
            label="data",
        )
        transformed_data = pd.Series(
            transformed,
            index=data.index,
            name=data.name,
        )
        transformed_data.index = data.index
        return BoxCoxResult(data=transformed_data, lmbda=fitted_lmbda)

    resolved_lmbdas = _resolve_dataframe_lmbdas(data, lmbda)
    transformed_columns = []
    fitted_lmbdas = []
    for position, column in enumerate(data.columns):
        column_lmbda = resolved_lmbdas.iloc[position]
        resolved_column_lmbda = None if column_lmbda is None else float(column_lmbda)
        transformed, fitted_lmbda = _transform_values(
            values[:, position],
            resolved_column_lmbda,
            label=f"column {column!r}",
        )
        transformed_columns.append(transformed)
        fitted_lmbdas.append(fitted_lmbda)

    transformed_data = pd.DataFrame(
        np.column_stack(transformed_columns),
        index=data.index,
        columns=data.columns,
    )
    transformed_data.index = data.index
    transformed_data.columns = data.columns
    fitted = pd.Series(
        fitted_lmbdas,
        index=data.columns,
        dtype=float,
        name="lmbda",
    )
    return BoxCoxResult(data=transformed_data, lmbda=fitted)

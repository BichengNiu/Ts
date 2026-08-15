"""General-purpose utility functions shared across all TsTests modules.

Provides input parsing and validation. Structural-break-specific utilities
live in ``_break_utils.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from Ts.TsUtils._validation import (
    as_1d_float as _as_1d_float,
    validate_alpha,
    validate_positive_int,
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_MODELS = ("intercept", "slope", "both")


def _validate_model(model: str) -> None:
    """Validate that *model* is one of the allowed values.

    Parameters
    ----------
    model : str
        Model specification to validate.

    Raises
    ------
    ValueError
        If *model* is not one of ``"intercept"``, ``"slope"``, or ``"both"``.
    """
    if model not in _VALID_MODELS:
        raise ValueError(
            f"model must be 'intercept', 'slope', or 'both', got {model!r}"
        )


def _validate_lags(lags) -> int:
    """Validate that *lags* is a positive integer (shared TsUtils validator)."""
    return validate_positive_int("lags", lags)


def _validate_alpha(alpha) -> float:
    """Validate that *alpha* is a significance level in (0, 1).

    Delegates to the shared :func:`Ts.TsUtils._validation.validate_alpha`.
    """
    return validate_alpha(alpha)


def _normalise_name_selection(selection, valid_names, *, label):
    """Resolve ``None``, one name, or an iterable of names into a unique tuple."""
    if selection is None:
        return tuple(valid_names)
    if isinstance(selection, str):
        selected = (selection,)
    else:
        try:
            selected = tuple(selection)
        except TypeError as error:
            raise TypeError(
                f"{label} must be a name or an iterable of names"
            ) from error
    if not selected:
        raise ValueError(f"{label} must contain at least one input")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{label} must be unique")
    unknown = [name for name in selected if name not in valid_names]
    if unknown:
        raise ValueError(f"{label} contains unknown input {unknown[0]!r}")
    return selected


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def _clean_1d(data: Any, *, name: str = "data") -> np.ndarray:
    """Drop NaN observations from 1-D input and reject infinities."""
    values = _as_1d_float(data, name=name)
    if np.any(np.isinf(values)):
        raise ValueError(f"{name} must not contain infinite values")
    return values[~np.isnan(values)]


def _clean_2d(data: Any, *, name: str = "data") -> np.ndarray:
    """Drop rows containing NaN from 2-D input and reject infinities."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {values.shape}")
    if np.any(np.isinf(values)):
        raise ValueError(f"{name} must not contain infinite values")
    return values[~np.any(np.isnan(values), axis=1)]


def _parse_input(
    data: Any,
    time_index: Any = None,
    y_col: str | int | None = None,
    time_col: str | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse *data* and *time_index* into 1-D float64 arrays.

    Supports DataFrame, Series, list, and ndarray inputs. For DataFrame
    inputs, *y_col* / *time_col* can be column names (str) or column
    indices (int).

    Parameters
    ----------
    data : array-like
        The time series. Can be a DataFrame, Series, list, or ndarray.
    time_index : array-like, optional
        Time index. If provided, takes precedence over *time_col*.
    y_col : str or int, optional
        Column name or index for the dependent variable when *data* is a
        DataFrame.
    time_col : str or int, optional
        Column name or index for the time index when *data* is a DataFrame.
        Only used when *time_index* is None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y_array, time_array)`` — two 1-D float64 ndarrays of equal length.

    Raises
    ------
    ValueError
        If *data* is a multi-column DataFrame and *y_col* is not specified.
    """

    def _get_col(_df: pd.DataFrame, _col: str | int) -> np.ndarray:
        if isinstance(_col, int):
            return _df.iloc[:, _col].values.astype(float).ravel()
        return _df[_col].values.astype(float).ravel()

    # --- Parse y ---
    if isinstance(data, pd.DataFrame):
        if y_col is not None:
            y = _get_col(data, y_col)
        elif data.shape[1] == 1:
            y = data.iloc[:, 0].values.astype(float).ravel()
        else:
            raise ValueError(
                f"DataFrame has {data.shape[1]} columns. "
                "Use y_col to specify the column for y."
            )
    elif isinstance(data, pd.Series):
        y = data.values.astype(float).ravel()
    else:
        y = _as_1d_float(data)

    # --- Parse time_index ---
    if time_index is not None:
        if isinstance(time_index, pd.DataFrame):
            if time_index.shape[1] != 1:
                raise ValueError("time_index DataFrame must contain exactly one column")
            t = time_index.iloc[:, 0].to_numpy(dtype=float)
        elif isinstance(time_index, pd.Series):
            t = time_index.to_numpy(dtype=float)
        else:
            t = _as_1d_float(time_index, name="time_index")
    elif time_col is not None and isinstance(data, pd.DataFrame):
        t = _get_col(data, time_col)
    else:
        t = np.arange(len(y), dtype=float)

    if len(y) != len(t):
        raise ValueError(
            f"time_index length ({len(t)}) must match data length ({len(y)})"
        )
    if len(y) == 0:
        raise ValueError("data must contain at least one observation")
    if not np.all(np.isfinite(y)):
        raise ValueError("data must contain only finite values")
    if not np.all(np.isfinite(t)):
        raise ValueError("time_index must contain only finite values")

    return y, t

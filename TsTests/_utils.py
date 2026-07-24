"""General-purpose utility functions shared across all TsTests modules.

Provides input parsing and validation. Structural-break-specific utilities
live in ``_break_utils.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

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
        y = np.asarray(data, dtype=float).ravel()

    # --- Parse time_index ---
    if time_index is not None:
        if isinstance(time_index, (pd.DataFrame, pd.Series)):
            t = time_index.values.astype(float).ravel()
        else:
            t = np.asarray(time_index, dtype=float).ravel()
    elif time_col is not None and isinstance(data, pd.DataFrame):
        t = _get_col(data, time_col)
    else:
        t = np.arange(len(y), dtype=float)

    return y, t

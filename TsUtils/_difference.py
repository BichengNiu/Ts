"""Composable differences for pandas time-series containers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from ._validation import validate_bool, validate_int


def _as_float_data(data):
    """Validate a pandas time-series container and return a float copy."""
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("data must be a pandas Series or DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one row and one series")
    if isinstance(data, pd.DataFrame) and not data.columns.is_unique:
        raise ValueError("DataFrame columns must be unique")

    dtypes = [data.dtype] if isinstance(data, pd.Series) else list(data.dtypes)
    if any(
        not is_numeric_dtype(dtype)
        or is_bool_dtype(dtype)
        or is_complex_dtype(dtype)
        for dtype in dtypes
    ):
        raise TypeError("data must contain only real numeric, non-boolean values")

    converted = data.astype(float)
    values = converted.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("data contains infinite values")
    return converted


def difference(data, *, order=1, log=False, lag=1):
    """Return an ordinary, log, or explicit-lag difference.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Numeric observations. A DataFrame is transformed independently by
        column. Missing values are allowed and propagate through differences.
    order : {1, 2}, default 1
        Number of times to apply the lag operator.
    log : bool, default False
        Apply the natural logarithm before differencing. Every non-missing
        observation must be strictly positive.
    lag : int, default 1
        Positive positional lag. Use the observations per year for a
        year-over-year difference, such as 12 for monthly or 4 for quarterly
        data.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        A new floating-point object with the same index, labels, and shape.

    Notes
    -----
    With lag ``s``, the first-order result is ``(1 - L^s) x_t`` and the
    second-order result is ``(1 - L^s)^2 x_t``. Log differences apply the same
    operator to ``log(x_t)``.

    Examples
    --------
    >>> import pandas as pd
    >>> from Ts.TsUtils import difference
    >>> series = pd.Series([100.0, 110.0, 121.0], name="index")
    >>> difference(series).tolist()
    [nan, 10.0, 11.0]
    >>> difference(series, log=True).round(6).tolist()
    [nan, 0.09531, 0.09531]

    Use ``lag=4`` for a year-over-year change in quarterly data.

    >>> quarterly = pd.Series([1, 2, 3, 4, 6], dtype=float)
    >>> difference(quarterly, lag=4).tolist()
    [nan, nan, nan, nan, 5.0]
    """
    order = validate_int("order", order, minimum=1)
    if order not in {1, 2}:
        raise ValueError(f"order must be 1 or 2, got {order}")
    lag = validate_int("lag", lag, minimum=1)
    log = validate_bool("log", log)
    transformed = _as_float_data(data)

    values = transformed.to_numpy(dtype=float, na_value=np.nan)
    if log and np.any(values[~np.isnan(values)] <= 0.0):
        raise ValueError("log differences require strictly positive values")
    if log:
        transformed = np.log(transformed)

    for _ in range(order):
        transformed = transformed.diff(periods=lag)

    return transformed

"""Rescaled-range estimation of the Hurst exponent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._summary import _as_numeric_series
from ._validation import validate_choice

_MIN_OBSERVATIONS = 20


def _selected_raw_data(data, variable):
    """Return the pre-conversion selection for dtype checks."""
    if isinstance(data, pd.DataFrame):
        if variable is None:
            return data.iloc[:, 0] if data.shape[1] == 1 else None
        if variable not in data.columns:
            return None
        return data.loc[:, variable]
    if isinstance(data, pd.Series):
        return data if variable is None else None
    return np.asarray(data)


def _block_sizes(nobs: int) -> tuple[int, ...]:
    """Return powers-of-two block sizes with at least two usable scales."""
    maximum = nobs // 2
    sizes = tuple(
        2**power for power in range(2, int(np.log2(maximum)) + 1) if 2**power <= maximum
    )
    if len(sizes) < 2:
        raise ValueError(
            "data must contain enough observations for at least two "
            "rescaled-range scales"
        )
    return sizes


def _mean_rescaled_range(values: np.ndarray, block_size: int) -> float:
    """Return the mean R/S statistic over complete blocks."""
    nblocks = len(values) // block_size
    statistics = []
    for block in np.array_split(values[: nblocks * block_size], nblocks):
        centered = block - np.mean(block)
        standard_deviation = np.std(block, ddof=1)
        if standard_deviation <= np.finfo(float).eps:
            continue
        cumulative_deviation = np.cumsum(centered)
        spread = np.ptp(cumulative_deviation)
        statistics.append(spread / standard_deviation)
    if not statistics:
        raise ValueError(
            "Hurst exponent is not defined because all usable blocks have zero variance"
        )
    return float(np.mean(statistics))


def hurst_exponent(data, *, variable=None, missing="drop") -> float:
    """Estimate the Hurst exponent with the classical rescaled-range method.

    Parameters
    ----------
    data : array-like, pandas.Series, or pandas.DataFrame
        One real numeric time series. A multi-column DataFrame requires
        ``variable``.
    variable : hashable, optional
        DataFrame column to analyse.
    missing : {"drop", "raise"}, default "drop"
        Whether to remove missing observations before estimation or reject
        the input when any missing observation is present. Missing values are
        never interpolated, and infinite values are always rejected.

    Returns
    -------
    float
        Estimated Hurst exponent. Values below 0.5 suggest anti-persistence,
        values near 0.5 are consistent with weak dependence, and values
        above 0.5 suggest persistence. These are descriptive heuristics, not
        hypothesis-test decisions.

    Notes
    -----
    The estimate regresses the logarithm of the mean rescaled range on the
    logarithm of powers-of-two block sizes from 4 through at most half of the
    effective sample. At least 20 effective observations are required. The
    estimate can be unstable for short or structurally changing series and
    should be treated as a descriptive diagnostic rather than proof of
    long-range dependence.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsUtils import hurst_exponent
    >>> values = np.random.default_rng(7).normal(size=200)
    >>> 0.3 < hurst_exponent(values) < 0.7
    True
    """
    missing = validate_choice("missing", missing, ("drop", "raise"))
    selected = _selected_raw_data(data, variable)
    selected_dtype = getattr(selected, "dtype", None)
    if selected_dtype is not None and pd.api.types.is_bool_dtype(selected_dtype):
        raise TypeError("data must contain numeric, non-boolean values")
    if selected_dtype is not None and pd.api.types.is_complex_dtype(selected_dtype):
        raise TypeError("data must contain real numeric values")
    series = _as_numeric_series(data, variable=variable)
    if missing == "raise" and series.isna().any():
        raise ValueError("data must not contain missing values")

    values = series.dropna().to_numpy(dtype=float, copy=True)
    if len(values) < _MIN_OBSERVATIONS:
        raise ValueError(
            f"data must contain at least {_MIN_OBSERVATIONS} effective "
            f"observations; got {len(values)}"
        )
    if np.ptp(values) <= np.finfo(float).eps * max(1.0, np.max(np.abs(values))):
        raise ValueError("Hurst exponent is not defined for a constant series")

    sizes = _block_sizes(len(values))
    rs_values = np.array(
        [_mean_rescaled_range(values, size) for size in sizes],
        dtype=float,
    )
    if not np.isfinite(rs_values).all() or (rs_values <= 0).any():
        raise ValueError("Hurst exponent produced no finite rescaled-range values")

    estimate = np.polyfit(np.log(sizes), np.log(rs_values), deg=1)[0]
    if not np.isfinite(estimate):
        raise ValueError("Hurst exponent estimate is not finite")
    return float(estimate)


__all__ = ["hurst_exponent"]

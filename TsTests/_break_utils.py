"""Utility functions specific to structural-break unit root tests.

Extracted from ``_utils.py`` — these helpers deal with break-dummy construction,
lag selection for ADF-type regressions, and regression-data assembly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Break dummy construction
# ---------------------------------------------------------------------------


def _make_perron_break_dummies(
    T: int,
    break_idx: int,
    model: str,
) -> dict[str, np.ndarray]:
    """Create the Perron known-break dummies for the selected model."""
    t = np.arange(T)
    dl = (t > break_idx).astype(float)
    dp = (t == break_idx + 1).astype(float)
    dt = np.maximum(t - break_idx, 0).astype(float)
    return {
        "intercept": {"DL": dl, "DP": dp},
        "slope": {"DL": dl, "DT": dt},
        "both": {"DL": dl, "DP": dp, "DT": dt},
    }[model]


def _make_zivot_break_dummies(
    T: int,
    break_idx: int,
    model: str,
) -> dict[str, np.ndarray]:
    """Create the Zivot-Andrews unknown-break dummies for the selected model."""
    t = np.arange(T)
    dl = (t > break_idx).astype(float)
    dt = np.maximum(t - break_idx, 0).astype(float)
    return {
        "intercept": {"DL": dl},
        "slope": {"DT": dt},
        "both": {"DL": dl, "DT": dt},
    }[model]


def _validate_nonnegative_int(value, *, name: str) -> int:
    """Return *value* as int after rejecting booleans and negative values."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _validate_time_axis(time_index: np.ndarray) -> None:
    """Require a strictly increasing observation-label axis."""
    if np.any(np.diff(time_index) <= 0):
        raise ValueError("time_index must be strictly increasing and unique")


def _locate_known_break(time_index: np.ndarray, break_year: float) -> int:
    """Locate a known break that matches exactly one observation label."""
    if isinstance(break_year, bool) or not np.isscalar(break_year):
        raise TypeError("break_year must be a finite scalar")
    try:
        break_value = float(break_year)
    except (TypeError, ValueError) as exc:
        raise TypeError("break_year must be a finite scalar") from exc
    if not np.isfinite(break_value):
        raise ValueError("break_year must be finite")
    matches = np.flatnonzero(np.isclose(time_index, break_value, rtol=0.0, atol=1e-12))
    if len(matches) != 1:
        raise ValueError("break_year must match exactly one time_index value")
    return int(matches[0])


# ---------------------------------------------------------------------------
# Lag selection
# ---------------------------------------------------------------------------


def _select_lags_by_tstat(
    y: np.ndarray,
    break_dummies: dict[str, np.ndarray],
    max_lags: int,
    crit: float = 1.60,
) -> int:
    """Select lag length by the general-to-specific t-statistic method.

    Starting from *max_lags*, the last lag is dropped if its t-statistic
    is less than *crit* in absolute value. This is the method used in
    Perron (1989) and Zivot & Andrews (1992).

    Parameters
    ----------
    y : np.ndarray
        The time series (1-D).
    break_dummies : dict[str, np.ndarray]
        Break dummy variables.
    max_lags : int
        Maximum number of lagged differences to consider.
    crit : float
        Critical value for the t-statistic threshold. Default 1.60.

    Returns
    -------
    int
        Selected number of lags.
    """
    T = len(y)
    dy = np.diff(y)
    y_lag1 = y[:-1]  # y_{t-1}

    # Deterministic trend is positional (0-based).
    trend = np.arange(T, dtype=float)

    for k in range(max_lags, -1, -1):
        # Build regressor matrix
        regs = [y_lag1[k:]]  # start from k because we need k initial diffs
        regs.append(np.ones(T - 1 - k))  # constant
        # Add break dummies (trimmed to match t=2..T, then trimmed by k)
        regs.extend(
            break_dummies[name][1:][k:]
            for name in ("DL", "DP", "DT")
            if name in break_dummies
        )
        # Add lagged differences
        regs.extend(dy[k - j : T - 1 - j] for j in range(1, k + 1))
        # Add time trend (positional)
        regs.append(trend[k + 1 :])

        X = np.column_stack(regs)
        y_dep = dy[k:]  # Δy_t, trimmed

        if X.shape[0] <= X.shape[1]:
            continue  # not enough obs

        try:
            res = sm.OLS(y_dep, X).fit()
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

        if k == 0:
            return 0

        # The last regressor is the time trend; the second-to-last
        # is the highest lagged difference dy_lagk.
        # regs = [y_lag1, const, dummies..., dy_lag1..dy_lagk, trend]
        last_lagdiff_idx = len(regs) - 2  # dy_lagk
        t_val = res.tvalues[last_lagdiff_idx]
        if abs(t_val) >= crit:
            return k

    return 0


def _select_lags_by_ic(
    y: np.ndarray,
    break_dummies: dict[str, np.ndarray],
    max_lags: int,
    ic_type: str,
) -> tuple[int, np.ndarray]:
    """Select lag length by AIC or BIC criterion.

    For each k = 0, ..., max_lags, estimate the augmented regression and
    compute the information criterion. Return the k that minimises it,
    together with the full IC sequence.

    Parameters
    ----------
    y : np.ndarray
        The time series (1-D).
    break_dummies : dict[str, np.ndarray]
        Break dummy variables.
    max_lags : int
        Maximum number of lagged differences to consider.
    ic_type : str
        ``"aic"`` or ``"bic"``.

    Returns
    -------
    tuple[int, np.ndarray]
        ``(best_k, ic_values)``, where *best_k* is the selected number of
        lags and *ic_values* is a 1-D array of length ``max_lags + 1``
        containing the IC value for each k.
    """
    best_ic = np.inf
    best_k = 0
    ic_values = np.full(max_lags + 1, np.nan)

    for k in range(max_lags + 1):
        df = _build_regression_data(y, break_dummies, k)

        reg_cols = _get_regression_columns(break_dummies, k)

        X = df[reg_cols].values
        y_dep = df["dy"].values

        if X.shape[0] <= X.shape[1]:
            continue

        try:
            res = sm.OLS(y_dep, X).fit()
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

        T_eff = len(y_dep)
        n_params = X.shape[1]
        ssr = np.sum(res.resid**2)
        log_sigma2 = np.log(ssr / T_eff)

        if ic_type == "aic":
            ic = log_sigma2 + 2 * n_params / T_eff
        else:  # bic
            ic = log_sigma2 + n_params * np.log(T_eff) / T_eff

        ic_values[k] = ic

        if ic < best_ic:
            best_ic = ic
            best_k = k

    return best_k, ic_values


# ---------------------------------------------------------------------------
# Regression data builder
# ---------------------------------------------------------------------------


def _build_regression_data(
    y: np.ndarray,
    break_dummies: dict[str, np.ndarray],
    lags: int,
) -> pd.DataFrame:
    """Build a DataFrame with all regressors for the ADF-type regression.

    The regression is::

        Δy_t = α + β·t + θ₁·DL_t + θ₂·DP_t + θ₃·DT_t
             + ρ·y_{t-1} + Σ_{j=1}^{k} γ_j·Δy_{t-j} + ε_t

    Parameters
    ----------
    y : np.ndarray
        Original time series.
    break_dummies : dict[str, np.ndarray]
        Break dummy variables.
    lags : int
        Number of lagged differences to include.

    Returns
    -------
    pd.DataFrame
        Regression-ready data (aligned, NaN rows dropped).
    """
    T = len(y)
    dy = np.diff(y)
    y_lag1 = y[:-1]  # y_{t-1}

    data = {"dy": dy, "y_lag1": y_lag1}

    # Constant
    data["const"] = np.ones(T - 1)

    # Deterministic trend is positional (0-based).
    data["trend"] = np.arange(1, T, dtype=float)

    # Break dummies (t=2..T)
    for name in ["DL", "DP", "DT"]:
        if name in break_dummies:
            data[name] = break_dummies[name][1:]

    # Lagged differences
    for j in range(1, lags + 1):
        data[f"dy_lag{j}"] = np.concatenate([np.zeros(j), dy[: T - 1 - j]])

    df = pd.DataFrame(data)
    # Drop rows with NaN (from lag construction)
    return df.iloc[lags:].copy()


# ---------------------------------------------------------------------------
# DRY helpers
# ---------------------------------------------------------------------------


def _get_regression_columns(
    break_dummies: dict[str, np.ndarray],
    lags: int,
) -> list[str]:
    """Return the ordered list of regressor column names.

    Parameters
    ----------
    break_dummies : dict[str, np.ndarray]
        Break dummy variables (used only to check which dummy keys exist).
    lags : int
        Number of lagged differences.

    Returns
    -------
    list[str]
        Column names in regression order.
    """
    cols = ["const", "trend"]
    cols.extend(name for name in ("DL", "DP", "DT") if name in break_dummies)
    cols.append("y_lag1")
    cols.extend(f"dy_lag{j}" for j in range(1, lags + 1))
    return cols


def _extract_rho_stats(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    reg_cols: list[str],
) -> tuple[float, float, float]:
    """Extract ρ̂, s.e.(ρ̂), and t(ρ̂) from the fitted regression.

    Parameters
    ----------
    results : RegressionResultsWrapper
        Fitted OLS results from statsmodels.
    reg_cols : list[str]
        Ordered list of regressor column names.

    Returns
    -------
    tuple[float, float, float]
        ``(rho_hat, rho_se, t_stat)``.
    """
    y_lag1_idx = reg_cols.index("y_lag1")
    rho_hat = float(results.params[y_lag1_idx])
    rho_se = float(results.bse[y_lag1_idx])
    t_stat = rho_hat / rho_se
    return rho_hat, rho_se, t_stat


def _extract_coefficients(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    reg_cols: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract coefficient estimates and p-values from the fitted regression.

    Parameters
    ----------
    results : RegressionResultsWrapper
        Fitted OLS results from statsmodels.
    reg_cols : list[str]
        Ordered list of regressor column names.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        ``(coefficients, pvalues)``.
    """
    coefs = {name: float(results.params[i]) for i, name in enumerate(reg_cols)}
    pvals = {name: float(results.pvalues[i]) for i, name in enumerate(reg_cols)}
    return coefs, pvals

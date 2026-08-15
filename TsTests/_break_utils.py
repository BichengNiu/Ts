"""Utility functions specific to structural-break unit root tests.

Extracted from ``_utils.py`` — these helpers deal with break-dummy construction,
lag selection for ADF-type regressions, and regression-data assembly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from Ts.TsUtils._validation import validate_int


# ---------------------------------------------------------------------------
# Break dummy construction
# ---------------------------------------------------------------------------


def _break_level_and_trend(T: int, break_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the shared positional index, level dummy, and trend dummy."""
    t = np.arange(T)
    dl = (t > break_idx).astype(float)
    dt = np.maximum(t - break_idx, 0).astype(float)
    return t, dl, dt


def _make_perron_break_dummies(
    T: int,
    break_idx: int,
    model: str,
) -> dict[str, np.ndarray]:
    """Create the Perron known-break dummies for the selected model."""
    _, dl, dt = _break_level_and_trend(T, break_idx)
    dp = (np.arange(T) == break_idx + 1).astype(float)
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
    _, dl, dt = _break_level_and_trend(T, break_idx)
    return {
        "intercept": {"DL": dl},
        "slope": {"DT": dt},
        "both": {"DL": dl, "DT": dt},
    }[model]


def _validate_lag_parameters(
    lags: int | None,
    max_lags: int,
    lag_crit: float,
    lag_method: str,
) -> tuple[int | None, int, float, str]:
    """Validate and normalise the shared lag-selection parameters."""
    lags_norm = None if lags is None else validate_int("lags", lags, minimum=0)
    max_lags_norm = validate_int("max_lags", max_lags, minimum=0)
    if isinstance(lag_crit, bool) or not np.isscalar(lag_crit):
        raise TypeError("lag_crit must be a positive finite scalar")
    lag_crit_norm = float(lag_crit)
    if not np.isfinite(lag_crit_norm) or lag_crit_norm <= 0:
        raise ValueError("lag_crit must be a positive finite scalar")
    if lag_method not in ("tstat", "aic", "bic"):
        raise ValueError(
            f"lag_method must be 'tstat', 'aic', or 'bic', got {lag_method!r}"
        )
    return lags_norm, max_lags_norm, lag_crit_norm, lag_method


def _validate_trim(trim):
    """Return a validated structural-break trimming fraction in (0, 0.5)."""
    if isinstance(trim, bool) or not np.isscalar(trim):
        raise TypeError("trim must be a finite scalar between 0 and 0.5")
    trim = float(trim)
    if not np.isfinite(trim) or not 0 < trim < 0.5:
        raise ValueError("trim must be between 0 and 0.5")
    return trim


def _format_break_test_summary(
    header: str,
    break_label: str,
    stat_label: str,
    model: str,
    break_year: float,
    lags: int,
    nobs: int,
    rho_hat: float,
    rho_se: float,
    statistic: float,
    cv_01: float,
    cv_05: float,
    cv_10: float,
) -> str:
    """Format the common summary body of a structural-break unit root test."""
    return (
        f"{header} — Model {model}\n"
        f"{'=' * 50}\n"
        f"{break_label:<20}: {break_year}\n"
        f"Number of lags (k)   : {lags}\n"
        f"Effective obs. (T)   : {nobs}\n"
        f"\n"
        f"ρ̂ (coeff on y_t-1)   : {rho_hat:.4f}\n"
        f"s.e.(ρ̂)              : {rho_se:.4f}\n"
        f"{stat_label:<20}: {statistic:.3f}\n"
        f"\n"
        f"Critical values:\n"
        f"  1%                 : {cv_01:.3f}\n"
        f"  5%                 : {cv_05:.3f}\n"
        f"  10%                : {cv_10:.3f}\n"
        f"\n"
        f"Conclusion (5%): "
        f"{'Reject H0 (unit root); evidence favors stationarity around a breaking trend' if statistic < cv_05 else 'Cannot reject H0 (unit root)'}\n"
    )


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


def _select_lag_by_tstat(
    max_lags: int,
    tstat_of: Callable[[int], float | None],
    crit: float = 1.60,
) -> int:
    """General-to-specific lag selection over a t-statistic oracle.

    ``tstat_of(lag)`` returns the t-statistic of the highest lag at *lag*,
    or ``None`` when that lag specification cannot be estimated. Returns
    the largest lag whose t-statistic reaches *crit* in absolute value,
    else 0.
    """
    for lag in range(max_lags, 0, -1):
        tstat = tstat_of(lag)
        if tstat is None:
            continue
        if abs(tstat) >= crit:
            return lag
    return 0


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

    def tstat_of(k: int) -> float | None:
        fitted = _fit_unitroot_ols(y, break_dummies, k)
        if fitted is None:
            return None
        res, columns = fitted
        # The final column is the highest lagged difference dy_lagk.
        return float(res.tvalues[len(columns) - 1])

    return _select_lag_by_tstat(max_lags, tstat_of, crit)


def _ic_from_ssr(ssr: float, n_effective: int, n_params: int, ic_type: str) -> float:
    """Return the AIC or BIC value from residual SSR."""
    log_sigma2 = np.log(ssr / n_effective)
    if ic_type == "aic":
        return log_sigma2 + 2 * n_params / n_effective
    return log_sigma2 + n_params * np.log(n_effective) / n_effective


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
        fitted = _fit_unitroot_ols(y, break_dummies, k)
        if fitted is None:
            continue
        res, columns = fitted

        T_eff = int(res.nobs)
        n_params = len(columns)
        ssr = np.sum(res.resid**2)
        ic = _ic_from_ssr(ssr, T_eff, n_params, ic_type)

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


def _fit_unitroot_ols(
    y: np.ndarray,
    break_dummies: dict[str, np.ndarray],
    lags: int,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, list[str]] | None:
    """Fit the augmented break regression or return None when impossible.

    Builds the lag-augmented design from *y* and *break_dummies*, fits OLS,
    and returns ``(results, columns)``.  Returns ``None`` when too few
    observations remain, the design matrix is rank deficient, or the OLS
    fit fails — callers decide whether to skip or raise.
    """
    frame = _build_regression_data(y, break_dummies, lags)
    columns = _get_regression_columns(break_dummies, lags)
    X = frame[columns].values
    y_dep = frame["dy"].values
    if X.shape[0] <= X.shape[1]:
        return None
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    try:
        return sm.OLS(y_dep, X).fit(), columns
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None


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

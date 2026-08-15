"""Shared simulation engine for GARCH-family processes.

Internal helpers and the core recursion loop used by all GARCH-family
simulation functions (:func:`simulate_garch`, :func:`simulate_igarch`,
:func:`simulate_gjr_garch`, :func:`simulate_egarch`,
:func:`simulate_garch_m`).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats as scipy_stats

from Ts.TsUtils._validation import (
    normalize_coefficients,
    validate_choice,
    validate_int,
    validate_real,
    validate_sample,
)

from ._garch_result import SimGARCHResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unconditional_variance(omega, persistence_sum, *, label, terms_text):
    """Return omega / (1 - persistence) with a non-stationarity fallback.

    When *persistence_sum* reaches one the process is non-stationary
    (or IGARCH) and the unconditional variance is undefined; warn and fall
    back to *omega* as the initial variance.
    """
    if persistence_sum >= 1.0:
        warnings.warn(
            f"{label} process is non-stationary: {terms_text} = "
            f"{persistence_sum} >= 1. Using omega = {omega} as initial variance.",
            RuntimeWarning,
            stacklevel=2,
        )
        return omega
    return omega / (1.0 - persistence_sum)


def _validate_garch_inputs(
    p,
    q,
    omega,
    alpha,
    beta,
    *,
    mean_model,
    mean_choices,
    o=None,
    gamma=None,
    q_minimum=0,
    omega_positive=True,
    alpha_default,
    beta_default,
    gamma_default=None,
):
    """Validate the shared order and coefficient inputs of a GARCH simulator."""
    p = validate_int("p", p, minimum=1)
    q = validate_int("q", q, minimum=q_minimum)
    if o is not None:
        o = validate_int("o", o, minimum=0)
    omega = validate_real("omega", omega, positive=omega_positive)
    validate_choice("mean_model", mean_model, mean_choices)
    alpha = normalize_coefficients(
        "alpha", alpha, length=p, default=alpha_default, nonnegative=True
    )
    if gamma_default is not None:
        gamma = normalize_coefficients("gamma", gamma, length=o, default=gamma_default)
    beta = normalize_coefficients(
        "beta", beta, length=q, default=beta_default, nonnegative=True
    )
    return p, q, o, omega, alpha, gamma, beta


def _make_standard_variance_fn(omega, alpha, beta, p, q):
    """Return a variance_fn closure for standard GARCH/ARCH/IGARCH/GARCH-M."""

    def _variance_fn(t, eps_ar, sigma2_ar, state=None):
        var_t = omega
        for i in range(p):
            var_t += alpha[i] * eps_ar[t - 1 - i] ** 2
        for j in range(q):
            var_t += beta[j] * sigma2_ar[t - 1 - j]
        return var_t

    return _variance_fn


def _t_dist_df(dist_params: dict | None) -> float:
    """Return the validated Student-t degrees of freedom."""
    df = 5.0
    if dist_params is not None and "df" in dist_params:
        df = validate_real("dist_params['df']", dist_params["df"], positive=True)
    if df <= 2:
        raise ValueError(
            f"Student's t requires df > 2 for finite variance, got df={df}"
        )
    return df


def _generate_innovations(n_total, dist, dist_params, rng):
    """Generate i.i.d. innovations from the specified distribution.

    Parameters
    ----------
    n_total : int
        Number of draws.
    dist : str
        ``"normal"`` or ``"t"``.
    dist_params : dict or None
        Distribution parameters (e.g. ``{"df": 5}``).
    rng : numpy.random.Generator

    Returns
    -------
    np.ndarray
    """
    if dist == "t":
        df = _t_dist_df(dist_params)
        raw = scipy_stats.t.rvs(df=df, size=n_total, random_state=rng)
        return raw / np.sqrt(df / (df - 2))
    # normal (default)
    return rng.standard_normal(n_total)


def _compute_mean(y, mean_model, mean_const, mean_ar, t):
    """Compute the conditional mean mu_t.

    Parameters
    ----------
    y : np.ndarray
        Full series so far.
    mean_model : str
    mean_const : float
    mean_ar : list of float or None
    t : int
        Current time index.

    Returns
    -------
    float
    """
    if mean_model == "zero":
        return 0.0
    mu = mean_const
    if mean_model == "ar" and mean_ar and t > 0:
        for k, phi in enumerate(mean_ar):
            lag = k + 1
            if t - lag >= 0:
                mu += phi * y[t - lag]
    return mu


# ---------------------------------------------------------------------------
# Shared simulation loop
# ---------------------------------------------------------------------------


def _run_garch_simulation(
    n,
    p,
    q,
    omega,
    alpha,
    beta,
    variance_fn,
    mean_model,
    mean_const,
    mean_ar,
    dist,
    dist_params,
    seed,
    burn,
    *,
    init_sigma2_fn=None,
    max_lag=None,
    model_type,
    extra_params=None,
    garch_m_kappa=None,
    garch_m_form=None,
    state_array=None,
):
    """Shared simulation loop for all GARCH-family variants.

    Handles array allocation, innovation generation, the recursion loop,
    GARCH-M mean adjustment, burn-in discard, and result construction.
    Model-specific variance dynamics are provided via ``variance_fn``.

    Parameters
    ----------
    n : int
        Number of observations (after burn-in).
    p : int
        ARCH order.
    q : int
        GARCH order.
    omega : float
        Constant in the variance equation.
    alpha : list of float
        ARCH coefficients.
    beta : list of float
        GARCH coefficients.
    variance_fn : callable(t, eps, sigma2, state_array=None) -> float
        Returns conditional variance at time *t* given past ``eps`` and
        ``sigma2`` arrays. Receives *state_array* for models that need
        auxiliary state (e.g. EGARCH log-variance).
    mean_model : str
    mean_const : float
    mean_ar : list of float
    dist : str
    dist_params : dict or None
    seed : int or None
    burn : int
    init_sigma2_fn : callable() -> float, optional
        Returns the initial unconditional variance for pre-allocation.
        If ``None``, the standard GARCH formula ``omega / (1 - sum(alpha)
        - sum(beta))`` is used with the standard stationarity warning.
    max_lag : int, optional
        Minimum start index for the recursion loop.  Computed from
        ``max(p, q, len(mean_ar))`` when ``None``.
    model_type : str
        Explicit model type (e.g. ``"GARCH"``, ``"GJR-GARCH"``), stored in
        the result parameter mapping as its single source of truth.
    extra_params : dict, optional
        Additional model-specific parameters merged into result params.
    garch_m_kappa : float, optional
        ARCH-in-mean coefficient.
    garch_m_form : str, optional
        Form of conditional variance in the mean equation.
    state_array : np.ndarray, optional
        Pre-allocated auxiliary state array (e.g. log-variance for EGARCH).
        Passed to *variance_fn* which may read from and write to it.

    Returns
    -------
    SimGARCHResult
    """
    n, burn = validate_sample(n, burn)
    dist = validate_choice("dist", dist, ("normal", "t"))
    mean_const = validate_real("mean_const", mean_const)
    mean_ar = normalize_coefficients("mean_ar", mean_ar)
    if dist_params is not None and not isinstance(dist_params, dict):
        raise TypeError("dist_params must be a dict or None")

    total_n = n + burn
    max_lag_val = max(p, q, len(mean_ar)) if max_lag is None else max_lag

    rng = np.random.default_rng(seed)
    u = _generate_innovations(total_n, dist, dist_params, rng)

    # Initial (unconditional) variance
    if init_sigma2_fn is not None:
        sigma2_uncond = init_sigma2_fn()
    else:
        alpha_sum = sum(alpha)
        beta_sum = sum(beta)
        sigma2_uncond = _unconditional_variance(
            omega,
            alpha_sum + beta_sum,
            label="GARCH",
            terms_text=f"sum(alpha) + sum(beta) = {alpha_sum} + {beta_sum}",
        )

    if not np.isfinite(sigma2_uncond) or sigma2_uncond <= 0:
        raise ValueError(
            "initial conditional variance must be finite and positive, "
            f"got {sigma2_uncond}"
        )

    sigma2 = np.full(total_n, sigma2_uncond)
    eps = np.zeros(total_n)
    y = np.zeros(total_n)

    for t in range(max_lag_val, total_n):
        var_t = variance_fn(t, eps, sigma2, state_array)
        if not np.isfinite(var_t) or var_t <= 0:
            raise ValueError(
                f"conditional variance must remain finite and positive; "
                f"got {var_t} at t={t}"
            )

        sigma2[t] = var_t
        sigma_t = np.sqrt(var_t)
        eps[t] = sigma_t * u[t]

        mu = _compute_mean(y, mean_model, mean_const, mean_ar, t)
        if garch_m_kappa is not None and garch_m_kappa != 0.0:
            if garch_m_form == "var":
                mu += garch_m_kappa * var_t
            elif garch_m_form == "log":
                mu += garch_m_kappa * np.log(max(var_t, 1e-10))
            else:  # "vol" (default)
                mu += garch_m_kappa * sigma_t
        y[t] = mu + eps[t]

    # Result params
    result_params = {
        "model_type": model_type,
        "p": p,
        "q": q,
        "omega": omega,
        "alpha": list(alpha),
        "beta": list(beta),
        "mean_model": mean_model,
        "mean_const": mean_const,
        "mean_ar": list(mean_ar),
        "dist": dist,
        "dist_params": dist_params,
        "seed": seed,
        "n": n,
        "burn": burn,
    }
    if extra_params:
        result_params.update(extra_params)
    if garch_m_kappa is not None:
        result_params["garch_m_kappa"] = garch_m_kappa
        result_params["garch_m_form"] = garch_m_form

    return SimGARCHResult(
        data=y[burn:],
        residuals=eps[burn:],
        conditional_volatility=np.sqrt(sigma2[burn:]),
        params=result_params,
    )

"""Extended GARCH-family simulation: GJR-GARCH, EGARCH, GARCH-M.

Provides simulation functions for asymmetric and non-standard GARCH variants
that go beyond the standard ARCH/GARCH/IGARCH in :mod:`_garch`.
"""

from __future__ import annotations

import numpy as np
from Ts.TsUtils._validation import validate_choice, validate_real, validate_sample

from ._garch_core import (
    _make_standard_variance_fn,
    _run_garch_simulation,
    _t_dist_df,
    _unconditional_variance,
    _validate_garch_inputs,
)
from ._garch_result import SimGARCHResult


# ---------------------------------------------------------------------------
# GJR-GARCH engine
# ---------------------------------------------------------------------------


def _simulate_gjr_garch(
    n,
    p,
    q,
    o,
    omega,
    alpha,
    gamma,
    beta,
    mean_model,
    mean_const,
    dist,
    dist_params,
    seed,
    burn,
):
    """Core simulation for GJR-GARCH(p,o,q) processes.

    GJR-GARCH(p,o,q):
        sigma2_t = omega + sum_{i=1}^{p} alpha_i * eps2_{t-i}
                  + sum_{k=1}^{o} gamma_k * I_{t-k} * eps2_{t-k}
                  + sum_{j=1}^{q} beta_j * sigma2_{t-j}
        I_{t-k} = 1 if eps_{t-k} < 0 else 0

    Stationarity condition: sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1.

    Returns
    -------
    SimGARCHResult

    """

    def _init_sigma2_fn():
        alpha_sum = sum(alpha)
        gamma_half_sum = 0.5 * sum(gamma)
        beta_sum = sum(beta)
        return _unconditional_variance(
            omega,
            alpha_sum + gamma_half_sum + beta_sum,
            label="GJR-GARCH",
            terms_text=(
                f"sum(alpha) + 0.5*sum(gamma) + sum(beta) = "
                f"{alpha_sum} + {gamma_half_sum} + {beta_sum}"
            ),
        )

    standard_variance = _make_standard_variance_fn(omega, alpha, beta, p, q)

    def _variance_fn(t, eps_ar, sigma2_ar, _state=None):
        var_t = standard_variance(t, eps_ar, sigma2_ar)
        for k in range(o):
            if eps_ar[t - 1 - k] < 0:
                var_t += gamma[k] * eps_ar[t - 1 - k] ** 2
        return var_t

    return _run_garch_simulation(
        n=n,
        p=p,
        q=q,
        omega=omega,
        alpha=alpha,
        beta=beta,
        variance_fn=_variance_fn,
        init_sigma2_fn=_init_sigma2_fn,
        max_lag=max(p, q, o),
        mean_model=mean_model,
        mean_const=mean_const,
        mean_ar=[],
        dist=dist,
        dist_params=dist_params,
        seed=seed,
        burn=burn,
        model_type="GJR-GARCH",
        extra_params={"o": o, "gamma": list(gamma)},
    )


# ---------------------------------------------------------------------------
# EGARCH engine
# ---------------------------------------------------------------------------


def _simulate_egarch(
    n,
    p,
    q,
    o,
    omega,
    alpha,
    gamma,
    beta,
    mean_model,
    mean_const,
    dist,
    dist_params,
    seed,
    burn,
):
    """Core simulation for EGARCH(p,o,q) processes.

    EGARCH(p,o,q):
        log(sigma2_t) = omega
                       + sum_{i=1}^{p} alpha_i * (|z_{t-i}| - E|z|)
                       + sum_{k=1}^{o} gamma_k * z_{t-k}
                       + sum_{j=1}^{q} beta_j * log(sigma2_{t-j})
        z_t = eps_t / sigma_t ~ iid(0,1)

    Returns
    -------
    SimGARCHResult

    """
    from scipy.special import gamma as gamma_func

    # Validate sample sizes before any allocation so invalid n/burn report
    # the public contract error instead of a NumPy dimension error.
    n, burn = validate_sample(n, burn)

    # Pre-compute E|z| for the chosen distribution
    if dist == "t":
        df = _t_dist_df(dist_params)
        exp_abs_z = (
            2.0
            * np.sqrt(df - 2.0)
            * gamma_func((df + 1.0) / 2.0)
            / ((df - 1.0) * gamma_func(df / 2.0) * np.sqrt(np.pi))
        )
    else:
        exp_abs_z = np.sqrt(2.0 / np.pi)

    total_n = n + burn
    ln_sigma2 = np.full(total_n, omega)

    def _variance_fn(t, eps_ar, sigma2_ar, state):
        var_ln = omega
        for i in range(p):
            z_lag = eps_ar[t - 1 - i] / max(np.sqrt(sigma2_ar[t - 1 - i]), 1e-8)
            var_ln += alpha[i] * (abs(z_lag) - exp_abs_z)
        for k in range(o):
            z_lag = eps_ar[t - 1 - k] / max(np.sqrt(sigma2_ar[t - 1 - k]), 1e-8)
            var_ln += gamma[k] * z_lag
        for j in range(q):
            var_ln += beta[j] * state[t - 1 - j]

        state[t] = var_ln
        sigma2_t = np.exp(var_ln)
        if sigma2_t > 1e100:
            sigma2_t = 1e100
        return sigma2_t

    return _run_garch_simulation(
        n=n,
        p=p,
        q=q,
        omega=omega,
        alpha=alpha,
        beta=beta,
        variance_fn=_variance_fn,
        init_sigma2_fn=lambda: np.exp(omega),
        max_lag=max(p, q, o),
        mean_model=mean_model,
        mean_const=mean_const,
        mean_ar=[],
        dist=dist,
        dist_params=dist_params,
        seed=seed,
        burn=burn,
        model_type="EGARCH",
        extra_params={"o": o, "gamma": list(gamma)},
        state_array=ln_sigma2,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_gjr_garch(
    n: int = 200,
    p: int = 1,
    q: int = 1,
    o: int = 1,
    omega: float = 0.10,
    alpha: list[float] | None = None,
    gamma: list[float] | None = None,
    beta: list[float] | None = None,
    mean_model: str = "constant",
    mean_const: float = 0.0,
    dist: str = "normal",
    dist_params: dict | None = None,
    seed: int | None = None,
    burn: int = 100,
) -> SimGARCHResult:
    """Simulate a GJR-GARCH(p,o,q) time series with leverage effects.

    The GJR-GARCH model:
        y_t = mu + eps_t
        eps_t = sigma_t * u_t,    u_t ~ D(0,1)
        sigma2_t = omega + sum_{i=1}^{p} alpha_i * eps2_{t-i}
                  + sum_{k=1}^{o} gamma_k * I_{t-k} * eps2_{t-k}
                  + sum_{j=1}^{q} beta_j * sigma2_{t-j}
        I_{t-k} = 1 if eps_{t-k} < 0 else 0

    Negative shocks increase volatility more than positive shocks of the
    same magnitude (leverage effect).  When ``o = 0`` the model reduces to
    standard GARCH(p,q).

    Parameters
    ----------
    n : int
        Number of observations to generate (after burn-in).
    p : int
        ARCH order.
    q : int
        GARCH order. Use ``q = 0`` for pure ARCH with leverage.
    o : int
        Asymmetric (GJR) order. Use ``o = 0`` for symmetric GARCH.
    omega : float
        Constant term in the variance equation (omega > 0).
    alpha : float or list of float, optional
        ARCH coefficients [alpha_1, ..., alpha_p]. Defaults to ``[0.10] * p``.
    gamma : float or list of float, optional
        Asymmetric (leverage) coefficients [gamma_1, ..., gamma_o].
        Defaults to ``[0.10] * o``.
    beta : float or list of float, optional
        GARCH coefficients [beta_1, ..., beta_q]. Defaults to ``[0.70] * q``.
    mean_model : str
        Mean equation: ``"constant"`` or ``"zero"``.
    mean_const : float
        Constant term in the mean equation.
    dist : str
        Innovation distribution: ``"normal"`` or ``"t"``.
    dist_params : dict, optional
        Additional distribution parameters.
    seed : int, optional
        Random seed for reproducibility.
    burn : int
        Number of burn-in observations to discard.

    Returns
    -------
    SimGARCHResult

    Examples
    --------
    >>> from Ts.TsSims import simulate_gjr_garch
    >>> result = simulate_gjr_garch(
    ...     n=50, alpha=[0.1], gamma=[0.15], beta=[0.7], seed=42
    ... )
    >>> result.params["gamma"]
    [0.15]
    """
    p, q, o, omega, alpha, gamma, beta = _validate_garch_inputs(
        p,
        q,
        omega,
        alpha,
        beta,
        o=o,
        gamma=gamma,
        mean_model=mean_model,
        mean_choices=("constant", "zero"),
        alpha_default=0.10,
        beta_default=0.70,
        gamma_default=0.10,
    )

    return _simulate_gjr_garch(
        n=n,
        p=p,
        q=q,
        o=o,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        mean_model=mean_model,
        mean_const=mean_const,
        dist=dist,
        dist_params=dist_params,
        seed=seed,
        burn=burn,
    )


def simulate_egarch(
    n: int = 200,
    p: int = 1,
    q: int = 1,
    o: int = 1,
    omega: float = 0.0,
    alpha: list[float] | None = None,
    gamma: list[float] | None = None,
    beta: list[float] | None = None,
    mean_model: str = "constant",
    mean_const: float = 0.0,
    dist: str = "normal",
    dist_params: dict | None = None,
    seed: int | None = None,
    burn: int = 100,
) -> SimGARCHResult:
    """Simulate an EGARCH(p,o,q) time series with log-variance dynamics.

    The EGARCH model:
        y_t = mu + eps_t
        eps_t = sigma_t * z_t,    z_t ~ iid(0,1)
        log(sigma2_t) = omega
                       + sum_{i=1}^{p} alpha_i * (|z_{t-i}| - E|z|)
                       + sum_{k=1}^{o} gamma_k * z_{t-k}
                       + sum_{j=1}^{q} beta_j * log(sigma2_{t-j})

    Variance is guaranteed positive by the exponential transformation.
    When ``o = 0`` the model is symmetric (no leverage effect).

    Parameters
    ----------
    n : int
        Number of observations to generate (after burn-in).
    p : int
        ARCH (magnitude) order.
    q : int
        GARCH (persistence) order.
    o : int
        Asymmetric (leverage) order. Use ``o = 0`` for symmetric EGARCH.
    omega : float
        Constant term in the log-variance equation.
    alpha : float or list of float, optional
        Magnitude coefficients [alpha_1, ..., alpha_p]. Defaults to
        ``[0.15] * p``.
    gamma : float or list of float, optional
        Asymmetric (sign) coefficients [gamma_1, ..., gamma_o].
        Defaults to ``[0.05] * o``.
    beta : float or list of float, optional
        Persistence coefficients [beta_1, ..., beta_q]. Defaults to
        ``[0.30] * q``.
    mean_model : str
        Mean equation: ``"constant"`` or ``"zero"``.
    mean_const : float
        Constant term in the mean equation.
    dist : str
        Innovation distribution: ``"normal"`` or ``"t"``.
    dist_params : dict, optional
        Additional distribution parameters.
    seed : int, optional
        Random seed for reproducibility.
    burn : int
        Number of burn-in observations to discard.

    Returns
    -------
    SimGARCHResult

    Examples
    --------
    >>> from Ts.TsSims import simulate_egarch
    >>> result = simulate_egarch(
    ...     n=50, alpha=[0.15], gamma=[-0.1], beta=[0.8], seed=42
    ... )
    >>> result.model_type
    'EGARCH'
    """
    p, q, o, omega, alpha, gamma, beta = _validate_garch_inputs(
        p,
        q,
        omega,
        alpha,
        beta,
        o=o,
        gamma=gamma,
        mean_model=mean_model,
        mean_choices=("constant", "zero"),
        omega_positive=False,
        alpha_default=0.15,
        beta_default=0.30,
        gamma_default=0.05,
    )

    return _simulate_egarch(
        n=n,
        p=p,
        q=q,
        o=o,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        mean_model=mean_model,
        mean_const=mean_const,
        dist=dist,
        dist_params=dist_params,
        seed=seed,
        burn=burn,
    )


def simulate_garch_m(
    n: int = 200,
    p: int = 1,
    q: int = 1,
    omega: float = 0.10,
    alpha: list[float] | None = None,
    beta: list[float] | None = None,
    garch_m_kappa: float = 0.20,
    garch_m_form: str = "vol",
    mean_model: str = "constant",
    mean_const: float = 0.0,
    dist: str = "normal",
    dist_params: dict | None = None,
    seed: int | None = None,
    burn: int = 100,
) -> SimGARCHResult:
    """Simulate a GARCH-M (ARCH-in-Mean) time series.

    The GARCH-M model:
        y_t = mu + kappa * f(sigma_t) + eps_t
        eps_t = sigma_t * u_t,    u_t ~ D(0,1)
        sigma2_t = omega + sum_{i=1}^{p} alpha_i * eps2_{t-i}
                  + sum_{j=1}^{q} beta_j * sigma2_{t-j}

    where f(.) is determined by ``garch_m_form``:
    - ``"vol"``: f(sigma_t) = sigma_t (conditional standard deviation)
    - ``"var"``: f(sigma_t) = sigma2_t (conditional variance)
    - ``"log"``: f(sigma_t) = log(sigma2_t) (log conditional variance)

    Parameters
    ----------
    n : int
        Number of observations to generate (after burn-in).
    p : int
        ARCH order.
    q : int
        GARCH order. Use ``q = 0`` for pure ARCH-M.
    omega : float
        Constant term in the variance equation (omega > 0).
    alpha : float or list of float, optional
        ARCH coefficients. Defaults to ``[0.20] * p``.
    beta : float or list of float, optional
        GARCH coefficients. Defaults to ``[0.60] * q``.
    garch_m_kappa : float
        ARCH-in-mean coefficient. Default 0.20.
    garch_m_form : str
        Form of conditional variance in the mean equation:
        ``"vol"``, ``"var"``, or ``"log"``. Default ``"vol"``.
    mean_model : str
        Mean equation: ``"constant"`` or ``"zero"``.
    mean_const : float
        Constant term in the mean equation.
    dist : str
        Innovation distribution: ``"normal"`` or ``"t"``.
    dist_params : dict, optional
        Additional distribution parameters.
    seed : int, optional
        Random seed for reproducibility.
    burn : int
        Number of burn-in observations to discard.

    Returns
    -------
    SimGARCHResult

    Examples
    --------
    >>> from Ts.TsSims import simulate_garch_m
    >>> result = simulate_garch_m(
    ...     n=50, garch_m_kappa=0.3, garch_m_form="var", seed=42
    ... )
    >>> result.params["garch_m_form"]
    'var'
    """
    p, q, _, omega, alpha, _, beta = _validate_garch_inputs(
        p,
        q,
        omega,
        alpha,
        beta,
        mean_model=mean_model,
        mean_choices=("constant", "zero"),
        alpha_default=0.20,
        beta_default=0.60,
    )
    garch_m_kappa = validate_real("garch_m_kappa", garch_m_kappa)
    validate_choice("garch_m_form", garch_m_form, ("vol", "var", "log"))

    model_type = "ARCH-M" if q == 0 else "GARCH-M"

    return _run_garch_simulation(
        n=n,
        p=p,
        q=q,
        omega=omega,
        alpha=alpha,
        beta=beta,
        variance_fn=_make_standard_variance_fn(omega, alpha, beta, p, q),
        mean_model=mean_model,
        mean_const=mean_const,
        mean_ar=[],
        dist=dist,
        dist_params=dist_params,
        seed=seed,
        burn=burn,
        model_type=model_type,
        garch_m_kappa=garch_m_kappa,
        garch_m_form=garch_m_form,
    )

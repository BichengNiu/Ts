"""Validated options shared by the SARIMAX-family model adapters.

This module contains only option normalization used by more than one model
family.  Model construction, input preparation, and prediction remain in the
owning model modules.
"""

from __future__ import annotations

import numpy as np

from Ts.TsUtils._validation import validate_int

from ._distributed_lag import _normalise_lag_order


_SARIMAX_OPTIMIZERS = frozenset(
    {
        "basinhopping",
        "bfgs",
        "cg",
        "lbfgs",
        "ncg",
        "newton",
        "nm",
        "powell",
    }
)
_SARIMAX_COVARIANCE_TYPES = frozenset(
    {"approx", "oim", "opg", "robust", "robust_approx"}
)


def _normalise_order(order):
    """Return a validated non-seasonal ``(p, d, q)`` order."""
    if not isinstance(order, (tuple, list)) or len(order) != 3:
        raise ValueError(f"order must be a tuple of (p, d, q), got {order}")
    p, d, q = order
    return (
        _normalise_lag_order(p, "p"),
        validate_int("d", d, minimum=0),
        _normalise_lag_order(q, "q"),
    )


def _normalise_seasonal_order(seasonal_order):
    """Return a validated seasonal ``(P, D, Q, s)`` order."""
    if not isinstance(seasonal_order, (tuple, list)) or len(seasonal_order) != 4:
        raise ValueError(
            "seasonal_order must be a tuple of (P, D, Q, s), "
            f"got {seasonal_order}"
        )
    P, D, Q, s = seasonal_order
    return (
        _normalise_lag_order(P, "P"),
        validate_int("D", D, minimum=0),
        _normalise_lag_order(Q, "Q"),
        validate_int("s", s, minimum=0),
    )


def _normalise_log(log):
    """Return an explicit boolean log-transform flag."""
    if not isinstance(log, (bool, np.bool_)):
        raise TypeError("log must be a boolean")
    return bool(log)


def _normalise_fit_method(method):
    """Return a supported statsmodels optimizer name."""
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method = method.strip().lower()
    if not method:
        raise ValueError("method must be a non-empty string")
    if method not in _SARIMAX_OPTIMIZERS:
        supported = ", ".join(sorted(_SARIMAX_OPTIMIZERS))
        raise ValueError(f"method must be one of: {supported}")
    return method


def _normalise_maxiter(maxiter):
    """Return a strictly positive optimizer iteration limit."""
    if isinstance(maxiter, (bool, np.bool_)) or not isinstance(
        maxiter,
        (int, np.integer),
    ):
        raise TypeError("maxiter must be a positive integer")
    maxiter = int(maxiter)
    if maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")
    return maxiter


def _normalise_cov_type(cov_type):
    """Return a supported covariance estimator name."""
    if not isinstance(cov_type, str):
        raise TypeError("cov_type must be a string")
    cov_type = cov_type.strip().lower()
    if cov_type not in _SARIMAX_COVARIANCE_TYPES:
        allowed = ", ".join(sorted(_SARIMAX_COVARIANCE_TYPES))
        raise ValueError(f"cov_type must be one of: {allowed}")
    return cov_type


def _normalise_require_convergence(require_convergence):
    """Return an explicit boolean convergence requirement."""
    if not isinstance(require_convergence, (bool, np.bool_)):
        raise TypeError("require_convergence must be a boolean")
    return bool(require_convergence)


__all__ = [
    "_SARIMAX_COVARIANCE_TYPES",
    "_SARIMAX_OPTIMIZERS",
    "_normalise_cov_type",
    "_normalise_fit_method",
    "_normalise_log",
    "_normalise_maxiter",
    "_normalise_order",
    "_normalise_require_convergence",
    "_normalise_seasonal_order",
]

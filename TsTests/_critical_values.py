"""Critical value tables and interpolation for structural break unit root tests.

Sources
-------
- Perron (1989) Table IV.B — finite-sample critical values
- Zivot & Andrews (1992) Table 2 — asymptotic critical values
- Lee & Strazicich (2003) Table 1 — two-break minimum LM critical values
"""

from __future__ import annotations

import numpy as np

from ._utils import _validate_model

# ---------------------------------------------------------------------------
# Lee & Strazicich (2003) Table 1 — two-break minimum LM critical values
# ---------------------------------------------------------------------------

_LS_MODEL_A_CRITICAL = {
    "1%": -4.545,
    "5%": -3.842,
    "10%": -3.504,
}

_LS_MODEL_C_CRITICAL = {
    (0.2, 0.4): {"1%": -6.16, "5%": -5.59, "10%": -5.27},
    (0.2, 0.6): {"1%": -6.41, "5%": -5.74, "10%": -5.32},
    (0.2, 0.8): {"1%": -6.33, "5%": -5.71, "10%": -5.33},
    (0.4, 0.6): {"1%": -6.45, "5%": -5.67, "10%": -5.31},
    (0.4, 0.8): {"1%": -6.42, "5%": -5.65, "10%": -5.32},
    (0.6, 0.8): {"1%": -6.32, "5%": -5.73, "10%": -5.32},
}

# ---------------------------------------------------------------------------
# Perron (1989) Table IV.B — t-alpha critical values by break fraction
# ---------------------------------------------------------------------------

# Model A (intercept break)
_PERRON_MODEL_INTERCEPT_CRIT = {
    0.1: {0.01: -4.30, 0.025: -3.93, 0.05: -3.68, 0.10: -3.40},
    0.2: {0.01: -4.39, 0.025: -4.08, 0.05: -3.77, 0.10: -3.47},
    0.3: {0.01: -4.39, 0.025: -4.03, 0.05: -3.76, 0.10: -3.46},
    0.4: {0.01: -4.34, 0.025: -4.01, 0.05: -3.72, 0.10: -3.44},
    0.5: {0.01: -4.32, 0.025: -4.01, 0.05: -3.76, 0.10: -3.46},
    0.6: {0.01: -4.45, 0.025: -4.09, 0.05: -3.76, 0.10: -3.47},
    0.7: {0.01: -4.42, 0.025: -4.07, 0.05: -3.80, 0.10: -3.51},
    0.8: {0.01: -4.33, 0.025: -3.99, 0.05: -3.75, 0.10: -3.46},
    0.9: {0.01: -4.27, 0.025: -3.97, 0.05: -3.69, 0.10: -3.38},
}

# Model B (slope/trend break)
_PERRON_MODEL_SLOPE_CRIT = {
    0.1: {0.01: -4.27, 0.025: -3.94, 0.05: -3.65, 0.10: -3.36},
    0.2: {0.01: -4.41, 0.025: -4.08, 0.05: -3.80, 0.10: -3.49},
    0.3: {0.01: -4.51, 0.025: -4.17, 0.05: -3.87, 0.10: -3.58},
    0.4: {0.01: -4.55, 0.025: -4.20, 0.05: -3.94, 0.10: -3.66},
    0.5: {0.01: -4.56, 0.025: -4.26, 0.05: -3.96, 0.10: -3.68},
    0.6: {0.01: -4.57, 0.025: -4.20, 0.05: -3.95, 0.10: -3.66},
    0.7: {0.01: -4.51, 0.025: -4.13, 0.05: -3.85, 0.10: -3.57},
    0.8: {0.01: -4.38, 0.025: -4.07, 0.05: -3.82, 0.10: -3.50},
    0.9: {0.01: -4.26, 0.025: -3.96, 0.05: -3.68, 0.10: -3.35},
}

# Model C (both intercept and slope break)
_PERRON_MODEL_BOTH_CRIT = {
    0.1: {0.01: -4.38, 0.025: -4.01, 0.05: -3.75, 0.10: -3.45},
    0.2: {0.01: -4.65, 0.025: -4.32, 0.05: -3.99, 0.10: -3.66},
    0.3: {0.01: -4.78, 0.025: -4.46, 0.05: -4.17, 0.10: -3.87},
    0.4: {0.01: -4.81, 0.025: -4.48, 0.05: -4.22, 0.10: -3.95},
    0.5: {0.01: -4.90, 0.025: -4.53, 0.05: -4.24, 0.10: -3.96},
    0.6: {0.01: -4.88, 0.025: -4.49, 0.05: -4.24, 0.10: -3.95},
    0.7: {0.01: -4.75, 0.025: -4.44, 0.05: -4.18, 0.10: -3.86},
    0.8: {0.01: -4.70, 0.025: -4.31, 0.05: -4.04, 0.10: -3.69},
    0.9: {0.01: -4.41, 0.025: -4.10, 0.05: -3.80, 0.10: -3.46},
}

_PERRON_CRIT_MAP = {
    "intercept": _PERRON_MODEL_INTERCEPT_CRIT,
    "slope": _PERRON_MODEL_SLOPE_CRIT,
    "both": _PERRON_MODEL_BOTH_CRIT,
}

# ---------------------------------------------------------------------------
# Zivot & Andrews (1992) Table 2 — asymptotic critical values
# ---------------------------------------------------------------------------

# Model A (intercept)
_ZA_MODEL_INTERCEPT_CRIT = {
    0.01: -5.34,
    0.025: -5.04,
    0.05: -4.80,
    0.10: -4.58,
}

# Model B (trend/slope)
_ZA_MODEL_SLOPE_CRIT = {
    0.01: -4.93,
    0.025: -4.67,
    0.05: -4.42,
    0.10: -4.11,
}

# Model C (both)
_ZA_MODEL_BOTH_CRIT = {
    0.01: -5.57,
    0.025: -5.30,
    0.05: -5.08,
    0.10: -4.82,
}

_ZA_CRIT_MAP = {
    "intercept": _ZA_MODEL_INTERCEPT_CRIT,
    "slope": _ZA_MODEL_SLOPE_CRIT,
    "both": _ZA_MODEL_BOTH_CRIT,
}


# ---------------------------------------------------------------------------
# Critical-value helpers
# ---------------------------------------------------------------------------


def _perron_crit(
    model: str,
    break_fraction: float,
    significance: float,
) -> float:
    """Return a Perron (1989) t-alpha critical value.

    Raises
    ------
    ValueError
        If the model, significance level, or break fraction is unsupported.
    """
    _validate_model(model)
    if significance not in (0.01, 0.025, 0.05, 0.10):
        raise ValueError("significance must be one of 0.01, 0.025, 0.05, or 0.10")
    if not 0.1 <= break_fraction <= 0.9:
        raise ValueError("break_fraction must be between 0.1 and 0.9")
    crit_dict = _PERRON_CRIT_MAP[model]
    fractions = np.asarray(sorted(crit_dict), dtype=float)
    values = np.asarray(
        [crit_dict[fraction][significance] for fraction in fractions],
        dtype=float,
    )
    return float(np.interp(break_fraction, fractions, values))


def _za_crit(model: str, significance: float) -> float:
    """Return the Zivot-Andrews (1992) asymptotic critical value.

    Raises
    ------
    ValueError
        If *model* is not one of ``"intercept"``, ``"slope"``, or ``"both"``.
    """
    _validate_model(model)
    return _ZA_CRIT_MAP[model][significance]

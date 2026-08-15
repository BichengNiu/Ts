"""Public input-contract tests for TsSims simulators."""

import numpy as np
import pytest

from Ts.TsSims._cointegration import simulate_cointegrated
from Ts.TsSims._garch import simulate_garch, simulate_igarch
from Ts.TsSims._garch_ext import (
    simulate_egarch,
    simulate_garch_m,
    simulate_gjr_garch,
)
from Ts.TsSims._sarima import simulate_sarima
from Ts.TsSims._ts_ds import (
    simulate_difference_stationary,
    simulate_trend_stationary,
)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 0}, "n"),
        ({"burn": -1}, "burn"),
        ({"p": 0}, "p"),
        ({"q": -1}, "q"),
        ({"omega": 0}, "omega"),
        ({"mean_model": "silent-fallback"}, "mean_model"),
        ({"dist": "silent-fallback"}, "dist"),
        ({"alpha": [0.1, 0.2]}, "alpha"),
        ({"beta": [0.4, 0.2]}, "beta"),
        ({"alpha": [np.nan]}, "alpha"),
        ({"beta": [-0.1]}, "beta"),
    ],
)
def test_garch_rejects_invalid_public_inputs(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        simulate_garch(n=20, seed=1, **kwargs)


def test_garch_accepts_coefficients_matching_declared_orders():
    result = simulate_garch(
        n=20,
        p=2,
        q=2,
        alpha=[0.1, 0.1],
        beta=[0.2, 0.2],
        seed=1,
    )
    assert result.data.shape == (20,)


def test_garch_rejects_nonfinite_student_t_df():
    with pytest.raises(ValueError, match="df"):
        simulate_garch(n=20, dist="t", dist_params={"df": np.nan})


def test_igarch_rejects_unsupported_ar_mean_model():
    with pytest.raises(ValueError, match="mean_model"):
        simulate_igarch(n=20, mean_model="ar")


@pytest.mark.parametrize(
    "simulator",
    [simulate_gjr_garch, simulate_egarch, simulate_garch_m],
)
def test_extended_garch_rejects_invalid_distribution(simulator):
    with pytest.raises(ValueError, match="dist"):
        simulator(n=20, dist="gaussian-fallback")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 0}, "n"),
        ({"burn": -1}, "burn"),
    ],
)
def test_egarch_validates_sample_sizes_before_allocation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        simulate_egarch(seed=1, **kwargs)


def test_extended_garch_enforces_variant_coefficient_lengths():
    with pytest.raises(ValueError, match="gamma"):
        simulate_gjr_garch(n=20, o=2, gamma=[0.1])
    with pytest.raises(ValueError, match="alpha"):
        simulate_egarch(n=20, p=2, alpha=[0.1])


def test_garch_m_rejects_nonfinite_kappa():
    with pytest.raises(ValueError, match="garch_m_kappa"):
        simulate_garch_m(n=20, garch_m_kappa=np.inf)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"order": (1, 0)}, "order"),
        ({"order": (1, -1, 0)}, "order"),
        ({"seasonal_order": (1, 0, 0, 1)}, "period"),
        ({"order": (2, 0, 0), "ar": [0.5]}, "ar"),
        ({"order": (0, 0, 1), "ma": [np.inf]}, "ma"),
        ({"sigma2": 0}, "sigma2"),
    ],
)
def test_sarima_rejects_inconsistent_or_invalid_inputs(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        simulate_sarima(n=20, seed=1, **kwargs)


def test_sarima_accepts_matching_declared_orders():
    result = simulate_sarima(
        n=20,
        order=(2, 0, 1),
        seasonal_order=(1, 0, 1, 4),
        ar=[0.3, 0.1],
        ma=[0.2],
        seasonal_ar=[0.1],
        seasonal_ma=[0.1],
        seed=1,
    )
    assert result.data.shape == (20,)


@pytest.mark.parametrize(
    "call",
    [
        lambda: simulate_cointegrated(n=0),
        lambda: simulate_cointegrated(n=20, sigma=0),
        lambda: simulate_cointegrated(n=20, alpha=[[np.nan], [0.0]]),
        lambda: simulate_trend_stationary(n=0),
        lambda: simulate_trend_stationary(n=20, sigma=np.inf),
        lambda: simulate_difference_stationary(n=20, burn=-1),
        lambda: simulate_difference_stationary(n=20, sigma=0),
    ],
)
def test_other_simulators_reject_invalid_sizes_and_scales(call):
    with pytest.raises((TypeError, ValueError)):
        call()

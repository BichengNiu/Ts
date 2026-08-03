"""Tests for RDL simulation and its estimator contract."""

import numpy as np
import pandas as pd
import pytest

from Ts.TsModels import SARIMAX
from Ts.TsSims import RDLInputSpec, SimRDLResult, simulate_rdl


def test_rdl_input_spec_creates_matching_sparse_estimator_spec():
    spec = RDLInputSpec(
        numerator={0: 1.0, 2: -0.25},
        denominator={1: 0.4, 3: -0.1},
        delay=1,
    )

    model_spec = spec.model_spec

    assert model_spec.numerator_lags == (0, 2)
    assert model_spec.fixed_numerator_lags == (1,)
    assert model_spec.denominator_lags == (1, 3)
    assert model_spec.fixed_denominator_lags == (2,)
    assert model_spec.delay == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"numerator": {}}, "at least one"),
        ({"numerator": {-1: 1.0}}, "non-negative"),
        ({"numerator": {0: np.inf}}, "finite"),
        (
            {"numerator": {0: 1.0}, "denominator": {0: 0.2}},
            "positive",
        ),
    ],
)
def test_rdl_input_spec_rejects_invalid_coefficients(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        RDLInputSpec(**kwargs)


def test_simulate_rdl_returns_inputs_components_and_reproducible_data():
    specs = {
        "x": RDLInputSpec(numerator={0: 1.0}, denominator={1: 0.5}),
        "z": RDLInputSpec(numerator={0: -0.5, 2: 0.2}),
    }
    first = simulate_rdl(n=80, distributed_lags=specs, sigma2=0.2, seed=17)
    second = simulate_rdl(n=80, distributed_lags=specs, sigma2=0.2, seed=17)

    assert isinstance(first, SimRDLResult)
    assert first.get_exog().columns.tolist() == ["x", "z"]
    assert first.get_components().columns.tolist() == ["x", "z", "noise"]
    np.testing.assert_array_equal(first.data, second.data)
    pd.testing.assert_frame_equal(first.get_exog(), second.get_exog())
    np.testing.assert_allclose(first.data, first.get_components().sum(axis=1))
    assert first.distributed_lags["z"].fixed_numerator_lags == (1,)
    assert "Rational Distributed-Lag" in first.summary()


def test_simulate_rdl_uses_supplied_input_and_exact_filter_initialization():
    exog = pd.DataFrame({"x": np.ones(4)}, index=pd.RangeIndex(10, 14))
    result = simulate_rdl(
        n=4,
        distributed_lags={
            "x": RDLInputSpec(
                numerator={0: 1.0},
                denominator={1: 0.5},
                initialization="zero",
            )
        },
        exog=exog,
        sigma2=1e-12,
        seed=18,
        burn=0,
    )

    assert result.get_exog().index.equals(exog.index)
    np.testing.assert_allclose(
        result.get_components()["x"],
        [1.0, 1.5, 1.75, 1.875],
    )


def test_simulate_rdl_rejects_unstable_denominator_by_default():
    specs = {"x": RDLInputSpec(numerator={0: 1.0}, denominator={1: 1.1})}
    with pytest.raises(ValueError, match="unstable"):
        simulate_rdl(n=20, distributed_lags=specs, seed=19)

    result = simulate_rdl(
        n=20,
        distributed_lags=specs,
        seed=19,
        enforce_stability=False,
    )
    assert len(result.data) == 20


def test_simulated_multiple_input_rdl_recovers_joint_estimator_parameters():
    truth = {
        "x1": RDLInputSpec(
            numerator={0: 1.2},
            denominator={1: 0.35, 3: -0.1},
        ),
        "x2": RDLInputSpec(numerator={0: -0.7}, denominator={1: 0.2}),
    }
    simulated = simulate_rdl(
        n=800,
        distributed_lags=truth,
        sigma2=0.09,
        seed=2305,
    )

    fitted = SARIMAX(
        simulated.data,
        exog=simulated.get_exog(),
        order=(0, 0, 0),
        trend="n",
        distributed_lags=simulated.distributed_lags,
    ).fit(method="bfgs", maxiter=300, require_convergence=True)

    assert fitted.params["rdl.x1.omega.L0"] == pytest.approx(1.2, abs=0.07)
    assert fitted.params["rdl.x1.delta.L1"] == pytest.approx(0.35, abs=0.05)
    assert fitted.params["rdl.x1.delta.L3"] == pytest.approx(-0.1, abs=0.05)
    assert "rdl.x1.delta.L2" not in fitted.params
    assert fitted.fixed_params["rdl.x1.delta.L2"] == 0.0
    assert fitted.params["rdl.x2.omega.L0"] == pytest.approx(-0.7, abs=0.07)
    assert fitted.params["rdl.x2.delta.L1"] == pytest.approx(0.2, abs=0.07)
    gains = fitted.steady_state_gains.set_index("input")["estimate"]
    assert gains["x1"] == pytest.approx(1.2 / (1.0 - 0.35 + 0.1), abs=0.16)
    assert gains["x2"] == pytest.approx(-0.7 / (1.0 - 0.2), abs=0.13)

"""Tests for rational distributed-lag specifications and derived quantities."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter
import matplotlib.pyplot as plt

from Ts.TsModels import AutoSARIMAX, SARIMAX, SARIMAXResult
from Ts.TsModels._distributed_lag import (
    RationalLagResult,
    RationalLagSpec,
    _filter_input,
)


def test_integer_orders_expand_to_contiguous_active_lags():
    spec = RationalLagSpec(numerator=3, denominator=2)

    assert spec.numerator_lags == (0, 1, 2, 3)
    assert spec.denominator_lags == (1, 2)
    assert spec.initialization == "auto"
    assert spec.resolved_initialization == "steady_state"
    assert spec.fixed_numerator_lags == ()
    assert spec.fixed_denominator_lags == ()


def test_auto_initialization_resolves_finite_lags_to_conditional_likelihood():
    spec = RationalLagSpec(numerator=3, denominator=0)

    assert spec.initialization == "auto"
    assert spec.resolved_initialization == "conditional"


def test_sparse_orders_fix_omitted_polynomial_coefficients_at_zero():
    spec = RationalLagSpec(numerator=(0, 2, 3), denominator=(1, 3))

    assert spec.numerator_lags == (0, 2, 3)
    assert spec.denominator_lags == (1, 3)
    assert spec.fixed_numerator_lags == (1,)
    assert spec.fixed_denominator_lags == (2,)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"numerator": ()}, "numerator"),
        ({"numerator": (-1, 0)}, "non-negative"),
        ({"numerator": (0, 0)}, "unique"),
        ({"denominator": (0, 1)}, "positive"),
        ({"denominator": (1, 1)}, "unique"),
        ({"delay": -1}, "delay"),
        ({"initialization": "diffuse"}, "initialization"),
    ],
)
def test_invalid_specifications_are_rejected(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        RationalLagSpec(**kwargs)


def test_koyck_weights_gain_and_roots_match_closed_form():
    result = RationalLagResult(
        name="price",
        spec=RationalLagSpec(numerator=0, denominator=1),
        numerator={0: 2.0},
        denominator={1: 0.5},
    )

    np.testing.assert_allclose(result.weights(5).to_numpy(), [2, 1, 0.5, 0.25, 0.125])
    assert result.steady_state_gain == pytest.approx(4.0)
    assert result.is_stable
    np.testing.assert_allclose(result.denominator_roots, [2.0])


def test_boundary_and_near_boundary_denominators_are_diagnosed():
    spec = RationalLagSpec(numerator=0, denominator=1)
    boundary = RationalLagResult(
        name="x",
        spec=spec,
        numerator={0: 1.0},
        denominator={1: 1.0},
    )
    near_boundary = RationalLagResult(
        name="x",
        spec=spec,
        numerator={0: 1.0},
        denominator={1: 0.999},
    )

    assert not boundary.is_stable
    assert np.isnan(boundary.steady_state_gain)
    assert near_boundary.is_stable
    assert near_boundary.steady_state_gain == pytest.approx(1000.0)


def test_delay_prepends_zero_impulse_weights():
    result = RationalLagResult(
        name="advertising",
        spec=RationalLagSpec(numerator=(0, 2), denominator=0, delay=2),
        numerator={0: 1.5, 1: 0.0, 2: -0.25},
        denominator={},
    )

    np.testing.assert_allclose(result.weights(6), [0.0, 0.0, 1.5, 0.0, -0.25, 0.0])
    assert result.steady_state_gain == pytest.approx(1.25)


def test_coefficient_table_includes_fixed_zero_rows():
    result = RationalLagResult(
        name="x",
        spec=RationalLagSpec(numerator=(0, 2), denominator=(1, 3)),
        numerator={0: 1.0, 1: 0.0, 2: 0.5},
        denominator={1: 0.4, 2: 0.0, 3: -0.1},
        std_errors={
            "rdl.x.omega.L0": 0.1,
            "rdl.x.omega.L2": 0.2,
            "rdl.x.delta.L1": 0.05,
            "rdl.x.delta.L3": 0.04,
        },
        p_values={
            "rdl.x.omega.L0": 0.01,
            "rdl.x.omega.L2": 0.02,
            "rdl.x.delta.L1": 0.03,
            "rdl.x.delta.L3": 0.04,
        },
    )

    table = result.coefficients
    fixed = table.loc[table["fixed"]]

    assert fixed["parameter"].tolist() == ["rdl.x.omega.L1", "rdl.x.delta.L2"]
    assert np.all(fixed["estimate"] == 0.0)
    assert np.all(fixed["standard_error"].isna())
    assert result.fixed_params == {
        "rdl.x.omega.L1": 0.0,
        "rdl.x.delta.L2": 0.0,
    }


def test_gain_delta_method_uses_joint_transfer_parameter_covariance():
    names = ("rdl.x.omega.L0", "rdl.x.delta.L1")
    covariance = np.diag([0.04, 0.01])
    result = RationalLagResult(
        name="x",
        spec=RationalLagSpec(numerator=0, denominator=1),
        numerator={0: 2.0},
        denominator={1: 0.5},
        covariance=covariance,
        covariance_names=names,
    )

    gain = result.gain()
    expected_variance = (2.0**2) * 0.04 + (8.0**2) * 0.01

    assert gain["estimate"] == pytest.approx(4.0)
    assert gain["standard_error"] == pytest.approx(np.sqrt(expected_variance))
    assert gain["lower"] < gain["estimate"] < gain["upper"]


def test_rational_lag_result_plots_existing_impulse_weights_as_bars():
    result = RationalLagResult(
        name="price",
        spec=RationalLagSpec(numerator=0, denominator=1),
        numerator={0: 2.0},
        denominator={1: 0.5},
    )

    fig, ax = result.plot_impulse_response(steps=4)

    assert [bar.get_height() for bar in ax.patches] == pytest.approx(
        result.weights(4).tolist()
    )
    assert ax.get_title() == "price"
    assert ax.get_xlabel() == "Time lag"
    plt.close(fig)


def test_rational_lag_plot_reuses_external_axis():
    result = RationalLagResult(
        name="x",
        spec=RationalLagSpec(numerator=0, denominator=0),
        numerator={0: 1.0},
        denominator={},
    )
    fig, supplied = plt.subplots()

    returned_fig, returned_ax = result.plot_impulse_response(3, ax=supplied)

    assert returned_fig is fig
    assert returned_ax is supplied
    plt.close(fig)


def _result_with_structured_lags():
    inputs = {
        "price": RationalLagResult(
            "price",
            RationalLagSpec(0, 1),
            {0: 1.0},
            {1: 0.5},
        ),
        "income": RationalLagResult(
            "income",
            RationalLagSpec(0, 0),
            {0: -0.4},
            {},
        ),
    }
    return SARIMAXResult(
        model_type="SARIMAX",
        params={},
        std_errors={},
        p_values={},
        aic=0.0,
        bic=0.0,
        log_likelihood=0.0,
        residuals=np.zeros(20),
        fitted_values=np.zeros(20),
        nobs=20,
        data=np.zeros(20),
        _distributed_lag_results=inputs,
    )


def test_sarimax_result_facets_all_rdl_impulse_responses():
    result = _result_with_structured_lags()

    fig, axes = result.plot_impulse_response(steps=4)

    assert [axis.get_title() for axis in axes] == ["price", "income"]
    np.testing.assert_allclose(
        [bar.get_height() for bar in axes[0].patches],
        result.weights(4)["price"],
    )
    plt.close(fig)


def test_sarimax_result_can_select_one_rdl_impulse_response():
    result = _result_with_structured_lags()

    fig, ax = result.plot_impulse_response(3, inputs="income")

    assert ax.get_title() == "income"
    assert [bar.get_height() for bar in ax.patches] == pytest.approx(
        result.weights(3)["income"].tolist()
    )
    plt.close(fig)


def test_sarimax_impulse_plot_rejects_missing_or_unknown_rdl_inputs():
    empty = SARIMAXResult(
        model_type="SARIMAX",
        params={},
        std_errors={},
        p_values={},
        aic=0.0,
        bic=0.0,
        log_likelihood=0.0,
        residuals=np.zeros(20),
        fitted_values=np.zeros(20),
        nobs=20,
        data=np.zeros(20),
    )
    with pytest.raises(ValueError, match="no rational"):
        empty.plot_impulse_response(3)

    with pytest.raises(ValueError, match="unknown"):
        _result_with_structured_lags().plot_impulse_response(3, inputs="missing")


def test_filter_initialization_is_explicit():
    values = np.ones(4)
    numerator = np.array([1.0])
    denominator = np.array([1.0, -0.5])

    zero = _filter_input(values, numerator, denominator, initialization="zero")
    steady = _filter_input(
        values,
        numerator,
        denominator,
        initialization="steady_state",
    )

    np.testing.assert_allclose(zero, [1.0, 1.5, 1.75, 1.875])
    np.testing.assert_allclose(steady, [2.0, 2.0, 2.0, 2.0])


def test_auto_filter_uses_steady_state_for_a_rational_denominator():
    values = np.ones(4)
    numerator = np.array([1.0])
    denominator = np.array([1.0, -0.5])

    automatic = _filter_input(
        values,
        numerator,
        denominator,
        initialization="auto",
    )

    np.testing.assert_allclose(automatic, [2.0, 2.0, 2.0, 2.0])


def test_finite_sparse_rdl_matches_explicit_lagged_regression():
    rng = np.random.default_rng(2301)
    x = rng.normal(size=300)
    lag2 = np.r_[0.0, 0.0, x[:-2]]
    y = 1.25 * x - 0.45 * lag2 + rng.normal(scale=0.25, size=len(x))

    rdl = SARIMAX(
        y,
        exog=x[:, None],
        exog_names=["x"],
        order=(0, 0, 0),
        trend="n",
        distributed_lags={
            "x": RationalLagSpec(
                numerator=(0, 2),
                denominator=0,
                initialization="zero",
            )
        },
    ).fit(method="bfgs", maxiter=200, require_convergence=True)
    explicit = SARIMAX(
        y,
        exog=np.column_stack([x, lag2]),
        exog_names=["x0", "x2"],
        order=(0, 0, 0),
        trend="n",
    ).fit(method="bfgs", maxiter=200, require_convergence=True)

    assert rdl.params["rdl.x.omega.L0"] == pytest.approx(explicit.params["x0"])
    assert rdl.params["rdl.x.omega.L2"] == pytest.approx(explicit.params["x2"])
    assert "rdl.x.omega.L1" not in rdl.params
    assert rdl.fixed_params["rdl.x.omega.L1"] == 0.0
    assert "rdl.x.omega.L1" in rdl.summary()


def test_auto_finite_rdl_uses_complete_history_and_robust_ar_start():
    rng = np.random.default_rng(2312)
    nobs = 420
    x = rng.normal(loc=8.0, scale=1.5, size=nobs)
    lag1 = np.r_[0.0, x[:-1]]
    lag2 = np.r_[0.0, 0.0, x[:-2]]
    disturbance = lfilter([1.0], [1.0, -0.82], rng.normal(scale=0.2, size=nobs))
    y = 12.0 + 1.1 * x - 0.7 * lag1 + 0.35 * lag2 + disturbance

    rdl = SARIMAX(
        y,
        exog=pd.Series(x, name="x"),
        order=(1, 0, 0),
        trend="c",
        distributed_lags={"x": RationalLagSpec(numerator=2, denominator=0)},
    ).fit(method="bfgs", maxiter=400, require_convergence=True)
    explicit = SARIMAX(
        y[2:],
        exog=np.column_stack([x[2:], lag1[2:], lag2[2:]]),
        exog_names=["x0", "x1", "x2"],
        order=(1, 0, 0),
        trend="c",
    ).fit(method="bfgs", maxiter=400, require_convergence=True)

    assert rdl.likelihood_burn == 3
    assert rdl.effective_nobs == nobs - 3
    assert len(rdl.residuals) == rdl.effective_nobs
    assert rdl.level_intercept == pytest.approx(explicit.level_intercept, abs=0.12)
    assert rdl.params["ar.L1"] == pytest.approx(explicit.params["ar.L1"], abs=0.02)
    assert rdl.params["rdl.x.omega.L0"] == pytest.approx(
        explicit.params["x0"], abs=0.02
    )
    assert rdl.params["rdl.x.omega.L1"] == pytest.approx(
        explicit.params["x1"], abs=0.02
    )
    assert rdl.params["rdl.x.omega.L2"] == pytest.approx(
        explicit.params["x2"], abs=0.02
    )
    assert "initialization auto -> conditional" in rdl.summary()


def test_named_series_exog_uses_its_name_for_rdl_mapping():
    rng = np.random.default_rng(2311)
    x = pd.Series(rng.normal(size=300), name="leading_indicator")
    y = 1.3 * x.to_numpy() + rng.normal(scale=0.2, size=len(x))

    result = SARIMAX(
        y,
        exog=x,
        order=(0, 0, 0),
        trend="n",
        distributed_lags={
            "leading_indicator": RationalLagSpec(numerator=0, denominator=0)
        },
    ).fit(method="bfgs", maxiter=200, require_convergence=True)

    assert result.distributed_lag_names == ("leading_indicator",)
    assert result.params["rdl.leading_indicator.omega.L0"] == pytest.approx(
        1.3,
        abs=0.04,
    )


def test_koyck_model_recovers_coefficient_gain_and_weights():
    rng = np.random.default_rng(2302)
    x = rng.normal(size=500)
    dynamic_effect = lfilter([1.4], [1.0, -0.55], x)
    y = dynamic_effect + rng.normal(scale=0.35, size=len(x))

    result = SARIMAX(
        y,
        exog=x[:, None],
        exog_names=["x"],
        order=(0, 0, 0),
        trend="n",
        distributed_lags={"x": RationalLagSpec(numerator=0, denominator=1)},
    ).fit(method="bfgs", maxiter=300, require_convergence=True)

    assert result.params["rdl.x.omega.L0"] == pytest.approx(1.4, abs=0.08)
    assert result.params["rdl.x.delta.L1"] == pytest.approx(0.55, abs=0.06)
    assert result.steady_state_gains.loc[0, "estimate"] == pytest.approx(
        1.4 / (1.0 - 0.55),
        abs=0.25,
    )
    np.testing.assert_allclose(
        result.weights(3)["x"],
        [
            result.params["rdl.x.omega.L0"],
            result.params["rdl.x.omega.L0"] * result.params["rdl.x.delta.L1"],
            result.params["rdl.x.omega.L0"] * result.params["rdl.x.delta.L1"] ** 2,
        ],
    )


def test_multiple_rational_inputs_are_jointly_estimated():
    rng = np.random.default_rng(2303)
    x1 = rng.normal(size=500)
    x2 = rng.normal(size=500)
    y = (
        lfilter([1.2], [1.0, -0.45], x1)
        + lfilter([-0.7], [1.0, -0.2], x2)
        + rng.normal(scale=0.3, size=500)
    )

    result = SARIMAX(
        y,
        exog=np.column_stack([x1, x2]),
        exog_names=["x1", "x2"],
        order=(0, 0, 0),
        trend="n",
        distributed_lags={
            "x1": RationalLagSpec(numerator=0, denominator=1),
            "x2": RationalLagSpec(numerator=0, denominator=1),
        },
    ).fit(method="bfgs", maxiter=300, require_convergence=True)

    assert result.distributed_lag_names == ("x1", "x2")
    assert result.params["rdl.x1.omega.L0"] == pytest.approx(1.2, abs=0.08)
    assert result.params["rdl.x1.delta.L1"] == pytest.approx(0.45, abs=0.06)
    assert result.params["rdl.x2.omega.L0"] == pytest.approx(-0.7, abs=0.08)
    assert result.params["rdl.x2.delta.L1"] == pytest.approx(0.2, abs=0.08)
    assert result.weights(4).columns.tolist() == ["x1", "x2"]


def test_rdl_and_ordinary_exog_can_share_one_input_frame():
    rng = np.random.default_rng(2304)
    x = rng.normal(size=250)
    control = rng.normal(size=250)
    y = (
        lfilter([0.9], [1.0, -0.3], x)
        + 1.5 * control
        + rng.normal(
            scale=0.3,
            size=250,
        )
    )

    result = SARIMAX(
        y,
        exog=np.column_stack([x, control]),
        exog_names=["x", "control"],
        order=(0, 0, 0),
        trend="n",
        distributed_lags={"x": RationalLagSpec(numerator=0, denominator=1)},
    ).fit(method="bfgs", maxiter=300, require_convergence=True)

    assert result.params["control"] == pytest.approx(1.5, abs=0.08)
    assert "x" not in result.params
    assert result.params["rdl.x.omega.L0"] == pytest.approx(0.9, abs=0.08)


def test_invalid_rdl_mapping_and_gapped_samples_are_rejected():
    y = np.arange(20.0)
    x = np.arange(20.0)[:, None]

    with pytest.raises(ValueError, match="unknown exog"):
        SARIMAX(
            y,
            exog=x,
            exog_names=["x"],
            distributed_lags={"z": RationalLagSpec()},
        )
    with pytest.raises(TypeError, match="RationalLagSpec"):
        SARIMAX(
            y,
            exog=x,
            exog_names=["x"],
            distributed_lags={"x": {"numerator": 1}},
        )

    dated_y = pd.Series(y, index=pd.date_range("2000-01-01", periods=20, freq="YS"))
    dated_x = pd.DataFrame({"x": np.arange(20.0)}, index=dated_y.index)
    dated_x.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="consecutive observations"):
        SARIMAX(
            dated_y,
            exog=dated_x,
            distributed_lags={"x": RationalLagSpec()},
            missing="drop",
        )


def test_rdl_require_convergence_rejects_optimizer_failure(monkeypatch):
    def nonconverged(*args, **kwargs):
        return SimpleNamespace(mle_retvals={"converged": False})

    monkeypatch.setattr(
        "Ts.TsModels._sarimax._RationalLagSARIMAX.fit",
        nonconverged,
    )
    model = SARIMAX(
        np.arange(20.0),
        exog=np.arange(20.0)[:, None],
        exog_names=["x"],
        trend="n",
        distributed_lags={"x": RationalLagSpec()},
    )

    with pytest.raises(RuntimeError, match="failed to converge"):
        model.fit(require_convergence=True)


def test_future_forecast_continues_filter_over_full_input_history():
    rng = np.random.default_rng(2305)
    nobs = 300
    steps = 6
    dates = pd.date_range("2000-01-01", periods=nobs + steps, freq="MS")
    x = rng.normal(size=nobs + steps)
    complete_effect = lfilter([1.1], [1.0, -0.6], x)
    y = pd.Series(
        complete_effect[:nobs] + rng.normal(scale=0.15, size=nobs),
        index=dates[:nobs],
    )
    exog = pd.DataFrame({"x": x}, index=dates)

    result = SARIMAX(
        y,
        exog=exog,
        order=(0, 0, 0),
        trend="n",
        distributed_lags={"x": RationalLagSpec(numerator=0, denominator=1)},
    ).fit(method="bfgs", maxiter=300, require_convergence=True)
    forecast = result.predict(start=nobs, end=nobs + steps - 1)

    expected = result.distributed_lags["x"].filter(x)[-steps:]
    np.testing.assert_allclose(forecast.mean, expected, atol=1e-9)
    assert np.all(forecast.is_oos)


def test_future_forecast_combines_rdl_and_static_exog_once():
    rng = np.random.default_rng(2309)
    nobs = 250
    steps = 3
    x = rng.normal(size=nobs + steps)
    control = rng.normal(size=nobs + steps)
    y = (
        lfilter([0.9], [1.0, -0.3], x)[:nobs]
        + 1.5 * control[:nobs]
        + rng.normal(scale=0.01, size=nobs)
    )
    fitted = SARIMAX(
        y,
        exog=np.column_stack([x[:nobs], control[:nobs]]),
        exog_names=["x", "control"],
        order=(0, 0, 0),
        trend="n",
        distributed_lags={"x": RationalLagSpec(denominator=1)},
    ).fit(method="bfgs", maxiter=300, require_convergence=True)

    forecast = fitted.predict(
        start=nobs,
        end=nobs + steps - 1,
        future_exog=np.column_stack([x[nobs:], control[nobs:]]),
    )
    expected = (
        fitted.distributed_lags["x"].filter(x)[-steps:]
        + fitted.params["control"] * control[-steps:]
    )

    np.testing.assert_allclose(forecast.mean, expected, atol=1e-10)


def test_future_rdl_scenarios_use_each_complete_input_path():
    rng = np.random.default_rng(2306)
    nobs = 250
    steps = 4
    dates = pd.date_range("2010-01-01", periods=nobs, freq="QS")
    x = rng.normal(size=nobs)
    y = pd.Series(
        lfilter([0.8], [1.0, -0.5], x) + rng.normal(scale=0.2, size=nobs),
        index=dates,
    )
    fitted = SARIMAX(
        y,
        exog=pd.DataFrame({"x": x}, index=dates),
        order=(0, 0, 0),
        trend="n",
        distributed_lags={"x": RationalLagSpec(numerator=0, denominator=1)},
    ).fit(method="bfgs", maxiter=300, require_convergence=True)
    future_dates = pd.date_range(dates[-1], periods=steps + 1, freq="QS")[1:]

    with pytest.raises(ValueError, match="future exog is required"):
        fitted.predict(
            start=nobs,
            end=nobs + steps - 1,
            future_dates=future_dates,
        )

    scenarios = fitted.predict(
        start=nobs,
        end=nobs + steps - 1,
        future_dates=future_dates,
        future_exog={
            "zero": pd.DataFrame({"x": np.zeros(steps)}, index=future_dates),
            "one": pd.DataFrame({"x": np.ones(steps)}, index=future_dates),
        },
    )

    assert scenarios.scenarios.keys() == {"zero", "one"}
    assert not np.allclose(scenarios["zero"].mean, scenarios["one"].mean)


def test_rdl_oos_clone_preserves_specification():
    rng = np.random.default_rng(2307)
    x = rng.normal(size=180)
    y = lfilter([1.0], [1.0, -0.35], x) + rng.normal(scale=0.25, size=180)

    evaluation = SARIMAX(
        y,
        exog=x[:, None],
        exog_names=["x"],
        order=(0, 0, 0),
        trend="n",
        distributed_lags={"x": RationalLagSpec(numerator=0, denominator=1)},
    ).oos(
        estimation_period=(0, 139),
        validation_period=(140, 179),
    )

    assert len(evaluation.mean) == 40
    assert np.isfinite(evaluation.metrics["rmse"])


def test_auto_sarimax_keeps_fixed_rdl_spec_across_candidates():
    rng = np.random.default_rng(2308)
    x = rng.normal(size=200)
    y = lfilter([1.0], [1.0, -0.4], x) + rng.normal(scale=0.25, size=200)

    result = AutoSARIMAX(
        y,
        p=(0, 0),
        d=(0, 0),
        q=(0, 0),
        P=(0, 0),
        D=(0, 0),
        Q=(0, 0),
        trend="n",
        exog=x[:, None],
        exog_names=["x"],
        distributed_lags={"x": RationalLagSpec(numerator=0, denominator=1)},
    ).fit()

    assert result.distributed_lags["x"].spec.denominator_lags == (1,)
    assert result.steady_state_gains.loc[0, "input"] == "x"
    assert result.weights(3).columns.tolist() == ["x"]


def test_intervention_bootstrap_refit_preserves_rdl_backend():
    from Ts.TsModels import EventSpec
    from Ts.TsModels._intervention import _bootstrap_refit

    rng = np.random.default_rng(2310)
    dates = pd.date_range("2010-01-01", periods=100, freq="MS")
    x = rng.normal(size=100)
    policy = np.zeros(100)
    policy[50:] = 1.0
    y = lfilter([0.9], [1.0, -0.3], x) + 1.1 * policy + rng.normal(scale=0.2, size=100)
    fitted = SARIMAX(
        pd.Series(y, index=dates),
        exog=pd.DataFrame({"x": x}, index=dates),
        events=[EventSpec("policy", [dates[50]], "step")],
        order=(0, 0, 0),
        trend="n",
        distributed_lags={"x": RationalLagSpec(denominator=1)},
    ).fit(method="bfgs", maxiter=300, require_convergence=True)

    names, parameters = _bootstrap_refit(fitted, np.random.default_rng(2311))

    assert names == tuple(fitted.params)
    assert parameters.shape == (len(fitted.params),)

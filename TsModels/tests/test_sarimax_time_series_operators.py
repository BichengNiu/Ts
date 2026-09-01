"""Regression tests for SARIMAX exogenous time-series operators and IRFs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Ts.TsModels import AutoSARIMAX, SARIMAX, TimeSeriesOperator


def _operator_fixture(n=80):
    rng = np.random.default_rng(20260901)
    index = pd.date_range("2015-01-01", periods=n, freq="MS")
    exog = pd.DataFrame(
        {
            "lagged": rng.normal(size=n + 3),
            "differenced": np.cumsum(rng.normal(size=n + 3)),
            "seasonal": np.cumsum(rng.normal(size=n + 3)),
        },
        index=pd.date_range("2015-01-01", periods=n + 3, freq="MS"),
    )
    y = pd.Series(rng.normal(size=n), index=index, name="y")
    operators = {
        "lagged": TimeSeriesOperator(lag=1),
        "differenced": TimeSeriesOperator(difference=1),
        "seasonal": TimeSeriesOperator(seasonal_difference=1, seasonal_period=12),
    }
    return y, exog, operators


def test_per_variable_operators_trim_only_structural_leading_rows():
    y, exog, operators = _operator_fixture()

    model = SARIMAX(
        y,
        exog=exog,
        order=(0, 0, 0),
        trend="n",
        exog_operators=operators,
        missing="raise",
    )

    assert model.operator_burn == 12
    assert model.dropped_positions == ()
    assert len(model.data) == len(y) - 12
    np.testing.assert_allclose(model.exog[:, 0], exog["lagged"].iloc[11:-4])
    np.testing.assert_allclose(
        model.exog[:, 1], exog["differenced"].diff().iloc[12:-3]
    )
    np.testing.assert_allclose(
        model.exog[:, 2], exog["seasonal"].diff(12).iloc[12:-3]
    )


def test_future_exog_is_transformed_with_raw_history():
    y, exog, operators = _operator_fixture()
    result = SARIMAX(
        y,
        exog=exog,
        order=(0, 0, 0),
        trend="n",
        exog_operators=operators,
    ).fit(maxiter=100)

    future = exog.iloc[len(y) :].copy()
    prediction = result.predict(
        start=result.nobs,
        end=result.nobs + len(future) - 1,
        future_exog=future,
    )

    assert len(prediction.scenarios["custom"].mean) == len(future)
    assert np.isfinite(prediction.scenarios["custom"].mean).all()


def test_log_operator_transforms_exog_and_prefixes_parameter_name():
    rng = np.random.default_rng(20260903)
    dates = pd.date_range("2018-01-01", periods=48, freq="MS")
    exog = pd.DataFrame(
        {"x": np.exp(rng.normal(size=len(dates)))},
        index=dates,
    )
    y = pd.Series(rng.normal(size=len(dates)), index=dates, name="y")
    operator = TimeSeriesOperator(
        lag=1,
        difference=1,
        seasonal_difference=1,
        seasonal_period=12,
        log=True,
    )

    model = SARIMAX(
        y,
        exog=exog,
        order=(0, 0, 0),
        trend="n",
        exog_operators={"x": operator},
        missing="raise",
    )

    expected = np.log(exog["x"]).diff(12).diff().shift(1).iloc[14:]
    assert model.design_columns == ("log.L.D.S.x",)
    np.testing.assert_allclose(model.exog[:, 0], expected.to_numpy())

    result = model.fit(maxiter=100)
    assert "log.L.D.S.x" in result.params


def test_log_operator_requires_positive_exogenous_values():
    exog = pd.DataFrame({"x": [1.0, 2.0, 0.0] * 8})
    with pytest.raises(ValueError, match="strictly positive exogenous"):
        SARIMAX(
            np.ones(len(exog)),
            exog=exog,
            order=(0, 0, 0),
            trend="n",
            exog_operators={"x": TimeSeriesOperator(log=True)},
        )


def test_log_operator_future_exog_uses_raw_scale():
    rng = np.random.default_rng(20260904)
    n = 40
    exog = pd.DataFrame(
        {"x": np.exp(rng.normal(size=n + 3))},
        index=pd.date_range("2019-01-01", periods=n + 3, freq="MS"),
    )
    y = pd.Series(rng.normal(size=n), index=exog.index[:n], name="y")
    result = SARIMAX(
        y,
        exog=exog,
        order=(0, 0, 0),
        trend="n",
        exog_operators={"x": TimeSeriesOperator(log=True)},
    ).fit(maxiter=100)

    prediction = result.predict(
        start=result.nobs,
        end=result.nobs + 2,
        future_exog=exog.iloc[n:],
    )

    assert len(prediction.scenarios["custom"].mean) == 3
    assert np.isfinite(prediction.scenarios["custom"].mean).all()


def test_operator_mapping_validation_and_rdl_boundary():
    y, exog, _ = _operator_fixture()
    with pytest.raises(ValueError, match="unknown exogenous"):
        SARIMAX(y, exog=exog.iloc[: len(y)], exog_operators={"missing": TimeSeriesOperator()})
    with pytest.raises(TypeError, match="TimeSeriesOperator"):
        SARIMAX(y, exog=exog.iloc[: len(y)], exog_operators={"lagged": "L1"})
    with pytest.raises(TypeError, match="seasonal_period"):
        TimeSeriesOperator(seasonal_difference=1)


def test_auto_sarimax_reuses_operator_configuration_for_candidates():
    y, exog, operators = _operator_fixture()
    result = AutoSARIMAX(
        y,
        exog=exog,
        p=(0, 0),
        d=(0, 0),
        q=(0, 0),
        P=(0, 0),
        D=(0, 0),
        Q=(0, 0),
        trend="n",
        exog_operators=operators,
        maxiter=100,
        n_jobs=1,
    ).fit()

    assert result.best_result.exog_operators == operators


def test_innovation_irf_and_root_diagnostics_delegate_to_statsmodels():
    rng = np.random.default_rng(20260902)
    result = SARIMAX(rng.normal(size=100), order=(1, 0, 1), trend="n").fit()

    response = result.innovation_impulse_response(4)
    expected = result._statsmodels_result.impulse_responses(steps=4)
    np.testing.assert_allclose(response.to_numpy(), expected)
    np.testing.assert_allclose(
        result.innovation_impulse_response(4, cumulative=True).to_numpy(),
        np.cumsum(expected),
    )
    diagnostics = result.root_diagnostics
    assert set(diagnostics["component"]) == {"AR", "MA"}
    assert diagnostics["outside_unit_circle"].all()
    fig, ax = result.plot_innovation_impulse_response(4)
    try:
        assert len(ax.patches) == 5
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

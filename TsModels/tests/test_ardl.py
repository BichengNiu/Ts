"""ARDL estimation, selection, prediction, and public result contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.ardl import ARDL as StatsmodelsARDL
from statsmodels.tsa.ardl import ardl_select_order

from Ts.TsModels import ARDL, AutoARDL, AutoARDLResult, ARDLResult


def _sample(n: int = 100) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    innovations = rng.normal(scale=0.25, size=n)
    y = np.empty(n)
    y[:2] = 2.0
    for t in range(2, n):
        y[t] = (
            0.7
            + 0.45 * y[t - 1]
            - 0.15 * y[t - 2]
            + 0.8 * x1[t]
            - 0.3 * x1[t - 2]
            + 0.25 * x2[t - 1]
            + innovations[t]
        )
    return (
        pd.Series(y, index=dates, name="output"),
        pd.DataFrame({"x1": x1, "x2": x2}, index=dates),
    )


def test_manual_ardl_matches_statsmodels_for_sparse_per_input_lags():
    y, x = _sample()
    order = {"x1": [0, 2], "x2": [1]}

    expected = StatsmodelsARDL(
        y,
        lags=[1, 2],
        exog=x,
        order=order,
        trend="c",
        missing="none",
    ).fit(cov_type="HC1", use_t=True)
    result = ARDL(
        y,
        lags=[1, 2],
        exog=x,
        order=order,
        trend="c",
        missing="raise",
    ).fit(cov_type="HC1", use_t=True)

    assert isinstance(result, ARDLResult)
    assert result.ar_lags == (1, 2)
    assert result.distributed_lags == {"x1": (0, 2), "x2": (1,)}
    assert result.ardl_order == (2, 2, 1)
    assert result.hold_back == 2
    assert result.effective_nobs == expected.nobs
    assert result.dates.equals(y.index)
    assert result.exog_names == ("x1", "x2")
    assert_allclose(list(result.params.values()), expected.params, rtol=1e-9)
    assert_allclose(list(result.std_errors.values()), expected.bse, rtol=1e-9)
    assert_allclose(list(result.p_values.values()), expected.pvalues, rtol=1e-9)
    assert_allclose(result.residuals, expected.resid, rtol=1e-9)
    assert np.isnan(result.fitted_values[:2]).all()
    assert_allclose(result.fitted_values[2:], expected.fittedvalues, rtol=1e-9)
    assert result.converged
    assert result.optimizer == "conditional_mle"


def test_ardl_predict_validates_complete_named_future_inputs():
    y, x = _sample()
    result = ARDL(y, lags=1, exog=x, order={"x1": 1, "x2": 0}).fit()
    future_dates = pd.date_range(y.index[-1], periods=5, freq="MS")[1:]
    future = pd.DataFrame(
        {"x1": np.linspace(0.1, 0.4, 4), "x2": np.linspace(0.5, 0.8, 4)},
        index=future_dates,
    )

    prediction = result.predict(
        start=len(y),
        end=len(y) + 3,
        future_exog=future,
    )

    assert prediction.mean.shape == (4,)
    assert prediction.lower.shape == (4,)
    assert prediction.upper.shape == (4,)
    assert prediction.is_oos.tolist() == [True] * 4
    with pytest.raises(ValueError, match="columns"):
        result.predict(
            start=len(y),
            end=len(y) + 3,
            future_exog=future[["x1"]],
        )
    with pytest.raises(ValueError, match="4 rows"):
        result.predict(
            start=len(y),
            end=len(y) + 3,
            future_exog=future.iloc[:3],
        )


def test_log_ardl_returns_bias_adjusted_predictions_on_original_scale():
    y, x = _sample()
    positive = np.exp(y / 5.0)
    result = ARDL(
        positive,
        lags=1,
        exog=x[["x1"]],
        order=1,
        log=True,
    ).fit()

    future_dates = pd.date_range(y.index[-1], periods=3, freq="MS")[1:]
    future = pd.DataFrame({"x1": [0.1, 0.2]}, index=future_dates)
    prediction = result.predict(
        start=len(y),
        end=len(y) + 1,
        future_exog=future,
    )

    assert result.log
    assert np.nanmin(result.fitted_values) > 0.0
    assert np.all(prediction.mean > 0.0)
    assert np.all(prediction.lower > 0.0)
    assert np.all(prediction.upper > prediction.lower)


def test_auto_ardl_matches_statsmodels_selected_structure_and_exposes_table():
    y, x = _sample(80)
    expected = ardl_select_order(
        y,
        maxlag=2,
        exog=x,
        maxorder={"x1": 2, "x2": 1},
        trend="c",
        ic="bic",
        glob=False,
        missing="none",
    )

    result = AutoARDL(
        y,
        maxlag=2,
        exog=x,
        maxorder={"x1": 2, "x2": 1},
        trend="c",
        criterion="bic",
        search_method="hierarchical",
        missing="raise",
    ).fit()

    assert isinstance(result, AutoARDLResult)
    assert result.best_result.ar_lags == tuple(expected.model.ar_lags or ())
    assert result.best_result.distributed_lags == {
        name: tuple(lags) for name, lags in expected.model.dl_lags.items()
    }
    assert result.selection_criterion == "bic"
    assert result.search_method == "hierarchical"
    assert list(result.criterion_table.columns) == [
        "criterion",
        "target_lags",
        "input_lags",
    ]
    assert len(result.criterion_table) > 1
    assert result.predict(start=10, end=12).mean.shape == (3,)


def test_auto_ardl_preserves_inputs_excluded_by_selected_order():
    """A selected ``None`` input must not be restored as lag zero on refit."""
    index = pd.date_range("2000-01-01", periods=60, freq="MS")
    y = pd.Series(
        10.0 + np.arange(60) * 0.08 + np.sin(np.arange(60) / 3.0),
        index=index,
    )
    x = pd.DataFrame(
        {"policy": 1.0 + np.arange(60) * 0.05 + np.cos(np.arange(60) / 4.0)},
        index=index,
    )

    result = AutoARDL(
        y,
        maxlag=1,
        exog=x,
        maxorder={"policy": 0},
        trend="c",
        criterion="bic",
    ).fit()

    assert result.distributed_lags == {}
    assert not any("policy" in name for name in result.params)


def test_auto_ardl_global_parallel_matches_serial_and_statsmodels():
    """Global ARDL selection keeps the reference structure in parallel mode."""
    y, x = _sample(70)
    kwargs = {
        "maxlag": 1,
        "exog": x,
        "maxorder": {"x1": 1, "x2": 1},
        "trend": "c",
        "criterion": "bic",
        "search_method": "global",
        "missing": "raise",
    }
    expected = ardl_select_order(
        y,
        maxlag=kwargs["maxlag"],
        exog=x,
        maxorder=kwargs["maxorder"],
        trend=kwargs["trend"],
        ic=kwargs["criterion"],
        glob=True,
        missing="none",
    )
    serial = AutoARDL(y, n_jobs=1, **kwargs).fit()
    parallel = AutoARDL(y, n_jobs=2, **kwargs).fit()

    assert serial.best_result.ar_lags == tuple(expected.model.ar_lags or ())
    assert serial.best_result.distributed_lags == {
        name: tuple(lags) for name, lags in expected.model.dl_lags.items()
    }
    assert parallel.best_result.ar_lags == serial.best_result.ar_lags
    assert parallel.best_result.distributed_lags == serial.best_result.distributed_lags
    pd.testing.assert_frame_equal(
        parallel.criterion_table,
        serial.criterion_table,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_ardl_rejects_internal_missing_rows_that_break_lag_adjacency():
    y, x = _sample(40)
    y.iloc[20] = np.nan

    with pytest.raises(ValueError, match="consecutive observations"):
        ARDL(y, lags=1, exog=x, order=1, missing="drop")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"log": True}, "strictly positive"),
        ({"search_method": "beam"}, "search_method"),
        ({"criterion": "hqic"}, "criterion"),
    ],
)
def test_ardl_validation_errors_are_explicit(kwargs, message):
    y, x = _sample(40)
    if kwargs == {"log": True}:
        y.iloc[0] = 0.0
        with pytest.raises(ValueError, match=message):
            ARDL(y, lags=1, exog=x, order=1, **kwargs)
    else:
        with pytest.raises((TypeError, ValueError), match=message):
            AutoARDL(y, maxlag=2, exog=x, maxorder=2, **kwargs)

"""Tests for dated SARIMAX/ARIMAX input contracts."""

import numpy as np
import pandas as pd
import pytest

from Ts.TsModels import SARIMAX


def test_new_arimax_types_are_public():
    from Ts import EventSpec, PolicyEffectResult, ScenarioForecastResult
    from Ts.TsModels import (
        EventSpec as ModelEventSpec,
    )
    from Ts.TsModels import (
        PolicyEffectResult as ModelPolicyEffectResult,
    )
    from Ts.TsModels import (
        ScenarioForecastResult as ModelScenarioForecastResult,
    )

    assert EventSpec is ModelEventSpec
    assert PolicyEffectResult is ModelPolicyEffectResult
    assert ScenarioForecastResult is ModelScenarioForecastResult


def test_dataframe_exog_splits_history_and_default_future():
    y_dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    all_dates = pd.date_range("2025-01-01", periods=15, freq="MS")
    y = pd.Series(np.arange(12.0), index=y_dates)
    exog = pd.DataFrame({"price": np.arange(15.0)}, index=all_dates)

    model = SARIMAX(y, exog=exog, order=(0, 0, 0), trend="n")

    assert model.exog.shape == (12, 1)
    assert model.exog_names == ("price",)
    assert model.dates.equals(y_dates)
    assert model.future_exog.index.equals(all_dates[12:])
    assert model.future_exog.columns.tolist() == ["price"]


def test_ndarray_exog_requires_names_and_equal_length():
    with pytest.raises(ValueError, match="exog_names"):
        SARIMAX(np.arange(12.0), exog=np.ones((12, 1)))
    with pytest.raises(ValueError, match="12 observations"):
        SARIMAX(
            np.arange(12.0),
            exog=np.ones((11, 1)),
            exog_names=["x"],
        )


def test_one_dimensional_ndarray_exog_is_normalised_to_one_column():
    model = SARIMAX(
        np.arange(12.0),
        exog=np.linspace(-1.0, 1.0, 12),
        exog_names=["x"],
    )

    assert model.exog.shape == (12, 1)
    assert model.exog_names == ("x",)


def test_array_exog_rejects_more_than_two_dimensions():
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        SARIMAX(
            np.arange(12.0),
            exog=np.ones((12, 1, 1)),
            exog_names=["x"],
        )


def test_named_series_exog_uses_name_and_is_normalised_to_one_column():
    x = pd.Series(np.arange(12.0), name="leading_indicator")

    model = SARIMAX(np.arange(12.0), exog=x, trend="n")

    assert model.exog.shape == (12, 1)
    assert model.exog_names == ("leading_indicator",)


def test_dated_series_exog_splits_history_and_default_future():
    y_dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    all_dates = pd.date_range("2025-01-01", periods=15, freq="MS")
    y = pd.Series(np.arange(12.0), index=y_dates)
    exog = pd.Series(np.arange(15.0), index=all_dates, name="price")

    model = SARIMAX(y, exog=exog, order=(0, 0, 0), trend="n")

    assert model.exog.shape == (12, 1)
    assert model.exog_names == ("price",)
    assert model.future_exog.index.equals(all_dates[12:])
    assert model.future_exog.columns.tolist() == ["price"]


def test_unnamed_series_exog_requires_exactly_one_explicit_name():
    y = np.arange(12.0)
    exog = pd.Series(np.linspace(-1.0, 1.0, 12))

    with pytest.raises(ValueError, match=r"unnamed.*exog_names"):
        SARIMAX(y, exog=exog)

    model = SARIMAX(y, exog=exog, exog_names=["x"])
    assert model.exog_names == ("x",)


def test_named_series_exog_rejects_redundant_explicit_names():
    y = np.arange(12.0)
    exog = pd.Series(np.ones(12), name="x")

    with pytest.raises(ValueError, match=r"exog_names.*named Series"):
        SARIMAX(y, exog=exog, exog_names=["renamed"])


def test_missing_default_drops_y_exog_and_dates_jointly():
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    y = pd.Series(np.arange(12.0), index=dates)
    x = pd.DataFrame({"x": np.arange(12.0)}, index=dates)
    y.iloc[2] = np.nan
    x.iloc[4, 0] = np.nan

    model = SARIMAX(y, exog=x)

    assert model.missing == "drop"
    assert len(model.data) == 10
    assert dates[2] not in model.dates
    assert dates[4] not in model.dates
    assert model.exog.shape == (10, 1)
    assert model.dropped_positions == (2, 4)


def test_missing_raise_explicitly_rejects_non_finite_values():
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    y = pd.Series(np.arange(12.0), index=dates)
    x = pd.DataFrame({"x": np.arange(12.0)}, index=dates)
    y.iloc[1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        SARIMAX(y, exog=x, missing="raise")

    y.iloc[1] = 1.0
    x.iloc[3, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        SARIMAX(y, exog=x, missing="raise")


def test_unknown_missing_policy_fails():
    with pytest.raises(ValueError, match="missing"):
        SARIMAX(np.arange(12.0), missing="fill")


def test_ndarray_data_accepts_explicit_dates():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")

    model = SARIMAX(np.arange(12.0), dates=dates)

    assert model.dates.equals(dates)


def test_series_rejects_conflicting_dates_argument():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    y = pd.Series(np.arange(12.0), index=dates)

    with pytest.raises(ValueError, match=r"dates.*Series"):
        SARIMAX(y, dates=dates)


@pytest.mark.parametrize(
    "dates",
    [
        pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-02"] * 4),
        pd.date_range("2025-01-01", periods=12, freq="D")[::-1],
    ],
)
def test_dates_must_be_unique_and_increasing(dates):
    with pytest.raises(ValueError, match=r"unique|increasing"):
        SARIMAX(np.arange(12.0), dates=dates)


def test_dataframe_exog_must_cover_every_historical_date():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    y = pd.Series(np.arange(12.0), index=dates)
    x = pd.DataFrame({"x": np.arange(11.0)}, index=dates.delete(5))

    with pytest.raises(ValueError, match="2025-06-01"):
        SARIMAX(y, exog=x)


def test_dataframe_rejects_extra_historical_rows():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    y = pd.Series(np.arange(12.0), index=dates)
    extra_dates = dates.insert(1, pd.Timestamp("2025-01-15"))
    x = pd.DataFrame({"x": np.arange(13.0)}, index=extra_dates)

    with pytest.raises(ValueError, match="extra historical"):
        SARIMAX(y, exog=x)


def test_dataframe_columns_are_authoritative():
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    y = pd.Series(np.arange(12.0), index=dates)
    x = pd.DataFrame({"x": np.arange(12.0)}, index=dates)

    with pytest.raises(ValueError, match=r"exog_names.*DataFrame"):
        SARIMAX(y, exog=x, exog_names=["renamed"])


def test_exog_names_must_be_unique_nonempty_and_match_columns():
    y = np.arange(12.0)

    with pytest.raises(ValueError, match="one name per"):
        SARIMAX(y, exog=np.ones((12, 2)), exog_names=["x"])
    with pytest.raises(ValueError, match="unique"):
        SARIMAX(y, exog=np.ones((12, 2)), exog_names=["x", "x"])
    with pytest.raises(ValueError, match="non-empty"):
        SARIMAX(y, exog=np.ones((12, 1)), exog_names=[""])


def test_inputs_are_copied_and_plain_sarima_still_works():
    y = np.arange(12.0)
    x = np.arange(12.0).reshape(-1, 1)

    model = SARIMAX(y, exog=x, exog_names=["x"])
    plain = SARIMAX(y)
    expected_x = x.copy()
    y[:] = -1
    x[:] = -1

    assert np.all(model.data >= 0)
    np.testing.assert_array_equal(model.exog, expected_x)
    assert plain.exog is None
    assert plain.dates is None
    assert plain.future_exog is None


def test_fitted_result_owns_model_arrays_and_default_future_exog():
    model = _dated_model_with_future_exog()
    fitted = model.fit()
    original_forecast = fitted.predict(
        start=fitted.nobs,
        end=fitted.nobs + 2,
    ).mean.copy()

    assert not np.shares_memory(fitted.data, model.data)
    assert not np.shares_memory(fitted._ordinary_exog, model.exog)
    assert not np.shares_memory(fitted._design_matrix, model.design_matrix)
    assert fitted._dates is not model.dates
    assert fitted._default_future_exog is not model.future_exog

    model.data[:] = -100.0
    model.exog[:] = -100.0
    model.design_matrix[:] = -100.0
    model.future_exog.iloc[:, :] = -100.0

    np.testing.assert_allclose(
        fitted.predict(
            start=fitted.nobs,
            end=fitted.nobs + 2,
        ).mean,
        original_forecast,
    )


def _arimax_fixture(seed=42, n=300):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    x = rng.normal(size=n)
    error = np.empty(n)
    error[0] = rng.normal(scale=0.4)
    for index in range(1, n):
        error[index] = 0.45 * error[index - 1] + rng.normal(scale=0.4)
    y = 1.75 * x + error
    return (
        pd.Series(y, index=dates),
        pd.DataFrame({"x": x}, index=dates),
    )


def test_arimax_recovers_exogenous_coefficient():
    y, x = _arimax_fixture()

    result = SARIMAX(
        y,
        exog=x,
        order=(1, 0, 0),
        trend="n",
    ).fit()

    assert result.params["x"] == pytest.approx(1.75, abs=0.15)
    assert result.exog_names == ("x",)
    assert result.event_names == ()


def test_pulse_event_coefficient_is_estimated_by_name():
    from Ts.TsModels._intervention import EventSpec

    rng = np.random.default_rng(10)
    dates = pd.date_range("2000-01-01", periods=180, freq="MS")
    event_dates = dates[[20, 55, 90, 125, 160]]
    indicator = dates.isin(event_dates).astype(float)
    y = pd.Series(
        2.25 * indicator + rng.normal(scale=0.08, size=180),
        index=dates,
    )

    result = SARIMAX(
        y,
        events=[
            EventSpec(
                "policy",
                event_dates,
                "pulse",
                date_rule="exact",
            )
        ],
        order=(0, 0, 0),
        trend="n",
    ).fit()

    assert result.params["event__policy"] == pytest.approx(2.25, abs=0.15)
    assert result.event_names == ("policy",)
    assert result.design_columns == ("event__policy",)


def test_cumulative_step_coefficient_is_estimated():
    from Ts.TsModels._intervention import EventSpec

    rng = np.random.default_rng(11)
    dates = pd.date_range("2000-01-01", periods=180, freq="MS")
    step = np.zeros(180)
    step[40:] += 1
    step[110:] += 1
    y = pd.Series(
        1.4 * step + rng.normal(scale=0.08, size=180),
        index=dates,
    )

    result = SARIMAX(
        y,
        events=[
            EventSpec(
                "policy",
                dates[[40, 110]],
                "step",
                date_rule="exact",
            )
        ],
        order=(0, 0, 0),
        trend="n",
    ).fit()

    assert result.params["event__policy"] == pytest.approx(1.4, abs=0.1)


def test_event_window_omits_reference_column():
    from Ts.TsModels._intervention import EventSpec

    dates = pd.date_range("2000-01-01", periods=80, freq="MS")
    y = pd.Series(np.random.default_rng(5).normal(size=80), index=dates)
    result = SARIMAX(
        y,
        events=[
            EventSpec(
                "announcement",
                dates[[20, 45, 65]],
                "pulse",
                window=(-2, 2),
                reference=-1,
                date_rule="exact",
            )
        ],
        order=(0, 0, 0),
        trend="n",
    ).fit()

    assert "event__announcement__m1" not in result.design_columns
    assert result.design_columns == (
        "event__announcement__m2",
        "event__announcement__p0",
        "event__announcement__p1",
        "event__announcement__p2",
    )


def test_events_require_dates_and_event_spec_instances():
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="dated data"):
        SARIMAX(
            np.arange(12.0),
            events=[EventSpec("policy", ["2025-01-01"], "pulse")],
        )
    dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    with pytest.raises(TypeError, match="EventSpec"):
        SARIMAX(
            pd.Series(np.arange(12.0), index=dates),
            events=[{"name": "policy"}],
        )


def test_design_validation_rejects_unidentified_or_collinear_columns():
    from Ts.TsModels._intervention import EventSpec

    dates = pd.date_range("2025-01-01", periods=30, freq="MS")
    y = pd.Series(np.arange(30.0), index=dates)

    with pytest.raises(ValueError, match=r"all-zero.*event__future"):
        SARIMAX(
            y,
            events=[EventSpec("future", ["2030-01-01"], "pulse")],
        )

    x = np.arange(30.0)
    with pytest.raises(ValueError, match="rank deficient"):
        SARIMAX(
            y,
            exog=np.column_stack([x, 2 * x]),
            exog_names=["x", "twice_x"],
        )

    with pytest.raises(ValueError, match=r"constant.*trend"):
        SARIMAX(
            y,
            exog=np.ones((30, 1)),
            exog_names=["constant"],
            trend="c",
        )


def test_ordinary_exog_cannot_use_event_namespace():
    with pytest.raises(ValueError, match="event__"):
        SARIMAX(
            np.arange(12.0),
            exog=np.ones((12, 1)),
            exog_names=["event__policy"],
            trend="n",
        )


def _dated_model_with_future_exog(include_future=True):
    rng = np.random.default_rng(17)
    history_dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    future_dates = pd.date_range("2023-01-01", periods=3, freq="MS")
    x = rng.normal(size=39)
    y = pd.Series(
        1.5 * x[:36] + rng.normal(scale=0.05, size=36),
        index=history_dates,
    )
    exog_dates = history_dates.append(future_dates) if include_future else history_dates
    exog = pd.DataFrame(
        {"x": x if include_future else x[:36]},
        index=exog_dates,
    )
    return SARIMAX(
        y,
        exog=exog,
        order=(0, 0, 0),
        trend="n",
    )


def test_default_future_exog_returns_one_predict_result():
    from Ts.TsModels._base import PredictResult

    model = _dated_model_with_future_exog()
    result = model.fit().predict(
        start=len(model.data),
        end=len(model.data) + 2,
    )

    assert isinstance(result, PredictResult)
    assert result.mean.shape == (3,)
    assert np.all(result.is_oos)


def test_mapping_returns_default_and_named_scenarios():
    from Ts.TsModels._sarimax import ScenarioForecastResult

    model = _dated_model_with_future_exog()
    future_dates = model.future_exog.index
    custom = {
        "high": pd.DataFrame({"x": [2.0, 2.0, 2.0]}, index=future_dates),
        "low": pd.DataFrame({"x": [-2.0, -2.0, -2.0]}, index=future_dates),
    }

    result = model.fit().predict(
        start=len(model.data),
        end=len(model.data) + 2,
        future_exog=custom,
    )

    assert isinstance(result, ScenarioForecastResult)
    assert tuple(result.scenarios) == ("default", "high", "low")
    assert result.default_name == "default"
    assert result.dates.equals(future_dates)
    assert np.all(result["high"].mean > result["low"].mean)


def test_custom_only_mapping_has_no_default_scenario():
    from Ts.TsModels._sarimax import ScenarioForecastResult

    model = _dated_model_with_future_exog(include_future=False)
    dates = pd.date_range("2023-01-01", periods=3, freq="MS")
    result = model.fit().predict(
        start=len(model.data),
        end=len(model.data) + 2,
        future_exog={
            "high": pd.DataFrame({"x": [2.0] * 3}, index=dates),
            "low": pd.DataFrame({"x": [-2.0] * 3}, index=dates),
        },
    )

    assert isinstance(result, ScenarioForecastResult)
    assert result.default_name is None
    assert tuple(result.scenarios) == ("high", "low")


def test_dataframe_future_exog_is_one_custom_scenario():
    from Ts.TsModels._base import PredictResult

    model = _dated_model_with_future_exog(include_future=False)
    dates = pd.date_range("2023-01-01", periods=3, freq="MS")
    result = model.fit().predict(
        start=len(model.data),
        end=len(model.data) + 2,
        future_exog=pd.DataFrame({"x": [1.0, 1.0, 1.0]}, index=dates),
    )

    assert isinstance(result, PredictResult)
    assert result.mean.shape == (3,)


def test_series_future_exog_is_one_custom_scenario():
    from Ts.TsModels._base import PredictResult

    model = _dated_model_with_future_exog(include_future=False)
    dates = pd.date_range("2023-01-01", periods=3, freq="MS")
    result = model.fit().predict(
        start=len(model.data),
        end=len(model.data) + 2,
        future_exog=pd.Series([1.0, 1.0, 1.0], index=dates, name="x"),
    )

    assert isinstance(result, PredictResult)
    assert result.mean.shape == (3,)


def test_series_future_exog_rejects_the_wrong_name():
    model = _dated_model_with_future_exog(include_future=False)
    dates = pd.date_range("2023-01-01", periods=3, freq="MS")

    with pytest.raises(ValueError, match=r"name.*x"):
        model.fit().predict(
            start=len(model.data),
            end=len(model.data) + 2,
            future_exog=pd.Series([1.0, 1.0, 1.0], index=dates, name="wrong"),
        )


def test_future_scenarios_reject_reserved_or_invalid_names():
    fitted = _dated_model_with_future_exog().fit()
    dates = fitted._default_future_exog.index
    frame = pd.DataFrame({"x": [1.0] * 3}, index=dates)

    with pytest.raises(ValueError, match="reserved"):
        fitted.predict(
            start=fitted.nobs,
            end=fitted.nobs + 2,
            future_exog={"default": frame},
        )
    with pytest.raises(ValueError, match="must not be empty"):
        fitted.predict(
            start=fitted.nobs,
            end=fitted.nobs + 2,
            future_exog={},
        )
    with pytest.raises(ValueError, match="scenario name"):
        fitted.predict(
            start=fitted.nobs,
            end=fitted.nobs + 2,
            future_exog={"": frame},
        )


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame(
                {"wrong": [1.0] * 3},
                index=pd.date_range("2023-01-01", periods=3, freq="MS"),
            ),
            "columns",
        ),
        (
            pd.DataFrame(
                {"x": [1.0] * 2},
                index=pd.date_range("2023-01-01", periods=2, freq="MS"),
            ),
            "dates|rows",
        ),
        (
            pd.DataFrame(
                {"x": [1.0, np.nan, 1.0]},
                index=pd.date_range("2023-01-01", periods=3, freq="MS"),
            ),
            "non-finite",
        ),
        (
            pd.DataFrame(
                {"x": [1.0] * 3, "event__policy": [0.0] * 3},
                index=pd.date_range("2023-01-01", periods=3, freq="MS"),
            ),
            "event|columns",
        ),
    ],
)
def test_future_scenarios_require_exact_columns_dates_and_values(
    frame,
    message,
):
    fitted = _dated_model_with_future_exog(include_future=False).fit()

    with pytest.raises(ValueError, match=message):
        fitted.predict(
            start=fitted.nobs,
            end=fitted.nobs + 2,
            future_exog={"scenario": frame},
        )


def test_array_scenario_requires_future_dates_and_is_copied():
    fitted = _dated_model_with_future_exog(include_future=False).fit()
    values = np.ones((3, 1))

    with pytest.raises(ValueError, match="future_dates"):
        fitted.predict(
            start=fitted.nobs,
            end=fitted.nobs + 2,
            future_exog=values,
        )

    dates = pd.date_range("2023-01-01", periods=3, freq="MS")
    prediction = fitted.predict(
        start=fitted.nobs,
        end=fitted.nobs + 2,
        future_exog=values,
        future_dates=dates,
    )
    original = prediction.mean.copy()
    values[:] = 100
    np.testing.assert_array_equal(prediction.mean, original)


def test_one_dimensional_array_future_exog_supports_one_input():
    fitted = _dated_model_with_future_exog(include_future=False).fit()
    values = np.ones(3)
    dates = pd.date_range("2023-01-01", periods=3, freq="MS")

    prediction = fitted.predict(
        start=fitted.nobs,
        end=fitted.nobs + 2,
        future_exog=values,
        future_dates=dates,
    )

    assert prediction.mean.shape == (3,)


def test_exogenous_forecast_without_coverage_lists_missing_date():
    fitted = _dated_model_with_future_exog(include_future=False).fit()

    with pytest.raises(ValueError, match=r"future exog.*2023-01-01"):
        fitted.predict(start=fitted.nobs, end=fitted.nobs + 1)


def test_scenario_result_validates_access_and_plots():
    import matplotlib.pyplot as plt

    from Ts.TsModels._base import PredictResult
    from Ts.TsModels._sarimax import ScenarioForecastResult

    prediction = PredictResult(
        mean=np.array([1.0, 2.0]),
        lower=None,
        upper=None,
        is_oos=np.array([True, True]),
    )
    dates = pd.date_range("2025-01-01", periods=2, freq="MS")
    result = ScenarioForecastResult(
        scenarios={"a": prediction, "b": prediction},
        default_name=None,
        dates=dates,
    )

    assert result["a"] is prediction
    with pytest.raises(KeyError):
        result["missing"]
    fig, ax = result.plot()
    assert len(ax.lines) == 2
    plt.close(fig)

    with pytest.raises(ValueError, match="length"):
        ScenarioForecastResult(
            scenarios={"a": prediction},
            default_name=None,
            dates=dates[:1],
        )


def test_predict_accepts_date_bounds_across_sample_boundary():
    model = _dated_model_with_future_exog()
    prediction = model.fit().predict(
        start="2022-10-01",
        end="2023-03-01",
    )

    assert prediction.mean.shape == (6,)
    assert prediction.is_oos.tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_multi_scenario_mixed_prediction_keeps_all_dates():
    from Ts.TsModels._sarimax import ScenarioForecastResult

    model = _dated_model_with_future_exog()
    future_dates = model.future_exog.index
    prediction = model.fit().predict(
        start="2022-11-01",
        end="2023-03-01",
        future_exog={
            "high": pd.DataFrame({"x": [2.0] * 3}, index=future_dates),
            "low": pd.DataFrame({"x": [-2.0] * 3}, index=future_dates),
        },
    )

    assert isinstance(prediction, ScenarioForecastResult)
    assert prediction.dates.equals(pd.date_range("2022-11-01", periods=5, freq="MS"))


def test_future_event_is_generated_without_future_exog_column():
    from Ts.TsModels._intervention import EventSpec

    rng = np.random.default_rng(23)
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    future_dates = pd.date_range("2023-01-01", periods=3, freq="MS")
    event_dates = [dates[10], dates[22], future_dates[1]]
    historical_indicator = dates.isin(event_dates).astype(float)
    y = pd.Series(
        3.0 * historical_indicator + rng.normal(scale=0.03, size=36),
        index=dates,
    )
    fitted = SARIMAX(
        y,
        events=[
            EventSpec(
                "policy",
                event_dates,
                "pulse",
                date_rule="exact",
            )
        ],
        order=(0, 0, 0),
        trend="n",
    ).fit()

    prediction = fitted.predict(
        start=fitted.nobs,
        end=fitted.nobs + 2,
    )

    assert prediction.mean[1] > 2.5
    assert abs(prediction.mean[0]) < 0.2
    assert abs(prediction.mean[2]) < 0.2


def test_skipped_future_periods_use_full_default_exog_path():
    fitted = _dated_model_with_future_exog().fit()
    skipped = fitted.predict(start=fitted.nobs + 1, end=fitted.nobs + 2)
    full = fitted.predict(start=fitted.nobs, end=fitted.nobs + 2)

    np.testing.assert_allclose(skipped.mean, full.mean[1:])


def test_irregular_dates_require_explicit_future_dates():
    dates = pd.DatetimeIndex(
        pd.date_range("2020-01-01", periods=20, freq="D").delete(8)
    )
    y = pd.Series(np.arange(19.0), index=dates)
    fitted = SARIMAX(y, order=(0, 0, 0), trend="c").fit()

    with pytest.raises(ValueError, match="future_dates"):
        fitted.predict(start=fitted.nobs, end=fitted.nobs)

    future_dates = pd.DatetimeIndex(["2020-01-21", "2020-01-23"])
    prediction = fitted.predict(
        start=fitted.nobs,
        end=fitted.nobs + 1,
        future_dates=future_dates,
    )
    assert prediction.mean.shape == (2,)


def test_date_bound_must_exist_in_prediction_calendar():
    fitted = _dated_model_with_future_exog().fit()

    with pytest.raises(ValueError, match="prediction date"):
        fitted.predict(start="2022-12-15", end="2023-02-01")

"""Tests for conditional feedback checking of distributed-lag inputs."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from Ts.TsTests import FeedbackTest


def _feedback_data(n=240, seed=1729):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.45 * y[t - 1] + rng.normal(scale=0.7)
        x2[t] = 0.35 * x2[t - 1] + rng.normal(scale=0.8)
        x1[t] = (
            0.40 * x1[t - 1]
            - 0.15 * x1[t - 2]
            + 0.20 * x2[t - 1]
            + 0.75 * y[t - 1]
            + rng.normal(scale=0.45)
        )
    index = pd.date_range("2000-01-01", periods=n, freq="MS")
    return (
        pd.Series(y, index=index, name="output"),
        pd.DataFrame({"x1": x1, "x2": x2}, index=index),
    )


def _reference_x1_regression(y, exog, lags):
    columns = {}
    for name in exog.columns:
        for lag in range(1, lags + 1):
            columns[f"{name}.L{lag}"] = exog[name].shift(lag)
    for lag in range(1, lags + 1):
        columns[f"output.L{lag}"] = y.shift(lag)
    design = sm.add_constant(pd.DataFrame(columns), has_constant="add")
    frame = pd.concat([exog["x1"].rename("response"), design], axis=1).dropna()
    fitted = sm.OLS(frame["response"], frame.drop(columns="response")).fit()
    restriction = np.zeros((lags, len(fitted.params)))
    for row, lag in enumerate(range(1, lags + 1)):
        restriction[row, fitted.params.index.get_loc(f"output.L{lag}")] = 1.0
    return fitted, fitted.f_test(restriction)


def test_feedback_matches_independent_ols_and_joint_f_test():
    y, exog = _feedback_data()
    result = FeedbackTest(y, exog, lags=2, tested_inputs=["x1"]).fit()
    expected_regression, expected_f = _reference_x1_regression(y, exog, 2)

    equation = result.get("x1")
    assert equation.regression.params.index.tolist() == [
        "const",
        "x1.L1",
        "x1.L2",
        "x2.L1",
        "x2.L2",
        "output.L1",
        "output.L2",
    ]
    np.testing.assert_allclose(equation.regression.params, expected_regression.params)
    assert equation.f_statistic == pytest.approx(float(expected_f.fvalue))
    assert equation.pvalue == pytest.approx(float(expected_f.pvalue))
    assert equation.df_num == pytest.approx(float(expected_f.df_num))
    assert equation.df_denom == pytest.approx(float(expected_f.df_denom))
    assert equation.reject


def test_multiple_inputs_produce_one_conditional_equation_each():
    y, exog = _feedback_data()
    result = FeedbackTest(y, exog, lags=2).fit()

    assert result.input_names == ("x1", "x2")
    assert tuple(result.regressions) == ("x1", "x2")
    assert result.tests["input"].tolist() == ["x1", "x2"]
    assert result.tests.columns.tolist() == [
        "input",
        "f_statistic",
        "df_num",
        "df_denom",
        "p_value",
        "reject",
        "nobs",
    ]
    assert result.get("x1").pvalue < 0.001
    assert result.get("x2").pvalue > 0.05
    assert result.residuals.shape == (result.nobs, 2)


def test_tested_inputs_subset_keeps_other_inputs_as_lagged_controls():
    y, exog = _feedback_data()
    result = FeedbackTest(y, exog, lags=1, tested_inputs="x1").fit()

    assert result.input_names == ("x1",)
    assert "x2.L1" in result.get("x1").regression.params.index


def test_summary_reports_regression_and_joint_feedback_test():
    y, exog = _feedback_data()
    result = FeedbackTest(y, exog, lags=1, tested_inputs="x1").fit()

    report = result.summary()
    assert "OLS Regression Results" in report
    assert "R-squared" in report
    assert "F-statistic" in report
    assert "Joint Feedback F Test" in report
    assert "H0: output.L1 = 0" in report
    assert "x1" in report
    assert str(result) == report
    assert FeedbackTest(y, exog, lags=1, tested_inputs="x1").summary() == report


def test_array_inputs_require_and_preserve_names():
    y, exog = _feedback_data()
    with pytest.raises(ValueError, match="exog_names"):
        FeedbackTest(y.to_numpy(), exog.to_numpy(), lags=1)

    result = FeedbackTest(
        y.to_numpy(),
        exog.to_numpy(),
        lags=1,
        exog_names=["first", "second"],
        tested_inputs=["second"],
    ).fit()
    assert result.input_names == ("second",)
    assert "first.L1" in result.get("second").regression.params.index


def test_unnamed_series_accepts_one_explicit_exog_name():
    y, exog = _feedback_data(n=80)
    result = FeedbackTest(
        y,
        exog["x1"].rename(None),
        lags=1,
        exog_names=["input"],
    ).fit()

    assert result.input_names == ("input",)


def test_drop_missing_preserves_original_time_spacing():
    y, exog = _feedback_data(n=80)
    missing_date = y.index[20]
    y.loc[missing_date] = np.nan

    result = FeedbackTest(
        y,
        exog,
        lags=2,
        tested_inputs="x1",
        missing="drop",
    ).fit()
    used = result.get("x1").observation_index

    assert missing_date + pd.offsets.MonthBegin(1) not in used
    assert missing_date + pd.offsets.MonthBegin(2) not in used
    assert missing_date + pd.offsets.MonthBegin(3) in used


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"lags": 0}, ValueError, "lags"),
        ({"lags": True}, TypeError, "lags"),
        ({"lags": 1, "trend": "quadratic"}, ValueError, "trend"),
        ({"lags": 1, "missing": "ignore"}, ValueError, "missing"),
        ({"lags": 1, "alpha": 1.0}, ValueError, "alpha"),
        ({"lags": 1, "tested_inputs": ["unknown"]}, ValueError, "unknown"),
    ],
)
def test_invalid_options_are_rejected(kwargs, error, match):
    y, exog = _feedback_data(n=80)
    with pytest.raises(error, match=match):
        FeedbackTest(y, exog, **kwargs)


def test_strict_missing_and_index_mismatch_are_rejected():
    y, exog = _feedback_data(n=80)
    y.iloc[10] = np.nan
    with pytest.raises(ValueError, match="missing"):
        FeedbackTest(y, exog, lags=1)

    clean_y, clean_exog = _feedback_data(n=80)
    shifted = clean_exog.copy()
    shifted.index = shifted.index + pd.offsets.MonthBegin(1)
    with pytest.raises(ValueError, match="index"):
        FeedbackTest(clean_y, shifted, lags=1)


def test_insufficient_sample_and_rank_deficiency_are_explicit():
    y = np.arange(8.0)
    exog = pd.DataFrame({"x1": np.arange(8.0), "x2": np.arange(8.0)})
    with pytest.raises(ValueError, match="degrees of freedom"):
        FeedbackTest(y, exog, lags=2).fit()

    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    collinear = pd.DataFrame({"x1": x, "x2": 2.0 * x})
    with pytest.raises(ValueError, match="rank deficient"):
        FeedbackTest(rng.normal(size=80), collinear, lags=1).fit()

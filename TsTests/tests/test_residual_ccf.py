"""Tests for Box-Jenkins residual cross-correlation diagnostics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm
import matplotlib.pyplot as plt

from Ts.TsTests import ResidualCCFTest, ResidualCCFTestResult


def _manual_ccf(output, input_, lags):
    output = np.asarray(output, dtype=float)
    input_ = np.asarray(input_, dtype=float)
    nobs = min(len(output), len(input_))
    output = output[-nobs:]
    input_ = input_[-nobs:]
    output = output - output.mean()
    input_ = input_ - input_.mean()
    scale = output.std(ddof=0) * input_.std(ddof=0)
    return np.array(
        [
            np.dot(output[lag:], input_[: nobs - lag]) / nobs / scale
            for lag in range(lags + 1)
        ]
    )


def test_residual_ccf_matches_box_jenkins_formula_and_lag_direction():
    output = np.array([0.2, -0.4, 1.1, 0.7, -0.8, 0.3, 1.4, -0.2])
    input_ = np.array([1.0, -0.5, 0.2, 1.3, -0.7, 0.4, 0.8, -1.1])

    result = ResidualCCFTest(output, input_, lags=3, input_names=["x"]).fit()
    item = result.get("x")

    expected = _manual_ccf(output, input_, 3)
    np.testing.assert_allclose(item.correlations.to_numpy(), expected)
    assert item.correlations.index.tolist() == [0, 1, 2, 3]

    nobs = len(output)
    expected_s = nobs**2 * np.sum(
        expected**2 / (nobs - np.arange(4, dtype=float))
    )
    assert item.statistic == pytest.approx(expected_s)
    assert item.df == 4
    assert item.pvalue == pytest.approx(chi2.sf(expected_s, 4))


def test_residual_ccf_right_aligns_shorter_post_burn_innovations():
    output = np.arange(1.0, 11.0) ** 1.3
    input_ = np.array([0.3, -0.2, 0.8, 1.1, -0.5, 0.4, 0.7])

    item = ResidualCCFTest(
        output,
        {"x": input_},
        lags=2,
        transfer_params={"x": 1},
    ).fit().get("x")

    assert item.nobs == len(input_)
    np.testing.assert_allclose(
        item.correlations,
        _manual_ccf(output[-len(input_) :], input_, 2),
    )
    assert item.df == 2


def test_residual_ccf_reports_standard_errors_bands_and_significant_lags():
    rng = np.random.default_rng(720)
    input_ = rng.normal(size=240)
    output = np.roll(input_, 2) + rng.normal(scale=0.08, size=240)
    output[:2] = rng.normal(size=2)

    item = ResidualCCFTest(
        output,
        pd.Series(input_, name="shock"),
        lags=5,
        transfer_params=1,
        alpha=0.10,
    ).fit().get("shock")

    expected_se = 1.0 / np.sqrt(240)
    expected_band = norm.ppf(0.95) * expected_se
    np.testing.assert_allclose(item.standard_errors, expected_se)
    np.testing.assert_allclose(item.confidence_limits["lower"], -expected_band)
    np.testing.assert_allclose(item.confidence_limits["upper"], expected_band)
    assert 2 in item.significant_lags
    assert item.reject == (item.pvalue < 0.10)


def test_residual_ccf_supports_multiple_inputs_and_compact_test_table():
    rng = np.random.default_rng(721)
    output = rng.normal(size=100)
    inputs = {
        "price": rng.normal(size=100),
        "income": rng.normal(size=97),
    }

    result = ResidualCCFTest(
        output,
        inputs,
        lags=4,
        transfer_params={"price": 2, "income": 1},
    ).fit()

    assert isinstance(result, ResidualCCFTestResult)
    assert result.input_names == ("price", "income")
    assert result.get("price").nobs == 100
    assert result.get("income").nobs == 97
    assert result.tests.columns.tolist() == [
        "input",
        "s_statistic",
        "df",
        "p_value",
        "reject",
        "nobs",
        "transfer_params",
    ]
    assert result.tests["transfer_params"].tolist() == [2, 1]
    assert "Residual cross-correlation: price" in result.summary()
    assert ResidualCCFTest(output, inputs, lags=4).summary() == str(
        ResidualCCFTest(output, inputs, lags=4).fit()
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"lags": 0}, ValueError, "positive integer"),
        ({"lags": 2.5}, TypeError, "positive integer"),
        ({"lags": 3, "alpha": 0.0}, ValueError, "alpha"),
        ({"lags": 3, "transfer_params": 4}, ValueError, "degrees of freedom"),
        ({"lags": 8}, ValueError, "observations"),
    ],
)
def test_residual_ccf_rejects_invalid_configuration(kwargs, error, match):
    with pytest.raises(error, match=match):
        ResidualCCFTest(np.arange(8.0), np.arange(8.0) ** 2, **kwargs)


@pytest.mark.parametrize(
    ("output", "input_", "match"),
    [
        (np.ones(20), np.arange(20.0), "output residuals must vary"),
        (np.arange(20.0), np.ones(20), "input residuals must vary"),
        (np.arange(20.0), np.r_[np.arange(19.0), np.nan], "finite"),
        (np.arange(20.0), np.ones((2, 2, 5)), "one- or two-dimensional"),
    ],
)
def test_residual_ccf_rejects_invalid_residual_arrays(output, input_, match):
    with pytest.raises(ValueError, match=match):
        ResidualCCFTest(output, input_, lags=3)


def test_residual_ccf_validates_names_and_transfer_parameter_mapping():
    output = np.arange(20.0) ** 1.2
    inputs = np.column_stack([np.arange(20.0), np.arange(20.0) ** 2])

    with pytest.raises(ValueError, match="input_names"):
        ResidualCCFTest(output, inputs, lags=3)
    with pytest.raises(ValueError, match="one name per input"):
        ResidualCCFTest(output, inputs, lags=3, input_names=["x"])
    with pytest.raises(ValueError, match="unknown input"):
        ResidualCCFTest(
            output,
            {"x": inputs[:, 0]},
            lags=3,
            transfer_params={"z": 1},
        )
    with pytest.raises(KeyError, match="unknown input"):
        ResidualCCFTest(output, {"x": inputs[:, 0]}, lags=3).fit().get("z")


def test_residual_ccf_input_result_plots_exact_correlations_and_band():
    rng = np.random.default_rng(722)
    item = ResidualCCFTest(
        rng.normal(size=80),
        pd.Series(rng.normal(size=80), name="price"),
        lags=4,
    ).fit().get("price")

    fig, ax = item.plot_test()

    np.testing.assert_allclose(
        [patch.get_height() for patch in ax.patches], item.correlations
    )
    assert ax.get_ylabel() == "Residual CCF"
    assert ax.get_title() == "price"
    plt.close(fig)


def test_residual_ccf_multi_result_plots_selected_input_facets():
    rng = np.random.default_rng(723)
    result = ResidualCCFTest(
        rng.normal(size=90),
        {
            "price": rng.normal(size=90),
            "income": rng.normal(size=88),
        },
        lags=3,
    ).fit()

    fig, axes = result.plot_test(title="RDL residual CCF")

    assert axes.shape == (2,)
    assert [axis.get_title() for axis in axes] == ["price", "income"]
    assert fig._suptitle.get_text() == "RDL residual CCF"
    plt.close(fig)

    fig, ax = result.plot_test(inputs="income")
    assert ax.get_title() == "income"
    plt.close(fig)

    with pytest.raises(ValueError, match="unknown input"):
        result.plot_test(inputs="missing")
    with pytest.raises(ValueError, match="unique"):
        result.plot_test(inputs=["price", "price"])

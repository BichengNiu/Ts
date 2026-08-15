"""Tests for the Lee-Strazicich two-unknown-break minimum LM test."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from Ts.TsTests._lee_strazicich import (
    LeeStrazicichTwoBreakTest,
    LeeStrazicichTwoBreakTestResult,
    _fit_lm_regression,
    _ls_critical_values,
    _ls_deterministic_terms,
)


def _manual_lm_statistic(y, first_index, second_index, model, lags):
    nobs = len(y)
    positions = np.arange(nobs, dtype=float)
    du1 = (positions > first_index).astype(float)
    du2 = (positions > second_index).astype(float)
    if model == "A":
        z = np.column_stack([positions + 1, du1, du2])
    else:
        dt1 = np.maximum(positions - first_index, 0)
        dt2 = np.maximum(positions - second_index, 0)
        z = np.column_stack([positions + 1, du1, dt1, du2, dt2])

    dy = np.diff(y)
    dz = np.diff(z, axis=0)
    detrending_beta = np.linalg.lstsq(dz, dy, rcond=None)[0]
    initial_level = y[0] - z[0] @ detrending_beta
    detrended = y - initial_level - z @ detrending_beta
    ds = np.diff(detrended)
    regressors = [detrended[:-1][lags:], dz[lags:]]
    regressors.extend(ds[lags - lag : nobs - 1 - lag] for lag in range(1, lags + 1))
    x = np.column_stack(regressors)
    dependent = dy[lags:]
    beta = np.linalg.lstsq(x, dependent, rcond=None)[0]
    residuals = dependent - x @ beta
    sigma2 = residuals @ residuals / (len(dependent) - x.shape[1])
    standard_error = np.sqrt(sigma2 * np.linalg.inv(x.T @ x)[0, 0])
    return float(beta[0] / standard_error)


@pytest.mark.parametrize(
    ("model", "expected_names"),
    [
        ("A", ("trend", "DU1", "DU2")),
        ("C", ("trend", "DU1", "DT1", "DU2", "DT2")),
    ],
)
def test_model_designs_match_published_deterministic_terms(
    model,
    expected_names,
):
    terms, names = _ls_deterministic_terms(12, 3, 8, model)
    assert names == expected_names
    assert terms.shape == (12, len(expected_names))
    np.testing.assert_array_equal(terms[:4, 1], 0.0)
    np.testing.assert_array_equal(terms[4:, 1], 1.0)
    if model == "C":
        np.testing.assert_array_equal(terms[:4, 2], 0.0)
        np.testing.assert_array_equal(terms[4:, 2], np.arange(1, 9))


@pytest.mark.parametrize("model", ["A", "C"])
def test_fixed_candidate_regression_matches_manual_lm_formula(model):
    rng = np.random.default_rng(61)
    y = np.cumsum(rng.normal(size=55))
    terms, names = _ls_deterministic_terms(55, 14, 37, model)
    result = _fit_lm_regression(y, terms, names, lags=1)
    expected = _manual_lm_statistic(y, 14, 37, model, lags=1)
    assert result.statistic == pytest.approx(expected)


def test_joint_two_dimensional_search_matches_manual_grid():
    rng = np.random.default_rng(7)
    nobs = 42
    y = np.cumsum(rng.normal(size=nobs))
    y[15:] += 2.0
    y[29:] -= 3.0
    result = LeeStrazicichTwoBreakTest(
        y,
        model="C",
        lags=0,
        trim=0.2,
    ).fit()

    lower = int(np.round(0.2 * nobs))
    upper = int(np.round(0.8 * nobs))
    candidates = []
    for first_shift in range(lower, upper + 1):
        for second_shift in range(first_shift + 3, upper + 1):
            pair = (first_shift - 1, second_shift - 1)
            statistic = _manual_lm_statistic(y, *pair, "C", lags=0)
            candidates.append((statistic, pair))
    expected_statistic, expected_pair = min(candidates)

    assert result.statistic == pytest.approx(expected_statistic)
    assert result.break_indices == expected_pair
    assert len(result.all_candidate_statistics) == len(candidates)


@pytest.mark.parametrize(
    (
        "model",
        "seed",
        "expected_statistic",
        "expected_breaks",
        "expected_critical_cell",
    ),
    [
        ("A", 101, -4.333276822299765, (10, 33), None),
        ("C", 202, -4.280776832736949, (19, 35), (0.4, 0.8)),
    ],
)
def test_fixed_reference_cases(
    model,
    seed,
    expected_statistic,
    expected_breaks,
    expected_critical_cell,
):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(size=48))
    y[16:] += 1.75
    y[32:] -= 2.25

    result = LeeStrazicichTwoBreakTest(
        y,
        model=model,
        lags=1,
        trim=0.2,
    ).fit()

    assert result.statistic == pytest.approx(expected_statistic, abs=1e-12)
    assert result.break_indices == expected_breaks
    assert result.lags == 1
    assert result.critical_value_cell == expected_critical_cell


def test_model_a_critical_values_are_break_location_invariant():
    first, cell = _ls_critical_values("A", 0.25, 0.70)
    second, second_cell = _ls_critical_values("A", 0.40, 0.80)
    assert first == {"1%": -4.545, "5%": -3.842, "10%": -3.504}
    assert second == first
    assert cell is None
    assert second_cell is None


@pytest.mark.parametrize(
    ("fractions", "cell", "expected"),
    [
        ((0.19, 0.42), (0.2, 0.4), (-6.16, -5.59, -5.27)),
        ((0.38, 0.62), (0.4, 0.6), (-6.45, -5.67, -5.31)),
        ((0.61, 0.79), (0.6, 0.8), (-6.32, -5.73, -5.32)),
    ],
)
def test_model_c_critical_values_select_nearest_published_cell(
    fractions,
    cell,
    expected,
):
    critical, matched = _ls_critical_values("C", *fractions)
    assert matched == cell
    assert tuple(critical[level] for level in ("1%", "5%", "10%")) == expected


@pytest.mark.parametrize("lag_method", ["aic", "bic", "tstat"])
def test_automatic_lag_selection_is_candidate_specific(lag_method):
    rng = np.random.default_rng(13)
    y = np.cumsum(rng.normal(size=45))
    result = LeeStrazicichTwoBreakTest(
        y,
        model="A",
        max_lags=2,
        lag_method=lag_method,
        trim=0.2,
    ).fit()

    assert 0 <= result.lags <= 2
    assert np.all((result.all_candidate_lags >= 0) & (result.all_candidate_lags <= 2))
    if lag_method in ("aic", "bic"):
        assert result.ic_by_lag.shape == (3,)
    else:
        assert result.ic_by_lag is None


def test_result_contract_labels_and_plot():
    rng = np.random.default_rng(19)
    y = np.cumsum(rng.normal(size=50))
    labels = 1980 + np.arange(50) * 2
    result = LeeStrazicichTwoBreakTest(
        y,
        time_index=labels,
        model="A",
        lags=0,
        trim=0.2,
    ).fit()

    assert isinstance(result, LeeStrazicichTwoBreakTestResult)
    assert result.break_years == tuple(
        float(labels[index]) for index in result.break_indices
    )
    assert result.break_fractions == tuple(
        (index + 1) / 50 for index in result.break_indices
    )
    assert result.critical_values["5%"] == -3.842
    assert result.regression_time_index.shape == result.residuals.shape
    assert result.regression_time_index[0] == labels[1]  # drops lags + 1 head
    assert result.regression_time_index[-1] == labels[-1]
    assert "H0: Unit root with two structural breaks" in str(result)
    fig, ax = result.plot_test()
    assert ax.get_title() == "Lee-Strazicich Two-Unknown-Break LM Test"
    assert len(ax.lines) == 3
    plt.close(fig)


def test_time_labels_do_not_change_joint_search():
    rng = np.random.default_rng(24)
    y = np.cumsum(rng.normal(size=45))
    first = LeeStrazicichTwoBreakTest(
        y,
        model="A",
        lags=0,
        trim=0.2,
    ).fit()
    irregular = np.cumsum(np.linspace(1.0, 3.0, 45))
    second = LeeStrazicichTwoBreakTest(
        y,
        time_index=irregular,
        model="A",
        lags=0,
        trim=0.2,
    ).fit()
    assert second.statistic == pytest.approx(first.statistic)
    assert second.break_indices == first.break_indices


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"model": "B"}, ValueError, "model"),
        ({"lags": -1}, ValueError, "lags"),
        ({"max_lags": 1.5}, TypeError, "max_lags"),
        ({"lag_method": "hqic"}, ValueError, "lag_method"),
        ({"lag_crit": 0}, ValueError, "lag_crit"),
        ({"trim": 0.5}, ValueError, "trim"),
        ({"model": "C", "min_break_distance": 2}, ValueError, "at least 3"),
    ],
)
def test_configuration_validation(kwargs, error, message):
    y = np.arange(40, dtype=float) + np.sin(np.arange(40))
    with pytest.raises(error, match=message):
        LeeStrazicichTwoBreakTest(y, **kwargs)


def test_rejects_constant_data_and_empty_candidate_search():
    with pytest.raises(ValueError, match="non-constant"):
        LeeStrazicichTwoBreakTest(np.ones(40))

    y = np.arange(40, dtype=float) + np.sin(np.arange(40))
    with pytest.raises(ValueError, match="no estimable"):
        LeeStrazicichTwoBreakTest(
            y,
            model="A",
            lags=0,
            trim=0.2,
            min_break_distance=30,
        ).fit()

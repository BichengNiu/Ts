"""Cross-model regression tests for the explicit missing-data contract."""

import numpy as np
import pytest

from Ts.TsModels import (
    GARCH,
    SARIMA,
    STL,
    SVAR,
    VAR,
    VECM,
    AutoGARCH,
    AutoSARIMA,
)


def _univariate_data():
    data = np.arange(30.0)
    data[4] = np.nan
    data[17] = np.inf
    return data


def _multivariate_data():
    data = np.column_stack(
        [
            np.arange(30.0),
            np.arange(30.0) * 0.5 + 1.0,
        ]
    )
    data[4, 0] = np.nan
    data[17, 1] = np.inf
    return data


UNIVARIATE_FACTORIES = [
    pytest.param(
        lambda data, missing: SARIMA(data, missing=missing),
        id="sarima",
    ),
    pytest.param(
        lambda data, missing: AutoSARIMA(
            data,
            p=(0, 0),
            d=(0, 0),
            q=(0, 0),
            missing=missing,
        ),
        id="auto-sarima",
    ),
    pytest.param(
        lambda data, missing: GARCH(data, p=1, q=1, missing=missing),
        id="garch",
    ),
    pytest.param(
        lambda data, missing: AutoGARCH(
            data,
            p=(1, 1),
            q=(1, 1),
            missing=missing,
        ),
        id="auto-garch",
    ),
    pytest.param(
        lambda data, missing: STL(data, period=12, missing=missing),
        id="stl",
    ),
]


@pytest.mark.parametrize("factory", UNIVARIATE_FACTORIES)
def test_univariate_models_raise_by_default(factory):
    """Every univariate model rejects NaN and Inf unless drop is explicit."""
    with pytest.raises(ValueError, match="row positions: 4, 17"):
        factory(_univariate_data(), "raise")


@pytest.mark.parametrize("factory", UNIVARIATE_FACTORIES)
def test_univariate_models_drop_explicitly_and_record_positions(factory):
    """Explicit drop removes non-finite rows and records original positions."""
    model = factory(_univariate_data(), "drop")

    assert model.missing == "drop"
    assert model.dropped_positions == (4, 17)
    assert model.data.shape[0] == 28
    assert np.all(np.isfinite(model.data))


MULTIVARIATE_FACTORIES = [
    pytest.param(
        lambda data, missing: VAR(data, lags=1, missing=missing),
        id="var",
    ),
    pytest.param(
        lambda data, missing: VECM(
            data,
            lags=2,
            coint_rank=1,
            missing=missing,
        ),
        id="vecm",
    ),
    pytest.param(
        lambda data, missing: SVAR(
            data,
            lags=1,
            A=np.array([[1.0, np.nan], [0.0, 1.0]]),
            missing=missing,
        ),
        id="svar",
    ),
]


@pytest.mark.parametrize("factory", MULTIVARIATE_FACTORIES)
def test_multivariate_models_raise_by_default(factory):
    """Multivariate models default to raise instead of silent row deletion."""
    with pytest.raises(ValueError, match="row positions: 4, 17"):
        factory(_multivariate_data(), "raise")


@pytest.mark.parametrize("factory", MULTIVARIATE_FACTORIES)
def test_multivariate_models_drop_complete_rows_and_record_positions(factory):
    """Explicit drop applies complete-row deletion and records positions."""
    model = factory(_multivariate_data(), "drop")

    assert model.missing == "drop"
    assert model.dropped_positions == (4, 17)
    assert model.data.shape == (28, 2)
    assert np.all(np.isfinite(model.data))


@pytest.mark.parametrize("model_class", [SARIMA, GARCH, VAR, VECM, STL])
def test_unknown_missing_policy_is_rejected(model_class):
    """The shared contract accepts only raise and drop."""
    data = _multivariate_data() if model_class in {VAR, VECM} else _univariate_data()
    kwargs = {}
    if model_class is GARCH:
        kwargs = {"p": 1, "q": 1}
    elif model_class is VAR:
        kwargs = {"lags": 1}
    elif model_class is VECM:
        kwargs = {"lags": 2, "coint_rank": 1}
    elif model_class is STL:
        kwargs = {"period": 12}

    with pytest.raises(ValueError, match="missing must be"):
        model_class(data, missing="omit", **kwargs)


def test_garch_drop_uses_joint_data_and_exog_mask():
    """GARCH drops rows missing in either data or exogenous regressors."""
    data = np.arange(20.0)
    exog = np.arange(40.0).reshape(20, 2)
    data[3] = np.nan
    exog[7, 1] = np.inf

    model = GARCH(data, p=1, q=1, exog=exog, missing="drop")

    assert model.dropped_positions == (3, 7)
    np.testing.assert_array_equal(model.data, np.delete(data, [3, 7]))
    np.testing.assert_array_equal(model.exog, np.delete(exog, [3, 7], axis=0))


def test_var_select_order_uses_same_missing_policy():
    """VAR lag selection shares the model-level missing-data contract."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(60, 2))
    data[11, 0] = np.nan

    with pytest.raises(ValueError, match="row positions: 11"):
        VAR.select_order(data, max_lags=2)

    result = VAR.select_order(data, max_lags=2, missing="drop")
    assert result.dropped_positions == (11,)
    assert result.nobs == 59

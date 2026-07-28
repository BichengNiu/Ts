"""Shared public input-contract tests for TsTests."""

import numpy as np
import pytest

from Ts.TsTests._adf import ADFTest
from Ts.TsTests._engle_lm import EngleLMTest
from Ts.TsTests._johansen import JohansenTest
from Ts.TsTests._kpss import KPSSTest
from Ts.TsTests._ljungbox import LjungBoxTest
from Ts.TsTests._normality import NormalityTest
from Ts.TsTests._perron import PerronTest
from Ts.TsTests._phillips_perron import PhillipsPerronTest
from Ts.TsTests._toda_yamamoto import TodaYamamotoTest
from Ts.TsTests._zivot import ZivotAndrewsTest


_UNIVARIATE_FACTORIES = [
    lambda data: ADFTest(data, lags=1),
    lambda data: KPSSTest(data, lags=1),
    lambda data: PhillipsPerronTest(data),
    lambda data: NormalityTest(data),
    lambda data: LjungBoxTest(data, lags=2),
    lambda data: EngleLMTest(data, lags=2),
]


@pytest.mark.parametrize("factory", _UNIVARIATE_FACTORIES)
def test_univariate_tests_drop_nan_consistently(factory):
    data = np.arange(30, dtype=float)
    data[5] = np.nan
    test = factory(data)
    assert test.data.shape == (29,)


@pytest.mark.parametrize("factory", _UNIVARIATE_FACTORIES)
def test_univariate_tests_reject_infinity(factory):
    data = np.arange(30, dtype=float)
    data[5] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        factory(data)


@pytest.mark.parametrize("factory", _UNIVARIATE_FACTORIES)
def test_univariate_tests_reject_multidimensional_input(factory):
    with pytest.raises(ValueError, match="1-D"):
        factory(np.arange(30, dtype=float).reshape(15, 2))


def test_engle_lm_rejects_infinite_aligned_residuals():
    data = np.arange(30, dtype=float)
    residuals = np.ones(30)
    residuals[4] = np.inf
    with pytest.raises(ValueError, match="finite"):
        EngleLMTest(data, lags=2, residuals=residuals)


@pytest.mark.parametrize(
    "factory",
    [
        lambda data: JohansenTest(data, lags=1),
        lambda data: TodaYamamotoTest(data, p=1, d_max=0),
    ],
)
def test_multivariate_tests_drop_nan_rows_and_reject_infinity(factory):
    rng = np.random.default_rng(7)
    data = rng.normal(size=(40, 2))
    data[3, 0] = np.nan
    assert factory(data).data.shape == (39, 2)

    data[3, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        factory(data)


@pytest.mark.parametrize(
    "factory",
    [
        lambda data, time: PerronTest(data, break_year=15, time_index=time, lags=1),
        lambda data, time: ZivotAndrewsTest(data, time_index=time, lags=1),
    ],
)
def test_break_tests_reject_misaligned_or_nonfinite_time_axes(factory):
    data = np.arange(30, dtype=float)
    with pytest.raises(ValueError, match="length"):
        factory(data, np.arange(29, dtype=float))

    time = np.arange(30, dtype=float)
    time[4] = np.inf
    with pytest.raises(ValueError, match="time_index"):
        factory(data, time)


def test_break_tests_reject_nonfinite_data_without_compressing_time():
    data = np.arange(30, dtype=float)
    data[4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PerronTest(data, break_year=15)
    with pytest.raises(ValueError, match="finite"):
        ZivotAndrewsTest(data)


def test_zivot_rejects_constant_series_explicitly():
    with pytest.raises(ValueError, match="non-constant"):
        ZivotAndrewsTest(np.ones(40), lags=1).fit()


@pytest.mark.parametrize("lags", [-1, True, 1.5])
@pytest.mark.parametrize("test_class", [PerronTest, ZivotAndrewsTest])
def test_break_tests_reject_invalid_fixed_lags(test_class, lags):
    data = np.arange(40, dtype=float) + np.sin(np.arange(40))
    kwargs = {"break_year": 20} if test_class is PerronTest else {}
    error = TypeError if isinstance(lags, (bool, float)) else ValueError
    with pytest.raises(error, match="lags"):
        test_class(data, lags=lags, **kwargs)


@pytest.mark.parametrize("max_lags", [-1, True, 1.5])
@pytest.mark.parametrize("test_class", [PerronTest, ZivotAndrewsTest])
def test_break_tests_reject_invalid_max_lags(test_class, max_lags):
    data = np.arange(40, dtype=float) + np.sin(np.arange(40))
    kwargs = {"break_year": 20} if test_class is PerronTest else {}
    error = TypeError if isinstance(max_lags, (bool, float)) else ValueError
    with pytest.raises(error, match="max_lags"):
        test_class(data, max_lags=max_lags, **kwargs)


@pytest.mark.parametrize("trim", [-0.1, 0.0, 0.5, 0.8, np.nan])
def test_zivot_rejects_invalid_trim(trim):
    with pytest.raises(ValueError, match="trim"):
        ZivotAndrewsTest(np.arange(40, dtype=float), trim=trim)


@pytest.mark.parametrize(
    "time",
    [
        np.array([0.0, 1.0, 1.0, 3.0]),
        np.array([0.0, 2.0, 1.0, 3.0]),
    ],
)
@pytest.mark.parametrize("test_class", [PerronTest, ZivotAndrewsTest])
def test_break_tests_require_strictly_increasing_unique_time_axis(test_class, time):
    data = np.arange(4, dtype=float)
    kwargs = {"break_year": 1.0} if test_class is PerronTest else {}
    with pytest.raises(ValueError, match="strictly increasing"):
        test_class(data, time_index=time, **kwargs)


def test_perron_requires_exact_break_label():
    data = np.arange(40, dtype=float) + np.sin(np.arange(40))
    with pytest.raises(ValueError, match="match exactly one"):
        PerronTest(data, time_index=np.arange(40), break_year=20.5)


@pytest.mark.parametrize("break_year", [3, 37])
def test_perron_rejects_break_fraction_outside_table(break_year):
    data = np.arange(40, dtype=float) + np.sin(np.arange(40))
    with pytest.raises(ValueError, match="break fraction"):
        PerronTest(data, break_year=break_year)

"""Behavior tests for shared TsTests results, parsing, and plotting."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from Ts.TsPlots.unitroot_plot import (
    _render_critical_value_plot,
    _render_ic_plot,
    _render_tstat_plot,
)
from Ts.TsTests._base import BaseTest, BaseTestResult
from Ts.TsTests._utils import _parse_input


@dataclass
class _CriticalValueResult(BaseTestResult):
    """Test-only result that declares the optional critical-value contract."""

    critical_values: dict[str, float] = field(default_factory=dict)


class _LazyTest(BaseTest):
    def fit(self):
        self.result_ = BaseTestResult(
            statistic=1.25,
            pvalue=0.2,
            lags=1,
            nobs=20,
        )
        return self.result_


def test_base_test_result_formats_both_result_styles():
    result = BaseTestResult(statistic=1.25, pvalue=0.2, lags=1, nobs=20)
    text = result._format_conclusion("Example", "No effect")
    assert "P-value:        0.200000" in text

    critical_result = _CriticalValueResult(
        statistic=-4.2,
        pvalue=None,
        lags=2,
        nobs=18,
        critical_values={"5%": -3.5, "1%": -4.1},
    )
    text = critical_result._format_conclusion("Break test", "Unit root")
    assert "Critical Values:" in text
    assert "1%: -4.1000" in text

    no_lag_result = BaseTestResult(
        statistic=2.0,
        pvalue=0.1,
        lags=None,
        nobs=30,
    )
    assert "Lags:           N/A" in no_lag_result._format_conclusion(
        "Stability test",
        "Stable parameters",
    )


def test_base_test_summary_fits_lazily_once():
    test = _LazyTest()
    assert test.result_ is None
    assert "BaseTestResult" in test.summary()
    first = test.result_
    test.summary()
    assert test.result_ is first


def test_parse_input_selects_dataframe_value_and_time_columns():
    frame = pd.DataFrame(
        {
            "year": [2000, 2001, 2002],
            "value": [1.0, 2.0, 3.0],
        }
    )

    values, years = _parse_input(frame, y_col="value", time_col=0)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(years, [2000.0, 2001.0, 2002.0])

    values, default_time = _parse_input(frame[["value"]])
    np.testing.assert_array_equal(default_time, np.arange(3, dtype=float))
    np.testing.assert_array_equal(values, frame["value"])


def test_parse_input_handles_explicit_pandas_time_index():
    data = pd.Series([1.0, 2.0, 3.0])
    time_frame = pd.DataFrame({"time": [10.0, 11.0, 12.0]})
    values, times = _parse_input(data, time_index=time_frame)
    np.testing.assert_array_equal(values, data)
    np.testing.assert_array_equal(times, time_frame["time"])

    with pytest.raises(ValueError, match="exactly one column"):
        _parse_input(data, time_index=pd.DataFrame({"a": data, "b": data}))

    with pytest.raises(ValueError, match="Use y_col"):
        _parse_input(pd.DataFrame({"a": data, "b": data}))


def test_critical_value_plot_handles_empty_values_and_external_axes():
    result = SimpleNamespace(critical_values={}, statistic=-2.0)
    fig, ax = plt.subplots()
    returned_fig, returned_ax = _render_critical_value_plot(result, "Example", ax=ax)
    assert returned_fig is fig
    assert returned_ax is ax
    assert ax.texts[0].get_text() == "No critical values available"
    plt.close(fig)


def test_tstat_and_information_criterion_plots_support_both_axes_paths():
    result = SimpleNamespace(
        all_break_years=np.array([2000.0, 2001.0, 2002.0]),
        all_t_stats=np.array([-2.0, -4.0, -3.0]),
        cv_05=-3.5,
        cv_01=-4.2,
        model="intercept",
        ic_by_lag=np.array([5.0, np.nan, 2.0]),
        lags=2,
        lag_method="bic",
        break_year=2001.0,
    )

    fig, ax = plt.subplots()
    returned_fig, returned_ax = _render_tstat_plot(result, ax=ax)
    assert returned_fig is fig
    assert returned_ax is ax
    assert "Zivot-Andrews" in ax.get_title()
    plt.close(fig)

    fig, ax = _render_ic_plot(result)
    assert ax.get_ylabel() == "BIC"
    np.testing.assert_array_equal(ax.get_xticks(), [0, 2])
    plt.close(fig)

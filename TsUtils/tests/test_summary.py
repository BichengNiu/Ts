"""Tests for descriptive time-series summaries and diagnostics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from Ts import TimeSeriesSummary as RootTimeSeriesSummary
from Ts.TsUtils import TimeSeriesSummary


def test_summary_reports_descriptive_statistics_and_frequency():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index, name="value")

    analysis = TimeSeriesSummary(series)
    text = analysis.summary(plot=False)

    assert "Name               : value" in text
    assert "Observations       : 5" in text
    assert "Valid observations : 5" in text
    assert "Frequency          : D" in text
    assert "Minimum            : 1" in text
    assert "First quartile     : 2" in text
    assert "Median             : 3" in text
    assert "Third quartile     : 4" in text
    assert "Maximum            : 5" in text
    assert analysis.figure_ is None


def test_summary_reports_missing_count_ratio_and_timestamps():
    index = pd.date_range("2024-01-01", periods=4, freq="MS")
    series = pd.Series([1.0, np.nan, 3.0, np.nan], index=index)

    analysis = TimeSeriesSummary(series)
    text = analysis.summary()

    assert analysis.n_missing == 2
    assert analysis.missing_ratio == pytest.approx(0.5)
    assert analysis.missing_timestamps == (index[1], index[3])
    assert "Missing values     : 2" in text
    assert "Missing ratio      : 50.00%" in text
    assert index[1].isoformat() in text
    assert index[3].isoformat() in text
    assert all(
        "not computed because" in text_item.get_text()
        for axis in analysis.axes_.flat
        for text_item in axis.texts
    )
    plt.close(analysis.figure_)


def test_array_input_reports_missing_positions():
    analysis = TimeSeriesSummary([1.0, np.nan, 3.0])

    text = analysis.summary(plot=False)

    assert analysis.missing_timestamps == (1,)
    assert "Missing positions  : 1" in text
    assert "Frequency          : positional (step=1)" in text


def test_multivariate_dataframe_requires_and_uses_variable_name():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "sales": [1.0, 2.0, np.nan, 4.0, 5.0],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        index=index,
    )

    with pytest.raises(ValueError, match="variable must be specified"):
        TimeSeriesSummary(frame)

    analysis = TimeSeriesSummary(frame, variable="sales")
    text = analysis.summary(plot=False)

    assert analysis.series.name == "sales"
    assert analysis.n_missing == 1
    assert analysis.missing_timestamps == (index[2],)
    assert "Name               : sales" in text
    assert "Mean               : 3" in text


def test_dataframe_variable_must_identify_an_existing_unique_column():
    frame = pd.DataFrame({"sales": [1.0, 2.0], "price": [3.0, 4.0]})

    with pytest.raises(ValueError, match="'missing' is not a DataFrame column"):
        TimeSeriesSummary(frame, variable="missing")

    duplicated = pd.DataFrame([[1.0, 2.0]], columns=["sales", "sales"])
    with pytest.raises(ValueError, match="exactly one column"):
        TimeSeriesSummary(duplicated, variable="sales")


def test_frequency_is_inferred_from_an_unannotated_datetime_index():
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    analysis = TimeSeriesSummary(pd.Series([1.0, 2.0, 3.0], index=index))

    assert index.freq is None
    assert analysis.frequency == "D"


def test_summary_uses_existing_acf_and_pacf_plotters(monkeypatch):
    calls = []

    def fake_plotter(data, nlags, *, alpha, title, ax):
        calls.append((np.asarray(data, dtype=float), nlags, alpha, title, ax))
        return ax.figure, ax

    monkeypatch.setattr("Ts.TsUtils._summary.plot_acf", fake_plotter)
    monkeypatch.setattr("Ts.TsUtils._summary.plot_pacf", fake_plotter)
    series = pd.Series(np.arange(10.0))

    analysis = TimeSeriesSummary(series)
    analysis.summary()

    assert len(calls) == 2
    assert [call[3] for call in calls] == ["Level ACF", "Level PACF"]
    np.testing.assert_array_equal(calls[0][0], np.arange(10.0))
    assert analysis.axes_[1, 0].texts[0].get_text() == (
        "Not defined for a constant series."
    )
    assert analysis.axes_[1, 1].texts[0].get_text() == (
        "Not defined for a constant series."
    )
    plt.close(analysis.figure_)


def test_summary_passes_first_difference_to_existing_plotters(monkeypatch):
    calls = []

    def fake_plotter(data, nlags, *, alpha, title, ax):
        calls.append((np.asarray(data, dtype=float), title))
        return ax.figure, ax

    monkeypatch.setattr("Ts.TsUtils._summary.plot_acf", fake_plotter)
    monkeypatch.setattr("Ts.TsUtils._summary.plot_pacf", fake_plotter)
    series = pd.Series([1.0, 3.0, 2.0, 6.0, 5.0, 9.0, 7.0, 12.0])

    analysis = TimeSeriesSummary(series)
    analysis.summary()

    assert len(calls) == 4
    np.testing.assert_array_equal(
        calls[2][0],
        series.diff().dropna().to_numpy(),
    )
    plt.close(analysis.figure_)


def test_summary_creates_four_real_diagnostic_panels():
    rng = np.random.default_rng(1234)
    series = pd.Series(rng.normal(size=100).cumsum())

    analysis = TimeSeriesSummary(series, nlags=10)
    text = analysis.summary()

    assert isinstance(text, str)
    assert isinstance(analysis.figure_, plt.Figure)
    assert analysis.axes_.shape == (2, 2)
    assert [axis.get_title() for axis in analysis.axes_.flat] == [
        "Level ACF",
        "Level PACF",
        "First-difference ACF",
        "First-difference PACF",
    ]
    plt.close(analysis.figure_)


def test_rejects_unsupported_or_invalid_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        TimeSeriesSummary(np.ones((2, 2)))
    with pytest.raises(TypeError, match="numeric"):
        TimeSeriesSummary(["a", "b"])
    with pytest.raises(ValueError, match="infinite"):
        TimeSeriesSummary([1.0, np.inf])
    with pytest.raises(ValueError, match="at least one"):
        TimeSeriesSummary([])
    with pytest.raises(ValueError, match="only valid for DataFrame"):
        TimeSeriesSummary(pd.Series([1.0, 2.0]), variable="value")


def test_explicit_nlags_must_fit_first_difference_pacf():
    analysis = TimeSeriesSummary(np.arange(10.0), nlags=5)

    with pytest.raises(ValueError, match=r"nlags must be <= 3"):
        analysis.plot()
    plt.close(analysis.figure_)


def test_time_series_summary_is_exported_from_root_package():
    assert RootTimeSeriesSummary is TimeSeriesSummary

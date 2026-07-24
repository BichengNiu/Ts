"""Tests for Ts.TsModels._stl — STL decomposition and STLResult."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt


@pytest.fixture
def seasonal_data():
    """Generate a deterministic monthly series with trend and seasonality."""
    time = np.arange(120, dtype=float)
    seasonal = 2.0 * np.sin(2.0 * np.pi * time / 12.0)
    return 10.0 + 0.05 * time + seasonal


@pytest.fixture
def outlier_data(seasonal_data):
    """Add one large outlier to the deterministic seasonal series."""
    data = seasonal_data.copy()
    data[60] += 40.0
    return data


def test_stl_construction_is_public(seasonal_data):
    """STL is publicly importable and stores its construction contract.

    covers: TsModels/_stl.py [module]
    covers: TsModels/_stl.py::STL [class]
    covers: TsModels/_stl.py::STL.__init__ [function]
    covers: TsModels/__init__.py [module]
    covers: TsPlots/__init__.py [module]
    covers: TsSims/__init__.py [module]
    covers: TsTests/__init__.py [module]
    covers: __init__.py [module]
    """
    from Ts.TsModels import STL

    model = STL(seasonal_data, period=12)

    assert model.period == 12
    assert model.result_ is None


def test_stl_rejects_non_1d_data(seasonal_data):
    """STL rejects data that are not one-dimensional.

    covers: TsModels/_stl.py::STL.__init__ [function]
    """
    from Ts.TsModels import STL

    with pytest.raises(ValueError, match="one-dimensional"):
        STL(seasonal_data.reshape(-1, 1), period=12)


def test_stl_rejects_non_finite_data(seasonal_data):
    """STL rejects NaN and infinite observations.

    covers: TsModels/_stl.py::STL.__init__ [function]
    """
    from Ts.TsModels import STL

    invalid = seasonal_data.copy()
    invalid[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        STL(invalid, period=12)


@pytest.mark.parametrize("period", [1, True, 2.5])
def test_stl_rejects_invalid_period(seasonal_data, period):
    """STL requires a non-boolean integer period of at least two.

    covers: TsModels/_stl.py::STL.__init__ [function]
    """
    from Ts.TsModels import STL

    with pytest.raises(ValueError, match="period"):
        STL(seasonal_data, period=period)


def test_stl_rejects_less_than_two_cycles(seasonal_data):
    """STL requires at least two complete seasonal cycles.

    covers: TsModels/_stl.py::STL.__init__ [function]
    """
    from Ts.TsModels import STL

    with pytest.raises(ValueError, match="two complete cycles"):
        STL(seasonal_data[:23], period=12)


def test_stl_exposes_resolved_configuration(seasonal_data):
    """STL delegates and exposes all smoothing configuration.

    covers: TsModels/_stl.py::STL.__init__ [function]
    """
    from Ts.TsModels import STL

    model = STL(
        seasonal_data,
        period=12,
        seasonal=9,
        trend=19,
        low_pass=13,
        seasonal_deg=0,
        trend_deg=0,
        low_pass_deg=0,
        robust=True,
        seasonal_jump=2,
        trend_jump=3,
        low_pass_jump=2,
    )

    assert model.config == {
        "period": 12,
        "seasonal": 9,
        "seasonal_deg": 0,
        "seasonal_jump": 2,
        "trend": 19,
        "trend_deg": 0,
        "trend_jump": 3,
        "low_pass": 13,
        "low_pass_deg": 0,
        "low_pass_jump": 2,
        "robust": True,
    }


def test_stl_fit_returns_public_result(seasonal_data):
    """fit returns STLResult and stores it on the model.

    covers: TsModels/_stl.py::STL.fit [function]
    covers: TsModels/_stl.py::STLResult [class]
    """
    from Ts.TsModels import STL, STLResult

    model = STL(seasonal_data, period=12)
    result = model.fit()

    assert isinstance(result, STLResult)
    assert model.result_ is result


def test_stl_result_reconstructs_observed_series(seasonal_data):
    """STLResult exposes aligned components that reconstruct observations.

    covers: TsModels/_stl.py::STLResult.nobs [function]
    covers: TsModels/_stl.py::STLResult.fitted_values [function]
    """
    from Ts.TsModels import STL

    result = STL(seasonal_data, period=12).fit()

    assert result.nobs == seasonal_data.size
    assert result.trend.shape == seasonal_data.shape
    assert result.seasonal.shape == seasonal_data.shape
    assert result.residuals.shape == seasonal_data.shape
    assert result.weights.shape == seasonal_data.shape
    np.testing.assert_allclose(
        result.observed,
        result.fitted_values + result.residuals,
        atol=1e-12,
    )


def test_stl_result_summary_reports_robust_fit(outlier_data):
    """Result summary reports configuration and robust fit downweights outliers.

    covers: TsModels/_stl.py::STLResult.summary [function]
    """
    from Ts.TsModels import STL

    result = STL(outlier_data, period=12, robust=True).fit()
    text = result.summary()

    assert "STL Decomposition Result" in text
    assert "Observations       : 120" in text
    assert "Period             : 12" in text
    assert "Robust             : True" in text
    assert result.weights[60] < 0.5


def test_stl_result_summary_uses_real_lines(seasonal_data):
    """Result summary separates fields with real newline characters.

    covers: TsModels/_stl.py::STLResult.summary [function]
    """
    from Ts.TsModels import STL

    text = STL(seasonal_data, period=12).fit().summary()

    assert len(text.splitlines()) == 9


def test_stl_summary_fits_automatically(seasonal_data):
    """Model summary fits once and delegates to STLResult.summary.

    covers: TsModels/_stl.py::STL.summary [function]
    """
    from Ts.TsModels import STL

    model = STL(seasonal_data, period=12)
    text = model.summary()

    assert model.result_ is not None
    assert text == model.result_.summary()


def test_stl_result_plot_returns_four_panels(seasonal_data):
    """plot returns a figure with observed, trend, seasonal, and residual panels.

    covers: TsModels/_stl.py::STLResult.plot [function]
    """
    from matplotlib.figure import Figure

    from Ts.TsModels import STL

    result = STL(seasonal_data, period=12).fit()
    fig, axes = result.plot()

    assert isinstance(fig, Figure)
    assert len(axes) == 4
    assert [axis.get_title() for axis in axes] == [
        "Observed",
        "Trend",
        "Seasonal",
        "Residual",
    ]
    plt.close(fig)


def test_stl_result_plot_uses_tsplots_style(seasonal_data):
    """plot uses the shared TsPlots title, label, palette, and axis style.

    covers: TsModels/_stl.py::STLResult.plot [function]
    """
    from Ts.TsModels import STL
    from Ts.TsPlots.style import (
        AXIS_LABEL_FONTSIZE,
        DEFAULT_PALETTE,
        TITLE_FONTSIZE,
    )

    result = STL(seasonal_data, period=12).fit()
    fig, axes = result.plot(title="Styled STL")

    try:
        assert fig._suptitle.get_fontsize() == TITLE_FONTSIZE
        assert fig._suptitle.get_fontweight() == "bold"
        assert axes[-1].xaxis.label.get_fontsize() == AXIS_LABEL_FONTSIZE
        assert axes[0].lines[0].get_color() == DEFAULT_PALETTE[0]
        assert not axes[0].spines["top"].get_visible()
        assert not axes[0].spines["right"].get_visible()
    finally:
        plt.close(fig)


class TestSTLClassCoverage:
    """Class-level audit coverage for STL."""

    def test_stl_class_contract(self, seasonal_data):
        """STL exposes the expected model state.

        covers: TsModels/_stl.py::STL [class]
        """
        from Ts.TsModels import STL

        model = STL(seasonal_data, period=12)

        assert model.data.shape == seasonal_data.shape


class TestSTLResultClassCoverage:
    """Class-level audit coverage for STLResult."""

    def test_stl_result_class_contract(self, seasonal_data):
        """STLResult exposes all decomposition arrays.

        covers: TsModels/_stl.py::STLResult [class]
        """
        from Ts.TsModels import STL

        result = STL(seasonal_data, period=12).fit()

        assert result.observed.shape == seasonal_data.shape

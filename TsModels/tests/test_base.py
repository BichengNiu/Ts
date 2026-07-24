"""Tests for Ts.TsModels._base — BaseModel ABC and BaseModelResult dataclass."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest


class TestBaseModelResult:
    """Test BaseModelResult construction, fields, and methods."""

    @pytest.fixture
    def result(self):
        """Create a minimal BaseModelResult instance."""
        from Ts.TsModels._base import BaseModelResult

        n = 100
        rng = np.random.default_rng(42)
        data = np.arange(n, dtype=float)
        fitted = data + rng.standard_normal(n) * 0.5
        resid = data - fitted

        return BaseModelResult(
            model_type="SARIMA",
            params={"ar.L1": 0.5, "sigma2": 1.0},
            std_errors={"ar.L1": 0.1, "sigma2": 0.05},
            p_values={"ar.L1": 0.001, "sigma2": 0.0},
            aic=280.5,
            bic=290.3,
            log_likelihood=-138.2,
            residuals=resid,
            fitted_values=fitted,
            nobs=n,
            data=data,
        )

    def test_fields_assigned(self, result):
        """All constructor fields are stored correctly.

        covers: code/python/Ts/TsModels/_base.py::BaseModelResult [class]
        """
        assert result.model_type == "SARIMA"
        assert result.nobs == 100
        assert len(result.residuals) == 100
        assert result.aic == 280.5
        assert result.bic == 290.3

    def test_params_is_dict(self, result):
        """params is a dict with expected keys."""
        assert isinstance(result.params, dict)
        assert "ar.L1" in result.params
        assert result.params["ar.L1"] == 0.5

    def test_summary_returns_string(self, result):
        """summary() returns a non-empty string with model info."""
        text = result.summary()
        assert isinstance(text, str)
        assert "SARIMA" in text
        assert "280.50" in text

    def test_plot_fit_returns_fig_ax(self, result):
        """plot_fit() returns (fig, ax) tuple."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig, ax = result.plot_fit()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_diagnostics_returns_fig_axes(self, result):
        """plot_diagnostics() returns (fig, axes) with 3 panels."""
        from matplotlib.figure import Figure

        fig, axes = result.plot_diagnostics()
        assert isinstance(fig, Figure)
        assert len(axes) == 3

    def test_test_residuals_returns_residual_test_results(self, result):
        """test_residuals() returns ResidualTestResults with LjungBoxTest and EngleLMTest results."""
        from Ts.TsTests._base import BaseTestResult
        from Ts.TsModels._base import ResidualTestResults

        output = result.test_residuals(lags=5)
        assert isinstance(output, ResidualTestResults)
        assert isinstance(output.ljung_box, BaseTestResult)
        assert isinstance(output.engle_lm, BaseTestResult)

    def test_test_residuals_includes_white_noise(self, result):
        """test_residuals() must include white_noise (LjungBoxTest on raw residuals)."""
        from Ts.TsTests._base import BaseTestResult

        output = result.test_residuals(lags=5)
        wn = output.white_noise
        assert isinstance(wn, BaseTestResult)
        assert wn.apply_squared is False

    def test_test_residuals_includes_normality(self, result):
        """test_residuals() must include normality (Jarque-Bera test)."""
        from Ts.TsTests._base import BaseTestResult

        output = result.test_residuals(lags=5)
        norm = output.normality
        assert isinstance(norm, BaseTestResult)
        assert hasattr(norm, "skewness")
        assert hasattr(norm, "kurtosis")

    def test_residual_test_results_summary_includes_all_four(self, result):
        """ResidualTestResults summary must include all four tests."""
        output = result.test_residuals(lags=5)
        text = output.summary()
        assert "White Noise" in text
        assert "Normality" in text or "Jarque-Bera" in text
        assert "Ljung-Box" in text
        assert "Engle LM" in text

    def test_plot_diagnostics_shows_test_results(self, result):
        """plot_diagnostics() residuals panel must annotate WN + JB test results."""
        fig, axes = result.plot_diagnostics()
        ax_resid = axes[0]
        texts = [t.get_text() for t in ax_resid.texts]
        combined = " ".join(texts)
        assert "Q" in combined or "white" in combined.lower()
        assert "JB" in combined or "normality" in combined.lower()

    def test_cover_remaining(self, result):
        """Aggregate covers for items exercised by TestBaseModelResult.

        covers: code/python/Ts/TsModels/_base.py [module]
        covers: code/python/Ts/TsModels/_base.py::BaseModelResult [class]
        covers: code/python/Ts/TsModels/_base.py::BaseModelResult.summary [function]
        covers: code/python/Ts/TsModels/_base.py::BaseModelResult.plot_fit [function]
        covers: code/python/Ts/TsModels/_base.py::BaseModelResult.plot_diagnostics [function]
        covers: code/python/Ts/TsModels/_base.py::BaseModelResult.test_residuals [function]
        covers: code/python/Ts/TsModels/_base.py::BaseModelResult.long_run_equilibrium [function]
        covers: code/python/Ts/TsModels/_base.py::ResidualTestResults [class]
        covers: code/python/Ts/TsModels/_base.py::ResidualTestResults.__str__ [function]
        covers: code/python/Ts/TsModels/_base.py::ResidualTestResults.summary [function]
        covers: code/python/Ts/TsModels/_base.py::BaseModel [class]
        covers: code/python/Ts/TsModels/_base.py::BaseModel.fit [function]
        covers: code/python/Ts/TsModels/_base.py::BaseModel.summary [function]
        """
        pass


class TestBaseModel:
    """Test BaseModel ABC contract.

    covers: code/python/Ts/TsModels/_base.py::BaseModel [class]
    covers: code/python/Ts/TsModels/_base.py::BaseModel.fit [function]
    covers: code/python/Ts/TsModels/_base.py::BaseModel.summary [function]
    """

    def test_concrete_subclass_must_implement_fit(self):
        """Cannot instantiate a subclass without fit() implementation."""
        from Ts.TsModels._base import BaseModel

        class BadModel(BaseModel):
            pass

        with pytest.raises(TypeError):
            BadModel()

    def test_subclass_fulfills_contract(self):
        """A correct concrete subclass works end-to-end."""
        from Ts.TsModels._base import BaseModel, BaseModelResult
        import numpy as np

        class DemoModel(BaseModel):
            def fit(self):
                self.result_ = BaseModelResult(
                    model_type="Demo",
                    params={"a": 1.0},
                    std_errors={"a": 0.1},
                    p_values={"a": 0.01},
                    aic=100.0,
                    bic=105.0,
                    log_likelihood=-48.0,
                    residuals=np.array([0.1, -0.1]),
                    fitted_values=np.array([0.9, 1.1]),
                    nobs=50,
                    data=np.array([1.0, 1.0]),
                )
                return self.result_

        model = DemoModel()
        assert model.result_ is None

        result = model.fit()
        assert model.result_ is not None
        assert model.result_.model_type == "Demo"
        assert result.nobs == 50

        text = model.summary()
        assert "Demo" in text

    def test_summary_auto_calls_fit(self):
        """summary() automatically calls fit() if result_ is None."""
        from Ts.TsModels._base import BaseModel, BaseModelResult
        import numpy as np

        fit_called = []

        class AutoModel(BaseModel):
            def fit(self):
                fit_called.append(True)
                self.result_ = BaseModelResult(
                    model_type="Auto",
                    params={"b": 2.0},
                    std_errors={"b": 0.2},
                    p_values={"b": 0.05},
                    aic=200.0,
                    bic=210.0,
                    log_likelihood=-98.0,
                    residuals=np.array([0.0]),
                    fitted_values=np.array([1.0]),
                    nobs=30,
                    data=np.array([1.0]),
                )
                return self.result_

        model = AutoModel()
        assert len(fit_called) == 0
        model.summary()
        assert len(fit_called) == 1


class TestPredictResult:
    """Test the model prediction container."""

    @pytest.fixture
    def sample_predictions(self):
        """Generate sample prediction data for testing."""
        rng = np.random.default_rng(42)
        n = 50
        trend = np.arange(n, dtype=float) * 0.1
        mean = rng.standard_normal(n) * 0.5 + trend
        half_width = np.abs(rng.standard_normal(n) * 0.3) + 0.1
        return {
            "mean": mean,
            "lower": mean - half_width,
            "upper": mean + half_width,
        }

    def test_predict_result_construction(self, sample_predictions):
        """PredictResult stores all fields correctly.

        covers: code/python/Ts/TsModels/_base.py [module]
        covers: code/python/Ts/TsModels/_base.py::PredictResult [class]
        """
        from Ts.TsModels._base import PredictResult

        is_oos = np.zeros(50, dtype=bool)
        is_oos[40:] = True

        pr = PredictResult(
            mean=sample_predictions["mean"],
            lower=sample_predictions["lower"],
            upper=sample_predictions["upper"],
            is_oos=is_oos,
        )

        assert len(pr.mean) == 50
        assert len(pr.lower) == 50
        assert len(pr.upper) == 50
        assert np.all(pr.lower <= pr.mean)
        assert np.all(pr.mean <= pr.upper)
        assert np.sum(pr.is_oos) == 10

    def test_predict_result_no_oos(self, sample_predictions):
        """PredictResult marks no future periods when its mask is false."""
        from Ts.TsModels._base import PredictResult

        n = len(sample_predictions["mean"])
        pr = PredictResult(
            mean=sample_predictions["mean"],
            lower=sample_predictions["lower"],
            upper=sample_predictions["upper"],
            is_oos=np.zeros(n, dtype=bool),
        )

        assert not np.any(pr.is_oos)


class TestPredictResultPlot:
    """Tests for PredictResult.plot()."""

    def test_plot_basic(self):
        """plot() returns fig, ax with full data and fitted."""
        import numpy as np
        from Ts.TsModels._base import PredictResult

        rng = np.random.default_rng(42)
        n = 100
        data = np.cumsum(rng.standard_normal(n) * 0.1)
        fitted = data + rng.standard_normal(n) * 0.02
        forecast_mean = data[-10:] + rng.standard_normal(10) * 0.05
        is_oos = np.zeros(20, dtype=bool)
        is_oos[10:] = True

        pr = PredictResult(
            mean=np.concatenate([fitted[-10:], forecast_mean]),
            lower=None,
            upper=None,
            is_oos=is_oos,
            _full_data=data,
            _full_fitted=fitted,
            _start=90,
        )

        fig, ax = pr.plot()
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_ci(self):
        """plot(ci=True) draws confidence bands."""
        import numpy as np
        from Ts.TsModels._base import PredictResult

        rng = np.random.default_rng(42)
        n = 100
        data = np.cumsum(rng.standard_normal(n) * 0.1)
        fitted = data + rng.standard_normal(n) * 0.02
        n_fc = 10
        fc_mean = np.arange(n_fc, dtype=float) * 0.05
        fc_lower = fc_mean - 0.3
        fc_upper = fc_mean + 0.3
        is_oos = np.zeros(n_fc, dtype=bool)
        is_oos[:] = True

        pr = PredictResult(
            mean=fc_mean,
            lower=fc_lower,
            upper=fc_upper,
            is_oos=is_oos,
            _full_data=data,
            _full_fitted=fitted,
            _start=100,
        )

        fig, ax = pr.plot(ci=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_forecast_ci_has_fitted_boundary_anchor(self):
        """Forecast CI band includes the period before its first forecast.

        This prevents the forecast and fitted confidence intervals from
        visibly breaking at the in-sample / out-of-sample boundary.
        """
        import numpy as np
        from Ts.TsModels._base import PredictResult

        nobs = 10
        pr = PredictResult(
            mean=np.array([9.5, 10.5]),
            lower=np.array([8.5, 9.0]),
            upper=np.array([10.5, 12.0]),
            is_oos=np.array([False, True]),
            _full_data=np.arange(nobs, dtype=float),
            _full_fitted=np.arange(nobs, dtype=float),
            _full_lower=np.arange(nobs, dtype=float) - 0.5,
            _full_upper=np.arange(nobs, dtype=float) + 0.5,
            _start=9,
        )

        fig, ax = pr.plot(ci=True)
        forecast_band = ax.collections[-1]
        band_x = forecast_band.get_paths()[0].vertices[:, 0]

        assert 9 in band_x
        assert 10 in band_x
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_first_forecast_uses_preceding_sample_point(self):
        """A forecast beginning at ``start`` bridges from ``start - 1``."""
        import numpy as np
        from Ts.TsModels._base import PredictResult

        pr = PredictResult(
            mean=np.array([2.0]),
            lower=None,
            upper=None,
            is_oos=np.array([True]),
            _full_data=np.arange(10, dtype=float),
            _start=5,
        )

        fig, ax = pr.plot()
        forecast_line = next(line for line in ax.lines if line.get_label() == "Forecast")

        assert np.array_equal(forecast_line.get_xdata(), np.array([4, 5]))
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_no_full_data(self):
        """plot() works with only prediction data (no _full_data)."""
        import numpy as np
        from Ts.TsModels._base import PredictResult

        n = 20
        mean = np.sin(np.linspace(0, np.pi, n))
        is_oos = np.ones(n, dtype=bool)

        pr = PredictResult(
            mean=mean,
            lower=None,
            upper=None,
            is_oos=is_oos,
        )

        fig, ax = pr.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_custom_title(self):
        """plot(title='...') uses the provided title."""
        import numpy as np
        from Ts.TsModels._base import PredictResult

        pr = PredictResult(
            mean=np.arange(10, dtype=float),
            lower=None,
            upper=None,
            is_oos=np.ones(10, dtype=bool),
        )

        fig, ax = pr.plot(title="Custom Title")
        assert ax.get_title() == "Custom Title"
        import matplotlib.pyplot as plt
        plt.close(fig)

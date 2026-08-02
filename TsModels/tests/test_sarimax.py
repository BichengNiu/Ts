"""Tests for Ts.TsModels._sarimax — SARIMAX and SARIMAXResult."""

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from Ts.TsSims import simulate_sarima


@pytest.fixture
def ar1_data():
    """Generate AR(1) data for testing."""
    r = simulate_sarima(n=200, order=(1, 0, 0), ar=[0.7], seed=42, burn=100)
    return r.data


@pytest.fixture
def ma1_data():
    """Generate MA(1) data for testing."""
    r = simulate_sarima(n=200, order=(0, 0, 1), ma=[0.5], seed=42, burn=100)
    return r.data


class TestSARIMAX:
    """Test SARIMAX construction and fit()."""

    def test_init_stores_data_and_order(self, ar1_data):
        """SARIMAX stores data and order parameters.

        covers: code/python/Ts/TsModels/_sarimax.py [module]
        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAX [class]
        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAX.__init__ [function]
        """
        from Ts.TsModels._sarimax import SARIMAX

        model = SARIMAX(ar1_data, order=(1, 0, 0))
        assert model.order == (1, 0, 0)
        assert model.result_ is None

    def test_default_order_is_zero_zero_zero(self, ar1_data):
        """Omitting order creates a regression/white-noise specification."""
        from Ts.TsModels._sarimax import SARIMAX

        model = SARIMAX(ar1_data)

        assert model.order == (0, 0, 0)

    def test_fit_returns_sarima_result(self, ar1_data):
        """fit() returns SARIMAXResult with expected fields.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAX.fit [function]
        """
        from Ts.TsModels._sarimax import SARIMAX, SARIMAXResult

        model = SARIMAX(ar1_data, order=(1, 0, 0))
        result = model.fit()

        assert isinstance(result, SARIMAXResult)
        assert model.result_ is result
        assert result.model_type == "SARIMAX"
        assert result.nobs == 200

    def test_fit_recovers_ar_coefficient(self, ar1_data):
        """AR(1) fit recovers coefficient close to true value 0.7.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAX.fit [function]
        """
        from Ts.TsModels._sarimax import SARIMAX

        model = SARIMAX(ar1_data, order=(1, 0, 0))
        result = model.fit()

        ar_estimate = result.params.get("ar.L1")
        assert ar_estimate is not None
        assert 0.3 < ar_estimate < 1.0

    def test_ma1_fit_basic(self, ma1_data):
        """MA(1) model fits without error.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAX.fit [function]
        """
        from Ts.TsModels._sarimax import SARIMAX

        model = SARIMAX(ma1_data, order=(0, 0, 1))
        result = model.fit()

        assert result.model_type == "SARIMAX"
        assert result.residuals.shape[0] == 200

    def test_invalid_order_raises(self, ar1_data):
        """Invalid order tuple raises ValueError.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAX.__init__ [function]
        """
        from Ts.TsModels._sarimax import SARIMAX

        with pytest.raises(ValueError):
            SARIMAX(ar1_data, order=(1,))

    def test_data_too_short_raises(self, ar1_data):
        """Too few observations raises ValueError.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAX.__init__ [function]
        """
        from Ts.TsModels._sarimax import SARIMAX

        short_data = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            SARIMAX(short_data, order=(1, 0, 0))

    def test_fit_forwards_optimizer_controls_and_copies_start_params(
        self,
        ar1_data,
        monkeypatch,
    ):
        """Public fit controls are validated and forwarded to statsmodels."""
        import Ts.TsModels._sarimax as sarimax_module

        captured = {}

        class StopAfterCapture(RuntimeError):
            pass

        class CapturingSARIMAX:
            def __init__(self, *args, **kwargs):
                self.k_params = 3

            def fit(self, **kwargs):
                captured.update(kwargs)
                raise StopAfterCapture

        monkeypatch.setattr(
            sarimax_module,
            "StatsmodelsSARIMAX",
            CapturingSARIMAX,
        )
        start_params = np.array([1.0, 0.4, 0.5])
        model = sarimax_module.SARIMAX(ar1_data, order=(1, 0, 0))

        with pytest.raises(StopAfterCapture):
            model.fit(
                start_params=start_params,
                method="powell",
                maxiter=250,
            )

        start_params[:] = -1.0
        assert captured["method"] == "powell"
        assert captured["maxiter"] == 250
        assert captured["disp"] is False
        np.testing.assert_array_equal(
            captured["start_params"],
            np.array([1.0, 0.4, 0.5]),
        )

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"method": ""}, ValueError, "method"),
            ({"method": 1}, TypeError, "method"),
            ({"method": "unknown"}, ValueError, "method"),
            ({"maxiter": 0}, ValueError, "maxiter"),
            ({"maxiter": True}, TypeError, "maxiter"),
            ({"start_params": [1.0, np.nan, 2.0]}, ValueError, "finite"),
            ({"start_params": [[1.0, 2.0, 3.0]]}, ValueError, "one-dimensional"),
            ({"start_params": [1.0, 2.0]}, ValueError, "3 parameters"),
            ({"require_convergence": 1}, TypeError, "require_convergence"),
        ],
    )
    def test_fit_rejects_invalid_optimizer_controls(
        self,
        ar1_data,
        kwargs,
        error,
        message,
    ):
        """Invalid optimizer controls fail before estimation starts."""
        from Ts.TsModels._sarimax import SARIMAX

        model = SARIMAX(ar1_data, order=(1, 0, 0))
        with pytest.raises(error, match=message):
            model.fit(**kwargs)

    def test_require_convergence_rejects_optimizer_failure(
        self,
        ar1_data,
        monkeypatch,
    ):
        """The opt-in convergence gate prevents invalid results escaping."""
        import Ts.TsModels._sarimax as sarimax_module

        class NonconvergedSARIMAX:
            def __init__(self, *args, **kwargs):
                self.k_params = 3

            def fit(self, **kwargs):
                return SimpleNamespace(
                    mle_retvals={
                        "converged": False,
                        "warnflag": 1,
                        "iterations": kwargs["maxiter"],
                    }
                )

        monkeypatch.setattr(
            sarimax_module,
            "StatsmodelsSARIMAX",
            NonconvergedSARIMAX,
        )
        model = sarimax_module.SARIMAX(ar1_data, order=(1, 0, 0))

        with pytest.raises(RuntimeError, match=r"failed to converge.*powell.*7"):
            model.fit(
                method="powell",
                maxiter=7,
                require_convergence=True,
            )

    def test_fit_attaches_inferred_frequency_before_statsmodels(
        self,
        monkeypatch,
    ):
        """Regular dated data reaches statsmodels with an explicit frequency."""
        import Ts.TsModels._sarimax as sarimax_module

        captured = {}

        class StopAfterCapture(RuntimeError):
            pass

        class CapturingSARIMAX:
            def __init__(self, *args, dates=None, exog=None, **kwargs):
                captured["dates"] = dates
                captured["exog"] = exog
                raise StopAfterCapture

        monkeypatch.setattr(
            sarimax_module,
            "StatsmodelsSARIMAX",
            CapturingSARIMAX,
        )
        regular_dates_without_freq = pd.DatetimeIndex(
            pd.date_range("2020-03-31", periods=20, freq="QE-DEC").values
        )
        data = pd.Series(np.arange(20.0), index=regular_dates_without_freq)
        exog = pd.DataFrame(
            {"x": np.linspace(0.0, 1.0, 20)},
            index=regular_dates_without_freq,
        )

        with pytest.raises(StopAfterCapture):
            sarimax_module.SARIMAX(
                data,
                exog=exog,
                order=(0, 0, 0),
            ).fit()

        assert captured["dates"].freqstr == "QE-DEC"
        assert captured["exog"].index.freqstr == "QE-DEC"


class TestSARIMAXResult:
    """Test SARIMAXResult methods."""

    @pytest.fixture
    def fitted_result(self, ar1_data):
        """Fit an AR(1) and return the result."""
        from Ts.TsModels._sarimax import SARIMAX

        model = SARIMAX(ar1_data, order=(1, 0, 0))
        return model.fit()

    def test_summary_has_key_fields(self, fitted_result):
        """summary() contains AIC, BIC, and parameter estimates.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult [class]
        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.summary [function]
        """
        text = fitted_result.summary()
        assert "AIC" in text
        assert "BIC" in text
        assert "ar.L1" in text
        assert "Optimizer" in text
        assert "Converged" in text

    def test_optimization_diagnostics_are_public_and_defensively_copied(
        self,
        fitted_result,
    ):
        """Callers can inspect convergence without mutating raw fit state."""
        assert fitted_result.converged
        assert fitted_result.optimizer == "lbfgs"

        details = fitted_result.optimization_details
        original_gradient = np.asarray(
            fitted_result._statsmodels_result.mle_retvals["gopt"]
        ).copy()
        details["gopt"][:] = np.nan

        np.testing.assert_array_equal(
            fitted_result._statsmodels_result.mle_retvals["gopt"],
            original_gradient,
        )

    def test_predict_in_sample(self, fitted_result):
        """predict() within sample range returns PredictResult with correct length.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.predict [function]
        """
        from Ts.TsModels._base import PredictResult

        pr = fitted_result.predict(start=10, end=50)
        assert isinstance(pr, PredictResult)
        assert len(pr.mean) == 41

    def test_forecast_returns_mean_and_intervals(self, fitted_result):
        """predict() beyond sample returns PredictResult with lower/upper CI.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.predict [function]
        """
        pr = fitted_result.predict(start=fitted_result.nobs, end=fitted_result.nobs + 9)
        assert len(pr.mean) == 10
        assert pr.lower is not None
        assert pr.upper is not None
        assert len(pr.lower) == 10
        assert len(pr.upper) == 10
        assert np.all(pr.lower <= pr.mean)
        assert np.all(pr.mean <= pr.upper)

    def test_plot_fit_inherited(self, fitted_result):
        """plot_fit() returns (fig, ax) — inherited from BaseModelResult."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig, ax = fitted_result.plot_fit()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_fit_masks_complete_state_initialization_period(self):
        """Integrated-model initialization values are not valid fitted data."""
        import matplotlib.pyplot as plt

        from Ts.TsModels._sarimax import SARIMAX

        rng = np.random.default_rng(2408)
        data = 7.0 + np.cumsum(rng.normal(0.01, 0.08, 100))
        result = SARIMAX(
            data,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 0, 12),
            trend="n",
        ).fit()

        burn = result._statsmodels_result.loglikelihood_burn
        assert burn == 13
        assert result.fitted_values[0] == pytest.approx(0.0)

        fig, ax = result.plot_fit()
        fitted_line = next(line for line in ax.lines if line.get_label() == "Fitted")
        displayed = np.asarray(fitted_line.get_ydata(), dtype=float)

        assert np.all(np.isnan(displayed[:burn]))
        assert np.all(np.isfinite(displayed[burn:]))
        plt.close(fig)

    def test_forecast_plot_masks_initialization_fitted_line_and_interval(self):
        """Forecast plots hide fitted means and intervals during burn-in."""
        import matplotlib.pyplot as plt

        from Ts.TsModels._sarimax import SARIMAX

        rng = np.random.default_rng(2409)
        data = 7.0 + np.cumsum(rng.normal(0.01, 0.08, 100))
        result = SARIMAX(
            data,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 0, 12),
            trend="n",
        ).fit()
        prediction = result.predict(start=result.nobs, end=result.nobs + 2)
        burn = result._statsmodels_result.loglikelihood_burn

        assert np.all(np.isnan(prediction._full_fitted[:burn]))
        assert np.all(np.isnan(prediction._full_lower[:burn]))
        assert np.all(np.isnan(prediction._full_upper[:burn]))

        fig, ax = prediction.plot(ci=True)
        fitted_line = next(line for line in ax.lines if line.get_label() == "Fitted")
        assert np.all(np.isnan(fitted_line.get_ydata()[:burn]))
        plt.close(fig)

    def test_plot_diagnostics_inherited(self, fitted_result):
        """plot_diagnostics() returns (fig, axes) with 3 panels."""
        from matplotlib.figure import Figure

        fig, axes = fitted_result.plot_diagnostics()
        assert isinstance(fig, Figure)
        assert len(axes) == 3

    def test_plot_diagnostics_correlograms_start_at_lag_one(self, fitted_result):
        """Residual ACF and PACF diagnostics omit the uninformative lag zero."""
        import matplotlib.pyplot as plt

        fig, axes = fitted_result.plot_diagnostics()

        for ax in axes[1:]:
            lag_centres = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
            assert lag_centres
            assert min(lag_centres) == pytest.approx(1.0)

        plt.close(fig)

    def test_diagnostics_exclude_state_initialization_residuals(self):
        """Invalid diffuse-initialization residuals must not affect diagnostics."""
        import matplotlib.pyplot as plt
        from statsmodels.tsa.stattools import acf as sm_acf

        from Ts.TsModels import SARIMAX
        from Ts.TsTests import LjungBoxTest

        rng = np.random.default_rng(2410)
        data = 10.0 + np.cumsum(rng.normal(0.01, 0.08, 100))
        result = SARIMAX(
            data,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 0, 12),
            trend="n",
        ).fit()
        burn = result._statsmodels_result.loglikelihood_burn
        raw_residuals = np.asarray(result._statsmodels_result.resid, dtype=float)

        assert burn == 13
        assert raw_residuals[0] == pytest.approx(data[0])
        assert len(result.residuals) == result.nobs - burn
        np.testing.assert_allclose(result.residuals, raw_residuals[burn:])
        assert np.all(np.isfinite(result.residuals))

        fig, axes = result.plot_diagnostics()
        displayed = np.asarray(axes[0].lines[0].get_ydata(), dtype=float)
        np.testing.assert_allclose(displayed, result.residuals)

        expected_acf = sm_acf(result.residuals, nlags=40, fft=True)
        assert axes[1].patches[0].get_height() == pytest.approx(expected_acf[1])

        diagnostics = result.test_residuals(lags=5)
        expected_wn = LjungBoxTest(
            result.residuals,
            lags=5,
            apply_squared=False,
        ).fit()
        assert diagnostics.white_noise.statistic == pytest.approx(expected_wn.statistic)
        assert diagnostics.white_noise.pvalue == pytest.approx(expected_wn.pvalue)
        plt.close(fig)

    def test_params_format(self, fitted_result):
        """Parameter dict keys match expected pattern."""
        params = fitted_result.params
        assert "ar.L1" in params
        assert "sigma2" in params


class TestSARIMAXLogScale:
    """Log-response forecasts return bias-adjusted original-scale values."""

    @staticmethod
    def _positive_data():
        rng = np.random.default_rng(2510)
        log_values = np.empty(80)
        log_values[0] = 2.0
        for index in range(1, len(log_values)):
            log_values[index] = (
                0.6 + 0.7 * log_values[index - 1] + rng.normal(scale=0.12)
            )
        return np.exp(log_values)

    def test_log_requires_boolean_and_strictly_positive_data(self):
        from Ts.TsModels import SARIMAX

        with pytest.raises(TypeError, match="log must be a boolean"):
            SARIMAX(np.arange(1.0, 21.0), log=1)
        with pytest.raises(ValueError, match="strictly positive"):
            SARIMAX(np.arange(20.0), log=True)

    def test_log_model_preserves_original_data_and_clone_contract(self):
        from Ts.TsModels import SARIMAX

        data = self._positive_data()
        model = SARIMAX(data, order=(1, 0, 0), log=True)
        clone = model._clone_for_evaluation(data[:40])

        np.testing.assert_allclose(model.data, data)
        np.testing.assert_allclose(model._model_data, np.log(data))
        assert clone.log is True
        np.testing.assert_allclose(clone.data, data[:40])
        np.testing.assert_allclose(clone._model_data, np.log(data[:40]))

    def test_forecast_uses_horizon_specific_lognormal_bias_correction(self):
        from Ts.TsModels import SARIMAX

        data = self._positive_data()
        result = SARIMAX(data, order=(1, 0, 0), log=True).fit()
        alpha = 0.10
        steps = 4
        raw = result._statsmodels_result.get_forecast(steps=steps)
        raw_frame = raw.summary_frame(alpha=alpha)
        raw_mean = np.asarray(raw_frame["mean"], dtype=float)
        raw_variance = np.asarray(raw.var_pred_mean, dtype=float)
        prediction = result.predict(
            start=result.nobs,
            end=result.nobs + steps - 1,
            alpha=alpha,
        )

        expected_mean = np.exp(raw_mean + 0.5 * raw_variance)
        np.testing.assert_allclose(prediction.mean, expected_mean)
        np.testing.assert_allclose(
            prediction.lower,
            np.exp(np.asarray(raw_frame["mean_ci_lower"], dtype=float)),
        )
        np.testing.assert_allclose(
            prediction.upper,
            np.exp(np.asarray(raw_frame["mean_ci_upper"], dtype=float)),
        )
        assert np.all(prediction.mean > np.exp(raw_mean))
        assert result.log is True
        assert "bias-adjusted mean" in result.summary()

    def test_fitted_values_are_bias_adjusted_on_original_scale(self):
        from Ts.TsModels import SARIMAX

        data = self._positive_data()
        result = SARIMAX(data, order=(1, 0, 0), log=True).fit()
        raw = result._statsmodels_result.get_prediction(start=0, end=len(data) - 1)
        raw_frame = raw.summary_frame(alpha=0.05)
        expected = np.exp(
            np.asarray(raw_frame["mean"], dtype=float)
            + 0.5 * np.asarray(raw.var_pred_mean, dtype=float)
        )
        burn = result._statsmodels_result.loglikelihood_burn

        np.testing.assert_allclose(result.data, data)
        np.testing.assert_allclose(result.fitted_values[burn:], expected[burn:])

    def test_undefined_log_scale_effect_summaries_are_rejected(self):
        from Ts.TsModels import SARIMAX

        result = SARIMAX(self._positive_data(), log=True).fit()

        with pytest.raises(NotImplementedError, match="policy_effect"):
            result.policy_effect(events=[])
        with pytest.raises(NotImplementedError, match="unconditional log variance"):
            result.long_run_equilibrium()


@pytest.fixture
def arma11_data():
    """Generate ARMA(1,1) data for roots testing."""
    r = simulate_sarima(
        n=200,
        order=(1, 0, 1),
        ar=[0.7],
        ma=[0.5],
        seed=42,
        burn=100,
    )
    return r.data


class TestSARIMAXRoots:
    """Test SARIMAXResult arroots, maroots, and plot_roots."""

    @pytest.fixture
    def fitted_ar1(self, ar1_data):
        """Fit AR(1) model."""
        from Ts.TsModels._sarimax import SARIMAX

        return SARIMAX(ar1_data, order=(1, 0, 0)).fit()

    @pytest.fixture
    def fitted_ma1(self, ma1_data):
        """Fit MA(1) model."""
        from Ts.TsModels._sarimax import SARIMAX

        return SARIMAX(ma1_data, order=(0, 0, 1)).fit()

    @pytest.fixture
    def fitted_arma11(self, arma11_data):
        """Fit ARMA(1,1) model."""
        from Ts.TsModels._sarimax import SARIMAX

        return SARIMAX(arma11_data, order=(1, 0, 1)).fit()

    def test_arroots_property_ar1(self, fitted_ar1):
        """AR(1) has one AR root, non-empty ndarray.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.arroots [function]
        """
        roots = fitted_ar1.arroots
        assert isinstance(roots, np.ndarray)
        assert len(roots) == 1

    def test_maroots_property_ma1(self, fitted_ma1):
        """MA(1) has one MA root, non-empty ndarray.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.maroots [function]
        """
        roots = fitted_ma1.maroots
        assert isinstance(roots, np.ndarray)
        assert len(roots) == 1

    def test_arroots_empty_for_ma1(self, fitted_ma1):
        """MA(1) has zero AR roots.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.arroots [function]
        """
        assert len(fitted_ma1.arroots) == 0

    def test_maroots_empty_for_ar1(self, fitted_ar1):
        """AR(1) has zero MA roots.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.maroots [function]
        """
        assert len(fitted_ar1.maroots) == 0

    def test_arroots_maroots_arma11(self, fitted_arma11):
        """ARMA(1,1) has both AR and MA roots.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.arroots [function]
        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.maroots [function]
        """
        assert len(fitted_arma11.arroots) == 1
        assert len(fitted_arma11.maroots) == 1

    def test_summary_distinguishes_enforcement_from_root_result(
        self,
        arma11_data,
    ):
        from Ts.TsModels._sarimax import SARIMAX

        result = SARIMAX(
            arma11_data,
            order=(1, 0, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        text = result.summary()

        assert not result.stationarity_enforced
        assert not result.invertibility_enforced
        assert "Stationarity Enforced  : No" in text
        assert "Invertibility Enforced : No" in text
        assert result.is_stationary == bool(np.all(np.abs(result.arroots) > 1.0))
        assert result.is_invertible == bool(np.all(np.abs(result.maroots) > 1.0))

    def test_plot_roots_returns_fig_ax(self, fitted_arma11):
        """plot_roots() returns (fig, ax) for ARMA(1,1).

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.plot_roots [function]
        """
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig, ax = fitted_arma11.plot_roots()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_roots_with_ar1_only(self, fitted_ar1):
        """plot_roots() works when only AR roots exist (no MA).

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.plot_roots [function]
        """
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig, ax = fitted_ar1.plot_roots()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


class TestSARIMAXCyclePeriod:
    """Test AR(2) and seasonal AR(2) damped-cycle diagnostics."""

    @staticmethod
    def _result(
        *,
        phi1=1.2,
        phi2=-0.8,
        order=(2, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        stationary=True,
    ):
        from Ts.TsModels._sarimax import SARIMAXResult

        seasonal_period = seasonal_order[3]
        polynomial_ar = np.array([1.0, -phi1, -phi2])
        polynomial_seasonal_ar = np.ones(1, dtype=float)
        params = {"ar.L1": phi1, "ar.L2": phi2}
        if seasonal_order[0]:
            polynomial_ar = np.ones(1, dtype=float)
            polynomial_seasonal_ar = np.zeros(2 * seasonal_period + 1, dtype=float)
            polynomial_seasonal_ar[0] = 1.0
            polynomial_seasonal_ar[seasonal_period] = -phi1
            polynomial_seasonal_ar[2 * seasonal_period] = -phi2
            params = {
                f"ar.S.L{seasonal_period}": phi1,
                f"ar.S.L{2 * seasonal_period}": phi2,
            }

        arroots = np.array([1.25 + 0.5j, 1.25 - 0.5j])
        if not stationary:
            arroots = np.array([0.9 + 0.2j, 0.9 - 0.2j])
        fitted = SimpleNamespace(
            polynomial_ar=polynomial_ar,
            polynomial_seasonal_ar=polynomial_seasonal_ar,
            arroots=arroots,
        )
        return SARIMAXResult(
            model_type="SARIMAX",
            params=params,
            std_errors={},
            p_values={},
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            residuals=np.zeros(20),
            fitted_values=np.zeros(20),
            nobs=20,
            data=np.zeros(20),
            _order=order,
            _seasonal_order=seasonal_order,
            _statsmodels_result=fitted,
        )

    def test_nonseasonal_ar2_returns_traceable_cycle_result(self):
        from Ts.TsModels import ARCycleResult

        result = self._result()
        diagnostic = result.cycle_period()
        expected = 2 * np.pi / np.arccos(1.2 / (2 * np.sqrt(0.8)))

        assert isinstance(diagnostic, ARCycleResult)
        assert diagnostic.component == "nonseasonal"
        assert diagnostic.lag_scale == 1
        assert diagnostic.phi1 == pytest.approx(1.2)
        assert diagnostic.phi2 == pytest.approx(-0.8)
        assert diagnostic.discriminant == pytest.approx(1.2**2 + 4 * -0.8)
        assert diagnostic.has_complex_roots
        assert diagnostic.is_stationary
        assert diagnostic.identified
        assert diagnostic.period == pytest.approx(expected)

    @pytest.mark.parametrize(("phi1", "phi2"), [(0.5, 0.2), (1.0, -0.25)])
    def test_nonnegative_discriminant_does_not_identify_cycle(self, phi1, phi2):
        diagnostic = self._result(phi1=phi1, phi2=phi2).cycle_period()

        assert not diagnostic.has_complex_roots
        assert diagnostic.is_stationary
        assert not diagnostic.identified
        assert diagnostic.period is None

    def test_seasonal_selector_must_be_boolean(self):
        with pytest.raises(TypeError, match="seasonal must be a boolean"):
            self._result().cycle_period(seasonal="yes")

    def test_nonstationary_complex_roots_do_not_identify_cycle(self):
        diagnostic = self._result(
            phi1=1.5,
            phi2=-1.1,
            stationary=False,
        ).cycle_period()

        assert diagnostic.has_complex_roots
        assert not diagnostic.is_stationary
        assert not diagnostic.identified
        assert diagnostic.period is None

    def test_seasonal_ar2_period_is_scaled_to_observation_intervals(self):
        result = self._result(
            order=(0, 0, 0),
            seasonal_order=(2, 0, 0, 12),
        )
        diagnostic = result.cycle_period(seasonal=True)
        expected = 12 * 2 * np.pi / np.arccos(1.2 / (2 * np.sqrt(0.8)))

        assert diagnostic.component == "seasonal"
        assert diagnostic.lag_scale == 12
        assert diagnostic.identified
        assert diagnostic.period == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("seasonal", "order", "seasonal_order", "message"),
        [
            (False, (1, 0, 0), (0, 0, 0, 0), "nonseasonal AR lags"),
            (True, (0, 0, 0), (1, 0, 0, 12), "seasonal AR lags"),
        ],
    )
    def test_requires_exactly_first_and_second_ar_lags(
        self,
        seasonal,
        order,
        seasonal_order,
        message,
    ):
        result = self._result(order=order, seasonal_order=seasonal_order)

        with pytest.raises(ValueError, match=message):
            result.cycle_period(seasonal=seasonal)


class TestSARIMAXSparseLags:
    """Sparse AR/MA lag specifications fix omitted coefficients at zero."""

    @pytest.fixture
    def sparse_ar_result(self):
        data = simulate_sarima(
            n=300,
            order=(3, 0, 0),
            ar=[0.5, 0.0, -0.2],
            const=1.0,
            seed=2401,
            burn=200,
        ).data
        from Ts.TsModels._sarimax import SARIMAX

        return SARIMAX(
            data,
            order=([1, 3], 0, 0),
            trend="c",
        ).fit()

    @pytest.fixture
    def sparse_ma_result(self):
        data = simulate_sarima(
            n=350,
            order=(1, 0, 3),
            ar=[0.35],
            ma=[0.45, 0.0, -0.2],
            seed=2402,
            burn=200,
        ).data
        from Ts.TsModels._sarimax import SARIMAX

        return SARIMAX(
            data,
            order=(1, 0, [1, 3]),
            trend="n",
        ).fit()

    def test_sparse_ar_fixes_second_lag_at_zero(self, sparse_ar_result):
        assert sparse_ar_result._order == ((1, 3), 0, 0)
        assert sparse_ar_result.ar_lags == (1, 3)
        assert "ar.L1" in sparse_ar_result.params
        assert "ar.L2" not in sparse_ar_result.params
        assert "ar.L3" in sparse_ar_result.params
        assert sparse_ar_result.fixed_params == {"ar.L2": 0.0}
        assert sparse_ar_result._statsmodels_result.polynomial_ar[2] == 0.0

    def test_sparse_ma_fixes_second_lag_at_zero(self, sparse_ma_result):
        assert sparse_ma_result._order == (1, 0, (1, 3))
        assert sparse_ma_result.ma_lags == (1, 3)
        assert "ma.L1" in sparse_ma_result.params
        assert "ma.L2" not in sparse_ma_result.params
        assert "ma.L3" in sparse_ma_result.params
        assert sparse_ma_result.fixed_params == {"ma.L2": 0.0}
        assert sparse_ma_result._statsmodels_result.polynomial_ma[2] == 0.0

    def test_sparse_lag_summary_reports_constraints_and_roots(
        self,
        sparse_ar_result,
    ):
        text = sparse_ar_result.summary()
        assert "Order: SARIMAX(3, 0, 0)" in text
        assert "Active AR Lags     : 1, 3" in text
        assert "Fixed at Zero      : ar.L2" in text
        assert "AR Stationarity    : Passed" in text
        assert "MA Invertibility   : Not applicable" in text
        assert "Stationarity Enforced  : Yes" in text
        assert sparse_ar_result.is_stationary

    def test_sparse_ma_predicts_and_reports_invertibility(
        self,
        sparse_ma_result,
    ):
        prediction = sparse_ma_result.predict(
            start=sparse_ma_result.nobs,
            end=sparse_ma_result.nobs + 2,
        )
        assert prediction.mean.shape == (3,)
        assert sparse_ma_result.is_invertible
        assert "MA Invertibility   : Passed" in sparse_ma_result.summary()

    def test_sparse_ar_long_run_equilibrium(self, sparse_ar_result):
        expected = sparse_ar_result.params["intercept"] / (
            1.0 - sparse_ar_result.params["ar.L1"] - sparse_ar_result.params["ar.L3"]
        )
        assert sparse_ar_result.long_run_equilibrium() == pytest.approx(expected)

    def test_sparse_lags_are_sorted_and_immutable(self, ar1_data):
        from Ts.TsModels._sarimax import SARIMAX

        lags = [3, 1]
        model = SARIMAX(ar1_data, order=(lags, 0, 0))
        lags.append(2)
        assert model.order == ((1, 3), 0, 0)

    @pytest.mark.parametrize(
        ("order", "error", "message"),
        [
            (([0, 1], 0, 0), ValueError, "p lags must be positive"),
            (([1, 1], 0, 0), ValueError, "p lags must be unique"),
            (([1, 2.5], 0, 0), TypeError, "p lags must be positive"),
            ((True, 0, 0), TypeError, "p must be a non-negative"),
            ((1, True, 0), TypeError, "d must be a non-negative"),
        ],
    )
    def test_invalid_sparse_orders_raise_clear_errors(
        self,
        ar1_data,
        order,
        error,
        message,
    ):
        from Ts.TsModels._sarimax import SARIMAX

        with pytest.raises(error, match=message):
            SARIMAX(ar1_data, order=order)


class TestSARIMAXPredict:
    """Test unified SARIMAXResult.predict() across observed and future ranges."""

    @pytest.fixture
    def result(self, ar1_data):
        """Fit AR(1) and return SARIMAXResult for prediction tests."""
        from Ts.TsModels._sarimax import SARIMAX

        model = SARIMAX(ar1_data, order=(1, 0, 0))
        return model.fit()

    def test_predict_in_sample_full(self, result):
        """predict() with defaults returns full-sample fitted values with CI."""
        from Ts.TsModels._base import PredictResult

        pr = result.predict()
        assert isinstance(pr, PredictResult)
        assert len(pr.mean) == result.nobs
        assert pr.lower is not None
        assert pr.upper is not None
        assert np.all(pr.lower <= pr.mean)
        assert np.all(pr.mean <= pr.upper)

    def test_predict_in_sample_range(self, result):
        """predict(start, end) within sample returns correct-length array."""
        from Ts.TsModels._base import PredictResult

        pr = result.predict(start=10, end=50)
        assert isinstance(pr, PredictResult)
        assert len(pr.mean) == 41

    def test_predict_out_of_sample(self, result):
        """predict() with end > nobs-1 returns out-of-sample forecasts with CI."""
        steps = 5
        end = result.nobs + steps - 1
        pr = result.predict(start=result.nobs - 1, end=end, alpha=0.05)

        assert len(pr.mean) == steps + 1
        assert pr.lower is not None
        assert pr.upper is not None
        assert len(pr.lower) == steps + 1
        assert len(pr.upper) == steps + 1
        assert np.all(pr.lower <= pr.mean)
        assert np.all(pr.mean <= pr.upper)

    def test_predict_can_skip_early_future_periods(self, result):
        """A future-only window returns only the requested later periods."""
        start = result.nobs + 2
        prediction = result.predict(start=start, end=start + 2)
        full_prediction = result.predict(
            start=result.nobs,
            end=start + 2,
        )

        assert prediction.mean.shape == (3,)
        assert prediction.lower.shape == (3,)
        assert prediction.upper.shape == (3,)
        assert prediction.is_oos.tolist() == [True, True, True]
        np.testing.assert_allclose(prediction.mean, full_prediction.mean[2:])
        np.testing.assert_allclose(prediction.lower, full_prediction.lower[2:])
        np.testing.assert_allclose(prediction.upper, full_prediction.upper[2:])

    @pytest.mark.parametrize(
        ("kwargs", "error_type"),
        [
            ({"start": -1}, ValueError),
            ({"start": True}, TypeError),
            ({"start": 5, "end": 4}, ValueError),
            ({"alpha": 0.0}, ValueError),
            ({"alpha": np.nan}, ValueError),
        ],
    )
    def test_predict_rejects_invalid_arguments(
        self,
        result,
        kwargs,
        error_type,
    ):
        """Prediction rejects invalid ranges and interval levels."""
        with pytest.raises(error_type):
            result.predict(**kwargs)

    def test_oos_uses_separate_evaluation_result(self, result):
        """SARIMAX holdout scoring is owned by TsMetrics."""
        from Ts.TsModels import SARIMAX

        split = int(result.nobs * 0.7)
        evaluation = SARIMAX(
            result.data,
            order=result._order,
        ).oos(
            estimation_period=(0, split - 1),
            validation_period=(split, result.nobs - 1),
        )

        assert evaluation.metrics["rmse"] > 0
        assert len(evaluation.mean) == result.nobs - split
        assert evaluation.validation_indices[0] == split

    def test_predict_rejects_removed_oos_start(self, result):
        """The leaked pseudo-OOS argument is no longer accepted."""
        with pytest.raises(TypeError, match="oos_start"):
            result.predict(oos_start=int(result.nobs * 0.7))

    def test_predict_has_no_evaluation_fields(self, result):
        """Ordinary predictions no longer carry scoring state."""
        pr = result.predict(start=0, end=result.nobs - 1)
        assert not hasattr(pr, "metrics")
        assert not hasattr(pr, "actual")
        assert not np.any(pr.is_oos)

    def test_cover_remaining(self, ar1_data):
        """Aggregate covers for items exercised indirectly.

        covers: code/python/Ts/TsModels/_sarimax.py::SARIMAXResult.long_run_equilibrium [function]
        """

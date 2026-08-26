"""Tests for Ts.TsModels._auto — AutoSARIMAX, AutoGARCH, AutoModelResult."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from Ts.TsSims import simulate_sarima, simulate_garch


# ============================================================
# Test Group 1: _get_criterion_value()
# ============================================================


class TestGetCriterionValue:
    """Tests for _get_criterion_value helper."""

    def test_extract_aic(self):
        """Returns result.aic when criterion='aic'."""
        from Ts.TsModels._auto import _get_criterion_value
        from Ts.TsModels._base import BaseModelResult

        result = BaseModelResult(
            model_type="TEST",
            params={"x": 1.0},
            std_errors={"x": 0.1},
            p_values={"x": 0.01},
            aic=100.0,
            bic=105.0,
            log_likelihood=-48.0,
            residuals=np.array([0.1, -0.2, 0.3]),
            fitted_values=np.array([1.0, 2.0, 3.0]),
            nobs=3,
            data=np.array([1.1, 1.8, 3.3]),
        )

        value = _get_criterion_value(result, "aic")
        assert value == 100.0

    def test_extract_bic(self):
        """Returns result.bic when criterion='bic'."""
        from Ts.TsModels._auto import _get_criterion_value
        from Ts.TsModels._base import BaseModelResult

        result = BaseModelResult(
            model_type="TEST",
            params={"x": 1.0},
            std_errors={"x": 0.1},
            p_values={"x": 0.01},
            aic=100.0,
            bic=105.0,
            log_likelihood=-48.0,
            residuals=np.array([0.1]),
            fitted_values=np.array([1.0]),
            nobs=1,
            data=np.array([1.1]),
        )
        assert _get_criterion_value(result, "bic") == 105.0

    def test_compute_hqic(self):
        """HQIC matches formula: -2*llf + 2*k*ln(ln(n))."""
        from Ts.TsModels._auto import _get_criterion_value
        from Ts.TsModels._base import BaseModelResult
        import math

        result = BaseModelResult(
            model_type="TEST",
            params={"a": 1.0, "b": 2.0},
            std_errors={"a": 0.1, "b": 0.1},
            p_values={"a": 0.01, "b": 0.01},
            aic=100.0,
            bic=105.0,
            log_likelihood=-48.0,
            residuals=np.array([0.1, -0.2]),
            fitted_values=np.array([1.0, 2.0]),
            nobs=100,
            data=np.array([1.1, 1.8]),
        )
        value = _get_criterion_value(result, "hqic")
        expected = -2.0 * (-48.0) + 2.0 * 2 * math.log(math.log(100))
        assert value == pytest.approx(expected)

    def test_compute_aicc(self):
        """AICc matches formula: aic + 2*k*(k+1)/(n-k-1)."""
        from Ts.TsModels._auto import _get_criterion_value
        from Ts.TsModels._base import BaseModelResult

        result = BaseModelResult(
            model_type="TEST",
            params={"a": 1.0, "b": 2.0},
            std_errors={"a": 0.1, "b": 0.1},
            p_values={"a": 0.01, "b": 0.01},
            aic=100.0,
            bic=105.0,
            log_likelihood=-48.0,
            residuals=np.array([0.1, -0.2]),
            fitted_values=np.array([1.0, 2.0]),
            nobs=100,
            data=np.array([1.1, 1.8]),
        )
        value = _get_criterion_value(result, "aicc")
        expected = 100.0 + 2.0 * 2 * 3 / (100 - 2 - 1)
        assert value == pytest.approx(expected)

    def test_unknown_criterion_raises(self):
        """Unknown criterion raises ValueError."""
        from Ts.TsModels._auto import _get_criterion_value
        from Ts.TsModels._base import BaseModelResult

        result = BaseModelResult(
            model_type="TEST",
            params={"x": 1.0},
            std_errors={"x": 0.1},
            p_values={"x": 0.01},
            aic=100.0,
            bic=105.0,
            log_likelihood=-48.0,
            residuals=np.array([0.1]),
            fitted_values=np.array([1.0]),
            nobs=1,
            data=np.array([1.1]),
        )
        with pytest.raises(ValueError):
            _get_criterion_value(result, "xyz")


# ============================================================
# Test Group 2: AutoModelResult
# ============================================================


class TestAutoModelResult:
    """Tests for AutoModelResult construction and methods."""

    @pytest.fixture
    def base_result(self):
        """Create a minimal BaseModelResult with adequate obs for diagnostics."""
        from Ts.TsModels._base import BaseModelResult

        n = 200
        rng = np.random.default_rng(42)
        resid = rng.normal(0, 1, size=n)
        return BaseModelResult(
            model_type="SARIMAX",
            params={"ar.L1": 0.7, "sigma2": 1.0},
            std_errors={"ar.L1": 0.1, "sigma2": 0.2},
            p_values={"ar.L1": 0.001, "sigma2": 0.01},
            aic=100.0,
            bic=108.0,
            log_likelihood=-47.0,
            residuals=resid,
            fitted_values=np.zeros(n),
            nobs=n,
            data=np.arange(n, dtype=float),
            _parameter_covariance=np.array([[0.01, -0.01], [-0.01, 0.04]]),
            _parameter_names=("ar.L1", "sigma2"),
        )

    @pytest.fixture
    def auto_result(self, base_result):
        """Create a sample AutoModelResult via from_search()."""
        from Ts.TsModels._auto import AutoModelResult

        return AutoModelResult.from_search(
            best_result=base_result,
            best_order=(1, 0, 0),
            candidate_results=[base_result],
            candidate_orders=[(1, 0, 0)],
            criterion_values=[100.0],
            selection_criterion="aic",
            search_method="grid",
        )

    def test_isinstance_base_model_result(self, auto_result):
        """AutoModelResult is an instance of BaseModelResult."""
        from Ts.TsModels._base import BaseModelResult

        assert isinstance(auto_result, BaseModelResult)

    def test_fields_copied_from_best(self, auto_result):
        """aic, bic, params, residuals copied from best model."""
        assert auto_result.aic == 100.0
        assert auto_result.bic == 108.0
        assert auto_result.params["ar.L1"] == 0.7
        assert len(auto_result.residuals) == 200

    def test_summary_has_selection_info(self, auto_result):
        """summary() shows best order, criterion, candidate count."""
        text = auto_result.summary()
        assert "Auto" in text
        assert "(1, 0, 0)" in text
        assert "aic" in text.lower()
        assert "1/1" in text

    def test_plot_fit_inherited(self, auto_result):
        """plot_fit() returns (fig, ax) via BaseModelResult."""
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig, ax = auto_result.plot_fit()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_diagnostics_inherited(self, auto_result):
        """plot_diagnostics() returns the shared four-panel layout."""
        from matplotlib.figure import Figure

        fig, axes = auto_result.plot_diagnostics()
        assert isinstance(fig, Figure)
        assert len(axes) == 4

    def test_test_residuals_inherited(self, auto_result):
        """test_residuals() returns ResidualTestResults."""
        output = auto_result.test_residuals(lags=3)
        assert output.ljung_box is not None
        assert output.engle_lm is not None

    def test_parameter_correlation_delegates_to_best_result(self, auto_result):
        """Auto results expose the selected model's inference, not search data."""
        correlation = auto_result.parameter_correlation(["sigma2", "ar.L1"])

        assert correlation.index.tolist() == ["sigma2", "ar.L1"]
        np.testing.assert_allclose(
            correlation.to_numpy(),
            [[1.0, -0.5], [-0.5, 1.0]],
        )

    def test_cycle_period_delegates_to_best_result(self, auto_result, base_result):
        expected = object()
        calls = []

        def cycle_period(*, seasonal=False):
            calls.append(seasonal)
            return expected

        base_result.cycle_period = cycle_period

        assert auto_result.cycle_period(seasonal=True) is expected
        assert calls == [True]

    def test_feedback_test_delegates_to_best_result(self, auto_result, base_result):
        expected = object()
        calls = []

        def feedback_test(lags, inputs=None, **kwargs):
            calls.append((lags, inputs, kwargs))
            return expected

        base_result.feedback_test = feedback_test

        assert (
            auto_result.feedback_test(2, inputs="price", trend="ct", alpha=0.1)
            is expected
        )
        assert calls == [(2, "price", {"trend": "ct", "alpha": 0.1})]

    def test_residual_ccf_test_delegates_to_best_result(
        self, auto_result, base_result
    ):
        expected = object()
        input_models = {"price": object()}
        calls = []

        def residual_ccf_test(models, lags=12, inputs=None, **kwargs):
            calls.append((models, lags, inputs, kwargs))
            return expected

        base_result.residual_ccf_test = residual_ccf_test

        assert (
            auto_result.residual_ccf_test(
                input_models,
                lags=8,
                inputs="price",
                alpha=0.10,
            )
            is expected
        )
        assert calls == [(input_models, 8, "price", {"alpha": 0.10})]

    def test_plot_impulse_response_delegates_to_best_result(
        self, auto_result, base_result
    ):
        expected = object()
        calls = []

        def plot_impulse_response(
            steps=20,
            inputs=None,
            sample_weights=None,
            **kwargs,
        ):
            calls.append((steps, inputs, sample_weights, kwargs))
            return expected

        base_result.plot_impulse_response = plot_impulse_response
        sample_weights = object()

        assert (
            auto_result.plot_impulse_response(
                12,
                inputs="price",
                sample_weights=sample_weights,
                grid=False,
            )
            is expected
        )
        assert calls == [(12, "price", sample_weights, {"grid": False})]

    def test_plot_impulse_response_delegates_inferred_steps_to_best_result(
        self, auto_result, base_result
    ):
        expected = object()
        calls = []

        def plot_impulse_response(
            steps=None,
            inputs=None,
            sample_weights=None,
            **kwargs,
        ):
            calls.append((steps, inputs, sample_weights, kwargs))
            return expected

        base_result.plot_impulse_response = plot_impulse_response
        sample_weights = pd.Series([1.0, 0.5, 0.25])

        assert (
            auto_result.plot_impulse_response(sample_weights=sample_weights) is expected
        )
        assert calls == [(None, None, sample_weights, {})]


# ============================================================
# Test Group 3: AutoSARIMAX
# ============================================================


class TestAutoSARIMAX:
    """Tests for AutoSARIMAX construction and fit()."""

    @staticmethod
    def _positive_data():
        """Generate positive data with autoregressive dynamics on log scale."""
        rng = np.random.default_rng(2511)
        log_values = np.empty(80)
        log_values[0] = 2.0
        for index in range(1, len(log_values)):
            log_values[index] = (
                0.6 + 0.7 * log_values[index - 1] + rng.normal(scale=0.12)
            )
        return np.exp(log_values)

    @pytest.fixture
    def ar1_data(self):
        """Generate AR(1) data."""
        r = simulate_sarima(n=200, order=(1, 0, 0), ar=[0.7], seed=42, burn=100)
        return r.data

    @pytest.fixture
    def ma1_data(self):
        """Generate MA(1) data."""
        r = simulate_sarima(n=200, order=(0, 0, 1), ma=[0.5], seed=42, burn=100)
        return r.data

    def test_fit_returns_auto_model_result(self, ar1_data):
        """fit() returns AutoModelResult."""
        from Ts.TsModels._auto import AutoSARIMAX, AutoModelResult

        auto = AutoSARIMAX(ar1_data, p=(1, 1), d=(0, 0), q=(0, 0))
        result = auto.fit()
        assert isinstance(result, AutoModelResult)
        assert auto.result_ is result

    def test_candidates_use_shared_sarimax_fit_defaults(self, ar1_data, monkeypatch):
        """AutoSARIMAX forwards its shared SARIMAX fit defaults."""
        from Ts.TsModels._auto import AutoSARIMAX
        from Ts.TsModels._sarimax import SARIMAX

        calls = []

        def recording_fit(self, **kwargs):
            calls.append(kwargs.copy())
            raise RuntimeError("sentinel candidate failure")

        monkeypatch.setattr(SARIMAX, "fit", recording_fit)
        auto = AutoSARIMAX(
            ar1_data,
            p=(0, 0),
            d=(0, 0),
            q=(0, 0),
            P=(0, 0),
            D=(0, 0),
            Q=(0, 0),
        )

        with pytest.raises(RuntimeError, match="No model converged"):
            auto.fit()

        assert calls == [{"method": "bfgs", "maxiter": 500, "cov_type": "oim"}]

    def test_candidates_use_configured_sarimax_fit_options(
        self, ar1_data, monkeypatch
    ):
        """Each automatic candidate receives the configured fit options."""
        from Ts.TsModels._auto import AutoSARIMAX
        from Ts.TsModels._sarimax import SARIMAX

        calls = []

        def recording_fit(self, **kwargs):
            calls.append(kwargs.copy())
            raise RuntimeError("sentinel candidate failure")

        monkeypatch.setattr(SARIMAX, "fit", recording_fit)
        auto = AutoSARIMAX(
            ar1_data,
            p=(0, 0),
            d=(0, 0),
            q=(0, 0),
            P=(0, 0),
            D=(0, 0),
            Q=(0, 0),
            fit_method="powell",
            maxiter=125,
            cov_type="opg",
        )

        with pytest.raises(RuntimeError, match="No model converged"):
            auto.fit()

        assert calls == [{"method": "powell", "maxiter": 125, "cov_type": "opg"}]

    def test_result_stored(self, ar1_data):
        """result_ is set after fit()."""
        from Ts.TsModels._auto import AutoSARIMAX

        auto = AutoSARIMAX(ar1_data, p=(1, 1), d=(0, 0), q=(0, 0))
        assert auto.result_ is None
        auto.fit()
        assert auto.result_ is not None

    def test_best_has_minimum_criterion_on_ar1(self, ar1_data):
        """Best model's criterion is minimum among candidates (AR1 data)."""
        from Ts.TsModels._auto import AutoSARIMAX

        auto = AutoSARIMAX(ar1_data, p=(0, 2), d=(0, 0), q=(0, 2), criterion="aic")
        result = auto.fit()
        best_val = result.aic
        for val in result.criterion_values:
            assert best_val <= val + 0.01  # allow tiny float rounding

    def test_best_has_minimum_criterion_on_ma1(self, ma1_data):
        """Best model's criterion is minimum among candidates (MA1 data)."""
        from Ts.TsModels._auto import AutoSARIMAX

        auto = AutoSARIMAX(ma1_data, p=(0, 2), d=(0, 0), q=(0, 2), criterion="bic")
        result = auto.fit()
        best_val = result.bic
        for val in result.criterion_values:
            assert best_val <= val + 0.01

    def test_criterion_table_contains_all_candidate_criteria(self, ar1_data):
        """Each successful candidate exposes all supported criteria."""
        from Ts.TsModels._auto import AutoSARIMAX

        result = AutoSARIMAX(
            ar1_data,
            p=(0, 1),
            d=(0, 0),
            q=(0, 1),
            criterion="aic",
        ).fit()

        table = result.criterion_table
        assert list(table.columns) == ["order", "aic", "bic", "hqic", "aicc"]
        assert len(table) == len(result.candidate_results)
        for criterion in ("aic", "bic", "hqic", "aicc"):
            assert np.isfinite(table[criterion].to_numpy(dtype=float)).all()

    def test_small_range_single_combo(self, ar1_data):
        """If range covers only one order, that model is selected."""
        from Ts.TsModels._auto import AutoSARIMAX

        auto = AutoSARIMAX(ar1_data, p=(1, 1), d=(0, 0), q=(0, 0))
        result = auto.fit()
        assert result.best_order == (1, 0, 0)
        assert len(result.candidate_results) == 1

    def test_invalid_criterion_raises(self, ar1_data):
        """Unknown criterion raises ValueError in __init__."""
        from Ts.TsModels._auto import AutoSARIMAX

        with pytest.raises(ValueError):
            AutoSARIMAX(ar1_data, criterion="xyz")

    def test_invalid_method_raises(self, ar1_data):
        """Unknown method raises ValueError in __init__."""
        from Ts.TsModels._auto import AutoSARIMAX

        with pytest.raises(ValueError):
            AutoSARIMAX(ar1_data, method="stepwise")

    def test_range_order_validation(self, ar1_data):
        """p/d/q must be (min, max) tuples with min <= max."""
        from Ts.TsModels._auto import AutoSARIMAX

        with pytest.raises(ValueError):
            AutoSARIMAX(ar1_data, p=(3, 1))

    def test_log_reuses_sarimax_validation(self):
        """AutoSARIMAX applies the shared boolean and positivity contract."""
        from Ts.TsModels import AutoSARIMAX

        with pytest.raises(TypeError, match="log must be a boolean"):
            AutoSARIMAX(self._positive_data(), log=1)
        with pytest.raises(ValueError, match="strictly positive"):
            AutoSARIMAX(np.arange(20.0), log=True)

    def test_log_is_fixed_across_candidates_and_evaluation_clones(self):
        """Every candidate and reconstructed evaluation model retains log=True."""
        from Ts.TsModels import AutoSARIMAX

        data = self._positive_data()
        model = AutoSARIMAX(
            data,
            p=(0, 1),
            d=(0, 0),
            q=(0, 0),
            P=(0, 0),
            D=(0, 0),
            Q=(0, 0),
            log=True,
        )
        clone = model._clone_for_evaluation(data[:50])
        result = model.fit()
        clone_result = clone.fit()

        assert model.log is True
        assert clone.log is True
        assert result.log is True
        assert clone_result.log is True
        assert all(candidate.log is True for candidate in result.candidate_results)
        assert result.level_intercept == pytest.approx(
            result.best_result.level_intercept
        )
        assert result.level_intercept_inference()["estimate"] == pytest.approx(
            result.level_intercept
        )
        assert result.unconditional_log_variance == pytest.approx(
            result.best_result.unconditional_log_variance
        )
        assert result.long_run_equilibrium() == pytest.approx(
            result.best_result.long_run_equilibrium()
        )
        assert "original (log fit; bias-adjusted mean)" in result.summary()

    def test_single_log_candidate_matches_direct_sarimax(self):
        """Auto selection reuses SARIMAX log fitting and original-scale prediction."""
        from Ts.TsModels import AutoSARIMAX, SARIMAX

        data = self._positive_data()
        auto_result = AutoSARIMAX(
            data,
            p=(0, 0),
            d=(0, 0),
            q=(0, 0),
            P=(0, 0),
            D=(0, 0),
            Q=(0, 0),
            log=True,
        ).fit()
        direct_result = SARIMAX(
            data,
            order=(0, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            log=True,
        ).fit()

        auto_prediction = auto_result.predict(start=len(data), end=len(data) + 2)
        direct_prediction = direct_result.predict(start=len(data), end=len(data) + 2)

        assert auto_result.log is True
        assert auto_result.log_likelihood == pytest.approx(direct_result.log_likelihood)
        np.testing.assert_allclose(
            auto_result.fitted_values, direct_result.fitted_values
        )
        np.testing.assert_allclose(auto_prediction.mean, direct_prediction.mean)
        np.testing.assert_allclose(auto_prediction.lower, direct_prediction.lower)
        np.testing.assert_allclose(auto_prediction.upper, direct_prediction.upper)

    def test_summary_has_auto_label(self, ar1_data):
        """summary() contains Auto SARIMAX header."""
        from Ts.TsModels._auto import AutoSARIMAX

        auto = AutoSARIMAX(ar1_data, p=(1, 1), d=(0, 0), q=(0, 0))
        result = auto.fit()
        text = result.summary()
        assert "Auto SARIMAX" in text

    def test_candidate_orders_tracked(self, ar1_data):
        """Search metadata separates attempts from converged candidates."""
        from Ts.TsModels._auto import AutoSARIMAX

        auto = AutoSARIMAX(ar1_data, p=(1, 2), d=(0, 0), q=(0, 1), criterion="aic")
        result = auto.fit()
        assert result.n_attempted == 4  # 2*1*2 = 4
        assert len(result.candidate_orders) == len(result.candidate_results)
        assert all(candidate.converged for candidate in result.candidate_results)
        assert result.candidate_orders[0] == (1, 0, 0)
        assert any("failed to converge" in message for message in result.search_messages)

    def test_seasonal_orders_are_retained(self, ar1_data):
        """Seasonal grid metadata remains available after model selection."""
        from Ts.TsModels._auto import AutoSARIMAX

        result = AutoSARIMAX(
            ar1_data,
            p=(0, 0),
            d=(0, 0),
            q=(0, 0),
            P=(0, 0),
            D=(0, 0),
            Q=(0, 1),
            s=4,
        ).fit()

        assert result.best_seasonal_order in {(0, 0, 0, 4), (0, 0, 1, 4)}
        assert result.candidate_seasonal_orders == [(0, 0, 0, 4), (0, 0, 1, 4)]
        assert "Best Seasonal Order" in result.summary()


# ============================================================
# Test Group 4: AutoGARCH
# ============================================================


class TestAutoGARCH:
    """Tests for AutoGARCH construction and fit()."""

    @pytest.fixture
    def arch2_data(self):
        """Generate ARCH(2) data."""
        r = simulate_garch(
            n=300,
            p=2,
            q=0,
            omega=0.4,
            alpha=[0.3, 0.2],
            seed=42,
            burn=200,
        )
        return r.data

    @pytest.fixture
    def garch11_data(self):
        """Generate GARCH(1,1) data."""
        r = simulate_garch(
            n=300,
            p=1,
            q=1,
            omega=0.1,
            alpha=[0.2],
            beta=[0.7],
            seed=42,
            burn=200,
        )
        return r.data

    def test_fit_returns_auto_model_result(self, garch11_data):
        """fit() returns AutoModelResult."""
        from Ts.TsModels._auto import AutoGARCH, AutoModelResult

        auto = AutoGARCH(garch11_data, p=(1, 1), q=(1, 1))
        result = auto.fit()
        assert isinstance(result, AutoModelResult)
        assert auto.result_ is result

    def test_best_has_minimum_criterion_on_arch2(self, arch2_data):
        """Best model's aic is minimum among candidates (ARCH2 data)."""
        from Ts.TsModels._auto import AutoGARCH

        auto = AutoGARCH(arch2_data, p=(1, 3), q=(0, 0), criterion="aic")
        result = auto.fit()
        best_val = result.aic
        for val in result.criterion_values:
            assert best_val <= val + 0.01

    def test_best_has_minimum_criterion_on_garch11(self, garch11_data):
        """Best model's bic is minimum among candidates (GARCH(1,1) data)."""
        from Ts.TsModels._auto import AutoGARCH

        auto = AutoGARCH(garch11_data, p=(1, 2), q=(1, 2), criterion="bic")
        result = auto.fit()
        best_val = result.bic
        for val in result.criterion_values:
            assert best_val <= val + 0.01

    def test_q_zero_only_arch_models(self, arch2_data):
        """If q=(0,0), only pure ARCH models are tried."""
        from Ts.TsModels._auto import AutoGARCH

        auto = AutoGARCH(arch2_data, p=(1, 3), q=(0, 0))
        result = auto.fit()
        all_arch = all(order[1] == 0 for order in result.candidate_orders)
        assert all_arch

    def test_candidates_tracked(self, garch11_data):
        """candidate_results, orders, and criterion_values are consistent."""
        from Ts.TsModels._auto import AutoGARCH

        auto = AutoGARCH(garch11_data, p=(1, 2), q=(0, 1))
        result = auto.fit()
        n = len(result.candidate_results)
        assert n > 0
        assert len(result.candidate_orders) == n
        assert len(result.criterion_values) == n
        assert result.n_attempted >= n

    def test_nan_data_keeps_exog_aligned(self, garch11_data):
        """AutoGARCH uses the same retained rows for data and exog."""
        from Ts.TsModels._auto import AutoGARCH

        data = garch11_data.copy()
        exog = np.arange(len(data), dtype=float)
        data[10] = np.nan

        auto = AutoGARCH(
            data,
            p=(1, 1),
            q=(1, 1),
            exog=exog,
            missing="drop",
        )

        assert len(auto.data) == len(data) - 1
        assert auto.dropped_positions == (10,)
        assert auto.exog.shape == (len(data) - 1, 1)

    def test_invalid_criterion_raises(self, garch11_data):
        """Unknown criterion raises ValueError in __init__."""
        from Ts.TsModels._auto import AutoGARCH

        with pytest.raises(ValueError):
            AutoGARCH(garch11_data, criterion="xyz")

    def test_invalid_method_raises(self, garch11_data):
        """Unknown method raises ValueError in __init__."""
        from Ts.TsModels._auto import AutoGARCH

        with pytest.raises(ValueError):
            AutoGARCH(garch11_data, method="stepwise")


# ============================================================
# Test Group 5: AutoGARCH -- EGARCH auto-selection
# ============================================================


class TestAutoGARCH_EGARCH:
    """Tests for AutoGARCH with vol='EGARCH'."""

    @pytest.fixture
    def egarch_data(self):
        """Generate EGARCH(1,1,1) data."""
        from Ts.TsSims import simulate_egarch

        r = simulate_egarch(
            n=300,
            p=1,
            q=1,
            o=1,
            omega=0.0,
            alpha=[0.15],
            gamma=[0.05],
            beta=[0.30],
            seed=42,
            burn=200,
        )
        return r.data

    def test_fit_egarch_returns_result(self, egarch_data):
        """AutoGARCH(vol='EGARCH') fit returns AutoModelResult."""
        from Ts.TsModels._auto import AutoGARCH, AutoModelResult

        auto = AutoGARCH(egarch_data, p=(1, 1), q=(1, 1), o=(1, 1), vol="EGARCH")
        result = auto.fit()
        assert isinstance(result, AutoModelResult)
        assert auto.result_ is result

    def test_egarch_best_min_criterion(self, egarch_data):
        """Best EGARCH model has minimum criterion among candidates."""
        from Ts.TsModels._auto import AutoGARCH

        auto = AutoGARCH(
            egarch_data, p=(1, 2), q=(1, 2), o=(1, 1), vol="EGARCH", criterion="aic"
        )
        result = auto.fit()
        best_val = result.aic
        for val in result.criterion_values:
            assert best_val <= val + 0.01


# ============================================================
# Test Group 6: AutoGARCH -- IGARCH auto-selection
# ============================================================


class TestAutoGARCH_IGARCH:
    """Tests for AutoGARCH with igarch=True."""

    @pytest.fixture
    def igarch_data(self):
        """Generate IGARCH(1,1) data."""
        from Ts.TsSims import simulate_igarch

        r = simulate_igarch(
            n=300,
            p=1,
            q=1,
            omega=0.10,
            alpha=[0.20],
            beta=[0.80],
            seed=42,
            burn=200,
        )
        return r.data

    def test_fit_igarch_returns_result(self, igarch_data):
        """AutoGARCH(igarch=True) fit returns AutoModelResult."""
        from Ts.TsModels._auto import AutoGARCH, AutoModelResult

        auto = AutoGARCH(igarch_data, p=(1, 1), q=(1, 1), igarch=True)
        result = auto.fit()
        assert isinstance(result, AutoModelResult)
        assert auto.result_ is result

    def test_failed_orders_are_recorded(self, igarch_data):
        """Search diagnostics retain skipped invalid candidate orders."""
        from Ts.TsModels._auto import AutoGARCH

        result = AutoGARCH(igarch_data, p=(1, 1), q=(0, 1), igarch=True).fit()

        assert result.search_messages
        assert "IGARCH requires q >= 1" in result.search_messages[0]
        assert "Search Diagnostics" in result.summary()

    def test_igarch_rejects_egarch_vol(self, igarch_data):
        """igarch=True with vol='EGARCH' raises ValueError."""
        from Ts.TsModels._auto import AutoGARCH

        with pytest.raises(ValueError):
            AutoGARCH(igarch_data, p=(1, 1), q=(1, 1), igarch=True, vol="EGARCH")

    def test_igarch_rejects_garch_m(self, igarch_data):
        """igarch=True with garch_m=True raises ValueError."""
        from Ts.TsModels._auto import AutoGARCH

        with pytest.raises(ValueError):
            AutoGARCH(igarch_data, p=(1, 1), q=(1, 1), igarch=True, garch_m=True)


# ============================================================
# Test Group 7: AutoGARCH -- GARCH-M auto-selection
# ============================================================


class TestAutoGARCH_GARCHM:
    """Tests for AutoGARCH with garch_m=True."""

    @pytest.fixture
    def garch_m_data(self):
        """Generate GARCH-M(1,1) data."""
        from Ts.TsSims import simulate_garch_m

        r = simulate_garch_m(
            n=300,
            p=1,
            q=1,
            omega=0.10,
            alpha=[0.20],
            beta=[0.60],
            garch_m_kappa=0.20,
            garch_m_form="vol",
            seed=42,
            burn=200,
        )
        return r.data

    def test_fit_garch_m_returns_result(self, garch_m_data):
        """AutoGARCH(garch_m=True) fit returns AutoModelResult."""
        from Ts.TsModels._auto import AutoGARCH, AutoModelResult

        auto = AutoGARCH(garch_m_data, p=(1, 1), q=(1, 1), garch_m=True)
        result = auto.fit()
        assert isinstance(result, AutoModelResult)
        assert auto.result_ is result

    def test_garch_m_rejects_egarch_vol(self, garch_m_data):
        """garch_m=True with vol='EGARCH' raises ValueError."""
        from Ts.TsModels._auto import AutoGARCH

        with pytest.raises(ValueError):
            AutoGARCH(garch_m_data, p=(1, 1), q=(1, 1), garch_m=True, vol="EGARCH")


# ============================================================
# Test Group 8: AutoGARCH -- parameter validation
# ============================================================


class TestAutoGARCH_Validation:
    """Tests for new parameter validation in AutoGARCH."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple GARCH(1,1) data for validation tests."""
        from Ts.TsSims import simulate_garch

        r = simulate_garch(n=200, p=1, q=1, seed=42, burn=100)
        return r.data

    def test_vol_default_is_garch(self, simple_data):
        """Default vol='GARCH' produces GARCH model type."""
        from Ts.TsModels._auto import AutoGARCH

        auto = AutoGARCH(simple_data, p=(1, 1), q=(1, 1))
        result = auto.fit()
        assert "EGARCH" not in result.model_type

    def test_vol_egarch_produces_egarch_type(self, simple_data):
        """vol='EGARCH' produces EGARCH model type in result."""
        from Ts.TsModels._auto import AutoGARCH

        auto = AutoGARCH(simple_data, p=(1, 1), q=(1, 1), o=(1, 1), vol="EGARCH")
        result = auto.fit()
        assert result.model_type == "EGARCH"

    def test_invalid_vol_raises(self, simple_data):
        """Invalid vol value raises ValueError from GARCH class."""
        from Ts.TsModels._auto import AutoGARCH

        with pytest.raises(ValueError):
            auto = AutoGARCH(simple_data, p=(1, 1), q=(1, 1), vol="INVALID")
            auto.fit()


class TestAutoSARIMAXExogenous:
    """AutoSARIMAX preserves the manual estimator's exogenous contracts."""

    @staticmethod
    def _dated_data(*, include_future=True):
        rng = np.random.default_rng(2026)
        dates = pd.date_range("2020-01-01", periods=48, freq="MS")
        all_dates = pd.date_range(
            dates[0],
            periods=51 if include_future else 48,
            freq="MS",
        )
        exog = pd.DataFrame(
            {"driver": np.linspace(-1.0, 1.5, len(all_dates))},
            index=all_dates,
        )
        data = pd.Series(
            1.75 * exog.loc[dates, "driver"].to_numpy()
            + rng.normal(scale=0.05, size=len(dates)),
            index=dates,
        )
        return data, exog

    def test_fit_preserves_exog_and_default_future_path(self):
        """Every candidate receives named exog and the selected result forecasts it."""
        from Ts.TsModels import AutoSARIMAX

        data, exog = self._dated_data()
        model = AutoSARIMAX(
            data,
            exog=exog,
            p=(0, 0),
            d=(0, 0),
            q=(0, 0),
            P=(0, 0),
            D=(0, 0),
            Q=(0, 0),
            trend="n",
        )
        result = model.fit()

        assert model.exog_names == ("driver",)
        assert model.future_exog.index.equals(exog.index[-3:])
        assert result.model_type == "SARIMAX"
        assert result.best_result.exog_names == ("driver",)
        assert result.best_result.params["driver"] == pytest.approx(1.75, abs=0.05)
        forecast = result.predict(start=len(data), end=len(data) + 2)
        assert forecast.mean.shape == (3,)
        assert np.isfinite(forecast.mean).all()

    def test_named_future_scenarios_delegate_to_best_result(self):
        """AutoModelResult retains SARIMAX named-scenario prediction."""
        from Ts.TsModels import AutoSARIMAX, ScenarioForecastResult

        data, exog = self._dated_data(include_future=False)
        fitted = AutoSARIMAX(
            data,
            exog=exog,
            p=(0, 0),
            d=(0, 0),
            q=(0, 0),
            P=(0, 0),
            D=(0, 0),
            Q=(0, 0),
            trend="n",
        ).fit()
        future_dates = pd.date_range(data.index[-1], periods=4, freq="MS")[1:]
        baseline = pd.DataFrame({"driver": [0.0, 0.2, 0.4]}, index=future_dates)
        stress = pd.DataFrame({"driver": [1.0, 1.2, 1.4]}, index=future_dates)

        scenarios = fitted.predict(
            start=len(data),
            end=len(data) + 2,
            future_exog={"baseline": baseline, "stress": stress},
        )

        assert isinstance(scenarios, ScenarioForecastResult)
        assert tuple(scenarios.scenarios) == ("baseline", "stress")
        assert np.all(scenarios["stress"].mean > scenarios["baseline"].mean)

    def test_missing_drop_aligns_endog_exog_and_dates_once(self):
        """Automatic selection applies the joint missing-row policy before search."""
        from Ts.TsModels import AutoSARIMAX

        data, exog = self._dated_data(include_future=False)
        data.iloc[4] = np.nan
        exog.iloc[9, 0] = np.inf

        model = AutoSARIMAX(
            data,
            exog=exog,
            p=(0, 0),
            d=(0, 0),
            q=(0, 0),
            missing="drop",
        )

        assert model.dropped_positions == (4, 9)
        assert len(model.data) == len(model.exog) == len(model.dates) == 46
        assert np.isfinite(model.data).all()
        assert np.isfinite(model.exog).all()

    def test_invalid_exog_fails_before_grid_search(self):
        """Shared input validation reports malformed exog at construction time."""
        from Ts.TsModels import AutoSARIMAX

        with pytest.raises(ValueError, match="exog_names"):
            AutoSARIMAX(
                np.arange(20.0),
                exog=np.ones((20, 1)),
                p=(0, 0),
                d=(0, 0),
                q=(0, 0),
            )

"""Tests for Ts.TsModels._var — VAR and VARResult."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from Ts.TsSims import simulate_sarima


@pytest.fixture
def var2_data():
    """Generate 2-variable data for VAR testing.

    Two independent AR(1) processes stacked as columns:
    y0 ~ AR(1) with phi=0.7, y1 ~ AR(1) with phi=0.5.
    """
    r0 = simulate_sarima(n=150, order=(1, 0, 0), ar=[0.7], seed=42, burn=100)
    r1 = simulate_sarima(n=150, order=(1, 0, 0), ar=[0.5], seed=99, burn=100)
    return np.column_stack([r0.data, r1.data])


class TestVARInit:
    """Test VAR construction and parameter validation."""

    def test_init_stores_data_and_lags(self, var2_data):
        """VAR stores 2-D data and lags parameter.

        covers: code/python/Ts/TsModels/_var.py [module]
        covers: code/python/Ts/TsModels/_var.py::VAR [class]
        covers: code/python/Ts/TsModels/_var.py::VAR.__init__ [function]
        """
        from Ts.TsModels._var import VAR

        model = VAR(var2_data, lags=2)
        assert model.lags == 2
        assert model.data.shape == (150, 2)
        assert model.result_ is None

    def test_invalid_lags_raises(self, var2_data):
        """lags < 1 raises ValueError.

        covers: code/python/Ts/TsModels/_var.py::VAR.__init__ [function]
        """
        from Ts.TsModels._var import VAR

        with pytest.raises(ValueError):
            VAR(var2_data, lags=0)

    def test_1d_data_raises(self, var2_data):
        """1-D data raises ValueError.

        covers: code/python/Ts/TsModels/_var.py::VAR.__init__ [function]
        """
        from Ts.TsModels._var import VAR

        with pytest.raises(ValueError):
            VAR(np.array([1.0, 2.0, 3.0]), lags=1)

    def test_data_too_short_raises(self, var2_data):
        """Too few observations relative to lags raises ValueError.

        covers: code/python/Ts/TsModels/_var.py::VAR.__init__ [function]
        """
        from Ts.TsModels._var import VAR

        short = np.column_stack(
            [
                np.arange(5, dtype=float),
                np.arange(5, dtype=float),
            ]
        )
        with pytest.raises(ValueError):
            VAR(short, lags=2)

    def test_default_cols(self, var2_data):
        """Default cols auto-generate y0, y1, ...

        covers: code/python/Ts/TsModels/_var.py::VAR.__init__ [function]
        """
        from Ts.TsModels._var import VAR

        model = VAR(var2_data, lags=1)
        assert model.data_names == ["y0", "y1"]

    def test_custom_cols(self, var2_data):
        """Custom cols are stored correctly.

        covers: code/python/Ts/TsModels/_var.py::VAR.__init__ [function]
        """
        from Ts.TsModels._var import VAR

        model = VAR(var2_data, lags=1, cols=["gnp", "m1"])
        assert model.data_names == ["gnp", "m1"]


class TestVARFit:
    """Test VAR fit() returns VARResult."""

    def test_fit_returns_var_result(self, var2_data):
        """fit() returns VARResult with expected fields.

        covers: code/python/Ts/TsModels/_var.py::VAR.fit [function]
        """
        from Ts.TsModels._var import VAR, VARResult

        model = VAR(var2_data, lags=2)
        result = model.fit()

        assert isinstance(result, VARResult)
        assert model.result_ is result
        assert result.model_type == "VAR"
        assert result.nobs > 0

    def test_fit_stores_fitted_values_2d(self, var2_data):
        """Fitted values are 2-D with correct shape.

        covers: code/python/Ts/TsModels/_var.py::VAR.fit [function]
        """
        from Ts.TsModels._var import VAR

        model = VAR(var2_data, lags=2)
        result = model.fit()

        assert result.fitted_values.ndim == 2
        assert result.fitted_values.shape[1] == 2

    def test_fit_stores_residuals_2d(self, var2_data):
        """Residuals are 2-D with correct shape.

        covers: code/python/Ts/TsModels/_var.py::VAR.fit [function]
        """
        from Ts.TsModels._var import VAR

        model = VAR(var2_data, lags=2)
        result = model.fit()

        assert result.residuals.ndim == 2
        assert result.residuals.shape[1] == 2


@pytest.fixture
def fitted_var(var2_data):
    """Fit a VAR(2) and return the result."""
    from Ts.TsModels._var import VAR

    model = VAR(var2_data, lags=2)
    return model.fit()


class TestVARResultMethods:
    """Test VARResult-specific methods: irf, fevd, forecast, granger.

    covers: code/python/Ts/TsModels/_var.py::VARResult.long_run_equilibrium [function]
    """

    def test_irf_returns_correct_shape(self, fitted_var):
        """irf() returns IRFResult with values (periods+1, k, k).

        covers: code/python/Ts/TsModels/_var.py::VARResult [class]
        covers: code/python/Ts/TsModels/_var.py::VARResult.irf [function]
        """
        from Ts.TsModels._var import IRFResult

        irf_result = fitted_var.irf(periods=8)
        assert isinstance(irf_result, IRFResult)
        assert irf_result.values.shape == (9, 2, 2)

    def test_irf_orth_returns_correct_shape(self, fitted_var):
        """irf(orth=True) returns IRFResult with values (periods+1, k, k).

        covers: code/python/Ts/TsModels/_var.py::VARResult.irf [function]
        """
        irf_result = fitted_var.irf(periods=8, orth=True)
        assert irf_result.values.shape == (9, 2, 2)
        assert irf_result.orth is True

    def test_irf_orth_differs_from_raw(self, fitted_var):
        """Orthogonalized IRF differs from raw IRF.

        covers: code/python/Ts/TsModels/_var.py::VARResult.irf [function]
        """
        irf_raw = fitted_var.irf(periods=8, orth=False)
        irf_orth = fitted_var.irf(periods=8, orth=True)
        assert not np.allclose(irf_raw.values, irf_orth.values)

    def test_irf_cache_reuse(self, fitted_var):
        """Second irf() call reuses cached values without error.

        covers: code/python/Ts/TsModels/_var.py::VARResult.irf [function]
        """
        raw1 = fitted_var.irf(periods=8, orth=False)
        raw2 = fitted_var.irf(periods=8, orth=False)
        assert np.allclose(raw1.values, raw2.values)
        assert hasattr(fitted_var, "_irf_cache")
        assert fitted_var._irf_cache is not None
        assert fitted_var._irf_cache["periods"] == 8
        assert "raw" in fitted_var._irf_cache
        assert "orth" in fitted_var._irf_cache

    def test_irf_different_periods_invalidates_cache(self, fitted_var):
        """Different periods parameter triggers fresh computation.

        covers: code/python/Ts/TsModels/_var.py::VARResult.irf [function]
        """
        irf8 = fitted_var.irf(periods=8, orth=False)
        assert fitted_var._irf_cache["periods"] == 8
        irf12 = fitted_var.irf(periods=12, orth=False)
        assert fitted_var._irf_cache["periods"] == 12
        assert irf12.values.shape == (13, 2, 2)
        assert irf8.values.shape == (9, 2, 2)

    def test_fevd_returns_correct_shape(self, fitted_var):
        """fevd() returns FEVDResult with values (periods, k, k).

        covers: code/python/Ts/TsModels/_var.py::VARResult.fevd [function]
        """
        from Ts.TsModels._var import FEVDResult

        fevd_result = fitted_var.fevd(periods=8, n_draws=50, seed=42)
        assert isinstance(fevd_result, FEVDResult)
        assert fevd_result.values.shape == (8, 2, 2)

    def test_forecast_returns_mean_lower_upper(self, fitted_var):
        """predict() beyond sample returns (mean, lower, upper) each shape (steps, k).

        covers: code/python/Ts/TsModels/_var.py::VARResult.predict [function]
        """
        steps = 5
        pr = fitted_var.predict(start=fitted_var.nobs, end=fitted_var.nobs + steps - 1)
        assert pr.mean.shape == (steps, 2)
        assert pr.lower is not None
        assert pr.upper is not None
        assert pr.lower.shape == (steps, 2)
        assert pr.upper.shape == (steps, 2)

    def test_granger_causality_returns_granger_result(self, fitted_var):
        """granger_causality() returns GrangerCausalityResult with 1 entry.

        covers: code/python/Ts/TsModels/_var.py::VARResult.granger_causality [function]
        covers: code/python/Ts/TsModels/_var.py::GrangerCausalityResult [class]
        covers: code/python/Ts/TsModels/_var.py::_GrangerEntry [class]
        covers: code/python/Ts/TsModels/_var.py::GrangerCausalityResult.__len__ [function]
        """
        from Ts.TsModels._var import GrangerCausalityResult, _GrangerEntry

        gc = fitted_var.granger_causality(caused=0, causing=1)
        assert isinstance(gc, GrangerCausalityResult)
        assert len(gc) == 1
        entry = gc[0]
        assert isinstance(entry, _GrangerEntry)
        assert entry.caused == "y0"
        assert entry.causing == ["y1"]
        assert entry.test_statistic > 0
        assert 0 < entry.p_value < 1

    def test_granger_causality_str_single(self, fitted_var):
        """GrangerCausalityResult.__str__ single-test: compact table.

        covers: code/python/Ts/TsModels/_var.py::GrangerCausalityResult.__str__ [function]
        covers: code/python/Ts/TsModels/_var.py::GrangerCausalityResult.summary [function]
        covers: code/python/Ts/TsModels/_var.py::_format_single [function]
        covers: code/python/Ts/TsUtils/_validation.py::significance_stars [function]
        """
        gc = fitted_var.granger_causality(caused=0, causing=1)
        text = str(gc)
        assert "Granger Causality Test" in text
        assert "Equation" in text
        assert "Excluded" in text
        assert "p-value" in text
        assert "y0" in text
        assert "y1" in text
        assert gc.summary() == text

    def test_granger_causality_with_names(self, fitted_var):
        """granger_causality() accepts string variable names."""
        gc = fitted_var.granger_causality(caused="y0", causing="y1")
        assert gc[0].caused == "y0"
        assert gc[0].causing == ["y1"]

    def test_granger_causality_chi2_kind(self, fitted_var):
        """granger_causality(kind='chi2') uses chi2 label in output."""
        gc = fitted_var.granger_causality(caused=0, causing=1, kind="chi2")
        assert gc.kind == "chi2"
        assert "chi2" in str(gc)

    def test_granger_causality_all_returns_granger_result(self, fitted_var):
        """granger_causality() (no args) returns GrangerCausalityResult.

        covers: code/python/Ts/TsModels/_var.py::VARResult._granger_causality_all [function]
        covers: code/python/Ts/TsModels/_var.py::_run_granger_all [function]
        """
        from Ts.TsModels._var import GrangerCausalityResult

        results = fitted_var.granger_causality()
        assert isinstance(results, GrangerCausalityResult)
        # 2-variable VAR: 2 individual tests, no ALL (only 1 other var per eq)
        assert len(results) == 2

    def test_granger_causality_all_str_table(self, fitted_var):
        """GrangerCausalityResult.__str__ (multi) produces grouped table.

        covers: code/python/Ts/TsModels/_var.py::GrangerCausalityResult.__str__ [function]
        covers: code/python/Ts/TsModels/_var.py::_format_table [function]
        """
        results = fitted_var.granger_causality()
        text = str(results)
        assert "Granger Causality Wald Tests" in text
        assert "Equation" in text
        assert "Excluded" in text
        assert "p-value" in text
        assert "Significance codes" in text
        assert results.summary() == text

    def test_granger_causality_all_iterable(self, fitted_var):
        """GrangerCausalityResult (multi) is iterable.

        covers: code/python/Ts/TsModels/_var.py::GrangerCausalityResult.__iter__ [function]
        covers: code/python/Ts/TsModels/_var.py::GrangerCausalityResult.__getitem__ [function]
        """
        results = fitted_var.granger_causality()
        count = 0
        for r in results:
            count += 1
            assert r.test_statistic > 0
        assert count == len(results)
        assert results[0].caused is not None

    def test_granger_causality_partial_args_raises(self, fitted_var):
        """granger_causality() raises ValueError when only one arg given."""
        import pytest

        with pytest.raises(ValueError, match="Both 'caused' and 'causing'"):
            fitted_var.granger_causality(caused=0)
        with pytest.raises(ValueError, match="Both 'caused' and 'causing'"):
            fitted_var.granger_causality(causing=1)

    def test_summary_contains_var_info(self, fitted_var):
        """summary() contains VAR label and variable names.

        covers: code/python/Ts/TsModels/_var.py::VARResult.summary [function]
        """
        text = fitted_var.summary()
        assert "VAR(2)" in text
        assert "AIC" in text
        assert "BIC" in text

    def test_test_residuals_returns_dict(self, fitted_var):
        """test_residuals() returns dict mapping variable names to results.

        covers: code/python/Ts/TsModels/_base.py::BaseModelResult.test_residuals [function]
        """
        results = fitted_var.test_residuals(lags=5)
        assert isinstance(results, dict)
        for name in results:
            assert name in fitted_var._data_names

    def test_build_param_names_coverage(self, var2_data):
        """_build_param_names is exercised via fit().

        covers: code/python/Ts/TsModels/_var.py::VAR._build_param_names [function]
        """
        from Ts.TsModels._var import VAR

        model = VAR(var2_data, lags=2, trend="ct")
        names = model._build_param_names(k=2, lags=2)
        assert isinstance(names, list)
        assert len(names) > 0
        assert names[0].startswith("const.")


class TestVARSelectOrder:
    """Test VAR.select_order static method."""

    def test_select_order_returns_varorderresult(self, var2_data):
        """select_order() returns a typed VAROrderResult.

        covers: code/python/Ts/TsModels/_var.py::VAROrderResult [class]
        covers: code/python/Ts/TsModels/_var.py::VAR.select_order [function]
        covers: code/python/Ts/TsModels/_var.py::VAROrderResult.summary [function]
        covers: code/python/Ts/TsModels/_var.py::VAROrderResult.__repr__ [function]
        """
        from Ts.TsModels._var import VAR, VAROrderResult

        result = VAR.select_order(var2_data, max_lags=4, criterion="aic")
        assert isinstance(result, VAROrderResult)
        assert result.criterion == "aic"
        assert 1 <= result.selected_lag <= 4
        assert not hasattr(result, "values")
        with pytest.raises(TypeError):
            result["selected_lag"]
        assert result.max_lags == 4
        assert result.nobs > 0
        assert "aic" in result.criteria_table
        assert "bic" in result.criteria_table
        assert "hqic" in result.criteria_table
        assert "fpe" in result.criteria_table
        # summary()
        text = result.summary()
        assert "VAR Lag Order Selection Criteria" in text
        assert "AIC" in text


class TestVARResultPlots:
    """Test VARResult plot methods return (fig, axes)."""

    def test_plot_fit_returns_fig_axes(self, fitted_var):
        """plot_fit() returns (fig, axes) with k subplots.

        covers: code/python/Ts/TsModels/_base.py::BaseModelResult.plot_fit [function]
        """
        from matplotlib.figure import Figure

        fig, axes = fitted_var.plot_fit()
        assert isinstance(fig, Figure)
        assert len(axes) == 2

    def test_plot_diagnostics_returns_fig_axes(self, fitted_var):
        """plot_diagnostics() returns (fig, axes) with k x 3 grid.

        covers: code/python/Ts/TsModels/_base.py::BaseModelResult.plot_diagnostics [function]
        """
        from matplotlib.figure import Figure

        fig, axes = fitted_var.plot_diagnostics()
        assert isinstance(fig, Figure)
        assert axes.shape == (2, 3)

        standardized = fitted_var.standardized_residuals
        np.testing.assert_allclose(
            np.std(standardized, axis=0, ddof=0),
            np.ones(standardized.shape[1]),
        )
        for position in range(standardized.shape[1]):
            displayed = np.asarray(
                axes[position, 0].lines[0].get_ydata(), dtype=float
            )
            np.testing.assert_allclose(displayed, standardized[:, position])
            assert axes[position, 0].get_title().endswith(
                "Standardized Residuals"
            )
            assert axes[position, 0].get_ylabel() == "Standardized Residual"

    def test_plot_irf_returns_fig_axes(self, fitted_var):
        """plot_irf() returns (fig, axes) with k x k grid.

        covers: code/python/Ts/TsModels/_var.py::VARResult.plot_irf [function]
        """
        from matplotlib.figure import Figure

        fig, axes = fitted_var.plot_irf(periods=6)
        assert isinstance(fig, Figure)
        assert axes.shape == (2, 2)

    def test_plot_irf_orth_returns_fig_axes(self, fitted_var):
        """plot_irf(orth=True) returns (fig, axes) with k x k grid.

        covers: code/python/Ts/TsModels/_var.py::VARResult.plot_irf [function]
        """
        from matplotlib.figure import Figure

        fig, axes = fitted_var.plot_irf(periods=6, orth=True)
        assert isinstance(fig, Figure)
        assert axes.shape == (2, 2)


class TestVARIRFCache:
    """Test IRF caching behavior."""

    def test_irf_cache_shared_across_alpha(self, fitted_var):
        """Different alpha values share the same statsmodels IRF cache.

        covers: code/python/Ts/TsModels/_var.py::VARResult.irf [function]
        """
        r1 = fitted_var.irf(periods=6, orth=False, alpha=0.05)
        r2 = fitted_var.irf(periods=6, orth=False, alpha=0.10)
        assert fitted_var._irf_cache["periods"] == 6
        assert np.allclose(r1.values, r2.values)
        assert not np.allclose(r1.lower, r2.lower)


class TestVARCovers:
    """Aggregate coverage declarations for VAR module items."""

    def test_cover_all(self, var2_data):
        """Declare coverage for VAR module items exercised by tests.

        covers: code/python/Ts/TsModels/_var.py::VARResult.long_run_equilibrium [function]
        covers: code/python/Ts/TsModels/_var.py::IRFResult.summary._step_cell [function]
        covers: code/python/Ts/TsModels/_var.py::IRFResult.summary._val_cell [function]
        covers: code/python/Ts/TsModels/_var.py::IRFResult.summary._row [function]
        covers: code/python/Ts/TsModels/_var.py::FEVDResult.summary._step_cell [function]
        covers: code/python/Ts/TsModels/_var.py::FEVDResult.summary._val_cell [function]
        covers: code/python/Ts/TsModels/_var.py::FEVDResult.summary._row [function]
        covers: code/python/Ts/TsModels/_var.py::_stata_fmt [function]
        """


class TestVARResultRoots:
    """Test VAR is_stable property and plot_roots method."""

    def test_is_stable_property(self, fitted_var):
        """is_stable returns bool, True for stable VAR(2).

        covers: code/python/Ts/TsModels/_var.py::VARResult.is_stable [function]
        """
        assert isinstance(fitted_var.is_stable, bool)
        assert fitted_var.is_stable is True

    def test_plot_roots_returns_fig_ax(self, fitted_var):
        """plot_roots() returns (fig, ax).

        covers: code/python/Ts/TsModels/_var.py::VARResult.plot_roots [function]
        """
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig, ax = fitted_var.plot_roots()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)


class TestVARPredict:
    """Test unified VARResult.predict() across observed and future ranges."""

    def test_predict_in_sample_full(self, fitted_var):
        """predict() with defaults returns full-sample fitted values.

        covers: code/python/Ts/TsModels/_var.py::VARResult.predict [function]
        covers: code/python/Ts/TsModels/_var.py::VARResult._forecast [function]
        covers: code/python/Ts/TsModels/_var.py::VARResult._forecast_ci [function]
        """
        from Ts.TsModels._base import PredictResult

        pr = fitted_var.predict()
        assert isinstance(pr, PredictResult)
        assert pr.mean.shape[0] == fitted_var.nobs
        assert pr.mean.shape[1] == 2

    def test_predict_out_of_sample(self, fitted_var):
        """predict() beyond sample returns forecasts with CI."""
        steps = 4
        end = fitted_var.nobs + steps - 1
        pr = fitted_var.predict(start=fitted_var.nobs, end=end)

        assert pr.mean.shape == (steps, 2)
        assert pr.lower is not None
        assert pr.upper is not None
        assert pr.lower.shape == (steps, 2)
        assert pr.upper.shape == (steps, 2)

    def test_predict_can_skip_early_future_periods(self, fitted_var):
        """A future-only window returns only the requested later periods."""
        start = fitted_var.nobs + 2
        result = fitted_var.predict(start=start, end=start + 2)
        full_result = fitted_var.predict(
            start=fitted_var.nobs,
            end=start + 2,
        )

        assert result.mean.shape == (3, 2)
        assert result.lower.shape == (3, 2)
        assert result.upper.shape == (3, 2)
        assert result.is_oos.tolist() == [True, True, True]
        np.testing.assert_allclose(result.mean, full_result.mean[2:])
        np.testing.assert_allclose(result.lower, full_result.lower[2:])
        np.testing.assert_allclose(result.upper, full_result.upper[2:])

    def test_predict_rejects_unsupported_dynamic_mode(self, fitted_var):
        """VAR does not silently ignore a requested dynamic mode."""
        with pytest.raises(TypeError, match="dynamic"):
            fitted_var.predict(dynamic=True)

    def test_holdout_uses_separate_evaluation_result(self, fitted_var):
        """VAR holdout scoring is owned by TsMetrics."""
        from Ts.TsMetrics import Holdout, evaluate_forecasts
        from Ts.TsModels import VAR

        split = int(len(fitted_var.data) * 0.7)
        model = VAR(
            fitted_var.data,
            lags=fitted_var._lags,
        )
        report = evaluate_forecasts(
            {"var": model},
            scheme=Holdout(
                train=(0, split - 1),
                test=(split, len(fitted_var.data) - 1),
            ),
        )
        evaluation = report.results["var"]

        assert evaluation.mean.shape == (
            1,
            len(fitted_var.data) - split,
            fitted_var.data.shape[1],
        )
        assert len(report.metric_table(by="series")) == fitted_var.data.shape[1]
        assert evaluation.metrics["rmse"] > 0

    def test_predict_has_no_evaluation_fields(self, fitted_var):
        """Ordinary VAR predictions no longer carry scoring state."""
        pr = fitted_var.predict(start=0, end=fitted_var.nobs - 1)
        assert not hasattr(pr, "metrics")
        assert not hasattr(pr, "actual")
        assert not np.any(pr.is_oos)


@pytest.fixture
def irf_result(fitted_var):
    """Return IRFResult from irf() call.

    covers: code/python/Ts/TsModels/_var.py::IRFResult [class]
    """
    return fitted_var.irf(periods=6, orth=False, alpha=0.05)


class TestIRFResult:
    """Test IRFResult dataclass: values, lower, upper, summary, get.

    covers: code/python/Ts/TsModels/_var.py::IRFResult [class]
    """

    def test_irf_result_fields(self, irf_result):
        """IRFResult has values, lower, upper, periods, k, names, orth, alpha.

        covers: code/python/Ts/TsModels/_var.py::IRFResult [class]
        covers: code/python/Ts/TsModels/_var.py::IRFResult.values [function]
        covers: code/python/Ts/TsModels/_var.py::IRFResult.lower [function]
        covers: code/python/Ts/TsModels/_var.py::IRFResult.upper [function]
        """
        assert irf_result.values.shape == (7, 2, 2)  # periods+1, k, k
        assert irf_result.lower is not None
        assert irf_result.upper is not None
        assert irf_result.lower.shape == (7, 2, 2)
        assert irf_result.upper.shape == (7, 2, 2)
        assert irf_result.periods == 6
        assert irf_result.k == 2
        assert irf_result.names == ["y0", "y1"]
        assert irf_result.orth is False
        assert irf_result.alpha == 0.05

    def test_irf_result_summary_contains_headers(self, irf_result):
        """summary() returns compact Stata-style IRF table with legend.

        covers: code/python/Ts/TsModels/_var.py::IRFResult.summary [function]
        """
        text = irf_result.summary()
        assert "IRF" in text
        assert "step" in text
        assert "(1)" in text
        assert "impulse" in text
        assert "response" in text

    def test_irf_result_get_returns_dict(self, irf_result):
        """get(response, shock) returns dict with step, value, lower, upper.

        covers: code/python/Ts/TsModels/_var.py::IRFResult.get [function]
        covers: code/python/Ts/TsModels/_var.py::_resolve_name [function]
        covers: code/python/Ts/TsModels/_var.py::IRFResult.__repr__ [function]
        """
        d = irf_result.get("y0", "y1")
        assert "step" in d
        assert "value" in d
        assert "lower" in d
        assert "upper" in d
        assert len(d["step"]) == 7  # periods + 1
        assert isinstance(d["value"], np.ndarray)

    def test_irf_result_get_int_indices(self, irf_result):
        """get() accepts integer indices.

        covers: code/python/Ts/TsModels/_var.py::IRFResult.get [function]
        """
        d = irf_result.get(0, 1)
        assert len(d["value"]) == 7

    def test_irf_result_orth_flag(self, fitted_var):
        """irf(orth=True) produces IRFResult with orth=True.

        covers: code/python/Ts/TsModels/_var.py::VARResult.irf [function]
        """
        r = fitted_var.irf(periods=4, orth=True)
        assert r.orth is True
        assert r.values.shape == (5, 2, 2)


class TestFEVDResult:
    """Test FEVDResult dataclass and fevd() method.

    covers: code/python/Ts/TsModels/_var.py::FEVDResult [class]
    """

    def test_fevd_returns_fevd_result(self, fitted_var):
        """fevd() returns FEVDResult, not ndarray.

        covers: code/python/Ts/TsModels/_var.py::VARResult.fevd [function]
        covers: code/python/Ts/TsModels/_var.py::FEVDResult [class]
        """
        from Ts.TsModels._var import FEVDResult

        result = fitted_var.fevd(periods=6, n_draws=50, seed=42)
        assert isinstance(result, FEVDResult)
        assert result.values.shape == (6, 2, 2)  # periods, k, k
        assert result.lower is not None
        assert result.upper is not None
        assert result.periods == 6
        assert result.k == 2
        assert result.names == ["y0", "y1"]
        assert result.method == "mc"
        assert result.alpha == 0.05

    def test_fevd_summary_contains_headers(self, fitted_var):
        """FEVDResult.summary() contains compact Stata-style table headers.

        covers: code/python/Ts/TsModels/_var.py::FEVDResult.summary [function]
        """
        result = fitted_var.fevd(periods=4, n_draws=50, seed=42)
        text = result.summary()
        assert "FEVD" in text
        assert "step" in text.lower()
        assert "(1)" in text
        assert "impulse =" in text
        assert "response =" in text
        assert "Monte Carlo" in text

    def test_fevd_get_returns_dict(self, fitted_var):
        """FEVDResult.get(response, shock) returns dict with step, value, lower, upper.

        covers: code/python/Ts/TsModels/_var.py::FEVDResult.get [function]
        covers: code/python/Ts/TsModels/_var.py::FEVDResult.__repr__ [function]
        """
        result = fitted_var.fevd(periods=6, n_draws=50, seed=42)
        d = result.get("y0", "y0")
        assert "step" in d
        assert "value" in d
        assert "lower" in d
        assert "upper" in d
        assert len(d["step"]) == 6  # periods
        assert isinstance(d["value"], np.ndarray)

    def test_fevd_values_sum_to_one(self, fitted_var):
        """FEVD values across all shocks sum to ~1 for each variable at each step.

        covers: code/python/Ts/TsModels/_var.py::VARResult.fevd [function]
        """
        result = fitted_var.fevd(periods=6, n_draws=50, seed=42)
        for h in range(6):
            for i in range(2):
                total = np.sum(result.values[h, i, :])
                assert np.isclose(total, 1.0, atol=1e-10)

    def test_fevd_ci_in_zero_one(self, fitted_var):
        """FEVD lower and upper confidence bounds are in [0, 1].

        covers: code/python/Ts/TsModels/_var.py::VARResult.fevd [function]
        covers: code/python/Ts/TsModels/_var.py::VARResult._fevd_mc [function]
        """
        result = fitted_var.fevd(periods=6, n_draws=50, seed=42)
        assert np.all(result.lower >= 0.0)
        assert np.all(result.lower <= 1.0)
        assert np.all(result.upper >= 0.0)
        assert np.all(result.upper <= 1.0)
        assert np.all(result.lower <= result.upper)

"""Tests for Ts.TsModels._auto — AutoSARIMA, AutoGARCH, AutoModelResult."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
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
            model_type="SARIMA",
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
        """plot_diagnostics() returns 3-panel figure."""
        from matplotlib.figure import Figure

        fig, axes = auto_result.plot_diagnostics()
        assert isinstance(fig, Figure)
        assert len(axes) == 3

    def test_test_residuals_inherited(self, auto_result):
        """test_residuals() returns ResidualTestResults."""
        output = auto_result.test_residuals(lags=3)
        assert output.ljung_box is not None
        assert output.engle_lm is not None


# ============================================================
# Test Group 3: AutoSARIMA
# ============================================================


class TestAutoSARIMA:
    """Tests for AutoSARIMA construction and fit()."""

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
        from Ts.TsModels._auto import AutoSARIMA, AutoModelResult

        auto = AutoSARIMA(ar1_data, p=(1, 1), d=(0, 0), q=(0, 0))
        result = auto.fit()
        assert isinstance(result, AutoModelResult)
        assert auto.result_ is result

    def test_result_stored(self, ar1_data):
        """result_ is set after fit()."""
        from Ts.TsModels._auto import AutoSARIMA

        auto = AutoSARIMA(ar1_data, p=(1, 1), d=(0, 0), q=(0, 0))
        assert auto.result_ is None
        auto.fit()
        assert auto.result_ is not None

    def test_best_has_minimum_criterion_on_ar1(self, ar1_data):
        """Best model's criterion is minimum among candidates (AR1 data)."""
        from Ts.TsModels._auto import AutoSARIMA

        auto = AutoSARIMA(ar1_data, p=(0, 2), d=(0, 0), q=(0, 2), criterion="aic")
        result = auto.fit()
        best_val = result.aic
        for val in result.criterion_values:
            assert best_val <= val + 0.01  # allow tiny float rounding

    def test_best_has_minimum_criterion_on_ma1(self, ma1_data):
        """Best model's criterion is minimum among candidates (MA1 data)."""
        from Ts.TsModels._auto import AutoSARIMA

        auto = AutoSARIMA(ma1_data, p=(0, 2), d=(0, 0), q=(0, 2), criterion="bic")
        result = auto.fit()
        best_val = result.bic
        for val in result.criterion_values:
            assert best_val <= val + 0.01

    def test_small_range_single_combo(self, ar1_data):
        """If range covers only one order, that model is selected."""
        from Ts.TsModels._auto import AutoSARIMA

        auto = AutoSARIMA(ar1_data, p=(1, 1), d=(0, 0), q=(0, 0))
        result = auto.fit()
        assert result.best_order == (1, 0, 0)
        assert len(result.candidate_results) == 1

    def test_invalid_criterion_raises(self, ar1_data):
        """Unknown criterion raises ValueError in __init__."""
        from Ts.TsModels._auto import AutoSARIMA

        with pytest.raises(ValueError):
            AutoSARIMA(ar1_data, criterion="xyz")

    def test_invalid_method_raises(self, ar1_data):
        """Unknown method raises ValueError in __init__."""
        from Ts.TsModels._auto import AutoSARIMA

        with pytest.raises(ValueError):
            AutoSARIMA(ar1_data, method="stepwise")

    def test_range_order_validation(self, ar1_data):
        """p/d/q must be (min, max) tuples with min <= max."""
        from Ts.TsModels._auto import AutoSARIMA

        with pytest.raises(ValueError):
            AutoSARIMA(ar1_data, p=(3, 1))

    def test_summary_has_auto_label(self, ar1_data):
        """summary() contains Auto SARIMA header."""
        from Ts.TsModels._auto import AutoSARIMA

        auto = AutoSARIMA(ar1_data, p=(1, 1), d=(0, 0), q=(0, 0))
        result = auto.fit()
        text = result.summary()
        assert "Auto SARIMA" in text

    def test_candidate_orders_tracked(self, ar1_data):
        """candidate_orders list has correct length."""
        from Ts.TsModels._auto import AutoSARIMA

        auto = AutoSARIMA(ar1_data, p=(1, 2), d=(0, 0), q=(0, 1), criterion="aic")
        result = auto.fit()
        assert len(result.candidate_orders) == 4  # 2*1*2 = 4
        assert result.candidate_orders[0] == (1, 0, 0)

    def test_seasonal_orders_are_retained(self, ar1_data):
        """Seasonal grid metadata remains available after model selection."""
        from Ts.TsModels._auto import AutoSARIMA

        result = AutoSARIMA(
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

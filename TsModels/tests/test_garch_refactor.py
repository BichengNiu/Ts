"""Characterization tests for _garch.py refactoring.

These tests capture the exact output of current implementations as a baseline.
After refactoring, all tests must still PASS with identical values.
Mode B: BASELINE verification.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


# ============================================================================
# Shared fixtures — fixed seeds for deterministic output
# ============================================================================


@pytest.fixture(scope="module")
def garch11_result():
    """Fit GARCH(1,1) with fixed seed. Output MUST NOT change after refactoring."""
    from Ts.TsSims import simulate_garch
    from Ts.TsModels._garch import GARCH

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
    model = GARCH(r.data, p=1, q=1)
    return model.fit()


@pytest.fixture(scope="module")
def arch2_result():
    """Fit ARCH(2) with fixed seed. Output MUST NOT change after refactoring."""
    from Ts.TsSims import simulate_garch
    from Ts.TsModels._garch import GARCH

    r = simulate_garch(
        n=200,
        p=2,
        q=0,
        omega=0.4,
        alpha=[0.3, 0.2],
        seed=42,
        burn=200,
    )
    model = GARCH(r.data, p=2, q=0)
    return model.fit()


@pytest.fixture(scope="module")
def igarch11_result():
    """Fit IGARCH(1,1) with fixed seed. Output MUST NOT change after refactoring."""
    from Ts.TsSims import simulate_igarch
    from Ts.TsModels._garch import GARCH

    r = simulate_igarch(
        n=300,
        p=1,
        q=1,
        omega=0.05,
        alpha=[0.3],
        seed=42,
        burn=200,
    )
    model = GARCH(r.data, p=1, q=1, igarch=True)
    return model.fit()


@pytest.fixture(scope="module")
def sim_garch_data():
    """Simulated GARCH(1,1) data with fixed seed."""
    from Ts.TsSims._garch import simulate_garch

    return simulate_garch(
        n=200,
        p=1,
        q=1,
        omega=0.1,
        alpha=[0.2],
        beta=[0.7],
        seed=42,
        burn=200,
    )


@pytest.fixture(scope="module")
def sim_egarch_data():
    """Simulated EGARCH(1,1,1) data with fixed seed."""
    from Ts.TsSims._garch_ext import simulate_egarch

    return simulate_egarch(
        n=200,
        p=1,
        q=1,
        o=1,
        omega=0.0,
        alpha=[0.20],
        gamma=[0.10],
        beta=[0.30],
        seed=42,
        burn=200,
    )


# ============================================================================
# GARCH(1,1) baseline
# ============================================================================


class TestGARCH11Baseline:
    """GARCH(1,1) estimation output must remain unchanged."""

    def test_model_type_is_garch(self, garch11_result):
        assert garch11_result.model_type == "GARCH"

    def test_params_keys(self, garch11_result):
        assert "mu" in garch11_result.params
        assert "omega" in garch11_result.params
        assert "alpha[1]" in garch11_result.params
        assert "beta[1]" in garch11_result.params

    def test_aic_bic_are_finite(self, garch11_result):
        assert np.isfinite(garch11_result.aic)
        assert np.isfinite(garch11_result.bic)
        assert garch11_result.aic > 0
        assert garch11_result.bic > 0

    def test_residuals_length(self, garch11_result):
        assert len(garch11_result.residuals) == garch11_result.nobs

    def test_conditional_volatility_not_none(self, garch11_result):
        assert garch11_result.conditional_volatility is not None
        assert len(garch11_result.conditional_volatility) == garch11_result.nobs

    def test_summary_contains_key_sections(self, garch11_result):
        text = garch11_result.summary()
        assert "GARCH(1,1)" in text
        assert "Model Estimation Result" in text
        assert "Parameter Estimates" in text
        assert "Observations" in text
        assert "AIC" in text
        assert "BIC" in text

    def test_forecast_returns_arrays(self, garch11_result):
        pr = garch11_result.predict(
            start=garch11_result.nobs, end=garch11_result.nobs + 4
        )
        assert len(pr.mean) == 5
        assert np.all(pr.mean >= 0)

    def test_test_persistence_returns_dict(self, garch11_result):
        stab = garch11_result.test_persistence()
        assert "chi2" in stab
        assert "pvalue" in stab
        assert "persistence_sum" in stab

    def test_exact_aic_value(self, garch11_result):
        """Precision test: AIC must match exactly after refactoring."""
        assert garch11_result.aic == pytest.approx(811.612854, rel=1e-4)

    def test_exact_bic_value(self, garch11_result):
        """Precision test: BIC must match exactly after refactoring."""
        assert garch11_result.bic == pytest.approx(826.427984, rel=1e-4)

    def test_exact_omega_value(self, garch11_result):
        assert garch11_result.params["omega"] == pytest.approx(0.196234, rel=1e-3)

    def test_exact_alpha1_value(self, garch11_result):
        assert garch11_result.params["alpha[1]"] == pytest.approx(0.279406, rel=1e-3)

    def test_exact_beta1_value(self, garch11_result):
        assert garch11_result.params["beta[1]"] == pytest.approx(0.524120, rel=1e-3)


# ============================================================================
# ARCH(2) baseline
# ============================================================================


class TestARCH2Baseline:
    """ARCH(2) estimation output must remain unchanged."""

    def test_model_type_is_arch(self, arch2_result):
        assert arch2_result.model_type == "ARCH"

    def test_params_keys(self, arch2_result):
        assert "mu" in arch2_result.params
        assert "omega" in arch2_result.params
        assert "alpha[1]" in arch2_result.params
        assert "alpha[2]" in arch2_result.params

    def test_summary_shows_individual_ic(self, arch2_result):
        """ARCH(p) summary must show per-lag IC comparison."""
        text = arch2_result.summary()
        assert "Lowest AIC" in text
        assert "Lowest BIC" in text

    def test_exact_aic_value(self, arch2_result):
        assert arch2_result.aic == pytest.approx(513.823279, rel=1e-4)

    def test_exact_alpha1_value(self, arch2_result):
        assert arch2_result.params["alpha[1]"] == pytest.approx(0.400933, rel=1e-3)


# ============================================================================
# IGARCH(1,1) baseline
# ============================================================================


class TestIGARCH11Baseline:
    """IGARCH(1,1) estimation output must remain unchanged."""

    def test_model_type_is_igarch(self, igarch11_result):
        assert igarch11_result.model_type == "IGARCH"

    def test_summary_shows_igarch(self, igarch11_result):
        text = igarch11_result.summary()
        assert "IGARCH(1,1)" in text

    def test_forecast_returns_arrays(self, igarch11_result):
        pr = igarch11_result.predict(
            start=igarch11_result.nobs, end=igarch11_result.nobs + 4
        )
        assert len(pr.mean) == 5

    def test_exact_omega_value(self, igarch11_result):
        assert igarch11_result.params["omega"] == pytest.approx(0.131236, rel=5e-3)


# ============================================================================
# ResidualTestResults baseline
# ============================================================================


class TestResidualTestResultsBaseline:
    """ResidualTestResults output must remain unchanged."""

    @pytest.fixture(scope="module")
    def diag_output(self, garch11_result):
        return garch11_result.test_residuals(lags=5)

    def test_returns_correct_type(self, diag_output):
        from Ts.TsModels._base import ResidualTestResults

        assert type(diag_output) is ResidualTestResults

    def test_has_all_four_tests_via_attribute(self, diag_output):
        assert diag_output.white_noise is not None
        assert diag_output.normality is not None
        assert diag_output.ljung_box is not None
        assert diag_output.engle_lm is not None

    def test_white_noise_is_ljungbox_result(self, diag_output):
        from Ts.TsTests._base import BaseTestResult

        assert isinstance(diag_output.white_noise, BaseTestResult)
        assert diag_output.white_noise.apply_squared is False

    def test_normality_has_skewness_kurtosis(self, diag_output):
        assert hasattr(diag_output.normality, "skewness")
        assert hasattr(diag_output.normality, "kurtosis")

    def test_all_statistics_are_finite(self, diag_output):
        assert np.isfinite(diag_output.white_noise.statistic)
        assert np.isfinite(diag_output.normality.statistic)
        assert np.isfinite(diag_output.ljung_box.statistic)
        assert np.isfinite(diag_output.engle_lm.statistic)

    def test_summary_contains_all_four_tests(self, diag_output):
        text = diag_output.summary()
        assert "White Noise" in text
        assert "Normality" in text
        assert "ARCH Effect" in text

    def test_exact_white_noise_stat(self, diag_output):
        assert diag_output.white_noise.statistic == pytest.approx(6.388309, rel=1e-2)


# ============================================================================
# Simulate GARCH baseline
# ============================================================================


class TestSimulateGARCHBaseline:
    """simulate_garch output must remain unchanged."""

    def test_model_type(self, sim_garch_data):
        assert sim_garch_data.model_type == "GARCH"

    def test_data_length(self, sim_garch_data):
        assert len(sim_garch_data.data) == 200

    def test_has_conditional_volatility(self, sim_garch_data):
        assert sim_garch_data.conditional_volatility is not None
        assert len(sim_garch_data.conditional_volatility) == 200

    def test_exact_first_value(self, sim_garch_data):
        assert sim_garch_data.data[0] == pytest.approx(0.259511, rel=1e-4)

    def test_exact_last_value(self, sim_garch_data):
        assert sim_garch_data.data[-1] == pytest.approx(-0.534081, rel=1e-4)

    def test_to_dataframe_columns(self, sim_garch_data):
        df = sim_garch_data.to_dataframe()
        assert list(df.columns) == ["data", "residuals", "volatility"]
        assert len(df) == 200


# ============================================================================
# Simulate EGARCH baseline
# ============================================================================


class TestSimulateEGARCHBaseline:
    """simulate_egarch output must remain unchanged."""

    def test_model_type(self, sim_egarch_data):
        assert sim_egarch_data.model_type == "EGARCH"

    def test_data_length(self, sim_egarch_data):
        assert len(sim_egarch_data.data) == 200

    def test_exact_first_value(self, sim_egarch_data):
        assert sim_egarch_data.data[0] == pytest.approx(0.305241, rel=1e-4)

    def test_exact_last_value(self, sim_egarch_data):
        assert sim_egarch_data.data[-1] == pytest.approx(-0.696067, rel=1e-4)

    def test_summary_shows_egarch(self, sim_egarch_data):
        text = sim_egarch_data.summary()
        assert "EGARCH" in text
        assert "gamma" in text


# ============================================================================
# Import path baseline
# ============================================================================


class TestImportPaths:
    """Public import paths must continue to work after refactoring."""

    def test_import_garch_from_tsmodels(self):
        from Ts.TsModels import GARCH

        assert GARCH is not None

    def test_import_garch_result_from_tsmodels(self):
        from Ts.TsModels import GARCHResult

        assert GARCHResult is not None

    def test_import_compare_models_from_tsmodels(self):
        from Ts.TsModels import compare_models

        assert compare_models is not None

    def test_import_sarima_from_tsmodels(self):
        from Ts.TsModels import SARIMA

        assert SARIMA is not None

    def test_import_simulate_garch_from_tssims(self):
        from Ts.TsSims import simulate_garch

        assert simulate_garch is not None

    def test_import_simulate_egarch_from_tssims(self):
        from Ts.TsSims import simulate_egarch

        assert simulate_egarch is not None

    def test_import_simulate_igarch_from_tssims(self):
        from Ts.TsSims import simulate_igarch

        assert simulate_igarch is not None

"""Public API contracts for the SARIMAX estimator family."""

from __future__ import annotations

import importlib.util

import numpy as np


def test_sarimax_types_are_the_only_public_arima_estimators():
    """Only canonical SARIMAX estimator names are publicly exported."""
    import Ts
    import Ts.TsModels as models

    assert Ts.SARIMAX is models.SARIMAX
    assert Ts.SARIMAXResult is models.SARIMAXResult
    assert Ts.AutoSARIMAX is models.AutoSARIMAX
    for removed in ("SARIMA", "SARIMAResult", "AutoSARIMA"):
        assert not hasattr(Ts, removed)
        assert not hasattr(models, removed)


def test_sarimax_fit_uses_canonical_result_and_model_type():
    """The estimator returns the canonical result and model label."""
    from Ts.TsModels import SARIMAX, SARIMAXResult

    rng = np.random.default_rng(2026)
    data = rng.normal(size=60)
    result = SARIMAX(data, order=(1, 0, 0)).fit()

    assert isinstance(result, SARIMAXResult)
    assert result.model_type == "SARIMAX"
    summary = result.summary()
    assert "SARIMAX Model Estimation Result" in summary
    assert "Order: SARIMAX(1, 0, 0)" in summary


def test_removed_private_module_has_no_import_path():
    """The old estimator module is removed rather than retained as a shim."""
    assert importlib.util.find_spec("Ts.TsModels._sarima") is None
    assert importlib.util.find_spec("Ts.TsModels._sarimax") is not None

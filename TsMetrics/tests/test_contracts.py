"""Public contract tests for unified forecast evaluation."""

import numpy as np
import pytest

from Ts.TsMetrics import (
    ForecastComparisonResult,
    ForecastEvaluationResult,
    Holdout,
    RollingOrigin,
    directional_accuracy,
    evaluate_forecasts,
    mpe,
    relative_win_rate,
    trend_correlation,
)


def test_unified_types_are_exported_from_package_root():
    from Ts import (
        ForecastComparisonResult as RootComparison,
        ForecastEvaluationResult as RootEvaluation,
        Holdout as RootHoldout,
        RollingOrigin as RootRolling,
        directional_accuracy as RootDirectionalAccuracy,
        evaluate_forecasts as root_evaluate,
        mpe as RootMPE,
        relative_win_rate as RootRelativeWinRate,
        trend_correlation as RootTrendCorrelation,
    )

    assert RootComparison is ForecastComparisonResult
    assert RootEvaluation is ForecastEvaluationResult
    assert RootHoldout is Holdout
    assert RootRolling is RollingOrigin
    assert RootDirectionalAccuracy is directional_accuracy
    assert root_evaluate is evaluate_forecasts
    assert RootMPE is mpe
    assert RootRelativeWinRate is relative_win_rate
    assert RootTrendCorrelation is trend_correlation


def test_legacy_evaluation_names_are_not_public():
    import Ts
    from Ts import TsMetrics

    names = {
        "oos",
        "backtest",
        "compare_forecasts",
        "evaluate_models_oos",
        "OOSResult",
        "BacktestResult",
        "ComparisonResult",
        "OOSComparisonResult",
    }
    assert names.isdisjoint(Ts.__all__)
    assert names.isdisjoint(TsMetrics.__all__)
    assert all(not hasattr(Ts, name) for name in names)
    assert all(not hasattr(TsMetrics, name) for name in names)


def test_result_rejects_half_an_interval():
    splits = RollingOrigin(initial_window=10).split(11)
    with pytest.raises(ValueError, match="both be set"):
        ForecastEvaluationResult(
            mean=np.ones((1, 1)),
            actual=np.ones((1, 1)),
            lower=np.zeros((1, 1)),
            upper=None,
            splits=splits,
            failures=[],
            model_type="TEST",
            target="observed",
        )


def test_base_model_does_not_retain_legacy_convenience_methods():
    from Ts.TsModels import BaseModel

    assert not hasattr(BaseModel, "oos")
    assert not hasattr(BaseModel, "backtest")

"""Ts — Time Series Econometrics Toolkit.

This package consolidates six sub-packages under a unified namespace.

- :mod:`Ts.TsPlots`  — shared plotting (series, scatter, ACF/PACF)
- :mod:`Ts.TsSims`   — synthetic data generation (SARIMA, GARCH, TS/DS)
- :mod:`Ts.TsUtils`  — preprocessing and identification diagnostics
- :mod:`Ts.TsModels` — model estimation (SARIMAX, GARCH)
- :mod:`Ts.TsMetrics` — forecast metrics and leakage-free evaluation
- :mod:`Ts.TsTests`  — statistical tests (unit root, structural break, ARCH)

Quick start
-----------
>>> from Ts import STL, interpolate_missing, eacf, SARIMAX, AutoSARIMAX, ADFTest
>>> from Ts.TsPlots import plot_series, plot_scatter, plot_acf, plot_pacf
>>> from Ts.TsSims import simulate_sarima, simulate_garch
>>> from Ts.TsUtils import STL, interpolate_missing, eacf
>>> from Ts.TsModels import SARIMAX, GARCH, AutoSARIMAX, AutoGARCH
>>> from Ts.TsMetrics import (
...     rmse, oos, backtest, compare_forecasts, evaluate_models_oos
... )
>>> from Ts.TsTests import ADFTest, KPSSTest, LjungBoxTest
"""

# ---------------------------------------------------------------------------
# TsPlots — plotting
# ---------------------------------------------------------------------------
from .TsPlots import plot_series, plot_scatter, plot_acf, plot_pacf

# ---------------------------------------------------------------------------
# TsUtils — preprocessing and identification diagnostics
# ---------------------------------------------------------------------------
from .TsUtils import (
    BoxCoxResult,
    EACFResult,
    InterpolationResult,
    STL,
    STLResult,
    TimeSeriesSummary,
    boxcox,
    difference,
    eacf,
    interpolate_missing,
)

# ---------------------------------------------------------------------------
# TsSims — simulation
# ---------------------------------------------------------------------------
from .TsSims import (
    BaseSimResult,
    SimSARIMAResult,
    SimGARCHResult,
    SimTSDSResult,
    simulate_sarima,
    simulate_garch,
    simulate_igarch,
    simulate_gjr_garch,
    simulate_egarch,
    simulate_garch_m,
    simulate_trend_stationary,
    simulate_difference_stationary,
)

# ---------------------------------------------------------------------------
# TsModels — estimation
# ---------------------------------------------------------------------------
from .TsModels import (
    BaseModel,
    BaseModelResult,
    ResidualTestResults,
    SARIMAX,
    SARIMAXResult,
    ScenarioForecastResult,
    EventSpec,
    PolicyEffectResult,
    GARCH,
    AutoSARIMAX,
    AutoGARCH,
    AutoModelResult,
    compare_models,
)

from .TsMetrics import (
    BacktestResult,
    ComparisonResult,
    OOSComparisonResult,
    OOSResult,
    backtest,
    compare_forecasts,
    compute_metrics,
    evaluate_models_oos,
    mae,
    mape,
    mse,
    oos,
    rmse,
    smape,
    theil_u1,
)

# ---------------------------------------------------------------------------
# TsTests — statistical tests
# ---------------------------------------------------------------------------
from .TsTests import (
    BaseTest,
    BaseTestResult,
    # Unit root tests
    ADFTest,
    ADFTestResult,
    PhillipsPerronTest,
    PhillipsPerronTestResult,
    KPSSTest,
    KPSSTestResult,
    # Structural break tests
    PerronTest,
    PerronTestResult,
    ZivotAndrewsTest,
    ZivotAndrewsTestResult,
    LeeStrazicichTwoBreakTest,
    LeeStrazicichTwoBreakTestResult,
    ChowTest,
    ChowTestResult,
    CUSUMTest,
    CUSUMTestResult,
    BaiPerronTest,
    BaiPerronTestResult,
    # ARCH-effect tests
    LjungBoxTest,
    LjungBoxTestResult,
    EngleLMTest,
    EngleLMTestResult,
    # Normality test
    NormalityTest,
    NormalityTestResult,
)

__all__ = [  # noqa: RUF022 - public API is grouped by subpackage
    # TsPlots
    "plot_series",
    "plot_scatter",
    "plot_acf",
    "plot_pacf",
    # TsUtils
    "BoxCoxResult",
    "STL",
    "STLResult",
    "TimeSeriesSummary",
    "boxcox",
    "difference",
    "eacf",
    "EACFResult",
    "interpolate_missing",
    "InterpolationResult",
    # TsSims — base
    "BaseSimResult",
    # TsSims — result classes
    "SimSARIMAResult",
    "SimGARCHResult",
    "SimTSDSResult",
    # TsSims — functions
    "simulate_sarima",
    "simulate_garch",
    "simulate_igarch",
    "simulate_gjr_garch",
    "simulate_egarch",
    "simulate_garch_m",
    "simulate_trend_stationary",
    "simulate_difference_stationary",
    # TsModels — base
    "BaseModel",
    "BaseModelResult",
    "ResidualTestResults",
    # TsModels — models
    "SARIMAX",
    "SARIMAXResult",
    "ScenarioForecastResult",
    "EventSpec",
    "PolicyEffectResult",
    "GARCH",
    "AutoSARIMAX",
    "AutoGARCH",
    "AutoModelResult",
    # TsModels — utilities
    "compare_models",
    # TsMetrics
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "theil_u1",
    "compute_metrics",
    "oos",
    "backtest",
    "compare_forecasts",
    "evaluate_models_oos",
    "OOSResult",
    "BacktestResult",
    "ComparisonResult",
    "OOSComparisonResult",
    # TsTests — base
    "BaseTest",
    "BaseTestResult",
    # TsTests — unit root
    "ADFTest",
    "ADFTestResult",
    "PhillipsPerronTest",
    "PhillipsPerronTestResult",
    "KPSSTest",
    "KPSSTestResult",
    # TsTests — structural break
    "PerronTest",
    "PerronTestResult",
    "ZivotAndrewsTest",
    "ZivotAndrewsTestResult",
    "LeeStrazicichTwoBreakTest",
    "LeeStrazicichTwoBreakTestResult",
    "ChowTest",
    "ChowTestResult",
    "CUSUMTest",
    "CUSUMTestResult",
    "BaiPerronTest",
    "BaiPerronTestResult",
    # TsTests — ARCH-effects
    "LjungBoxTest",
    "LjungBoxTestResult",
    "EngleLMTest",
    "EngleLMTestResult",
    # TsTests — normality
    "NormalityTest",
    "NormalityTestResult",
]

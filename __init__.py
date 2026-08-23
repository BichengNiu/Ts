"""Ts — Time Series Econometrics Toolkit.

This package consolidates six sub-packages under a unified namespace.

- :mod:`Ts.TsPlots`  — shared plotting (series, scatter, ACF/PACF, lag response)
- :mod:`Ts.TsSims`   — synthetic data generation (SARIMA, RDL, GARCH, TS/DS)
- :mod:`Ts.TsUtils`  — preprocessing and identification diagnostics
- :mod:`Ts.TsModels` — model estimation (SARIMAX/RDL, ARDL, GARCH)
- :mod:`Ts.TsMetrics` — forecast metrics and leakage-free evaluation
- :mod:`Ts.TsTests`  — statistical tests (unit root, breaks, ARCH, feedback)

The top-level namespace is a curated convenience API, not the union of every
subpackage export. Import the complete model, simulation, plotting, and test
APIs from their corresponding subpackages.

Quick start
-----------
>>> from Ts import (
...     ADFTest, AutoSARIMAX, SARIMAX, STL, eacf,
...     calendar_table, interpolate_missing, seasonal_dummies,
... )
>>> from Ts.TsPlots import plot_series, plot_scatter, plot_acf, plot_pacf
>>> from Ts.TsSims import simulate_sarima, simulate_garch
>>> from Ts.TsUtils import (
...     STL, calendar_table, eacf, interpolate_missing, seasonal_dummies,
... )
>>> from Ts.TsModels import ARDL, AutoARDL, SARIMAX, GARCH, AutoSARIMAX, AutoGARCH
>>> from Ts.TsMetrics import Holdout, RollingOrigin, evaluate_forecasts, rmse
>>> from Ts.TsTests import ADFTest, KPSSTest, LjungBoxTest
"""

# ---------------------------------------------------------------------------
# TsPlots — plotting
# ---------------------------------------------------------------------------
from .TsPlots import plot_acf, plot_lag_response, plot_pacf, plot_scatter, plot_series

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
    calendar_table,
    difference,
    eacf,
    interpolate_missing,
    seasonal_dummies,
)

# ---------------------------------------------------------------------------
# TsSims — simulation
# ---------------------------------------------------------------------------
from .TsSims import (
    BaseSimResult,
    RDLInputSpec,
    SimCointegratedResult,
    SimRDLResult,
    SimSARIMAResult,
    SimGARCHResult,
    SimTSDSResult,
    simulate_sarima,
    simulate_rdl,
    simulate_garch,
    simulate_igarch,
    simulate_gjr_garch,
    simulate_egarch,
    simulate_garch_m,
    simulate_cointegrated,
    simulate_trend_stationary,
    simulate_difference_stationary,
)

# ---------------------------------------------------------------------------
# TsModels — estimation
# ---------------------------------------------------------------------------
from .TsModels import (
    ARDL,
    ARDLResult,
    AutoARDL,
    AutoARDLResult,
    BaseModel,
    BaseModelResult,
    ResidualTestResults,
    SARIMAX,
    SARIMAXResult,
    RationalLagResult,
    RationalLagSpec,
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
    ForecastComparisonResult,
    ForecastEvaluationResult,
    Holdout,
    RollingOrigin,
    compute_metrics,
    evaluate_forecasts,
    mae,
    mape,
    mse,
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
    FeedbackEquationResult,
    FeedbackTest,
    FeedbackTestResult,
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
    "plot_lag_response",
    # TsUtils
    "BoxCoxResult",
    "STL",
    "STLResult",
    "TimeSeriesSummary",
    "boxcox",
    "calendar_table",
    "difference",
    "eacf",
    "EACFResult",
    "interpolate_missing",
    "InterpolationResult",
    "seasonal_dummies",
    # TsSims — base
    "BaseSimResult",
    # TsSims — result classes
    "SimSARIMAResult",
    "SimRDLResult",
    "SimGARCHResult",
    "SimTSDSResult",
    "SimCointegratedResult",
    "RDLInputSpec",
    # TsSims — functions
    "simulate_sarima",
    "simulate_rdl",
    "simulate_garch",
    "simulate_igarch",
    "simulate_gjr_garch",
    "simulate_egarch",
    "simulate_garch_m",
    "simulate_cointegrated",
    "simulate_trend_stationary",
    "simulate_difference_stationary",
    # TsModels — base
    "BaseModel",
    "BaseModelResult",
    "ResidualTestResults",
    # TsModels — models
    "SARIMAX",
    "SARIMAXResult",
    "ARDL",
    "ARDLResult",
    "AutoARDL",
    "AutoARDLResult",
    "RationalLagSpec",
    "RationalLagResult",
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
    "Holdout",
    "RollingOrigin",
    "evaluate_forecasts",
    "ForecastEvaluationResult",
    "ForecastComparisonResult",
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
    "FeedbackEquationResult",
    "FeedbackTest",
    "FeedbackTestResult",
    # TsTests — normality
    "NormalityTest",
    "NormalityTestResult",
]

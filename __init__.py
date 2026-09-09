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
...     ADFTest, AutoSARIMAX, SARIMAX, STL, eacf, hurst_exponent,
...     calendar_table, interpolate_missing, seasonal_dummies,
... )
>>> from Ts.TsPlots import plot_series, plot_scatter, plot_acf, plot_pacf
>>> from Ts.TsSims import simulate_sarima, simulate_garch
>>> from Ts.TsUtils import (
...     STL, calendar_table, eacf, hurst_exponent, interpolate_missing,
...     seasonal_dummies,
... )
>>> from Ts.TsModels import ARDL, AutoARDL, SARIMAX, GARCH, AutoSARIMAX, AutoGARCH
>>> from Ts.TsMetrics import Holdout, RollingOrigin, evaluate_forecasts, rmse
>>> from Ts.TsTests import ADFTest, KPSSTest, LjungBoxTest
"""

# The root package is a convenience namespace.  Importing it must not import
# every optional plotting, modelling, and statistical dependency: users who
# need only one subpackage should pay for that subpackage only.
from importlib import import_module


_SUBPACKAGES = frozenset(
    {"TsPlots", "TsUtils", "TsSims", "TsModels", "TsMetrics", "TsTests"}
)

_LAZY_EXPORTS = {
    # TsPlots
    "plot_acf": ("TsPlots", "plot_acf"),
    "plot_lag_response": ("TsPlots", "plot_lag_response"),
    "plot_pacf": ("TsPlots", "plot_pacf"),
    "plot_scatter": ("TsPlots", "plot_scatter"),
    "plot_series": ("TsPlots", "plot_series"),
    # TsUtils
    "BoxCoxResult": ("TsUtils", "BoxCoxResult"),
    "EACFResult": ("TsUtils", "EACFResult"),
    "InterpolationResult": ("TsUtils", "InterpolationResult"),
    "STL": ("TsUtils", "STL"),
    "STLResult": ("TsUtils", "STLResult"),
    "TimeSeriesSummary": ("TsUtils", "TimeSeriesSummary"),
    "boxcox": ("TsUtils", "boxcox"),
    "calendar_table": ("TsUtils", "calendar_table"),
    "difference": ("TsUtils", "difference"),
    "eacf": ("TsUtils", "eacf"),
    "hurst_exponent": ("TsUtils", "hurst_exponent"),
    "interpolate_missing": ("TsUtils", "interpolate_missing"),
    "seasonal_dummies": ("TsUtils", "seasonal_dummies"),
    # TsSims
    "BaseSimResult": ("TsSims", "BaseSimResult"),
    "RDLInputSpec": ("TsSims", "RDLInputSpec"),
    "SimCointegratedResult": ("TsSims", "SimCointegratedResult"),
    "SimRDLResult": ("TsSims", "SimRDLResult"),
    "SimSARIMAResult": ("TsSims", "SimSARIMAResult"),
    "SimGARCHResult": ("TsSims", "SimGARCHResult"),
    "SimTSDSResult": ("TsSims", "SimTSDSResult"),
    "simulate_sarima": ("TsSims", "simulate_sarima"),
    "simulate_rdl": ("TsSims", "simulate_rdl"),
    "simulate_garch": ("TsSims", "simulate_garch"),
    "simulate_igarch": ("TsSims", "simulate_igarch"),
    "simulate_gjr_garch": ("TsSims", "simulate_gjr_garch"),
    "simulate_egarch": ("TsSims", "simulate_egarch"),
    "simulate_garch_m": ("TsSims", "simulate_garch_m"),
    "simulate_cointegrated": ("TsSims", "simulate_cointegrated"),
    "simulate_trend_stationary": ("TsSims", "simulate_trend_stationary"),
    "simulate_difference_stationary": ("TsSims", "simulate_difference_stationary"),
    # TsModels
    "ARDL": ("TsModels", "ARDL"),
    "ARDLResult": ("TsModels", "ARDLResult"),
    "AutoARDL": ("TsModels", "AutoARDL"),
    "AutoARDLResult": ("TsModels", "AutoARDLResult"),
    "BaseModel": ("TsModels", "BaseModel"),
    "BaseModelResult": ("TsModels", "BaseModelResult"),
    "ResidualTestResults": ("TsModels", "ResidualTestResults"),
    "SARIMAX": ("TsModels", "SARIMAX"),
    "SARIMAXResult": ("TsModels", "SARIMAXResult"),
    "RationalLagResult": ("TsModels", "RationalLagResult"),
    "RationalLagSpec": ("TsModels", "RationalLagSpec"),
    "ScenarioForecastResult": ("TsModels", "ScenarioForecastResult"),
    "EventSpec": ("TsModels", "EventSpec"),
    "PolicyEffectResult": ("TsModels", "PolicyEffectResult"),
    "GARCH": ("TsModels", "GARCH"),
    "AutoSARIMAX": ("TsModels", "AutoSARIMAX"),
    "AutoGARCH": ("TsModels", "AutoGARCH"),
    "AutoModelResult": ("TsModels", "AutoModelResult"),
    "compare_models": ("TsModels", "compare_models"),
    # TsMetrics
    "ForecastComparisonResult": ("TsMetrics", "ForecastComparisonResult"),
    "ForecastEvaluationResult": ("TsMetrics", "ForecastEvaluationResult"),
    "Holdout": ("TsMetrics", "Holdout"),
    "RollingOrigin": ("TsMetrics", "RollingOrigin"),
    "compute_metrics": ("TsMetrics", "compute_metrics"),
    "directional_accuracy": ("TsMetrics", "directional_accuracy"),
    "evaluate_forecasts": ("TsMetrics", "evaluate_forecasts"),
    "mae": ("TsMetrics", "mae"),
    "mape": ("TsMetrics", "mape"),
    "mse": ("TsMetrics", "mse"),
    "mpe": ("TsMetrics", "mpe"),
    "relative_win_rate": ("TsMetrics", "relative_win_rate"),
    "rmse": ("TsMetrics", "rmse"),
    "smape": ("TsMetrics", "smape"),
    "theil_u1": ("TsMetrics", "theil_u1"),
    "trend_correlation": ("TsMetrics", "trend_correlation"),
    # TsTests
    "BaseTest": ("TsTests", "BaseTest"),
    "BaseTestResult": ("TsTests", "BaseTestResult"),
    "ADFTest": ("TsTests", "ADFTest"),
    "ADFTestResult": ("TsTests", "ADFTestResult"),
    "PhillipsPerronTest": ("TsTests", "PhillipsPerronTest"),
    "PhillipsPerronTestResult": ("TsTests", "PhillipsPerronTestResult"),
    "KPSSTest": ("TsTests", "KPSSTest"),
    "KPSSTestResult": ("TsTests", "KPSSTestResult"),
    "PerronTest": ("TsTests", "PerronTest"),
    "PerronTestResult": ("TsTests", "PerronTestResult"),
    "ZivotAndrewsTest": ("TsTests", "ZivotAndrewsTest"),
    "ZivotAndrewsTestResult": ("TsTests", "ZivotAndrewsTestResult"),
    "LeeStrazicichTwoBreakTest": ("TsTests", "LeeStrazicichTwoBreakTest"),
    "LeeStrazicichTwoBreakTestResult": ("TsTests", "LeeStrazicichTwoBreakTestResult"),
    "ChowTest": ("TsTests", "ChowTest"),
    "ChowTestResult": ("TsTests", "ChowTestResult"),
    "CUSUMTest": ("TsTests", "CUSUMTest"),
    "CUSUMTestResult": ("TsTests", "CUSUMTestResult"),
    "BaiPerronTest": ("TsTests", "BaiPerronTest"),
    "BaiPerronTestResult": ("TsTests", "BaiPerronTestResult"),
    "LjungBoxTest": ("TsTests", "LjungBoxTest"),
    "LjungBoxTestResult": ("TsTests", "LjungBoxTestResult"),
    "EngleLMTest": ("TsTests", "EngleLMTest"),
    "EngleLMTestResult": ("TsTests", "EngleLMTestResult"),
    "FeedbackEquationResult": ("TsTests", "FeedbackEquationResult"),
    "FeedbackTest": ("TsTests", "FeedbackTest"),
    "FeedbackTestResult": ("TsTests", "FeedbackTestResult"),
    "NormalityTest": ("TsTests", "NormalityTest"),
    "NormalityTestResult": ("TsTests", "NormalityTestResult"),
}


def __getattr__(name):
    """Load a subpackage or convenience export only when first requested."""
    if name in _SUBPACKAGES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    """Expose lazy convenience exports to interactive discovery tools."""
    return sorted(set(globals()) | _SUBPACKAGES | set(_LAZY_EXPORTS))

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
    "hurst_exponent",
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
    "mpe",
    "directional_accuracy",
    "relative_win_rate",
    "trend_correlation",
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

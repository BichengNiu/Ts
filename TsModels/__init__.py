"""TsModels — Time series model estimation toolkit.

This package provides unified interfaces for estimating SARIMAX, GARCH, VAR,
SVAR, and VECM models. Result objects integrate with
TsPlots for plotting and, where applicable, TsTests for diagnostics.

Main interfaces
---------------
SARIMAX
    SARIMAX model estimation via statsmodels SARIMAX.
GARCH
    GARCH(p,q) model estimation via the ``arch`` library.  Handles both
    pure ARCH (q = 0) and GARCH (q >= 1) volatility models.
VAR
    Vector Autoregression estimation via statsmodels VAR.
VECM
    Vector Error Correction Model estimation via statsmodels VECM.
SVAR
    Structural VAR with short-run (A/B) and long-run (Blanchard-Quah)
    identification restrictions.

Result classes
--------------
SARIMAXResult
    Container for SARIMAX estimation output. Provides ``.summary()``,
    ``.predict()``, ``.plot_fit()``, ``.plot_diagnostics()``,
    ``.test_residuals()``.
GARCHResult
    Container for GARCH estimation output. Provides ``.summary()``,
    ``.predict()``, ``.plot_fit()``, ``.plot_diagnostics()``,
    ``.test_residuals()``, ``.conditional_volatility``.
SVARResult
    Container for SVAR estimation output (extends VARResult). Provides
    ``.A``, ``.B``, ``.sirf()``, ``.structural_residuals``.

Quick start
-----------
>>> from Ts.TsModels import SARIMAX, GARCH
>>> from Ts.TsSims import simulate_sarima, simulate_garch

>>> # AR(1) estimation
>>> data = simulate_sarima(n=200, order=(1, 0, 0), ar=[0.7], seed=42).data
>>> model = SARIMAX(data, order=(1, 0, 0))
>>> result = model.fit()
>>> print(result.summary())
>>> result.plot_diagnostics()

>>> # GARCH(1,1) estimation
>>> data = simulate_garch(n=300, seed=42).data
>>> model = GARCH(data, p=1, q=1)
>>> result = model.fit()
>>> result.plot_fit()

>>> # Pure ARCH(2) estimation (GARCH with q=0)
>>> data = simulate_garch(n=200, p=2, q=0, seed=42).data
>>> model = GARCH(data, p=2, q=0)
>>> result = model.fit()
"""

from ._auto import AutoGARCH, AutoModelResult, AutoSARIMAX
from ._backcast import BackcastResult
from ._base import BaseModel, BaseModelResult, PredictResult, ResidualTestResults
from ._compare import compare_models
from ._distributed_lag import RationalLagResult, RationalLagSpec
from ._garch import GARCH
from ._garch_result import GARCHResult
from ._intervention import EventSpec, PolicyEffectResult
from ._sarimax import ARCycleResult, SARIMAX, SARIMAXResult, ScenarioForecastResult
from ._svar import SVAR, SVARResult
from ._var import (
    VAR,
    FEVDResult,
    GrangerCausalityResult,
    IRFResult,
    VAROrderResult,
    VARResult,
)
from ._vecm import (
    VECM,
    VECMOrderResult,
    VECMResult,
)

__all__ = [  # noqa: RUF022 - public API is grouped by model family
    "BackcastResult",
    "BaseModel",
    "BaseModelResult",
    "PredictResult",
    "ResidualTestResults",
    "SARIMAX",
    "SARIMAXResult",
    "ARCycleResult",
    "ScenarioForecastResult",
    "RationalLagSpec",
    "RationalLagResult",
    "EventSpec",
    "PolicyEffectResult",
    "GARCH",
    "GARCHResult",
    "VAR",
    "VARResult",
    "VAROrderResult",
    "GrangerCausalityResult",
    "IRFResult",
    "FEVDResult",
    "VECM",
    "VECMResult",
    "VECMOrderResult",
    "SVAR",
    "SVARResult",
    "AutoSARIMAX",
    "AutoGARCH",
    "AutoModelResult",
    "compare_models",
]

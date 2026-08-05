"""TsTests — Time series statistical tests.

This package implements statistical tests for time series analysis:

**Unit root tests**

- :class:`ADFTest` — Augmented Dickey-Fuller unit root test.
- :class:`PhillipsPerronTest` — Phillips-Perron unit root test.
- :class:`KPSSTest` — KPSS stationarity test (H0: stationary).

**Structural break unit root tests**

- :class:`PerronTest` — Perron (1989) test with a *known* break date.
- :class:`ZivotAndrewsTest` — Zivot & Andrews (1992) test with an
  *unknown* break date (endogenously selected).
- :class:`LeeStrazicichTwoBreakTest` — minimum LM unit-root test with
  *two unknown* break dates.

**Regression parameter-stability tests**

- :class:`ChowTest` — coefficient stability at a *known* break date.
- :class:`CUSUMTest` — unknown regression instability from OLS residuals.
- :class:`BaiPerronTest` — *multiple unknown* regression break dates.

**ARCH-effect tests**

- :class:`LjungBoxTest` — Ljung-Box Q-test on squared data for ARCH effects.
- :class:`EngleLMTest` — Engle's Lagrange Multiplier test for ARCH effects.

**Granger causality tests**

- :class:`TodaYamamotoTest` — Toda-Yamamoto (1995) Granger causality test
  for possibly integrated or cointegrated VAR systems.
- :class:`FeedbackTest` — conditional predictive-feedback F tests for
  stochastic distributed-lag inputs.

**Transfer-function diagnostics**

- :class:`ResidualCCFTest` — residual cross-correlations and the joint
  Box-Jenkins S* adequacy test for prewhitened transfer inputs.

Quick start
-----------
>>> from Ts.TsTests import (
...     ADFTest, PhillipsPerronTest, KPSSTest,
...     PerronTest, ZivotAndrewsTest, LeeStrazicichTwoBreakTest,
...     ChowTest, CUSUMTest, BaiPerronTest,
... )
"""

from ._adf import ADFTest, ADFTestResult
from ._bai_perron import BaiPerronTest, BaiPerronTestResult
from ._base import BaseMultiTestResult, BaseTest, BaseTestResult
from ._chow import ChowTest, ChowTestResult
from ._cusum import CUSUMTest, CUSUMTestResult
from ._engle_lm import EngleLMTest, EngleLMTestResult
from ._feedback import FeedbackEquationResult, FeedbackTest, FeedbackTestResult
from ._johansen import JohansenTest, JohansenTestResult
from ._kpss import KPSSTest, KPSSTestResult
from ._lee_strazicich import (
    LeeStrazicichTwoBreakTest,
    LeeStrazicichTwoBreakTestResult,
)
from ._ljungbox import LjungBoxTest, LjungBoxTestResult
from ._normality import NormalityTest, NormalityTestResult
from ._perron import PerronTest, PerronTestResult
from ._phillips_perron import PhillipsPerronTest, PhillipsPerronTestResult
from ._residual_ccf import (
    ResidualCCFInputResult,
    ResidualCCFTest,
    ResidualCCFTestResult,
)
from ._toda_yamamoto import TodaYamamotoTest, TodaYamamotoTestResult
from ._zivot import ZivotAndrewsTest, ZivotAndrewsTestResult

__all__ = [
    "ADFTest",
    "ADFTestResult",
    "BaiPerronTest",
    "BaiPerronTestResult",
    "BaseMultiTestResult",
    "BaseTest",
    "BaseTestResult",
    "CUSUMTest",
    "CUSUMTestResult",
    "ChowTest",
    "ChowTestResult",
    "EngleLMTest",
    "EngleLMTestResult",
    "FeedbackEquationResult",
    "FeedbackTest",
    "FeedbackTestResult",
    "JohansenTest",
    "JohansenTestResult",
    "KPSSTest",
    "KPSSTestResult",
    "LeeStrazicichTwoBreakTest",
    "LeeStrazicichTwoBreakTestResult",
    "LjungBoxTest",
    "LjungBoxTestResult",
    "NormalityTest",
    "NormalityTestResult",
    "PerronTest",
    "PerronTestResult",
    "PhillipsPerronTest",
    "PhillipsPerronTestResult",
    "ResidualCCFInputResult",
    "ResidualCCFTest",
    "ResidualCCFTestResult",
    "TodaYamamotoTest",
    "TodaYamamotoTestResult",
    "ZivotAndrewsTest",
    "ZivotAndrewsTestResult",
]

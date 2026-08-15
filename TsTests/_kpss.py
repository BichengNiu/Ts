"""KPSS (Kwiatkowski-Phillips-Schmidt-Shin) stationarity test.

Wraps :func:`statsmodels.tsa.stattools.kpss` in the TsTests
:class:`BaseTest` / :class:`BaseTestResult` framework.

.. important::
   KPSS has **reversed** hypotheses compared to ADF / PP:

   - H0: The series is **stationary** (level or trend stationary).
   - H1: The series has a **unit root** (non-stationary).

Reference
---------
Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992).
"Testing the Null Hypothesis of Stationarity against the Alternative of
a Unit Root." *Journal of Econometrics*, 54(1-3), 159–178.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from Ts.TsPlots.unitroot_plot import _render_critical_value_plot
from Ts.TsUtils._validation import validate_choice

from ._base import BaseTest, BaseTestResult
from ._utils import _clean_1d


@dataclass
class KPSSTestResult(BaseTestResult):
    """Container for KPSS test results.

    .. note::
       KPSS H0 = **stationarity**. A small p-value means *reject*
       stationarity, i.e. the series has a unit root.

    In addition to the fields inherited from :class:`BaseTestResult`:

    Parameters
    ----------
    trend : str
        Trend specification (``"c"`` for level-stationary,
        ``"ct"`` for trend-stationary).
    critical_values : dict
        Critical values keyed by significance level.
    statistic, pvalue, lags, nobs, residuals : see BaseTestResult
        Common statistic, p-value, bandwidth, effective sample size, and
        optional residuals inherited from ``BaseTestResult``.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import KPSSTest
    >>> result = KPSSTest(np.random.default_rng(42).normal(size=100)).fit()
    >>> result.trend
    'c'
    """

    trend: str = "c"
    critical_values: dict = field(default_factory=dict)

    def __str__(self) -> str:
        # KPSS: reject H0 (stationarity) when pvalue < 0.05
        if self.pvalue is not None:
            reject = self.pvalue < 0.05
        else:
            cv_5 = self.critical_values.get("5%")
            reject = cv_5 is not None and self.statistic > cv_5

        conclusion = (
            "Reject H0 (stationarity) -> series has a unit root"
            if reject
            else "Cannot reject H0 -> series is stationary"
        )

        lines = [
            self._format_conclusion("KPSS Test", "Series is stationary"),
            f"  Trend              : {self.trend}",
            f"  Lags (NW trunc.)   : {self.lags}",
            f"  Observations       : {self.nobs}",
            f"  Conclusion (5%): {conclusion}",
        ]

        return "\n".join(lines)

    def plot_test(self, ax=None):
        """Plot the KPSS test statistic against critical values.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import KPSSTest
        >>> result = KPSSTest(np.random.default_rng(42).normal(size=80)).fit()
        >>> fig, ax = result.plot_test()
        """
        return _render_critical_value_plot(self, "KPSS", ax)


class KPSSTest(BaseTest):
    """KPSS (Kwiatkowski-Phillips-Schmidt-Shin) stationarity test.

    Wraps :func:`statsmodels.tsa.stattools.kpss`.

    .. important::
       **Reversed hypotheses** — KPSS tests the null of **stationarity**,
       unlike ADF / PP which test the null of a **unit root**.

       - H0: The series is stationary (level or trend stationary).
       - H1: The series has a unit root (is non-stationary).

    Parameters
    ----------
    data : array-like
        The time series to test.
    trend : str, optional
        ``"c"`` — level stationary (default). The series is stationary
        around a constant mean.
        ``"ct"`` — trend stationary. The series is stationary around
        a deterministic trend.
    lags : int, optional
        Number of Newey-West lags. If ``None`` (default), automatic
        selection via ``"legacy"`` method is used.
    nlags : str, optional
        Method for automatic lag selection when *lags* is ``None``:
        ``"legacy"`` (default) or ``"auto"``.

    Attributes
    ----------
    result_ : KPSSTestResult
        Full test results after calling :meth:`fit`.

    Examples
    --------
    KPSS reverses the ADF/PP null: a large p-value does not reject
    stationarity.

    >>> import numpy as np
    >>> from Ts.TsTests import KPSSTest
    >>> data = np.random.default_rng(42).normal(size=100)
    >>> result = KPSSTest(data, trend="c", lags=3).fit()
    >>> result.lags
    3
    """

    _VALID_TRENDS = ("c", "ct")

    def __init__(
        self,
        data,
        trend: str = "c",
        lags: int | None = None,
        nlags: str = "legacy",
    ):
        self.data = _clean_1d(data)
        self.trend = validate_choice("trend", trend, self._VALID_TRENDS)
        self.lags = lags
        self.nlags = validate_choice("nlags", nlags, ("legacy", "auto"))
        self.result_: KPSSTestResult | None = None

    def fit(self) -> KPSSTestResult:
        """Run the KPSS test.

        Returns
        -------
        KPSSTestResult

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import KPSSTest
        >>> result = KPSSTest(np.random.default_rng(42).normal(size=80)).fit()
        >>> 0.0 <= result.pvalue <= 1.0
        True
        """
        from statsmodels.tsa.stattools import kpss

        kwargs = {"regression": self.trend}
        if self.lags is not None:
            kwargs["nlags"] = self.lags
        else:
            kwargs["nlags"] = self.nlags

        kpss_result = kpss(self.data, **kwargs)

        statistic = float(kpss_result[0])
        pvalue = float(kpss_result[1])
        used_lags = int(kpss_result[2])
        crit_vals = {str(k): float(v) for k, v in kpss_result[3].items()}

        self.result_ = KPSSTestResult(
            statistic=statistic,
            pvalue=pvalue,
            lags=used_lags,
            nobs=len(self.data),
            trend=self.trend,
            critical_values=crit_vals,
        )
        return self.result_

"""ARIMA-residual outlier detection for additive (AO), level-shift (LS),
and innovational (IO) outliers.

The procedure follows the classic iterative method: a fitted ARIMA model
produces one-step-ahead prediction errors ``e_t``; every candidate outlier
is estimated by an intercept-free regression of ``e_t`` on its whitened
footprint regressor, and the most significant candidate is removed from the
residuals before the scan repeats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._base import BaseTestResult
from ._utils import _as_1d_float

_OUTLIER_TYPES = ("AO", "LS", "IO")


# ---------------------------------------------------------------------------
# Whitening weights
# ---------------------------------------------------------------------------


def _max_lag(component):
    """Return the largest lag of an integer or active-lags component."""
    if isinstance(component, (int, np.integer)):
        return int(component)
    return max(component, default=0)


def _ar_ma_coefficients(params, order, seasonal_order):
    """Return complete AR and MA coefficient arrays from fitted parameters.

    The arrays follow the ``1 - phi_1 B - ...`` convention of the whitening
    operator. Seasonal polynomials and the regular / seasonal differencing
    operators are expanded into the same lag space.
    """
    p, d, q = order
    P, D, Q, s = seasonal_order

    ar = np.zeros(_max_lag(p) + 1)
    ar[0] = 1.0
    for lag in range(1, len(ar)):
        ar[lag] = params.get(f"ar.L{lag}", 0.0)
    ma = np.zeros(_max_lag(q) + 1)
    ma[0] = 1.0
    for lag in range(1, len(ma)):
        ma[lag] = params.get(f"ma.L{lag}", 0.0)

    P_lags = _max_lag(P)
    Q_lags = _max_lag(Q)
    if P_lags:
        seasonal_ar = np.zeros(P_lags * s + 1)
        seasonal_ar[0] = 1.0
        for k in range(1, P_lags + 1):
            seasonal_ar[k * s] = params.get(f"ar.S.L{k}", 0.0)
        ar = np.convolve(ar, seasonal_ar)
    if Q_lags:
        seasonal_ma = np.zeros(Q_lags * s + 1)
        seasonal_ma[0] = 1.0
        for k in range(1, Q_lags + 1):
            seasonal_ma[k * s] = params.get(f"ma.S.L{k}", 0.0)
        ma = np.convolve(ma, seasonal_ma)

    for _ in range(d):
        ar = np.convolve(ar, np.array([1.0, -1.0]))
    if D:
        seasonal_difference = np.zeros(s + 1)
        seasonal_difference[0] = 1.0
        seasonal_difference[s] = -1.0
        for _ in range(D):
            ar = np.convolve(ar, seasonal_difference)
    return ar, ma


def _pi_weights(ar, ma, n):
    """Return the first ``n`` pi-weights of the whitening operator.

    ``pi(B) = ar(B) / ma(B)`` expands as
    ``pi(B) = 1 - pi_1 B - pi_2 B^2 - ...`` so that ``pi(B) z_t = a_t``.
    With ``ar(B) = 1 - sum ar_m B^m`` and ``ma(B) = 1 + sum ma_m B^m``
    the coefficients satisfy ``pi_m = ar_m + ma_m - sum_j pi_j ma_{m-j}``.
    """
    pi = np.zeros(n)
    pi[0] = 1.0
    active_ma = [lag for lag in range(1, len(ma)) if ma[lag] != 0.0]
    if not active_ma:
        length = min(n, len(ar))
        pi[:length] = ar[:length]
        return pi
    for m in range(1, n):
        acc = ar[m] if m < len(ar) else 0.0
        for lag in active_ma:
            if lag < m:
                acc -= pi[m - lag] * ma[lag]
        pi[m] = acc + (ma[m] if m < len(ma) else 0.0)
    return pi


def _c_weights(pi):
    """Return the c-weights of ``c(B) = pi(B) / (1 - B)``.

    ``c_1 = pi_1 - 1`` and ``c_m = c_{m-1} + pi_m``; the zero lag is unused.
    """
    c = np.cumsum(pi) - 2.0
    c[0] = 0.0
    return c


# ---------------------------------------------------------------------------
# Footprints and per-type scanning
# ---------------------------------------------------------------------------


def _whitened_regressor(i, weights, n):
    """Return the whitened outlier regressor for candidate time ``i``.

    The regressor has value 1 at time ``i`` and value ``-weights[j]`` at
    time ``i + j``, truncated at the sample end.
    """
    regressor = np.zeros(n)
    regressor[i] = 1.0
    available = n - i - 1
    if available > 0:
        regressor[i + 1 :] = -weights[1 : available + 1]
    return regressor


def _outlier_footprint(omega, weights, i, n):
    """Return the residual footprint of one detected outlier.

    The footprint is ``omega`` at time ``i`` and ``-omega * weights[j]`` at
    time ``i + j``.  Passing an all-zero ``weights`` yields the pure-pulse
    innovational-outlier footprint.
    """
    footprint = np.zeros(n)
    footprint[i] = omega
    available = n - i - 1
    if available > 0:
        footprint[i + 1 :] = -omega * weights[1 : available + 1]
    return footprint


def _scan_outlier_type(e, weights, sigma):
    """Scan one weighted outlier type across every candidate time.

    Returns ``(omega, L, standard_error)`` arrays.  The estimates use the
    intercept-free regression ``e = omega * W + a`` whose regressor is
    truncated at the sample end, so the normalization
    ``1 + sum weights^2`` stays finite even when the weights do not decay.
    """
    n = len(e)
    omega = np.zeros(n)
    lstat = np.zeros(n)
    standard_error = np.zeros(n)
    for i in range(n):
        available = n - i - 1
        if available > 0:
            w = weights[1 : available + 1]
            numerator = e[i] - float(np.dot(w, e[i + 1 :]))
            denominator = 1.0 + float(np.dot(w, w))
        else:
            numerator = e[i]
            denominator = 1.0
        omega[i] = numerator / denominator
        scale = sigma * np.sqrt(denominator)
        lstat[i] = numerator / scale
        standard_error[i] = scale / denominator
    return omega, lstat, standard_error


def _scan_io(e, sigma):
    """Scan innovational outliers: the raw residual is the estimator."""
    n = len(e)
    return e.copy(), e / sigma, np.full(n, sigma)


def _initial_skip(order, seasonal_order):
    """Return the number of leading residuals excluded from candidate times.

    One-step-ahead errors early in the sample are contaminated by the
    state-space initialization (the first prediction relies on the prior
    mean), so those times are excluded from detection.  The skip covers
    the model memory depth including differencing and seasonal lags.
    """
    p, d, q = order
    P, D, Q, s = seasonal_order
    memory = (
        max(_max_lag(p), _max_lag(q))
        + d
        + _max_lag(P) * s
        + _max_lag(Q) * s
        + D * s
    )
    return max(1, memory)


def _full_scan(e, pi, c, sigma):
    """Return stacked (L, omega, standard_error) arrays over all types."""
    omega_ao, l_ao, se_ao = _scan_outlier_type(e, pi, sigma)
    omega_ls, l_ls, se_ls = _scan_outlier_type(e, c, sigma)
    omega_io, l_io, se_io = _scan_io(e, sigma)
    return (
        np.stack([l_ao, l_ls, l_io]),
        np.stack([omega_ao, omega_ls, omega_io]),
        np.stack([se_ao, se_ls, se_io]),
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class OutlierDetectorResult(BaseTestResult):
    """Result of ARIMA-residual AO/LS/IO outlier detection.

    Parameters
    ----------
    statistic, pvalue, lags, nobs, residuals : see BaseTestResult
        ``statistic`` is the maximum |L| of the final scan; residuals are
        the original one-step-ahead prediction errors.
    events : pandas.DataFrame
        Detected events with columns ``time``, ``type`` (one of
        ``"AO"``, ``"LS"``, ``"IO"``), ``omega``, ``standard_error``, and
        ``L``, in detection order.
    pi_weights : numpy.ndarray
        Whitening-operator coefficients ``pi(B) = 1 - pi_1 B - ...``.
    c_weights : numpy.ndarray
        Level-shift weights ``c(B) = pi(B) / (1 - B)``.
    adjusted_residuals : numpy.ndarray
        Residuals after subtracting every detected footprint.
    l_statistics : pandas.DataFrame
        Final-scan L statistics for every candidate time and type.
    scan_history : pandas.DataFrame
        One row per detection round with the winning event and round sigma.
    sigma : float
        Residual standard deviation of the final round.
    critical_value : float
        Detection threshold applied to |L|.
    model : SARIMAXResult
        ARIMA model fitted to the series before detection.

    Examples
    --------
    >>> import numpy as np
    >>> from Ts.TsTests import OutlierDetector
    >>> from Ts.TsSims import simulate_sarima
    >>> data = simulate_sarima(n=120, order=(1, 0, 0), ar=[0.7], seed=42).data
    >>> data = data.copy()
    >>> data[60] += 6.0
    >>> result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)
    >>> list(result.events.columns)
    ['time', 'type', 'omega', 'standard_error', 'L']
    """

    events: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            [], columns=["time", "type", "omega", "standard_error", "L"]
        )
    )
    pi_weights: np.ndarray | None = None
    c_weights: np.ndarray | None = None
    adjusted_residuals: np.ndarray | None = None
    l_statistics: pd.DataFrame | None = None
    scan_history: pd.DataFrame | None = None
    sigma: float | None = None
    critical_value: float | None = None
    model: object | None = None

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "  ARIMA Outlier Detection",
            "=" * 50,
            f"  Critical Value  : {self.critical_value}",
            f"  Events Detected : {len(self.events)}",
            f"  Max |L| (final) : {self.statistic:.6f}",
            f"  Final Residual SD: {self.sigma:.6f}",
        ]
        if len(self.events):
            lines.extend(["", "Detected Events:", str(self.events.to_string(index=False))])
        return "\n".join(lines)

    def summary(self) -> str:
        """Return a formatted detection report (identical to ``str``).

        Returns
        -------
        str
            Text report with the critical value, detected events, the
            final maximum |L|, and the final residual standard deviation.

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsTests import OutlierDetector
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=120, order=(1, 0, 0), ar=[0.7], seed=42).data
        >>> data = data.copy()
        >>> data[60] += 6.0
        >>> report = OutlierDetector(order=(1, 0, 0)).fit_detect(data).summary()
        >>> "ARIMA Outlier Detection" in report
        True
        """
        return str(self)

    def plot(self):
        """Plot the residuals with detected events and the L statistics.

        Returns
        -------
        fig, axes : tuple
            Two stacked axes: the original one-step-ahead prediction
            errors with the detected outliers marked, and the final-scan
            L statistics of every type with the critical-value line.

        Examples
        --------
        >>> from Ts.TsTests import OutlierDetector
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=120, order=(1, 0, 0), ar=[0.7], seed=42).data
        >>> data = data.copy()
        >>> data[60] += 6.0
        >>> result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)
        >>> fig, axes = result.plot()
        """
        import matplotlib.pyplot as plt

        from Ts.TsPlots.style import DEFAULT_PALETTE, FIGSIZE, style_axes

        marker = {"AO": "o", "LS": "^", "IO": "s"}
        positions = np.arange(len(self.residuals))
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=FIGSIZE, sharex=True)

        residuals_ax = axes[0]
        residuals_ax.plot(positions, self.residuals, color=DEFAULT_PALETTE[0])
        if self.l_statistics is not None and len(self.events):
            index = self.l_statistics.index
            event_positions = index.get_indexer(self.events["time"])
            for event_type, color in zip(
                ("AO", "LS", "IO"), DEFAULT_PALETTE[1:4], strict=True
            ):
                selected = event_positions[
                    (self.events["type"] == event_type).to_numpy()
                ]
                if len(selected):
                    residuals_ax.scatter(
                        selected,
                        np.asarray(self.residuals)[selected],
                        marker=marker[event_type],
                        color=color,
                        s=60,
                        label=event_type,
                    )
        residuals_ax.axhline(0.0, color="black", linewidth=0.8)
        residuals_ax.set_ylabel("residuals")
        residuals_ax.legend(frameon=False, ncols=3)
        residuals_ax.set_title("ARIMA residual outlier detection")

        if self.l_statistics is not None:
            statistics_ax = axes[1]
            for event_type, color in zip(
                ("AO", "LS", "IO"), DEFAULT_PALETTE[1:4], strict=True
            ):
                statistics_ax.plot(
                    positions,
                    self.l_statistics[event_type],
                    color=color,
                    label=event_type,
                )
            threshold = self.critical_value or 3.5
            statistics_ax.axhline(
                threshold, color="black", linestyle="--", linewidth=0.8
            )
            statistics_ax.axhline(
                -threshold, color="black", linestyle="--", linewidth=0.8
            )
            statistics_ax.set_xlabel("time")
            statistics_ax.set_ylabel("L statistic")
            statistics_ax.legend(frameon=False, ncols=3)

        for ax in axes:
            style_axes(ax, grid=False)
        fig.tight_layout(pad=1.5)
        return fig, axes


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class OutlierDetector:
    """Detect additive, level-shift, and innovational outliers in an ARIMA.

    Parameters
    ----------
    order : tuple
        ``(p, d, q)`` non-seasonal ARIMA order passed to
        :class:`Ts.TsModels.SARIMAX`.
    seasonal_order : tuple
        ``(P, D, Q, s)`` seasonal order. Default ``(0, 0, 0, 0)``.
    trend : str
        Deterministic trend specification ``"n"``, ``"c"``, or ``"t"``.
        Default ``"c"``.
    critical_value : float, default 3.5
        Detection threshold for |L|. Detection stops when no candidate
        exceeds it.
    max_events : int or None, default None
        Upper bound on the number of detected events. ``None`` scans until
        no candidate exceeds the critical value.

    Attributes
    ----------
    result_ : OutlierDetectorResult or None
        Detection result populated by :meth:`fit_detect`.

    Examples
    --------
    >>> from Ts.TsTests import OutlierDetector
    >>> from Ts.TsSims import simulate_sarima
    >>> data = simulate_sarima(n=120, order=(1, 0, 0), ar=[0.7], seed=42).data
    >>> data = data.copy()
    >>> data[60] += 6.0
    >>> result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)
    >>> result.events.iloc[0]["type"]
    'AO'
    """

    def __init__(
        self,
        order=(0, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        critical_value=3.5,
        max_events=None,
    ):
        if isinstance(critical_value, (bool, np.bool_)) or not isinstance(
            critical_value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("critical_value must be a positive number")
        if critical_value <= 0.0:
            raise ValueError("critical_value must be positive")
        if max_events is not None:
            if isinstance(max_events, (bool, np.bool_)) or not isinstance(
                max_events,
                (int, np.integer),
            ):
                raise TypeError("max_events must be a positive integer or None")
            if max_events < 1:
                raise ValueError("max_events must be a positive integer or None")
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.critical_value = float(critical_value)
        self.max_events = None if max_events is None else int(max_events)
        self.result_: OutlierDetectorResult | None = None

    @staticmethod
    def _time_labels(index, burn, length):
        """Return time labels aligned with the usable residuals."""
        if index is not None:
            return index[burn : burn + length]
        return np.arange(burn, burn + length)

    def fit_detect(self, series) -> OutlierDetectorResult:
        """Fit the ARIMA model and iteratively detect outliers.

        Parameters
        ----------
        series : array-like or pandas.Series
            Observed time series with at least 10 finite observations. A
            Series index is used as the event time labels.

        Returns
        -------
        OutlierDetectorResult

        Raises
        ------
        ValueError
            If the series is invalid or the fitted residual standard
            deviation is non-positive.

        Examples
        --------
        >>> from Ts.TsTests import OutlierDetector
        >>> from Ts.TsSims import simulate_sarima
        >>> data = simulate_sarima(n=120, order=(1, 0, 0), ar=[0.7], seed=42).data
        >>> data = data.copy()
        >>> data[60] += 6.0
        >>> result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)
        >>> result.events.iloc[0]["time"]
        60
        """
        values = _as_1d_float(series)
        index = series.index if isinstance(series, pd.Series) else None
        if len(values) < 10:
            raise ValueError(f"Need at least 10 observations, got {len(values)}")
        if not np.all(np.isfinite(values)):
            raise ValueError("series must contain only finite values")

        from Ts.TsModels import SARIMAX

        model = SARIMAX(
            values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            missing="raise",
        ).fit()

        residuals = np.asarray(model.residuals, dtype=float)
        if residuals.ndim != 1 or len(residuals) == 0:
            raise ValueError("fitted model produced no usable residuals")
        ar, ma = _ar_ma_coefficients(
            model.params,
            model._order,
            model._seasonal_order,
        )
        pi = _pi_weights(ar, ma, len(residuals))
        c = _c_weights(pi)
        e = residuals.copy()
        sigma = float(np.sqrt(np.mean(e**2)))
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("residual standard deviation must be positive and finite")

        burn = int(model.likelihood_burn)
        labels = self._time_labels(index, burn, len(residuals))
        skip = _initial_skip(model._order, model._seasonal_order)

        event_rows = []
        scan_rows = []
        l_matrix = None
        omega_matrix = None
        se_matrix = None
        while True:
            l_matrix, omega_matrix, se_matrix = _full_scan(e, pi, c, sigma)
            flat = np.abs(l_matrix)
            if skip > 0:
                flat[:, :skip] = -np.inf
            maximum = float(flat.max())
            kind_index, position = np.unravel_index(int(np.argmax(flat)), l_matrix.shape)
            if maximum <= self.critical_value:
                break
            if self.max_events is not None and len(event_rows) >= self.max_events:
                break
            kind = _OUTLIER_TYPES[kind_index]
            omega = float(omega_matrix[kind_index, position])
            standard_error = float(se_matrix[kind_index, position])
            lstat = float(l_matrix[kind_index, position])
            event_rows.append(
                {
                    "time": labels[position],
                    "type": kind,
                    "omega": omega,
                    "standard_error": standard_error,
                    "L": lstat,
                }
            )
            scan_rows.append(
                {
                    "iteration": len(scan_rows) + 1,
                    "time": labels[position],
                    "type": kind,
                    "omega": omega,
                    "standard_error": standard_error,
                    "L": lstat,
                }
            )
            if kind == "IO":
                footprint = np.zeros(len(e))
                footprint[position] = omega
            else:
                weights = pi if kind == "AO" else c
                footprint = _outlier_footprint(omega, weights, position, len(e))
            e = e - footprint
            sigma = float(np.sqrt(np.mean(e**2)))

        events = pd.DataFrame(
            event_rows,
            columns=["time", "type", "omega", "standard_error", "L"],
        )
        scan_history = pd.DataFrame(
            scan_rows,
            columns=["iteration", "time", "type", "omega", "standard_error", "L"],
        )
        l_frame = pd.DataFrame(
            np.column_stack([l_matrix[row] for row in range(3)]),
            index=labels,
            columns=list(_OUTLIER_TYPES),
        )
        self.result_ = OutlierDetectorResult(
            statistic=maximum,
            pvalue=None,
            lags=None,
            nobs=len(residuals),
            residuals=residuals.copy(),
            events=events,
            pi_weights=pi.copy(),
            c_weights=c.copy(),
            adjusted_residuals=e.copy(),
            l_statistics=l_frame,
            scan_history=scan_history,
            sigma=sigma,
            critical_value=self.critical_value,
            model=model,
        )
        return self.result_

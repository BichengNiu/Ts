"""Base class for all simulation result containers in TsSims.

Provides :class:`BaseSimResult` — a common dataclass that defines the
uniform interface shared by :class:`SimSARIMAResult`, :class:`SimGARCHResult`,
and :class:`SimTSDSResult`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BaseSimResult:
    """Common result container for all TsSims simulation output.

    Subclasses override :meth:`summary` and :meth:`plot` to provide
    model-specific formatting and visualisation.

    Parameters
    ----------
    data : np.ndarray
        Generated time series of length *n*.
    residuals : np.ndarray
        Innovations (or residuals) used in generation.
    params : dict
        All parameters used for simulation.

    Examples
    --------
    Concrete simulators return subclasses with this shared interface.

    >>> from Ts.TsSims import BaseSimResult, simulate_sarima
    >>> result = simulate_sarima(n=20, seed=42)
    >>> isinstance(result, BaseSimResult)
    True
    >>> result.get_data().shape
    (20,)
    """

    data: np.ndarray
    residuals: np.ndarray
    params: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Concrete methods — shared across all simulation result types
    # ------------------------------------------------------------------

    def get_data(self) -> pd.Series | pd.DataFrame:
        """Return the generated time series.

        Returns a :class:`pd.Series` for univariate data or a
        :class:`pd.DataFrame` with columns ``"y0"``, ``"y1"``, ...
        for multivariate data.

        Returns
        -------
        pd.Series or pd.DataFrame

        Examples
        --------
        >>> from Ts.TsSims import simulate_sarima
        >>> series = simulate_sarima(n=20, seed=42).get_data()
        >>> (series.name, len(series))
        ('y', 20)
        """
        if self.data.ndim == 1:
            return pd.Series(self.data, name="y")
        cols = [f"y{i}" for i in range(self.data.shape[1])]
        return pd.DataFrame(self.data, columns=cols)

    def get_params(self) -> dict:
        """Return a deep copy of the simulation parameters.

        Returns
        -------
        dict

        Examples
        --------
        >>> from Ts.TsSims import simulate_sarima
        >>> result = simulate_sarima(n=20, order=(1, 0, 0), seed=42)
        >>> result.get_params()["order"]
        (1, 0, 0)
        """
        return copy.deepcopy(self.params)

    # ------------------------------------------------------------------
    # Override points — subclasses provide model-specific output
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted parameter summary string.

        Subclasses override this to include model-type headers and
        coefficient detail.

        Returns
        -------
        str

        Examples
        --------
        >>> from Ts.TsSims import BaseSimResult
        >>> result = BaseSimResult(data=[1, 2], residuals=[0, 0])
        >>> result.summary()
        'BaseSimResult(n=2)'
        """
        return f"{type(self).__name__}(n={len(self.data)})"

    def plot(self, title=None, **kwargs):
        """Plot the generated time series.

        Subclasses override this for model-specific visualisation (e.g.
        multi-panel volatility plots).

        Parameters
        ----------
        title : str, optional
            Chart title.
        **kwargs
            Forwarded to the underlying plotting function.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes or numpy.ndarray of Axes

        Examples
        --------
        >>> import numpy as np
        >>> from Ts.TsSims import BaseSimResult
        >>> result = BaseSimResult(np.arange(5.0), np.zeros(5))
        >>> fig, ax = result.plot(title="Simulation")
        >>> ax.get_title()
        'Simulation'
        """
        from Ts.TsPlots import plot_series

        if title is None:
            title = f"{type(self).__name__} Simulation"
        kwargs.setdefault("facet", False)
        fig, ax = plot_series(
            self.data,
            title=title,
            ytitle="Value",
            xtitle="Time",
            show_legend=False,
            **kwargs,
        )
        return fig, ax

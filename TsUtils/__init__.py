"""Time-series preprocessing and identification diagnostics.

The package owns decomposition, data-cleaning, and pre-fit diagnostic helpers
without estimating predictive models.
"""

from ._difference import difference
from ._boxcox import BoxCoxResult, boxcox
from ._eacf import EACFResult, eacf
from ._interpolation import InterpolationResult, interpolate_missing
from ._seasonal_dummies import seasonal_dummies
from ._stl import STL, STLResult
from ._summary import TimeSeriesSummary

__all__ = [
    "STL",
    "BoxCoxResult",
    "EACFResult",
    "InterpolationResult",
    "STLResult",
    "TimeSeriesSummary",
    "boxcox",
    "difference",
    "eacf",
    "interpolate_missing",
    "seasonal_dummies",
]

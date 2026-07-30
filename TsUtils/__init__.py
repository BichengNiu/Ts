"""Time-series preprocessing and identification diagnostics.

The package owns decomposition, data-cleaning, and pre-fit diagnostic helpers
without estimating predictive models.
"""

from ._difference import difference
from ._eacf import EACFResult, eacf
from ._interpolation import InterpolationResult, interpolate_missing
from ._stl import STL, STLResult
from ._summary import TimeSeriesSummary

__all__ = [
    "STL",
    "EACFResult",
    "InterpolationResult",
    "STLResult",
    "TimeSeriesSummary",
    "difference",
    "eacf",
    "interpolate_missing",
]
